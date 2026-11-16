#include "tile/dialect.h"
#include "tile/analysis.h"
#include "tile/module.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace mlir;

namespace carbon {
namespace {

Type scalar(const std::string& name) {
    auto& context = ir::context();
    if (name == "f16") return Float16Type::get(&context);
    if (name == "bf16") return BFloat16Type::get(&context);
    if (name == "f32") return Float32Type::get(&context);
    if (name == "i32") return IntegerType::get(&context, 32);
    if (name == "ui8") return IntegerType::get(&context, 8, IntegerType::Unsigned);
    if (name == "si8") return IntegerType::get(&context, 8, IntegerType::Signed);
    if (name == "ui16") return IntegerType::get(&context, 16, IntegerType::Unsigned);
    if (name == "si16") return IntegerType::get(&context, 16, IntegerType::Signed);
    if (name == "i1") return IntegerType::get(&context, 1);
    if (name == "index") return IndexType::get(&context);
    throw nb::value_error(("unsupported element type " + name).c_str());
}

std::string elementName(Type type) {
    if (type.isF16()) return "f16";
    if (type.isBF16()) return "bf16";
    if (type.isF32()) return "f32";
    if (ir::storageInteger(type))
        return (type.isSignedInteger() ? "si" : "ui") + std::to_string(type.getIntOrFloatBitWidth());
    if (type.isInteger(32)) return "i32";
    if (type.isInteger(1)) return "i1";
    if (type.isIndex()) return "index";
    throw nb::value_error("unsupported element type");
}

// A staged value's type as Python reads it: (kind, element, layout, shape, strides, offset, extent, address space).
nb::tuple describe(Type type) {
    auto none = nb::none();
    if (auto tile = dyn_cast<ir::TileType>(type))
        return nb::make_tuple("tile", elementName(tile.getElementType()), tile.getLayout(), none, none, 0, 0, "");
    if (auto view = dyn_cast<ir::ViewType>(type))
        return nb::make_tuple("view", elementName(view.getElementType()), none, view.getShape().asArrayRef().vec(),
                              view.getStrides().asArrayRef().vec(), view.getOffsetBytes(), view.getExtentBytes(),
                              view.getAddressSpace().str());
    if (auto buffer = dyn_cast<MemRefType>(type))
        return nb::make_tuple("buffer", elementName(buffer.getElementType()), none, none, none, 0, 0, "device");
    return nb::make_tuple("scalar", elementName(type), none, none, none, 0, 0, "");
}

Location location(nb::handle where) {
    auto& context = ir::context();
    if (where.is_none())
        return UnknownLoc::get(&context);
    auto [file, line, column] = nb::cast<std::tuple<std::string, unsigned, unsigned>>(where);
    return FileLineColLoc::get(&context, file, line, column);
}

Attribute attribute(nb::handle value) {
    auto& context = ir::context();
    Builder builder(&context);
    if (nb::isinstance<nb::bool_>(value))
        return builder.getBoolAttr(nb::cast<bool>(value));
    if (nb::isinstance<nb::int_>(value))
        return builder.getI64IntegerAttr(nb::cast<int64_t>(value));
    if (nb::isinstance<nb::str>(value))
        return builder.getStringAttr(nb::cast<std::string>(value));
    if (nb::isinstance<nb::tuple>(value))
        return builder.getDenseI64ArrayAttr(nb::cast<std::vector<int64_t>>(value));
    if (nb::isinstance<nb::list>(value))
        return builder.getDenseI32ArrayAttr(nb::cast<std::vector<int32_t>>(value));
    throw nb::type_error("unsupported attribute value");
}

SmallVector<Type> inferred(OperationState& state) {
    auto name = state.name;
    auto info = name.getRegisteredInfo();
    auto* inference = info ? info->getInterface<InferTypeOpInterface>() : nullptr;
    if (!inference)
        throw nb::value_error((name.getStringRef() + " needs explicit result types").str().c_str());
    auto& context = ir::context();
    ir::Diagnostics diagnostics(context, false);
    auto attributes = state.attributes.getDictionary(&context);
    auto bytes = name.getOpPropertyByteSize();
    auto storage = std::make_unique<char[]>(bytes);
    OpaqueProperties properties(bytes ? storage.get() : nullptr);
    if (bytes)
        name.initOpProperties(properties, nullptr);
    auto invalid = [&] { return emitError(state.location); };
    SmallVector<Type> types;
    bool valid = (!bytes || succeeded(name.setOpPropertiesFromAttribute(name, properties, attributes, invalid))) &&
                 succeeded(inference->inferReturnTypes(&context, state.location, state.operands, attributes,
                                                       properties, state.regions, types));
    if (bytes)
        name.destroyOpProperties(properties);
    if (!valid)
        throw nb::value_error(diagnostics.text.c_str());
    return types;
}

// Branch blocks remain detached until their parent operation is created.
class Staging {
  public:
    Staging(int64_t threads, std::optional<int64_t> maxGrid, bool indirect)
        : module_(ModuleOp::create(UnknownLoc::get(&ir::context()))), builder_(&ir::context()) {
        auto& context = ir::context();
        function_ = func::FuncOp::create(UnknownLoc::get(&context), "carbon_kernel",
                                         FunctionType::get(&context, {}, {}));
        function_->setAttr("carbon.threads", builder_.getI64IntegerAttr(threads));
        function_->setAttr("carbon.layout_ir", builder_.getI64IntegerAttr(2));
        if (maxGrid)
            function_->setAttr("carbon.max_grid", builder_.getI64IntegerAttr(*maxGrid));
        if (indirect)
            function_->setAttr("carbon.indirect", builder_.getUnitAttr());
        module_->push_back(function_);
        entry_ = function_.addEntryBlock();
        builder_.setInsertionPointToEnd(entry_);
    }

    Value argument(Type type) {
        auto index = function_.getNumArguments();
        function_.insertArgument(index, type, {}, UnknownLoc::get(&ir::context()));
        return function_.getArgument(index);
    }

    void setRange(Value value, int64_t minimum, int64_t maximum) {
        contract(value, "carbon.range", builder_.getDenseI64ArrayAttr({minimum, maximum}));
    }

    void setMinimumBytes(Value value, int64_t bytes) {
        contract(ir::bindingRoot(value), "carbon.min_bytes", builder_.getI64IntegerAttr(bytes));
    }

    void checkStages(const std::vector<std::vector<Value>>& stages) {
        if (stages.size() != 2 || stages[0].empty() || stages[0].size() != stages[1].size())
            throw nb::value_error("double buffering requires two equally sized nonempty stages");
        llvm::DenseSet<Value> allocations;
        for (const auto& stage : stages) {
            for (size_t index = 0; index < stage.size(); ++index) {
                auto value = stage[index];
                auto allocation = ir::scratchAllocation(ir::viewRoot(value));
                if (!allocation || allocation->getParentOfType<func::FuncOp>() != function_)
                    throw nb::value_error("pipeline stages require scratch allocated in the current kernel");
                if (value.getType() != stages[0][index].getType())
                    throw nb::value_error("corresponding pipeline buffers must have identical types");
                if (!allocations.insert(allocation.getResult()).second)
                    throw nb::value_error("pipeline buffers must use distinct scratch allocations");
            }
        }
    }

    std::vector<Value> create(const std::string& name, const std::vector<Value>& operands,
                              std::optional<std::vector<Type>> results, nb::dict attributes, nb::handle where,
                              bool atEntry) {
        OperationState state(location(where), name);
        state.addOperands(operands);
        for (auto [key, value] : attributes)
            state.addAttribute(nb::cast<std::string>(key), attribute(value));
        if (results)
            state.addTypes(*results);
        else
            state.addTypes(inferred(state));
        OpBuilder::InsertionGuard guard(builder_);
        if (atEntry)
            builder_.setInsertionPoint(entry_, firstOperation());
        auto* operation = builder_.create(state);
        ir::Diagnostics diagnostics(ir::context(), false);
        if (failed(mlir::verify(operation, false)))
            throw nb::value_error(diagnostics.text.c_str());
        return {operation->result_begin(), operation->result_end()};
    }

    Value constant(nb::handle value, Type type, nb::handle where) {
        TypedAttr literal;
        if (isa<FloatType>(type))
            literal = builder_.getFloatAttr(type, nb::cast<double>(value));
        else if (nb::isinstance<nb::bool_>(value))
            literal = builder_.getIntegerAttr(type, nb::cast<bool>(value));
        else
            literal = builder_.getIntegerAttr(
                type, APInt(isa<IndexType>(type) ? 64 : type.getIntOrFloatBitWidth(), nb::cast<uint64_t>(value)));
        return arith::ConstantOp::create(builder_, location(where), literal);
    }

    // Exact bits of an f32 constant, NaN payloads included.
    Value bits(uint32_t word, nb::handle where) {
        auto type = Float32Type::get(&ir::context());
        return arith::ConstantOp::create(builder_, location(where),
                                         FloatAttr::get(type, APFloat(APFloat::IEEEsingle(), APInt(32, word))));
    }

    Value compare(const std::string& predicate, Value left, Value right, nb::handle where) {
        if (isa<FloatType>(left.getType())) {
            auto kind = arith::symbolizeCmpFPredicate(predicate);
            if (!kind)
                throw nb::value_error("unsupported floating-point comparison");
            return arith::CmpFOp::create(builder_, location(where), *kind, left, right);
        }
        auto kind = arith::symbolizeCmpIPredicate(predicate);
        if (!kind)
            throw nb::value_error("unsupported integer comparison");
        return arith::CmpIOp::create(builder_, location(where), *kind, left, right);
    }

    Block* enter(Block* block) {
        if (!block) {
            arms_.push_back(std::make_unique<Block>());
            block = arms_.back().get();
        }
        saved_.push_back(builder_.saveInsertionPoint());
        builder_.setInsertionPointToEnd(block);
        return block;
    }

    void leave() {
        builder_.restoreInsertionPoint(saved_.back());
        saved_.pop_back();
    }

    std::vector<Value> branch(Value condition, const std::vector<Type>& results, Block* then, Block* otherwise,
                              const std::vector<Value>& thenValues, const std::vector<Value>& otherwiseValues,
                              nb::handle where) {
        auto at = location(where);
        if (then == otherwise || !owned(then) || !owned(otherwise))
            throw nb::value_error("a branch arm was used twice");
        auto conditional = scf::IfOp::create(builder_, at, TypeRange(results), condition, true);
        for (auto [region, block, values] : {std::tuple(&conditional.getThenRegion(), then, &thenValues),
                                             std::tuple(&conditional.getElseRegion(), otherwise, &otherwiseValues)}) {
            region->getBlocks().clear();
            region->push_back(release(block));
            auto end = OpBuilder::atBlockEnd(block);
            scf::YieldOp::create(end, at, *values);
        }
        return {conditional.result_begin(), conditional.result_end()};
    }

    // A loop whose body is the insertion point until `close`; returns the counter and the carried values.
    std::vector<Value> loop(Value lower, Value upper, Value step, const std::vector<Value>& initial, nb::handle where) {
        auto loop = scf::ForOp::create(builder_, location(where), lower, upper, step, initial);
        auto* body = loop.getBody();
        if (body->mightHaveTerminator())
            body->getTerminator()->erase();
        loops_.push_back(loop);
        saved_.push_back(builder_.saveInsertionPoint());
        builder_.setInsertionPointToEnd(body);
        return {body->args_begin(), body->args_end()};
    }

    std::vector<Value> close(const std::vector<Value>& yielded, nb::handle where) {
        auto loop = loops_.back();
        loops_.pop_back();
        scf::YieldOp::create(builder_, location(where), yielded);
        leave();
        return {loop.result_begin(), loop.result_end()};
    }

    void finish() { func::ReturnOp::create(builder_, UnknownLoc::get(&ir::context())); }

    std::string fingerprint() {
        std::string bytes;
        llvm::raw_string_ostream stream(bytes);
        if (failed(writeBytecodeToFile(*module_, stream)))
            throw std::runtime_error("cannot serialize the staged kernel");
        auto digest = llvm::SHA256::hash(llvm::arrayRefFromStringRef(bytes));
        return llvm::toHex(digest, true);
    }

    ModuleOp module() { return *module_; }

  private:
    void contract(Value value, StringRef name, Attribute attribute) {
        auto argument = dyn_cast<BlockArgument>(value);
        if (!argument || argument.getOwner() != entry_)
            throw nb::value_error("contracts apply to kernel arguments");
        function_.setArgAttr(argument.getArgNumber(), name, attribute);
        ir::Diagnostics diagnostics(ir::context(), false);
        if (failed(ir::verifyContracts(function_, argument.getArgNumber())))
            throw nb::value_error(diagnostics.text.c_str());
    }

    Block::iterator firstOperation() {
        auto at = entry_->begin();
        while (at != entry_->end() && isa<ir::ScratchOp>(*at))
            ++at;
        return at;
    }

    bool owned(Block* block) {
        return llvm::any_of(arms_, [&](auto& arm) { return arm.get() == block; });
    }

    Block* release(Block* block) {
        auto found = llvm::find_if(arms_, [&](auto& arm) { return arm.get() == block; });
        auto* raw = found->release();
        arms_.erase(found);
        return raw;
    }

    OwningOpRef<ModuleOp> module_;
    OpBuilder builder_;
    func::FuncOp function_;
    Block* entry_;
    std::vector<std::unique_ptr<Block>> arms_;
    std::vector<OpBuilder::InsertPoint> saved_;
    std::vector<scf::ForOp> loops_;
};

}

ModuleOp stagedModule(nb::handle staging) {
    if (!nb::isinstance<Staging>(staging))
        throw nb::type_error("compilation takes a kernel staged by carbon.compiler, not IR text");
    return nb::cast<Staging&>(staging).module();
}

void bindBuilder(nb::module_& module) {
    nb::class_<Type>(module, "Type", "A compiler type, compared by identity.")
        .def("__eq__",
             [](Type type, nb::handle other) { return nb::isinstance<Type>(other) && nb::cast<Type>(other) == type; })
        .def("__hash__", [](Type type) { return size_t(type.getAsOpaquePointer()); })
        .def("__str__", [](Type type) {
            std::string text;
            llvm::raw_string_ostream stream(text);
            type.print(stream);
            return text;
        })
        .def("describe", &describe, "The element, layout and view metadata the builder reads from a type.");
    nb::class_<Value>(module, "Value", "A value of the module being staged.")
        .def_prop_ro("type", [](Value value) { return value.getType(); }, "The value's type.");
    nb::class_<Block>(module, "Block", "A detached block being staged as a branch arm.");

    module.def("scalar_type", &scalar, nb::arg("element"),
               "The scalar type named f16, bf16, f32, i32, i1, ui8, si8, ui16 or si16.");
    module.def("buffer_type", [](const std::string& element) {
        return Type(MemRefType::get({ShapedType::kDynamic}, scalar(element)));
    }, nb::arg("element"), "A flat device binding of `element`.");
    module.def("tile_type", [](const std::string& element, ir::LayoutAttr layout) {
        return Type(ir::TileType::get(&ir::context(), scalar(element), layout));
    }, nb::arg("element"), nb::arg("layout"), "A tile of `element` held as `layout` places it.");
    module.def("view_type", [](const std::string& element, std::vector<int64_t> shape, std::vector<int64_t> strides,
                               int64_t offset, int64_t extent, const std::string& space) {
        auto& context = ir::context();
        return Type(ir::ViewType::get(&context, scalar(element), DenseI64ArrayAttr::get(&context, shape),
                                      DenseI64ArrayAttr::get(&context, strides), offset, extent, space));
    }, nb::arg("element"), nb::arg("shape"), nb::arg("strides"), nb::arg("offset"), nb::arg("extent"),
       nb::arg("space"), "A view of `element`: -1 dimensions, offset and extent are runtime values.");

    nb::class_<Staging>(module, "Staging",
                        "One kernel as it is staged. Operations are verified as they are made; branch arms are "
                        "built in detached blocks between enter and leave, and loop bodies between loop and close.")
        .def(nb::init<int64_t, std::optional<int64_t>, bool>(), nb::arg("threads"), nb::arg("max_grid") = nb::none(),
             nb::arg("indirect") = false)
        .def("argument", &Staging::argument, nb::arg("type"), "Append a kernel argument of `type`.")
        .def("set_range", &Staging::setRange, nb::arg("value"), nb::arg("minimum"), nb::arg("maximum"),
             "Give a u32 kernel argument inclusive bounds, checked at dispatch.")
        .def("set_min_bytes", &Staging::setMinimumBytes, nb::arg("value"), nb::arg("bytes"),
             "Give a buffer argument a minimum size in bytes, checked at dispatch.")
        .def("check_stages", &Staging::checkStages, nb::arg("stages"),
             "Check that two double-buffer stages hold matching types in distinct scratch allocations.")
        .def("create", &Staging::create, nb::arg("name"), nb::arg("operands"), nb::arg("results").none(),
             nb::arg("attributes"), nb::arg("location").none(), nb::arg("at_entry") = false,
             "Make operation `name`; with `results` None, the operation infers its result types.")
        .def("constant", &Staging::constant, nb::arg("value"), nb::arg("type"), nb::arg("location").none(),
             "A constant of `type`.")
        .def("bits", &Staging::bits, nb::arg("word"), nb::arg("location").none(),
             "The f32 constant with exactly these bits, NaN payloads included.")
        .def("compare", &Staging::compare, nb::arg("predicate"), nb::arg("left"), nb::arg("right"),
             nb::arg("location").none(), "An arith comparison named by its MLIR predicate.")
        .def("enter", &Staging::enter, nb::arg("block").none() = nb::none(), nb::rv_policy::reference,
             "Stage into `block`, or a new detached block, until leave.")
        .def("leave", &Staging::leave, "Return to where staging was before the matching enter.")
        .def("branch", &Staging::branch, nb::arg("condition"), nb::arg("results"), nb::arg("then"),
             nb::arg("otherwise"), nb::arg("then_values"), nb::arg("otherwise_values"), nb::arg("location").none(),
             "An if whose arms are two detached blocks, each yielding its values.")
        .def("loop", &Staging::loop, nb::arg("lower"), nb::arg("upper"), nb::arg("step"), nb::arg("initial"),
             nb::arg("location").none(), "Open a for loop; returns its counter and carried values.")
        .def("close", &Staging::close, nb::arg("yielded"), nb::arg("location").none(),
             "Close the innermost loop, carrying `yielded`; returns the loop's results.")
        .def("finish", &Staging::finish, "End the kernel.")
        .def_prop_ro("fingerprint", &Staging::fingerprint,
                     "A digest of the module's bytecode: equal for equal kernels.")
        .def_prop_ro("ir", [](Staging& staging) { return ir::print(staging.module()); },
                     "The module as text, for reading.");
}

}
