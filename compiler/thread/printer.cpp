#include "thread/lowering.h"
#include "thread/dialect.h"
#include "tile/dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include <stdexcept>

using namespace mlir;

namespace carbon::msl {
namespace {

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument(message); }

// u32 floordiv or mod that gives 0 for a zero divisor, as the guarded op promises.
std::string integerDivisionExpression(const std::string& opcode, const std::string& left, const std::string& right) {
    return "(" + right + " != 0u ? (" + left + (opcode == "floordiv" ? " / " : " % ") + right + ") : 0u)";
}

// and, or and xor; shifts give 0 for counts of 32 or more, which MSL leaves undefined.
std::string bitwiseExpression(const std::string& opcode, const std::string& left, const std::string& right) {
    if (opcode == "shl" || opcode == "shr")
        return "(" + right + " < 32u ? (" + left + (opcode == "shl" ? " << (" : " >> (") + right + " & 31u)) : 0u)";
    return "(" + left + (opcode == "and" ? " & " : opcode == "or" ? " | " : " ^ ") + right + ")";
}

// NaN-propagating f32 maximum or minimum with -0 below +0; the keys order floats as unsigned integers.
std::string extremumDefinition(const std::string& name, const std::string& comparison) {
    return "inline float " + name + "(float a, float b) {\n"
           "    uint x = as_type<uint>(a), y = as_type<uint>(b);\n"
           "    if ((x & 0x7fffffffu) > 0x7f800000u || (y & 0x7fffffffu) > 0x7f800000u)\n"
           "        return as_type<float>(uint(0x7fc00000u));\n"
           "    uint kx = x >> 31u ? ~x : (x ^ 0x80000000u);\n"
           "    uint ky = y >> 31u ? ~y : (y ^ 0x80000000u);\n"
           "    return as_type<float>(kx " + comparison + " ky ? x : y);\n"
           "}\n";
}

std::string typeName(Type type) {
    if (auto pointer = dyn_cast<PointerType>(type))
        return pointer.getSpace().str() + " " + typeName(pointer.getElementType()) + "*";
    if (type.isF32()) return "float";
    if (type.isF16()) return "half";
    if (type.isBF16()) return "bfloat";
    if (type.isInteger(1)) return "bool";
    if (type.isInteger(8)) return "uchar";
    if (type.isInteger(16)) return "ushort";
    if (type.isInteger(32)) return "uint";
    if (type.isInteger(64)) return "long";
    if (type.isIndex()) return "ulong";
    fail("unsupported type in Metal printing");
}

// The MSL spelling of a constant; floats are spelled by their bits, so every payload survives the round trip.
std::string literal(Attribute value) {
    if (auto integer = dyn_cast<IntegerAttr>(value)) {
        auto type = integer.getType();
        auto bits = integer.getValue();
        if (type.isInteger(1)) return bits.isZero() ? "false" : "true";
        if (type.isInteger(8)) return "uchar(" + std::to_string(bits.getZExtValue()) + "u)";
        if (type.isInteger(16)) return "ushort(" + std::to_string(bits.getZExtValue()) + "u)";
        if (type.isInteger(32)) return std::to_string(bits.getZExtValue()) + "u";
        if (type.isInteger(64)) return "(" + std::to_string(bits.getSExtValue()) + "l)";
        if (type.isIndex()) return std::to_string(bits.getZExtValue()) + "ul";
    }
    if (auto floating = dyn_cast<FloatAttr>(value)) {
        auto bits = std::to_string(floating.getValue().bitcastToAPInt().getZExtValue()) + "u";
        if (!floating.getType().isF32())
            bits = "ushort(" + bits + ")";
        return "as_type<" + typeName(floating.getType()) + ">(" + bits + ")";
    }
    fail("unsupported constant in Metal printing");
}

bool signedType(Type type) { return type.isInteger(64); }

class Printer {
  public:
    std::string run(func::FuncOp function) {
        for (auto& operation : function.getBody().front())
            statement(operation);
        return body_;
    }

    bool usesMaximum() const { return usesMaximum_; }
    bool usesMinimum() const { return usesMinimum_; }
    bool usesTensors() const { return usesTensors_; }

  private:
    std::string name(Value value) {
        if (auto found = names_.find(value); found != names_.end())
            return found->second;
        auto* operation = value.getDefiningOp();
        if (auto constant = dyn_cast_or_null<arith::ConstantOp>(operation))
            return literal(constant.getValue());
        if (auto builtin = dyn_cast_or_null<BuiltinOp>(operation))
            return builtin.getName().str();
        if (auto field = dyn_cast_or_null<ParamOp>(operation))
            return "params." + field.getField().str();
        if (auto argument = dyn_cast_or_null<ArgumentOp>(operation))
            return "arg" + std::to_string(argument.getBinding());
        fail("Metal printing used a value before defining it");
    }

    std::string fresh() { return "v" + std::to_string(next_++); }

    void line(const std::string& text) { body_ += indent_ + text + "\n"; }

    void declare(Value value, const std::string& expression) {
        auto variable = fresh();
        line(typeName(value.getType()) + " " + variable + " = " + expression + ";");
        names_[value] = variable;
    }

    std::string asUnsigned(Value value) {
        return signedType(value.getType()) ? "ulong(" + name(value) + ")" : name(value);
    }

    std::string asSigned(Value value) {
        auto type = value.getType();
        if (type.isInteger(32)) return "int(" + name(value) + ")";
        if (type.isInteger(16)) return "short(" + name(value) + ")";
        if (type.isInteger(8)) return "as_type<char>(" + name(value) + ")";
        if (type.isIndex()) return "long(" + name(value) + ")";
        return name(value);
    }

    std::string binary(const std::string& left, const std::string& opcode, const std::string& right) {
        return "(" + left + " " + opcode + " " + right + ")";
    }

    bool arithmetic(Operation& operation) {
        if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
            if (isa<FloatAttr>(constant.getValue()))
                declare(constant.getResult(), literal(constant.getValue()));
            return true;
        }
        auto output = operation.getNumResults() ? operation.getResult(0) : Value();
        auto a = operation.getNumOperands() > 0 ? operation.getOperand(0) : Value();
        auto b = operation.getNumOperands() > 1 ? operation.getOperand(1) : Value();
        bool boolean = a && a.getType().isInteger(1);
        std::string expression;
        if (isa<arith::AddIOp, arith::AddFOp>(operation)) expression = binary(name(a), "+", name(b));
        else if (isa<arith::SubIOp, arith::SubFOp>(operation)) expression = binary(name(a), "-", name(b));
        else if (isa<arith::MulIOp, arith::MulFOp>(operation)) expression = binary(name(a), "*", name(b));
        else if (isa<arith::DivFOp>(operation)) expression = binary(name(a), "/", name(b));
        else if (isa<arith::NegFOp>(operation)) expression = "(-" + name(a) + ")";
        else if (isa<arith::DivUIOp>(operation)) expression = binary(asUnsigned(a), "/", asUnsigned(b));
        else if (isa<arith::RemUIOp>(operation)) expression = binary(asUnsigned(a), "%", asUnsigned(b));
        else if (isa<arith::DivSIOp>(operation)) expression = binary(asSigned(a), "/", asSigned(b));
        else if (isa<arith::RemSIOp>(operation)) expression = binary(asSigned(a), "%", asSigned(b));
        else if (isa<arith::AndIOp>(operation)) expression = binary(name(a), boolean ? "&&" : "&", name(b));
        else if (isa<arith::OrIOp>(operation)) expression = binary(name(a), boolean ? "||" : "|", name(b));
        else if (isa<arith::XOrIOp>(operation)) expression = binary(name(a), boolean ? "!=" : "^", name(b));
        else if (isa<arith::ShLIOp>(operation)) expression = binary(name(a), "<<", name(b));
        else if (isa<arith::ShRUIOp>(operation)) expression = binary(asUnsigned(a), ">>", name(b));
        else if (isa<arith::ShRSIOp>(operation)) expression = binary(asSigned(a), ">>", name(b));
        else if (isa<arith::MinUIOp>(operation)) expression = "min(" + asUnsigned(a) + ", " + asUnsigned(b) + ")";
        else if (isa<arith::MaxUIOp>(operation)) expression = "max(" + asUnsigned(a) + ", " + asUnsigned(b) + ")";
        else if (isa<arith::MinSIOp>(operation)) expression = "min(" + asSigned(a) + ", " + asSigned(b) + ")";
        else if (isa<arith::MaxSIOp>(operation)) expression = "max(" + asSigned(a) + ", " + asSigned(b) + ")";
        else if (auto choice = dyn_cast<arith::SelectOp>(operation)) {
            // A bfloat select moves bits: a bfloat ternary can flush subnormals or canonicalize NaNs.
            bool bits = output.getType().isBF16();
            auto pick = [&](Value value) { return bits ? "as_type<ushort>(" + name(value) + ")" : name(value); };
            expression = "(" + name(choice.getCondition()) + " ? " + pick(choice.getTrueValue()) + " : " +
                         pick(choice.getFalseValue()) + ")";
            if (bits)
                expression = "as_type<bfloat>" + expression;
        } else if (isa<arith::ExtUIOp, arith::TruncIOp, arith::IndexCastUIOp>(operation)) {
            if (output.getType().isInteger(1))
                expression = "bool(" + name(a) + " & 1u)";
            else
                expression = typeName(output.getType()) + "(" + name(a) + ")";
        } else if (isa<arith::ExtSIOp, arith::IndexCastOp, arith::SIToFPOp>(operation))
            expression = typeName(output.getType()) + "(" + asSigned(a) + ")";
        else if (isa<arith::BitcastOp>(operation))
            expression = "as_type<" + typeName(output.getType()) + ">(" + name(a) + ")";
        else if (auto compare = dyn_cast<arith::CmpIOp>(operation)) {
            using P = arith::CmpIPredicate;
            auto predicate = compare.getPredicate();
            static const std::map<P, std::string> symbols = {
                {P::eq, "=="}, {P::ne, "!="}, {P::ult, "<"}, {P::ule, "<="}, {P::ugt, ">"}, {P::uge, ">="},
                {P::slt, "<"}, {P::sle, "<="}, {P::sgt, ">"}, {P::sge, ">="}};
            bool isSigned = predicate == P::slt || predicate == P::sle || predicate == P::sgt || predicate == P::sge;
            bool isUnsigned = predicate == P::ult || predicate == P::ule || predicate == P::ugt || predicate == P::uge;
            auto left = isSigned ? asSigned(a) : isUnsigned ? asUnsigned(a) : name(a);
            auto right = isSigned ? asSigned(b) : isUnsigned ? asUnsigned(b) : name(b);
            expression = binary(left, symbols.at(predicate), right);
        } else if (auto compare = dyn_cast<arith::CmpFOp>(operation)) {
            using P = arith::CmpFPredicate;
            static const std::map<P, std::string> symbols = {
                {P::OLT, "<"}, {P::OLE, "<="}, {P::OGT, ">"}, {P::OGE, ">="}, {P::OEQ, "=="}, {P::UNE, "!="}};
            auto found = symbols.find(compare.getPredicate());
            if (found == symbols.end())
                fail("unsupported floating-point comparison in Metal printing");
            expression = binary(name(a), found->second, name(b));
        } else
            return false;
        declare(output, expression);
        return true;
    }

    std::string address(Value pointer, Value index) { return name(pointer) + "[" + name(index) + "]"; }

    std::string vectorType(Type element, size_t width) {
        auto name = width > 4 ? "uint" : typeName(element);
        if (width > 4)
            width /= 32 / element.getIntOrFloatBitWidth();
        return name + (width == 1 ? "" : std::to_string(width));
    }

    std::string vectorAccess(Value pointer, Value index, Type element, size_t width, bool constant) {
        auto space = cast<PointerType>(pointer.getType()).getSpace().str();
        return "*reinterpret_cast<" + space + (constant ? " const " : " ") + vectorType(element, width) +
               "*>(&" + address(pointer, index) + ")";
    }

    std::string vector(ValueRange items) {
        if (items.size() == 1)
            return name(items[0]);
        std::string list;
        auto element = items[0].getType();
        if (items.size() > 4) {
            std::string packed = element.getIntOrFloatBitWidth() == 8 ? "uint" : "uint2";
            for (size_t i = 0; i < items.size(); i += 4)
                list += (list.empty() ? "" : ", ") + ("as_type<" + packed + ">(" + vector(items.slice(i, 4)) + ")");
        } else
            for (auto item : items)
                list += (list.empty() ? "" : ", ") + name(item);
        return vectorType(element, items.size()) + "(" + list + ")";
    }

    void load(ValueRange results, const std::string& expression) {
        if (results.size() == 1)
            return declare(results[0], expression);
        auto variable = fresh();
        auto element = results[0].getType();
        line(vectorType(element, results.size()) + " " + variable + " = " + expression + ";");
        if (results.size() > 4) {
            for (size_t i = 0; i < results.size(); i += 4) {
                auto part = element.getIntOrFloatBitWidth() == 8 ? "[" + std::to_string(i / 4) + "]"
                                                               : i ? ".zw" : ".xy";
                load(results.slice(i, 4), "as_type<" + vectorType(element, 4) + ">(" + variable + part + ")");
            }
            return;
        }
        for (auto [index, item] : llvm::enumerate(results))
            names_[item] = variable + "[" + std::to_string(index) + "]";
    }

    std::vector<std::string> declared(ValueRange results) {
        std::vector<std::string> variables;
        for (auto result : results) {
            variables.push_back(fresh());
            line(typeName(result.getType()) + " " + variables.back() + ";");
            names_[result] = variables.back();
        }
        return variables;
    }

    void elements(const std::string& tensor, ValueRange values) {
        for (auto [index, item] : llvm::enumerate(values))
            line("    " + tensor + "[" + std::to_string(index) + "] = " + name(item) + ";");
    }

    // An accumulator of zeros needs no seeding: the multiply mode writes the destination.
    static bool startsAtZero(ValueRange values) {
        return llvm::all_of(values, [](Value value) { return matchPattern(value, m_PosZeroFloat()); });
    }

    void results(const std::vector<std::string>& outputs, const std::string& tensor) {
        for (auto [index, output] : llvm::enumerate(outputs))
            line("    " + output + " = " + tensor + "[" + std::to_string(index) + "];");
    }

    std::string target(Value pointer, Value index, ValueRange values, bool constant) {
        if (values.size() == 1)
            return address(pointer, index);
        return vectorAccess(pointer, index, values[0].getType(), values.size(), constant);
    }

    void loop(scf::ForOp loop) {
        for (auto [argument, initial, output] :
             llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs(), loop.getResults())) {
            auto variable = fresh();
            line(typeName(argument.getType()) + " " + variable + " = " + name(initial) + ";");
            names_[argument] = variable;
            names_[output] = variable;
        }
        auto induction = fresh();
        names_[loop.getInductionVar()] = induction;
        line("for (" + typeName(loop.getInductionVar().getType()) + " " + induction + " = " +
             name(loop.getLowerBound()) + "; " + induction + " < " + name(loop.getUpperBound()) + "; " + induction +
             " += " + name(loop.getStep()) + ") {");
        indent_ += "    ";
        auto* body = loop.getBody();
        for (auto& nested : body->without_terminator())
            statement(nested);
        std::vector<std::pair<std::string, std::string>> updates;
        for (auto [argument, yielded] : llvm::zip(loop.getRegionIterArgs(), body->getTerminator()->getOperands())) {
            if (name(argument) == name(yielded))
                continue;
            auto temporary = fresh();
            line(typeName(argument.getType()) + " " + temporary + " = " + name(yielded) + ";");
            updates.emplace_back(name(argument), temporary);
        }
        for (auto [target, source] : updates)
            line(target + " = " + source + ";");
        indent_.resize(indent_.size() - 4);
        line("}");
    }

    void branch(scf::IfOp conditional) {
        std::vector<std::string> targets;
        for (auto output : conditional.getResults()) {
            auto variable = fresh();
            line(typeName(output.getType()) + " " + variable + ";");
            names_[output] = variable;
            targets.push_back(variable);
        }
        line("if (" + name(conditional.getCondition()) + ") {");
        for (auto [index, region] : llvm::enumerate(conditional->getRegions())) {
            if (region.empty())
                continue;
            auto& block = region.front();
            if (index) {
                if (llvm::hasSingleElement(block) && targets.empty())
                    continue;
                line("} else {");
            }
            indent_ += "    ";
            for (auto& nested : block.without_terminator())
                statement(nested);
            for (auto [target, yielded] : llvm::zip(targets, block.getTerminator()->getOperands()))
                line(target + " = " + name(yielded) + ";");
            indent_.resize(indent_.size() - 4);
        }
        line("}");
    }

    void statement(Operation& operation) {
        if (isa<func::ReturnOp, BuiltinOp, ParamOp, ArgumentOp>(operation))
            return;
        if (auto loopOp = dyn_cast<scf::ForOp>(operation))
            return loop(loopOp);
        if (auto conditional = dyn_cast<scf::IfOp>(operation))
            return branch(conditional);
        if (arithmetic(operation))
            return;
        if (auto scratch = dyn_cast<ScratchOp>(operation)) {
            auto type = typeName(scratch.getPointer().getType());
            return declare(scratch.getPointer(), "reinterpret_cast<" + type + ">(scratch + " +
                                                     std::to_string(scratch.getOffset()) + "u)");
        }
        if (auto cast = dyn_cast<ReinterpretOp>(operation))
            return declare(cast.getResult(), "reinterpret_cast<" + typeName(cast.getResult().getType()) + ">(" +
                                                 name(cast.getPointer()) + ")");
        if (auto access = dyn_cast<LoadOp>(operation))
            return load(access.getResults(), target(access.getPointer(), access.getIndex(), access.getResults(), true));
        if (auto access = dyn_cast<GuardedLoadOp>(operation))
            return load(access.getResults(), "(" + name(access.getCondition()) + " ? " +
                                                 target(access.getPointer(), access.getIndex(), access.getResults(),
                                                        true) + " : " + vector(access.getFallbacks()) + ")");
        if (auto store = dyn_cast<StoreOp>(operation))
            return line(target(store.getPointer(), store.getIndex(), store.getValues(), false) + " = " +
                        vector(store.getValues()) + ";");
        if (auto store = dyn_cast<GuardedStoreOp>(operation))
            return line("if (" + name(store.getCondition()) + ") " +
                        target(store.getPointer(), store.getIndex(), store.getValues(), false) + " = " +
                        vector(store.getValues()) + ";");
        if (auto atomic = dyn_cast<AtomicOp>(operation)) {
            bool f32 = atomic.getValue().getType().isF32();
            return declare(atomic.getResult(), "(" + name(atomic.getCondition()) + " ? atomic_fetch_" +
                atomic.getOpcode().str() + "_explicit(reinterpret_cast<device " +
                (f32 ? "atomic_float" : "atomic_uint") + "*>(" + name(atomic.getPointer()) + ") + " +
                name(atomic.getIndex()) + ", " + name(atomic.getValue()) + ", memory_order_relaxed) : " +
                (f32 ? "0.0f" : "0u") + ")");
        }
        if (auto atomic = dyn_cast<AtomicStoreOp>(operation))
            return line("if (" + name(atomic.getCondition()) + ") atomic_store_explicit(reinterpret_cast<device " +
                        (atomic.getValue().getType().isF32() ? "atomic_float" : "atomic_uint") + "*>(" +
                        name(atomic.getPointer()) + ") + " + name(atomic.getIndex()) + ", " +
                        name(atomic.getValue()) + ", memory_order_relaxed);");
        if (isa<BarrierOp>(operation))
            return line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        if (auto shuffle = dyn_cast<ShuffleOp>(operation)) {
            auto arguments = name(shuffle.getValue()) + ", ushort(" + name(shuffle.getLane()) + "))";
            return declare(shuffle.getResult(),
                           (shuffle.getButterfly() ? "simd_shuffle_xor(" : "simd_shuffle(") + arguments);
        }
        if (auto simd = dyn_cast<SimdOp>(operation))
            return declare(simd.getResult(), "simd_" + simd.getKind().str() + "(" + name(simd.getValue()) + ")");
        if (auto math = dyn_cast<MathOp>(operation)) {
            auto input = name(math.getValue());
            auto function = math.getFunction();
            if (function == "reciprocal")
                return declare(math.getResult(), "(1.0f / " + input + ")");
            bool exact = function == "abs" || function == "floor" || function == "ceil";
            return declare(math.getResult(), (exact ? "metal::" : "precise::") + function.str() + "(" + input + ")");
        }
        if (auto maximum = dyn_cast<MaximumOp>(operation)) {
            usesMaximum_ = true;
            return declare(maximum.getResult(),
                           "carbon_maximum(" + name(maximum.getLhs()) + ", " + name(maximum.getRhs()) + ")");
        }
        if (auto minimum = dyn_cast<MinimumOp>(operation)) {
            usesMinimum_ = true;
            return declare(minimum.getResult(),
                           "carbon_minimum(" + name(minimum.getLhs()) + ", " + name(minimum.getRhs()) + ")");
        }
        if (auto fma = dyn_cast<FmaOp>(operation))
            return declare(fma.getResult(), "metal::fma(" + name(fma.getA()) + ", " + name(fma.getB()) + ", " +
                                                name(fma.getC()) + ")");
        if (auto convert = dyn_cast<ConvertOp>(operation))
            return declare(convert.getResult(), typeName(convert.getResult().getType()) + "(" +
                                                    name(convert.getValue()) + ")");
        if (auto guarded = dyn_cast<GuardedIntegerOp>(operation)) {
            auto opcode = guarded.getOpcode().str();
            auto left = name(guarded.getLhs()), right = name(guarded.getRhs());
            return declare(guarded.getResult(), opcode == "floordiv" || opcode == "mod" ?
                integerDivisionExpression(opcode, left, right) : bitwiseExpression(opcode, left, right));
        }
        if (auto matmul = dyn_cast<TensorMatmulOp>(operation)) {
            usesTensors_ = true;
            auto outputs = declared(matmul.getResults());
            auto tensors = matmul.getTensors();
            bool direct = tensors.size() == 3 * tensorOperands;
            line("{");
            for (auto [index, label] : llvm::enumerate(std::array<const char*, 3>{"left", "right", "destination"})) {
                if (index == 2 && !direct)
                    break;
                auto values = tensors.slice(tensorOperands * index, tensorOperands);
                auto pointer = cast<PointerType>(values[0].getType());
                auto element = pointer.getSpace().str() + " " + typeName(pointer.getElementType());
                line("    auto " + std::string(label) + " = tensor<" + element +
                     ", dextents<int32_t, 2>, tensor_inline>(const_cast<" + element + "*>(" + name(values[0]) +
                     ") + (" + name(values[1]) + "), dextents<int32_t, 2>(int(" + name(values[2]) + "), int(" +
                     name(values[3]) + ")), array<int32_t, 2>({1, int(" + name(values[4]) + ")}));");
            }
            bool zero = direct || startsAtZero(matmul.getAccumulator());
            line("    constexpr auto descriptor = matmul2d_descriptor(" + std::to_string(matmul.getM()) + ", " +
                 std::to_string(matmul.getN()) + ", static_cast<int>(dynamic_extent), " +
                 (matmul.getTransposeLeft() ? "true, " : "false, ") +
                 (matmul.getTransposeRight() ? "true, " : "false, ") +
                 (matmul.getRelaxed() ? "true" : "false") + ", matmul2d_descriptor::mode::" +
                 (zero ? "multiply" : "multiply_accumulate") + ");");
            line("    matmul2d<descriptor, execution_simdgroups<" + std::to_string(matmul.getGroups()) +
                 ">> operation;");
            if (direct) {
                line("    operation.run(left, right, destination);");
                return line("}");
            }
            line("    auto accumulator = operation.get_destination_cooperative_tensor<decltype(left), "
                 "decltype(right), " + typeName(matmul.getAccumulator()[0].getType()) + ">();");
            if (!zero)
                elements("accumulator", matmul.getAccumulator());
            line("    operation.run(left, right, accumulator);");
            results(outputs, "accumulator");
            return line("}");
        }
        if (auto matrix = dyn_cast<MatrixLoadOp>(operation)) {
            auto outputs = declared(matrix.getResults());
            line("{");
            line("    simdgroup_matrix<" + typeName(matrix.getResults()[0].getType()) + ", 8, 8> matrix;");
            line("    simdgroup_load(matrix, " + name(matrix.getPointer()) + " + " + name(matrix.getIndex()) +
                 ", ulong(" + std::to_string(matrix.getStride()) + "), ulong2(0), " +
                 (matrix.getTransposed() ? "true" : "false") + ");");
            results(outputs, "matrix.thread_elements()");
            return line("}");
        }
        if (auto mma = dyn_cast<MmaOp>(operation)) {
            auto outputs = declared(mma.getResults());
            auto element = [&](ValueRange values) { return typeName(values[0].getType()); };
            line("{");
            if (mma.getUnit() == "simdgroup") {
                for (auto [label, values] : {std::pair("left", mma.getLeft()), std::pair("right", mma.getRight()),
                                             std::pair("accumulator", mma.getAccumulator())}) {
                    line("    simdgroup_matrix<" + element(values) + ", 8, 8> " + label + ";");
                    for (auto [index, item] : llvm::enumerate(values))
                        line("    " + std::string(label) + ".thread_elements()[" + std::to_string(index) + "] = " +
                             name(item) + ";");
                }
                line("    simdgroup_multiply_accumulate(accumulator, left, right, accumulator);");
                for (auto [index, output] : llvm::enumerate(outputs))
                    line("    " + output + " = accumulator.thread_elements()[" + std::to_string(index) + "];");
                return line("}");
            }
            usesTensors_ = true;
            bool zero = startsAtZero(mma.getAccumulator());
            auto types = element(mma.getLeft()) + ", " + element(mma.getRight()) + ", " + element(mma.getAccumulator());
            line("    constexpr auto descriptor = matmul2d_descriptor(" + std::to_string(mma.getM()) + ", " +
                 std::to_string(mma.getN()) + ", " + std::to_string(mma.getK()) + ", false, false, " +
                 (mma.getRelaxed() ? "true" : "false") + ", matmul2d_descriptor::mode::" +
                 (zero ? "multiply" : "multiply_accumulate") + ");");
            line("    matmul2d<descriptor, execution_simdgroup> operation;");
            line("    auto left = operation.get_left_input_cooperative_tensor<" + types + ">();");
            line("    auto right = operation.get_right_input_cooperative_tensor<" + types + ">();");
            line("    auto accumulator = operation.get_destination_cooperative_tensor<decltype(left), "
                 "decltype(right), " + element(mma.getAccumulator()) + ">();");
            elements("left", mma.getLeft());
            elements("right", mma.getRight());
            if (!zero)
                elements("accumulator", mma.getAccumulator());
            line("    operation.run(left, right, accumulator);");
            results(outputs, "accumulator");
            return line("}");
        }
        fail("unsupported per-thread operation in Metal printing: " + operation.getName().getStringRef().str());
    }

    std::string body_;
    std::string indent_ = "    ";
    llvm::DenseMap<Value, std::string> names_;
    size_t next_ = 0;
    bool usesMaximum_ = false;
    bool usesMinimum_ = false;
    bool usesTensors_ = false;
};

Type memoryElement(Type type) {
    if (auto view = dyn_cast<ir::ViewType>(type))
        return view.getElementType();
    return cast<MemRefType>(type).getElementType();
}

}

std::string printMetal(ModuleOp module, func::FuncOp tile, const MemoryPlan& memory) {
    Printer printer;
    auto body = printer.run(*module.getOps<func::FuncOp>().begin());
    std::vector<std::string> arguments;
    // The constant block is the dispatch ABI that Artifact._pack_constants writes. Every field is four bytes, so
    // there is no padding: the runtime scalars in argument order, each binding's element count, then for each runtime
    // view its extents, its i64 strides as (lo, hi) u32 halves, and its offset in elements.
    std::string constants = "struct Constants {\n";
    size_t count = 0;
    for (auto argument : tile.getArguments())
        if (!isa<MemRefType, ir::ViewType>(argument.getType()))
            constants +=
                "    " + typeName(argument.getType()) + " arg" + std::to_string(argument.getArgNumber()) + ";\n";
    for (auto argument : tile.getArguments()) {
        if (!isa<MemRefType, ir::ViewType>(argument.getType()))
            continue;
        auto slot = count++;
        arguments.push_back("device " + std::string(memory.access[slot] == "read" ? "const " : "") +
                            typeName(memoryElement(argument.getType())) + "* arg" + std::to_string(slot) +
                            " [[buffer(" + std::to_string(slot) + ")]]");
        constants += "    uint bound" + std::to_string(slot) + ";\n";
    }
    size_t slot = 0;
    for (auto argument : tile.getArguments()) {
        if (!isa<MemRefType, ir::ViewType>(argument.getType()))
            continue;
        auto view = dyn_cast<ir::ViewType>(argument.getType());
        auto prefix = std::to_string(slot++);
        if (!view)
            continue;
        for (size_t axis = 0; axis < view.getShape().size(); ++axis)
            constants += "    uint shape" + prefix + "_" + std::to_string(axis) + ";\n";
        for (size_t axis = 0; axis < view.getShape().size(); ++axis) {
            auto suffix = prefix + "_" + std::to_string(axis);
            constants += "    uint stride_lo" + suffix + ";\n    uint stride_hi" + suffix + ";\n";
        }
        constants += "    uint offset" + prefix + ";\n";
    }
    // Under write checking, a u32 shadow per recorded binding and then the record buffer follow the bindings.
    for (size_t check = 0; check <= memory.checkedBindings.size() && !memory.checkedBindings.empty(); ++check)
        arguments.push_back("device uint* arg" + std::to_string(count + check) + " [[buffer(" +
                            std::to_string(count + check) + ")]]");
    if (!memory.checkedBindings.empty())
        count += memory.checkedBindings.size() + 1;
    arguments.push_back("constant Constants& params [[buffer(" + std::to_string(count) + ")]]");
    if (memory.scratchBytes)
        arguments.push_back("threadgroup uchar* scratch [[threadgroup(0)]]");
    arguments.push_back("uint lane [[thread_index_in_simdgroup]]");
    arguments.push_back("uint simd_group [[simdgroup_index_in_threadgroup]]");
    bool indirect = tile->hasAttr("carbon.indirect");
    arguments.push_back(indirect ? "uint3 position [[threadgroup_position_in_grid]]"
                                 : "uint group [[threadgroup_position_in_grid]]");
    std::string guard = indirect ? "    if (position.x >= " + std::to_string(memory.maxGrid) +
                                       "u || position.y || position.z)\n        return;\n    uint group = position.x;\n"
                                 : "";
    std::string tensors = printer.usesTensors() ? "#include <metal_tensor>\n#include "
        "<MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n" : "";
    std::string source = "#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n"
                         "using namespace metal;\n" + tensors + "#pragma clang fp contract(off)\n" +
                         (printer.usesMaximum() ? extremumDefinition("carbon_maximum", ">") : "") +
                         (printer.usesMinimum() ? extremumDefinition("carbon_minimum", "<") : "") + constants +
                         "};\nkernel void carbon_kernel(\n";
    for (size_t i = 0; i < arguments.size(); ++i)
        source += "    " + arguments[i] + (i + 1 == arguments.size() ? ") {\n" : ",\n");
    return source + guard + body + "}\n";
}

}
