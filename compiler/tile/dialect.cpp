#include "tile/dialect.h"
#include "tile/analysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include "CarbonDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "CarbonAttrs.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "CarbonTypes.cpp.inc"
#define GET_OP_CLASSES
#include "CarbonOps.cpp.inc"

namespace carbon::ir {

void CarbonDialect::initialize() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "CarbonAttrs.cpp.inc"
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "CarbonTypes.cpp.inc"
        >();
    addOperations<
#define GET_OP_LIST
#include "CarbonOps.cpp.inc"
        >();
}

mlir::LogicalResult IotaOp::verify() {
    if (!getResult().getType().getElementType().isInteger(32))
        return emitOpError("requires an i32 tile");
    return mlir::success();
}

// Whether no two elements of a static view share an address: sorted by stride, each axis steps past everything the
// smaller ones span. Runtime views are not checked here.
static bool uniqueView(ViewType view) {
    if (view.isDynamic() || llvm::is_contained(view.getShape().asArrayRef(), 0))
        return true;
    std::vector<std::pair<__int128, int64_t>> dimensions;
    for (size_t axis = 0; axis < view.getShape().size(); ++axis) {
        __int128 stride = view.getStrides()[axis];
        if (view.getShape()[axis] > 1)
            dimensions.emplace_back(stride < 0 ? -stride : stride, view.getShape()[axis]);
    }
    std::sort(dimensions.begin(), dimensions.end());
    __int128 span = 1;
    for (auto [stride, extent] : dimensions) {
        if (stride < span)
            return false;
        span += (extent - 1) * stride;
    }
    return true;
}

static mlir::LogicalResult memory(mlir::Operation* operation, mlir::Value buffer, mlir::ValueRange indices,
                                  mlir::Value mask, mlir::Value values, bool store) {
    if (indices.empty() || !llvm::all_of(indices.getTypes(), [](mlir::Type type) { return mlir::isa<TileType>(type); }))
        return operation->emitOpError("memory indices must be layout-aware tiles");
    auto index = mlir::cast<TileType>(indices.front().getType());
    auto predicate = mlir::dyn_cast<TileType>(mask.getType());
    auto data = mlir::dyn_cast<TileType>(values.getType());
    mlir::Type element;
    size_t rank = 1;
    if (auto view = mlir::dyn_cast<ViewType>(buffer.getType())) {
        element = view.getElementType();
        rank = view.getShape().size();
        auto rootView = mlir::dyn_cast<ViewType>(viewRoot(buffer).getType());
        if (store && rootView && rootView.getAddressSpace() == "threadgroup" && !uniqueView(rootView))
            return operation->emitOpError("scratch stores need unique addresses");
    } else if (auto view = mlir::dyn_cast<mlir::MemRefType>(buffer.getType())) {
        if (view.getRank() != 1 || !view.isDynamicDim(0) || !view.getLayout().isIdentity() ||
            view.getMemorySpace())
            return operation->emitOpError("requires a flat dynamic device binding");
        element = view.getElementType();
    }
    if (indices.size() != 1 && indices.size() != rank)
        return operation->emitOpError("memory accesses take flat indices or one coordinate tile per view axis");
    if (!predicate || !data || index.getLayout() != predicate.getLayout() ||
        llvm::any_of(indices.getTypes(), [&](mlir::Type type) { return type != index; }) ||
        index.getLayout() != data.getLayout() || !index.getElementType().isInteger(32) ||
        !predicate.getElementType().isInteger(1) || element != data.getElementType())
        return operation->emitOpError("requires matching index, mask and data tile ownership and elements");
    return mlir::success();
}

mlir::LogicalResult LoadOp::verify() {
    return memory(*this, getBuffer(), getIndices(), getMask(), getFallback(), false);
}

mlir::LogicalResult StoreOp::verify() {
    return memory(*this, getBuffer(), getIndices(), getMask(), getValues(), true);
}

static mlir::LogicalResult atomicMemory(mlir::Operation* operation, mlir::Value buffer, mlir::Value indices,
                                        mlir::Value mask, mlir::Value values, llvm::StringRef opcode) {
    auto view = mlir::dyn_cast<ViewType>(buffer.getType());
    auto element = mlir::cast<TileType>(values.getType()).getElementType();
    bool floating = element.isF32() && (opcode == "add" || opcode == "store");
    if ((view && view.getAddressSpace() != "device") || !(element.isInteger(32) || floating))
        return operation->emitOpError("atomics require u32 device storage, or f32 for atomic_add and atomic_store");
    return memory(operation, buffer, indices, mask, values, false);
}

mlir::LogicalResult AtomicStoreOp::verify() {
    return atomicMemory(*this, getBuffer(), getIndices(), getMask(), getValues(), "store");
}

mlir::LogicalResult AtomicOp::verify() {
    if (getOpcode() != "add" && getOpcode() != "min" && getOpcode() != "max")
        return emitOpError("atomic opcode must be add, min or max");
    return atomicMemory(*this, getBuffer(), getIndices(), getMask(), getValues(), getOpcode());
}

using namespace mlir;
using Digits = std::vector<std::pair<int64_t, int64_t>>;

static bool modeDigits(Attribute shape, Attribute stride, Digits& digits) {
    if (auto extent = dyn_cast<IntegerAttr>(shape)) {
        auto step = dyn_cast<IntegerAttr>(stride);
        if (!extent.getType().isInteger(64) || !step || !step.getType().isInteger(64) || extent.getInt() < 1)
            return false;
        digits.emplace_back(extent.getInt(), step.getInt());
        return true;
    }
    auto extents = dyn_cast<ArrayAttr>(shape), steps = dyn_cast<ArrayAttr>(stride);
    if (!extents || !steps || extents.empty() || extents.size() != steps.size())
        return false;
    for (size_t i = 0; i < extents.size(); ++i)
        if (!modeDigits(extents[i], steps[i], digits))
            return false;
    return true;
}

static Digits modeDigits(ModeAttr mode) {
    Digits result;
    modeDigits(mode.getShape(), mode.getStride(), result);
    return result;
}

LogicalResult ModeAttr::verify(llvm::function_ref<InFlightDiagnostic()> error, Attribute shape,
                               Attribute stride) {
    Digits digits;
    if (!shape || !stride || !modeDigits(shape, stride, digits))
        return error() << "mode needs matching nonempty trees of positive i64 extents and i64 strides";
    int64_t size = 1;
    for (auto [extent, step] : digits) {
        if (extent > UINT32_MAX / size)
            return error() << "mode size exceeds u32";
        size *= extent;
    }
    return success();
}

int64_t ModeAttr::size() const {
    int64_t result = 1;
    for (auto [extent, stride] : modeDigits(*this))
        result *= extent;
    return result;
}

Digits ModeAttr::digits() const { return modeDigits(*this); }

int64_t ModeAttr::offset(int64_t index) const {
    int64_t divisor = size(), result = 0;
    for (auto [extent, stride] : digits()) {
        divisor /= extent;
        result += (index / divisor % extent) * stride;
    }
    return result;
}

int64_t ModeAttr::inverse(int64_t index) const { return holder(index, 0); }

// A stride-0 digit repeats the element across every value of that digit; the holder keeps `preferred`'s value there.
int64_t ModeAttr::holder(int64_t index, int64_t preferred) const {
    int64_t divisor = size(), result = 0;
    for (auto [extent, stride] : digits()) {
        divisor /= extent;
        if (extent > 1)
            result += (stride ? index / stride % extent : preferred / divisor % extent) * divisor;
    }
    return result;
}

int64_t ModeAttr::repeats() const {
    int64_t result = 1;
    for (auto [extent, stride] : digits())
        if (!stride)
            result *= extent;
    return result;
}

bool ModeAttr::replicated() const { return repeats() > 1; }

int64_t LayoutAttr::index(int64_t group, int64_t lane, int64_t slot) const {
    return getGroups().offset(group) + getLanes().offset(lane) + getSlots().offset(slot);
}

LogicalResult LayoutAttr::verify(llvm::function_ref<InFlightDiagnostic()> error, DenseI64ArrayAttr shape,
                                 ModeAttr groups, ModeAttr lanes, ModeAttr slots) {
    if (!shape || !groups || !lanes || !slots)
        return error() << "layout needs shape and three ownership modes";
    int64_t logical = 1, capacity = 1;
    for (auto extent : shape.asArrayRef()) {
        if (extent <= 0 || extent > UINT32_MAX / logical)
            return error() << "layout size must fit positive u32";
        logical *= extent;
    }
    Digits strides;
    if (slots.replicated())
        return error() << "register slots cannot repeat elements; repeat them across lanes or SIMD groups";
    for (auto mode : {groups, lanes, slots}) {
        if (mode.size() > UINT32_MAX / capacity)
            return error() << "layout capacity exceeds u32";
        capacity *= mode.size();
        for (auto [extent, stride] : modeDigits(mode))
            if (extent > 1 && stride)
                strides.emplace_back(stride, extent);
            else if (extent > 1)
                capacity /= extent;
    }
    if (logical > capacity || lanes.size() != 32 || groups.size() > 32)
        return error() << "layout needs sufficient capacity and 1 to 32 full 32-lane SIMD groups";
    std::sort(strides.begin(), strides.end());
    int64_t expected = 1;
    for (auto [stride, extent] : strides) {
        if (stride != expected)
            return error() << "ownership must be a dense bijection without overlaps or gaps";
        expected *= extent;
    }
    return success();
}

int64_t LayoutAttr::size() const {
    auto shape = getShape().asArrayRef();
    return std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());
}

int64_t LayoutAttr::capacity() const { return getGroups().size() * getLanes().size() * getSlots().size(); }

int64_t LayoutAttr::threads() const { return getGroups().size() * getLanes().size(); }

int64_t LayoutAttr::groupOwner(int64_t index) const {
    return getGroups().inverse(index);
}

bool LayoutAttr::replicated() const { return getGroups().replicated() || getLanes().replicated(); }

LogicalResult TileType::verify(llvm::function_ref<InFlightDiagnostic()> error, Type element,
                               LayoutAttr layout) {
    if (!element || !layout || !(element.isF16() || element.isBF16() || element.isF32() ||
                                element.isInteger(32) || element.isInteger(1) || storageInteger(element)))
        return error() << "tile requires a layout and f16, bf16, f32, i32, i1 or 8- or 16-bit integer elements";
    return success();
}

LogicalResult ViewType::verify(llvm::function_ref<InFlightDiagnostic()> error, Type element,
                               DenseI64ArrayAttr shape, DenseI64ArrayAttr strides, int64_t offset,
                               int64_t extent, StringRef addressSpace) {
    if (!element || !(element.isF16() || element.isBF16() || element.isF32() || element.isInteger(32) ||
                      storageInteger(element)))
        return error() << "view requires f16, bf16, f32, i32 or 8- or 16-bit integer elements";
    if (!shape || !strides || (addressSpace != "device" && addressSpace != "threadgroup"))
        return error() << "view needs a shape, strides, and supported address space";
    int64_t bytes = element.getIntOrFloatBitWidth() / 8;
    if (extent == -1) {
        if (addressSpace != "device")
            return error() << "runtime view metadata requires device storage";
        if (offset < -1 || (offset > 0 && offset % bytes))
            return error() << "a runtime view's offset is -1 or fixed element-aligned bytes";
        if (llvm::any_of(shape.asArrayRef(), [](int64_t n) { return n < -1 || n > UINT32_MAX; }))
            return error() << "runtime view dimensions must be dynamic or fit u32";
        if (!strides.empty() && strides.size() != shape.size())
            return error() << "a runtime view fixes every stride or none";
        return success();
    }
    if (!shape || !strides ||
        shape.size() != strides.size() || offset < 0 || offset % bytes || extent < offset)
        return error() << "invalid static view descriptor";
    bool empty = llvm::is_contained(shape.asArrayRef(), 0);
    int64_t size = empty ? 0 : 1;
    __int128 lower = offset, upper = offset;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < 0 || shape[i] > UINT32_MAX || (size && shape[i] > UINT32_MAX / size))
            return error() << "view logical size must fit u32";
        size *= shape[i];
        if (empty)
            continue;
        __int128 span = __int128(shape[i] - 1) * strides[i] * bytes;
        lower += std::min<__int128>(0, span);
        upper += std::max<__int128>(0, span);
    }
    if (!empty && (lower < 0 || upper + bytes > extent))
        return error() << "view addresses exceed storage extent";
    return success();
}

bool ViewType::isDynamic() const { return getExtentBytes() == -1; }

LogicalResult MakeViewOp::verify() {
    auto result = getResult().getType();
    if (auto buffer = dyn_cast<MemRefType>(getBuffer().getType())) {
        if (buffer.getRank() != 1 || !buffer.isDynamicDim(0) || !buffer.getLayout().isIdentity() ||
            buffer.getMemorySpace() || buffer.getElementType() != result.getElementType() ||
            result.isDynamic() || result.getAddressSpace() != "device")
            return emitOpError("view requires a flat device binding of the same element type");
        return success();
    }
    auto source = cast<ViewType>(getBuffer().getType());
    if (!scratchAllocation(getBuffer()) || source.isDynamic() || result.isDynamic() ||
        source.getAddressSpace() != "threadgroup" || result.getAddressSpace() != "threadgroup" ||
        source.getElementType() != result.getElementType() || result.getExtentBytes() > source.getExtentBytes())
        return emitOpError("scratch view must preserve its allocation, element type and accessible extent");
    return success();
}

LogicalResult DimOp::verify() {
    auto type = getView().getType();
    auto view = dyn_cast<ViewType>(type);
    auto rank = view ? int64_t(view.getShape().size()) : cast<MemRefType>(type).getRank();
    if (getAxis() < 0 || getAxis() >= rank)
        return emitOpError("dimension axis must be within the view rank");
    return success();
}

LogicalResult BroadcastOp::verify() {
    if (getValue().getType() != getResult().getType().getElementType())
        return emitOpError("broadcast requires a matching scalar element type");
    return success();
}

LogicalResult CastOp::verify() {
    auto source = getValue().getType(), target = getResult().getType();
    auto a = dyn_cast<TileType>(source), b = dyn_cast<TileType>(target);
    if (a || b) {
        if (!a || !b || a.getLayout() != b.getLayout())
            return emitOpError("cast must preserve tile shape and ownership");
        source = a.getElementType();
        target = b.getElementType();
    }
    auto floating = [](Type element) { return element.isF16() || element.isBF16() || element.isF32(); };
    bool integer = source.isInteger(32) || storageInteger(source);
    if ((floating(source) && floating(target)) || (integer && target.isF32()) ||
        (storageInteger(source) && target.isInteger(32)) || (source.isInteger(32) && storageInteger(target)))
        return success();
    return emitOpError("cast converts floats to floats, u32 and 8- or 16-bit integers to f32, and u32 to and from "
                       "8- or 16-bit integers");
}

LogicalResult ReshapeOp::inferReturnTypes(MLIRContext* context, std::optional<Location> location, Adaptor adaptor,
                                          SmallVectorImpl<Type>& types) {
    auto tile = dyn_cast<TileType>(adaptor.getValue().getType());
    auto shape = adaptor.getShape();
    LayoutAttr layout;
    if (tile && llvm::all_of(shape, [](int64_t n) { return n >= 0; }))
        layout = tile.getLayout().reshape(shape);
    if (!layout)
        return emitOptionalError(location, "reshape must preserve the element count of a tile");
    types.push_back(TileType::get(context, tile.getElementType(), layout));
    return success();
}

LogicalResult StackOp::inferReturnTypes(MLIRContext* context, std::optional<Location> location, Adaptor adaptor,
                                        SmallVectorImpl<Type>& types) {
    auto inputs = adaptor.getInputs();
    auto tile = inputs.empty() ? TileType() : dyn_cast<TileType>(inputs.front().getType());
    int64_t axis = adaptor.getAxis();
    if (!tile || axis < 0 || axis > int64_t(tile.getLayout().getShape().size()))
        return emitOptionalError(location, "stack needs tiles and an axis within the new rank");
    auto layout = tile.getLayout().stack(inputs.size(), axis);
    if (!layout)
        return emitOptionalError(location, "tile axes split an ownership digit; convert the layout explicitly first");
    types.push_back(TileType::get(context, tile.getElementType(), layout));
    return success();
}

LogicalResult StackOp::verify() {
    if (llvm::any_of(getInputs(), [&](Value value) { return value.getType() != getInputs()[0].getType(); }))
        return emitOpError("stack inputs must have the same tile type and layout");
    return success();
}

LogicalResult TransposeOp::inferReturnTypes(MLIRContext* context, std::optional<Location> location, Adaptor adaptor,
                                            SmallVectorImpl<Type>& types) {
    auto tile = dyn_cast<TileType>(adaptor.getValue().getType());
    SmallVector<int64_t> sorted(adaptor.getAxes());
    llvm::sort(sorted);
    if (!tile || sorted.size() != tile.getLayout().getShape().size() ||
        llvm::any_of(llvm::enumerate(sorted), [](auto axis) { return axis.value() != int64_t(axis.index()); }))
        return emitOptionalError(location, "transpose axes must be a permutation of the tile axes");
    auto layout = tile.getLayout().transpose(adaptor.getAxes());
    if (!layout)
        return emitOptionalError(location, "tile axes split an ownership digit; convert the layout explicitly first");
    types.push_back(TileType::get(context, tile.getElementType(), layout));
    return success();
}

static bool positiveDivisor(Value value) {
    if (auto broadcast = value.getDefiningOp<BroadcastOp>())
        value = broadcast.getValue();
    auto constant = value.getDefiningOp<arith::ConstantOp>();
    auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
    return integer && integer.getType().isInteger(32) && !integer.getValue().isZero();
}

static bool comparison(StringRef opcode) {
    return opcode == "lt" || opcode == "le" || opcode == "gt" || opcode == "ge" || opcode == "eq" || opcode == "ne";
}

LogicalResult MapOp::inferReturnTypes(MLIRContext* context, std::optional<Location> location, Adaptor adaptor,
                                      SmallVectorImpl<Type>& types) {
    auto inputs = adaptor.getInputs();
    auto tile = inputs.empty() ? TileType() : dyn_cast<TileType>(inputs.front().getType());
    if (!tile)
        return emitOptionalError(location, "map requires tile operands");
    types.push_back(comparison(adaptor.getOpcode())
                        ? TileType::get(context, IntegerType::get(context, 1), tile.getLayout()) : tile);
    return success();
}

LogicalResult MapOp::verify() {
    auto opcode = getOpcode();
    bool unary = opcode == "neg";
    bool division = opcode == "floordiv" || opcode == "mod";
    bool floatingDivision = opcode == "truediv";
    bool compare = comparison(opcode);
    if (!(unary || compare || division || floatingDivision || opcode == "add" || opcode == "sub" || opcode == "mul") ||
        getInputs().size() != (unary ? 1 : 2))
        return emitOpError("invalid map opcode or arity");
    auto input = cast<TileType>(getInputs()[0].getType());
    for (auto value : getInputs())
        if (value.getType() != input)
            return emitOpError("map operands must have identical tile types");
    auto element = input.getElementType();
    if (division && (!element.isInteger(32) || !positiveDivisor(getInputs()[1])))
        return emitOpError("integer division requires u32 operands and a positive compile-time divisor");
    if (!(element.isF32() || element.isInteger(32)) || ((unary || floatingDivision) && !element.isF32()))
        return emitOpError(compare ? "comparison requires i32 or f32 inputs" : "unsupported map element type");
    return success();
}

static Type elementType(Type type) {
    if (auto tile = dyn_cast<TileType>(type))
        return tile.getElementType();
    return type;
}

LogicalResult MathOp::verify() {
    static const llvm::StringSet<> functions = {"exp", "exp2", "log", "log2", "reciprocal", "rsqrt", "sqrt", "sin",
                                                "cos", "tan", "tanh", "abs", "floor", "ceil"};
    if (!elementType(getValue().getType()).isF32())
        return emitOpError("math operations require an f32 scalar or tile; cast storage explicitly");
    if (!functions.contains(getOpcode()))
        return emitOpError("math requires exp, exp2, log, log2, reciprocal, rsqrt, sqrt, sin, cos, tan, tanh, abs, "
                           "floor or ceil");
    return success();
}

LogicalResult BitwiseOp::verify() {
    auto element = elementType(getResult().getType());
    bool shift = getOpcode() == "shl" || getOpcode() == "shr";
    if (!(element.isInteger(32) || (!shift && element.isInteger(1))) ||
        !(shift || getOpcode() == "and" || getOpcode() == "or" || getOpcode() == "xor"))
        return emitOpError("bitwise requires matching u32 or predicate values; shifts require u32");
    return success();
}

LogicalResult BitcastOp::verify() {
    auto source = getValue().getType(), target = getResult().getType();
    auto a = dyn_cast<TileType>(source), b = dyn_cast<TileType>(target);
    if ((a || b) && (!a || !b || a.getLayout() != b.getLayout()))
        return emitOpError("bitcast must preserve shape and ownership");
    for (auto element : {elementType(source), elementType(target)})
        if (!(element.isF32() || element.isInteger(32)))
            return emitOpError("bitcast requires f32 or u32 elements");
    return success();
}

LogicalResult IntegerDivOp::verify() {
    if (!elementType(getResult().getType()).isInteger(32) || !(getOpcode() == "floordiv" || getOpcode() == "mod"))
        return emitOpError("integer division requires identical u32 scalar or tile types and floordiv or mod");
    return success();
}

template <class Extremum>
static LogicalResult verifyExtremum(Extremum operation) {
    auto element = elementType(operation.getResult().getType());
    if (!(element.isF32() || element.isInteger(32)))
        return operation.emitOpError(operation->getName().stripDialect() +
                                     " requires identical f32 or u32 scalar or tile types");
    return success();
}

LogicalResult MaximumOp::verify() { return verifyExtremum(*this); }

LogicalResult MinimumOp::verify() { return verifyExtremum(*this); }

LogicalResult FmaOp::verify() {
    if (!elementType(getResult().getType()).isF32())
        return emitOpError("fma requires identical f32 scalar or tile types");
    return success();
}

LogicalResult SelectOp::verify() {
    auto type = getResult().getType(), condition = getCondition().getType();
    auto element = elementType(type);
    if (!(element.isF32() || element.isF16() || element.isBF16() || element.isInteger(32) || element.isInteger(1) ||
          storageInteger(element)))
        return emitOpError("select requires identical supported scalar or tile values");
    if (auto tile = dyn_cast<TileType>(type)) {
        auto predicate = dyn_cast<TileType>(condition);
        if (!predicate || !predicate.getElementType().isInteger(1) || predicate.getLayout() != tile.getLayout())
            return emitOpError("select predicate must have the same tile layout and i1 elements");
    } else if (!condition.isInteger(1))
        return emitOpError("scalar select requires an i1 predicate");
    return success();
}

static bool groupUniform(Value value, llvm::DenseSet<Value>& active, bool withinSimd) {
    if (isa<TileType, ViewType, MemRefType>(value.getType()))
        return false;
    // A carried cycle is uniform only if every incoming definition is uniform.
    if (!active.insert(value).second)
        return true;
    auto uniform = [&](Value input) { return groupUniform(input, active, withinSimd); };
    bool result = false;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
        auto parent = argument.getOwner()->getParentOp();
        if (isa<func::FuncOp>(parent))
            result = true;
        else if (auto loop = dyn_cast<scf::ForOp>(parent)) {
            result = uniform(loop.getLowerBound()) && uniform(loop.getUpperBound()) && uniform(loop.getStep());
            if (auto index = argument.getArgNumber())
                result = result && uniform(loop.getInitArgs()[index - 1]) &&
                         uniform(loop.getBody()->getTerminator()->getOperand(index - 1));
        }
    } else if (auto* operation = value.getDefiningOp()) {
        if (isa<ProgramIdOp, ItemOp, arith::ConstantOp>(operation))
            result = true;
        else if (isa<SimdIdOp>(operation))
            result = withinSimd;
        else if (auto load = dyn_cast<ScalarLoadOp>(operation))
            result = uniform(load.getIndex()) && uniform(load.getMask()) && uniform(load.getFallback());
        else if (auto dimension = dyn_cast<DimOp>(operation)) {
            auto memory = dimension.getView();
            auto view = dyn_cast<ViewType>(memory.getType());
            auto argument = dyn_cast<BlockArgument>(memory);
            result = (view && view.getShape()[dimension.getAxis()] >= 0) ||
                     (argument && isa<func::FuncOp>(argument.getOwner()->getParentOp()));
            if (auto subview = memory.getDefiningOp<SubviewOp>())
                result = llvm::all_of(subview.getOffsets(), uniform) && llvm::all_of(subview.getSizes(), uniform);
        } else if (operation->getName().getDialectNamespace() == "arith" ||
                 isa<CastOp, MathOp, MaximumOp, MinimumOp, FmaOp, SelectOp, BitwiseOp, BitcastOp,
                     IntegerDivOp>(operation))
            result = llvm::all_of(operation->getOperands(), uniform);
        else if (auto loop = dyn_cast<scf::ForOp>(operation)) {
            auto index = cast<OpResult>(value).getResultNumber();
            result = uniform(loop.getLowerBound()) && uniform(loop.getUpperBound()) && uniform(loop.getStep()) &&
                     uniform(loop.getInitArgs()[index]) && uniform(loop.getBody()->getTerminator()->getOperand(index));
        } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
            auto index = cast<OpResult>(value).getResultNumber();
            result = uniform(branch.getCondition()) && uniform(branch.thenYield().getOperand(index)) &&
                     uniform(branch.elseYield().getOperand(index));
        }
    }
    active.erase(value);
    return result;
}

bool groupUniform(Value value, bool withinSimd) {
    llvm::DenseSet<Value> active;
    return groupUniform(value, active, withinSimd);
}

LogicalResult ScalarLoadOp::verify() {
    Type element;
    if (auto view = dyn_cast<ViewType>(getBuffer().getType())) {
        if (view.getAddressSpace() != "device")
            return emitOpError("scalar loads require read-only device storage, not scratch");
        element = view.getElementType();
    } else if (auto buffer = dyn_cast<MemRefType>(getBuffer().getType())) {
        if (buffer.getRank() != 1 || !buffer.isDynamicDim(0) || !buffer.getLayout().isIdentity() ||
            buffer.getMemorySpace())
            return emitOpError("scalar loads require a flat dynamic device binding");
        element = buffer.getElementType();
    }
    if (!element || getFallback().getType() != element)
        return emitOpError("scalar load fallback and result must match the storage element");
    if (!groupUniform(getIndex()) || !groupUniform(getMask()) || !groupUniform(getFallback()))
        return emitOpError("scalar load index, mask and fallback must be threadgroup-uniform");
    auto function = (*this)->getParentOfType<func::FuncOp>();
    // A branch arm still being staged has no kernel yet; checking the whole module covers its binding.
    if (!function)
        return success();
    auto root = bindingRoot(getBuffer());
    auto binding = dyn_cast<BlockArgument>(root);
    if (!binding || binding.getOwner() != &function.getBody().front())
        return emitOpError("scalar loads must originate in a kernel device binding");
    bool written = false;
    function.walk([&](Operation* operation) {
        if (auto store = dyn_cast<StoreOp>(operation))
            written |= bindingRoot(store.getBuffer()) == root;
        else if (auto atomic = dyn_cast<AtomicOp>(operation))
            written |= bindingRoot(atomic.getBuffer()) == root;
        else if (auto atomic = dyn_cast<AtomicStoreOp>(operation))
            written |= bindingRoot(atomic.getBuffer()) == root;
    });
    if (written)
        return emitOpError("scalar load binding must be read-only throughout the kernel");
    return success();
}

static ViewType runtimeView(MLIRContext* context, Type element, ArrayRef<int64_t> shape) {
    return ViewType::get(context, element, DenseI64ArrayAttr::get(context, shape), DenseI64ArrayAttr::get(context, {}),
                         -1, -1, "device");
}

LogicalResult PartitionOp::inferReturnTypes(MLIRContext* context, std::optional<Location> location, Adaptor adaptor,
                                            SmallVectorImpl<Type>& types) {
    auto view = dyn_cast<ViewType>(adaptor.getView().getType());
    if (!view || view.getShape().size() != adaptor.getShape().size())
        return emitOptionalError(location, "partition needs a ranked view and a tile shape of its rank");
    types.push_back(runtimeView(context, view.getElementType(), adaptor.getShape()));
    return success();
}

LogicalResult PartitionOp::verify() {
    if (getView().getType().getAddressSpace() != "device")
        return emitOpError("partitions take device views, not scratch");
    int64_t size = 1;
    for (auto dimension : getShape()) {
        if (dimension <= 0 || dimension > UINT32_MAX / size)
            return emitOpError("partition shape must have a positive u32 size");
        size *= dimension;
    }
    if (!groupUniform(getIndex()))
        return emitOpError("partition index must be threadgroup-uniform");
    return success();
}

LogicalResult SubviewOp::inferReturnTypes(MLIRContext* context, std::optional<Location> location, Adaptor adaptor,
                                          SmallVectorImpl<Type>& types) {
    auto source = adaptor.getView().getType();
    auto view = dyn_cast<ViewType>(source);
    auto buffer = dyn_cast<MemRefType>(source);
    if (!view && !buffer)
        return emitOptionalError(location, "subview requires a flat device binding or ranked view");
    auto rank = view ? view.getShape().size() : 1;
    types.push_back(runtimeView(context, view ? view.getElementType() : buffer.getElementType(),
                                SmallVector<int64_t>(rank, -1)));
    return success();
}

LogicalResult SubviewOp::verify() {
    auto source = getView().getType();
    auto view = dyn_cast<ViewType>(source);
    auto buffer = dyn_cast<MemRefType>(source);
    if (view && view.getAddressSpace() != "device")
        return emitOpError("subviews take device views, not scratch");
    if (buffer && (buffer.getRank() != 1 || !buffer.isDynamicDim(0) ||
                   !buffer.getLayout().isIdentity() || buffer.getMemorySpace()))
        return emitOpError("subview requires a flat device binding or ranked view");
    auto rank = view ? view.getShape().size() : 1;
    if (getOffsets().size() != rank || getSizes().size() != rank)
        return emitOpError("subview offsets and sizes must match the source rank");
    auto uniform = [](Value value) { return groupUniform(value); };
    if (!llvm::all_of(getOffsets(), uniform) || !llvm::all_of(getSizes(), uniform))
        return emitOpError("subview offsets and sizes must be threadgroup-uniform");
    return success();
}

static bool uniformControl(Operation* operation, bool withinSimd = false) {
    if (auto loop = dyn_cast<scf::ForOp>(operation))
        return groupUniform(loop.getLowerBound(), withinSimd) && groupUniform(loop.getUpperBound(), withinSimd) &&
               groupUniform(loop.getStep(), withinSimd);
    if (auto branch = dyn_cast<scf::IfOp>(operation))
        return groupUniform(branch.getCondition(), withinSimd);
    return false;
}

// Whether the structured control flow around `operation`, up to its kernel, is uniform. A branch arm still being
// staged has no parent yet and ends the walk; checking the whole module sees the rest.
static bool uniformAncestors(Operation* operation, bool withinSimd = false) {
    for (auto* parent = operation->getParentOp(); parent && !isa<func::FuncOp>(parent); parent = parent->getParentOp())
        if (!uniformControl(parent, withinSimd))
            return false;
    return true;
}

static LogicalResult simdControl(Operation* operation) {
    if (!uniformAncestors(operation, true))
        return operation->emitOpError("matrix operations require SIMD-uniform control flow");
    return success();
}

// Operands stack one block per SIMD group, rows first: block g of each tile belongs to SIMD group g.
static bool perGroup(TileType tile, int64_t rows, int64_t columns) {
    auto layout = tile.getLayout();
    auto groups = layout.getGroups().size();
    if (layout.getShape().asArrayRef() != ArrayRef<int64_t>({groups * rows, columns}) ||
        layout.capacity() != layout.size())
        return false;
    for (int64_t group = 0; group < groups; ++group)
        if (layout.getGroups().offset(group) != group * rows * columns)
            return false;
    return true;
}

LogicalResult MmaOp::verify() {
    auto shape = getShape();
    auto a = getLeft().getType(), b = getRight().getType(), c = getAccumulator().getType();
    bool simdgroup = getUnit() == "simdgroup";
    if ((!simdgroup && getUnit() != "tensor") || shape.size() != 3 ||
        (simdgroup && shape != ArrayRef<int64_t>{8, 8, 8}))
        return emitOpError("mma unit must be simdgroup (8, 8, 8) or tensor with an (m, n, k) shape");
    auto [m, n, k] = std::tuple(shape[0], shape[1], shape[2]);
    if (!perGroup(a, m, k) || !perGroup(b, k, n) || !perGroup(c, m, n) ||
        a.getLayout().getGroups().size() != c.getLayout().getGroups().size() ||
        b.getLayout().getGroups().size() != c.getLayout().getGroups().size())
        return emitOpError("mma operands must stack one (m, k), (k, n) and (m, n) block per SIMD group");
    auto element = a.getElementType();
    bool narrow = element.isF16() || element.isBF16();
    if (b.getElementType() != element || !(narrow || element.isF32()) || !c.getElementType().isF32() ||
        (!simdgroup && !narrow))
        return emitOpError("mma multiplies f16, bf16 (or f32 on simdgroup) operands into f32");
    return simdControl(*this);
}

LogicalResult ScratchOp::verify() {
    auto type = getResult().getType();
    if (!isa<func::FuncOp>((*this)->getParentOp()) || type.isDynamic() ||
        type.getAddressSpace() != "threadgroup" || type.getOffsetBytes() != 0)
        return emitOpError("scratch needs static threadgroup storage at kernel scope");
    int64_t size = 1, bytes = type.getElementType().getIntOrFloatBitWidth() / 8;
    for (int64_t axis = int64_t(type.getShape().size()) - 1; axis >= 0; --axis) {
        auto extent = type.getShape()[axis];
        if (extent <= 0 || type.getStrides()[axis] != size)
            return emitOpError("scratch needs a contiguous positive shape");
        size *= extent;
    }
    if (type.getExtentBytes() != size * bytes)
        return emitOpError("scratch extent must match its shape and storage type");
    return success();
}

int64_t ScratchOp::allocationBytes() { return getResult().getType().getExtentBytes(); }

LogicalResult TensorMatmulOp::verify() {
    auto left = getLeft().getType(), right = getRight().getType();
    auto accumulator = getAccumulator().getType();
    if (left.getShape().size() != 2 || right.getShape().size() != 2)
        return emitOpError("tensor matmul needs rank-two views");
    auto [m, k] = std::pair(left.getShape()[0], left.getShape()[1]);
    auto shape = accumulator.getLayout().getShape().asArrayRef();
    if (right.getShape()[0] != k || shape != ArrayRef<int64_t>({m, right.getShape()[1]}))
        return emitOpError("tensor matmul needs views (M, K) and (K, N) and an (M, N) accumulator");
    auto element = left.getElementType();
    if (right.getElementType() != element || !(element.isF16() || element.isBF16() || element.isF32()) ||
        !accumulator.getElementType().isF32())
        return emitOpError("tensor matmul multiplies f16, bf16 or f32 operands into an f32 accumulator");
    for (auto memory : {getLeft(), getRight()}) {
        auto root = cast<ViewType>(viewRoot(memory).getType());
        auto strides = root.getStrides().asArrayRef();
        if (strides.size() != 2 || (strides[0] != 1 && strides[1] != 1))
            return emitOpError("tensor matmul views need a row- or column-major root view with fixed strides; "
                               "specialize ranked buffers with views=");
    }
    if (!uniformAncestors(*this))
        return emitOpError("tensor matmul requires threadgroup-uniform control flow");
    return success();
}

LogicalResult BarrierOp::verify() {
    if (!uniformAncestors(*this))
        return emitOpError("barrier requires threadgroup-uniform control flow");
    return success();
}

template <class ReductionOp>
static LogicalResult verifyReduction(ReductionOp operation) {
    auto source = operation.getValue().getType(), result = operation.getResult().getType();
    auto a = source.getLayout(), b = result.getLayout();
    auto shape = a.getShape().asArrayRef();
    auto axis = operation.getAxis();
    if (axis < 0 || axis >= int64_t(shape.size()) ||
        (operation.getOrder() != "pairwise" && operation.getOrder() != "fast"))
        return operation.emitOpError("reduction needs a valid axis and fast or pairwise order");
    if (source.getElementType() != result.getElementType() ||
        !(source.getElementType().isF32() || source.getElementType().isInteger(32)))
        return operation.emitOpError("reduction requires an f32 or u32 tile; cast storage values explicitly");
    SmallVector<int64_t> reduced(shape.begin(), shape.end());
    reduced.erase(reduced.begin() + axis);
    if (b.getShape().asArrayRef() != ArrayRef<int64_t>(reduced) ||
        a.getGroups().size() != b.getGroups().size())
        return operation.emitOpError("reduction must remove its axis while preserving participants");
    if (a.replicated() || b.replicated())
        return operation.emitOpError("reductions take and give tiles without repeated elements; convert the layout");
    bool shared = reductionNeedsScratch(a, b, axis);
    if (auto scratch = operation.getScratch()) {
        auto type = scratch.getType();
        if (!scratchAllocation(scratch) || type.isDynamic() || type.getShape() != a.getShape() ||
            type.getElementType() != source.getElementType() || !uniqueView(type))
            return operation.emitOpError(
                "reduction workspace must match input shape and accumulation type with unique addresses");
    } else if (shared)
        return operation.emitOpError("cross-SIMD reduction requires explicit scratch");
    if (!uniformAncestors(operation, !shared))
        return operation.emitOpError(shared ? "reduction requires threadgroup-uniform control flow"
                                            : "reduction requires SIMD-uniform control flow");
    return success();
}

LogicalResult ReduceSumOp::verify() { return verifyReduction(*this); }
LogicalResult ReduceMaxOp::verify() { return verifyReduction(*this); }

template <class ScanOp>
static LogicalResult verifyScan(ScanOp operation) {
    auto source = operation.getValue().getType();
    auto layout = source.getLayout();
    auto axis = operation.getAxis();
    if (axis < 0 || axis >= int64_t(layout.getShape().size()) ||
        (operation.getOrder() != "fast" && operation.getOrder() != "pairwise"))
        return operation.emitOpError("scan needs a valid axis and fast or pairwise order");
    if (!(source.getElementType().isF32() || source.getElementType().isInteger(32)))
        return operation.emitOpError("scans require an f32 or u32 tile; cast storage values explicitly");
    if (layout.replicated())
        return operation.emitOpError("scans take tiles without repeated elements; convert the layout");
    bool shared = scanNeedsScratch(layout, axis);
    if (auto scratch = operation.getScratch()) {
        auto type = scratch.getType();
        if (!scratchAllocation(scratch) || type.isDynamic() || type.getShape() != layout.getShape() ||
            type.getElementType() != source.getElementType() || !uniqueView(type))
            return operation.emitOpError("scan workspace must match input shape and type with unique addresses");
    } else if (shared)
        return operation.emitOpError("cross-SIMD scan requires explicit scratch");
    if (!uniformAncestors(operation, !shared))
        return operation.emitOpError(shared ? "scan requires threadgroup-uniform control flow"
                                            : "scan requires SIMD-uniform control flow");
    return success();
}

LogicalResult ScanSumOp::verify() { return verifyScan(*this); }
LogicalResult ScanMaxOp::verify() { return verifyScan(*this); }

LogicalResult BroadcastInDimOp::verify() {
    auto source = getValue().getType(), target = getResult().getType();
    auto a = source.getLayout(), b = target.getLayout();
    auto shape = a.getShape().asArrayRef(), output = b.getShape().asArrayRef(), axes = getAxesAttr().asArrayRef();
    if (source.getElementType() != target.getElementType() || a.getGroups().size() != b.getGroups().size() ||
        axes.size() != shape.size())
        return emitOpError("broadcast must preserve elements and participants");
    llvm::DenseSet<int64_t> mapped;
    for (size_t i = 0; i < axes.size(); ++i) {
        auto axis = axes[i];
        if (axis < 0 || axis >= int64_t(output.size()) || !mapped.insert(axis).second ||
            (shape[i] != 1 && shape[i] != output[axis]))
            return emitOpError("broadcast axes must be distinct and dimensions must match or expand one");
    }
    bool shared = broadcastCrossGroup(a, b, axes);
    if (auto scratch = getScratch()) {
        auto type = scratch.getType();
        auto storage =
            source.getElementType().isInteger(1) ? IntegerType::get(getContext(), 32) : source.getElementType();
        if (!scratchAllocation(scratch) || type.isDynamic() || type.getShape() != a.getShape() ||
            type.getElementType() != storage || !uniqueView(type))
            return emitOpError(
                "broadcast workspace must match the input shape with unique storage; predicates use u32");
    } else if (shared)
        return emitOpError("cross-SIMD broadcast requires explicit scratch");
    if (!uniformAncestors(*this, !shared))
        return emitOpError(shared ? "broadcast requires threadgroup-uniform control flow"
                                  : "broadcast requires SIMD-uniform control flow");
    return success();
}

LogicalResult ItemOp::inferReturnTypes(MLIRContext*, std::optional<Location> location, Adaptor adaptor,
                                       SmallVectorImpl<Type>& types) {
    auto tile = dyn_cast<TileType>(adaptor.getValue().getType());
    if (!tile)
        return emitOptionalError(location, "item requires a one-element tile");
    types.push_back(tile.getElementType());
    return success();
}

LogicalResult ItemOp::verify() {
    auto input = getValue().getType();
    auto layout = input.getLayout();
    auto element = input.getElementType();
    if (layout.size() != 1)
        return emitOpError("item requires a one-element tile");
    if (auto scratch = getScratch()) {
        auto type = scratch.getType();
        auto storage = element.isInteger(1) ? IntegerType::get(getContext(), 32) : element;
        if (!scratchAllocation(scratch) || type.isDynamic() || type.getElementType() != storage ||
            llvm::is_contained(type.getShape().asArrayRef(), 0))
            return emitOpError("item scratch must be nonempty with matching storage; predicates use u32");
    } else if (layout.getGroups().size() > 1)
        return emitOpError("cross-SIMD item requires explicit scratch");
    if (!uniformAncestors(*this))
        return emitOpError("item requires threadgroup-uniform control flow");
    return success();
}

LogicalResult ConvertLayoutOp::verify() {
    auto source = getValue().getType(), target = getResult().getType();
    auto a = source.getLayout(), b = target.getLayout();
    if (source.getElementType() != target.getElementType() || a.getShape() != b.getShape() ||
        a.getGroups().size() != b.getGroups().size())
        return emitOpError("conversion must preserve shape, element type and participants");
    if (getFree() && conversionMovesBetweenThreads(a, b))
        return emitOpError("conversion is not free: some element moves to another lane or SIMD group");
    if (auto scratch = getScratch()) {
        auto type = scratch.getType();
        if (getFree() || !scratchAllocation(scratch) || type.isDynamic() || type.getShape() != a.getShape() ||
            type.getElementType() != source.getElementType() || !uniqueView(type))
            return emitOpError("conversion scratch must match the tile shape and element type with unique storage");
    } else if (conversionCrossGroup(a, b))
        return emitOpError("cross-SIMD conversion needs explicit scratch and barriers");
    if (!uniformAncestors(*this, !getScratch()))
        return emitOpError("conversion requires uniform structured control flow across its participants");
    return success();
}

static bool carriedType(Type type) {
    return isa<TileType>(type) || type.isF16() || type.isBF16() || type.isF32() ||
           type.isInteger(32) || type.isInteger(1);
}

static bool u32Bound(Value value) {
    if (auto cast = value.getDefiningOp<arith::IndexCastUIOp>())
        return value.getType().isIndex() && cast->getOperand(0).getType().isInteger(32);
    if (auto constant = value.getDefiningOp<arith::ConstantOp>())
        if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
            return integer.getType().isIndex() && integer.getInt() >= 0 && uint64_t(integer.getInt()) <= UINT32_MAX;
    return false;
}

LogicalResult verifyContracts(func::FuncOp function, unsigned index) {
    auto type = function.getArgument(index).getType();
    if (auto attribute = function.getArgAttr(index, "carbon.range")) {
        auto range = dyn_cast<DenseI64ArrayAttr>(attribute);
        if (!type.isInteger(32) || !range || range.size() != 2 || range[0] < 0 || range[1] < range[0] ||
            uint64_t(range[1]) > UINT32_MAX)
            return function.emitOpError("carbon.range requires an inclusive u32 pair on a u32 argument");
    }
    if (auto attribute = function.getArgAttr(index, "carbon.min_bytes")) {
        auto bytes = dyn_cast<IntegerAttr>(attribute);
        auto memory = dyn_cast<MemRefType>(type);
        auto view = dyn_cast<ViewType>(type);
        if (!bytes || !bytes.getType().isInteger(64) || (!memory && !view))
            return function.emitOpError("carbon.min_bytes requires an i64 on a buffer argument");
        auto width = (memory ? memory.getElementType() : view.getElementType()).getIntOrFloatBitWidth() / 8;
        if (bytes.getInt() < 0 || bytes.getInt() % width || uint64_t(bytes.getInt() / width) > UINT32_MAX)
            return function.emitOpError("carbon.min_bytes requires whole u32-indexed elements");
    }
    return success();
}

LogicalResult verifyLayoutKernels(ModuleOp module) {
    for (auto function : module.getOps<func::FuncOp>()) {
        auto version = function->getAttrOfType<IntegerAttr>("carbon.layout_ir");
        auto threads = function->getAttrOfType<IntegerAttr>("carbon.threads");
        if (!version || !version.getType().isInteger(64) || version.getInt() != 2 || !threads ||
            !threads.getType().isInteger(64) || threads.getInt() < 32 || threads.getInt() > 1024 ||
            threads.getInt() % 32 || function->hasAttr("carbon.tile_size") || function->hasAttr("carbon.layout"))
            return function.emitOpError(
                "layout IR requires version 2, valid carbon.threads and no function-wide tile_size or layout");
        if (function.isExternal() || !llvm::hasSingleElement(function.getBody()) || function.getNumResults())
            return function.emitOpError("layout IR requires one kernel block without returned values");
        if (auto attribute = function->getAttr("carbon.max_grid")) {
            auto bound = dyn_cast<IntegerAttr>(attribute);
            if (!bound || !bound.getType().isInteger(64) || bound.getInt() < 1 ||
                uint64_t(bound.getInt()) > UINT32_MAX)
                return function.emitOpError("carbon.max_grid requires a positive u32 count in an i64 attribute");
        }
        for (auto argument : function.getArguments()) {
            auto type = argument.getType();
            if (auto buffer = dyn_cast<MemRefType>(type)) {
                if (buffer.getRank() != 1 || !buffer.isDynamicDim(0) || !buffer.getLayout().isIdentity() ||
                    buffer.getMemorySpace() ||
                    !(buffer.getElementType().isF16() || buffer.getElementType().isBF16() ||
                      buffer.getElementType().isF32() || buffer.getElementType().isInteger(32) ||
                      storageInteger(buffer.getElementType())))
                    return function.emitOpError(
                        "kernel buffers must be flat f16, bf16, f32, i32 or 8- or 16-bit integer device bindings");
            } else if (auto view = dyn_cast<ViewType>(type)) {
                if (!view.isDynamic())
                    return function.emitOpError("view arguments require runtime view metadata");
            } else if (!type.isF32() && !type.isInteger(32))
                return function.emitOpError("kernel arguments must be raw bindings or scalars");
            if (failed(verifyContracts(function, argument.getArgNumber())))
                return failure();
        }
        auto walk = function.walk([&](Operation* operation) -> WalkResult {
            if (isa<arith::DivUIOp, arith::RemUIOp>(operation) &&
                (!operation->getResult(0).getType().isInteger(32) ||
                 !positiveDivisor(operation->getOperand(1)))) {
                operation->emitOpError("integer division requires u32 operands and a positive compile-time divisor");
                return WalkResult::interrupt();
            }
            if (operation != function && operation->getNumRegions() && !isa<scf::ForOp, scf::IfOp>(operation)) {
                operation->emitOpError("only structured for loops and if branches are supported");
                return WalkResult::interrupt();
            }
            if (auto branch = dyn_cast<scf::IfOp>(operation)) {
                if (!groupUniform(branch.getCondition(), true) ||
                    !llvm::all_of(branch.getResultTypes(), carriedType)) {
                    operation->emitOpError("branch needs a SIMD-uniform condition and scalar or tile results");
                    return WalkResult::interrupt();
                }
            }
            if (auto loop = dyn_cast<scf::ForOp>(operation)) {
                auto step = loop.getStep().getDefiningOp<arith::ConstantOp>();
                auto increment = step ? dyn_cast<IntegerAttr>(step.getValue()) : IntegerAttr();
                if (!loop.getInductionVar().getType().isIndex() || !u32Bound(loop.getLowerBound()) ||
                    !u32Bound(loop.getUpperBound()) || !increment || increment.getInt() <= 0 ||
                    uint64_t(increment.getInt()) > UINT32_MAX || !groupUniform(loop.getLowerBound(), true) ||
                    !groupUniform(loop.getUpperBound(), true)) {
                    operation->emitOpError("loop needs SIMD-uniform u32 bounds and a positive constant u32 step");
                    return WalkResult::interrupt();
                }
                for (auto value : loop.getInitArgs()) {
                    if (!carriedType(value.getType())) {
                        operation->emitOpError("loop-carried values must be supported scalars or tiles");
                        return WalkResult::interrupt();
                    }
                }
            }
            for (auto type : operation->getResultTypes()) {
                if (isa<VectorType>(type)) {
                    operation->emitOpError("vector types are not part of layout IR; use tiles");
                    return WalkResult::interrupt();
                }
                if (auto tile = dyn_cast<TileType>(type)) {
                    if (tile.getLayout().getGroups().size() * 32 != threads.getInt()) {
                        operation->emitOpError("tile ownership must match carbon.threads");
                        return WalkResult::interrupt();
                    }
                }
            }
            return WalkResult::advance();
        });
        if (walk.wasInterrupted())
            return failure();
    }
    return success();
}

}
