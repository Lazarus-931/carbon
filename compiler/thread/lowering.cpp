#include "thread/lowering.h"
#include "thread/analysis.h"
#include "thread/dialect.h"
#include "tile/analysis.h"
#include "tile/dialect.h"
#include "tile/module.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <map>
#include <set>
#include <stdexcept>

using namespace mlir;

namespace carbon::msl {
namespace {

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument(message); }

// Narrow storage integers use signless registers; casts interpret their signedness.
Type registers(Type type) {
    return ir::storageInteger(type) ? IntegerType::get(type.getContext(), type.getIntOrFloatBitWidth()) : type;
}

Type memoryElement(Type type) {
    if (auto view = dyn_cast<ir::ViewType>(type))
        return registers(view.getElementType());
    return registers(cast<MemRefType>(type).getElementType());
}

int64_t viewSize(ir::ViewType view) {
    if (llvm::is_contained(view.getShape().asArrayRef(), 0))
        return 0;
    int64_t size = 1;
    for (auto extent : view.getShape().asArrayRef())
        size *= extent;
    return size;
}

bool anyIndex(ir::LayoutAttr layout, int64_t slot, const std::function<bool(int64_t)>& predicate) {
    for (int64_t group = 0; group < layout.getGroups().size(); ++group)
        for (int64_t lane = 0; lane < 32; ++lane) {
            auto index = layout.index(group, lane, slot);
            if (index < layout.size() && predicate(index))
                return true;
        }
    return false;
}

// Metal's 31 buffer slots include bindings, write-check buffers and the constant block.
void checkBindings(const MemoryPlan& memory) {
    auto checked = memory.checkedBindings.size();
    if (memory.access.size() + checked + (checked ? 1 : 0) + 1 > 31)
        fail("too many Metal buffer bindings");
}

// `shape` (cl.dim) keeps full partition tiles but clips subviews; `limits` clips accesses.
// `requested` keeps unclipped subview extents. `active` excludes out-of-grid partitions and inactive parents.
// Offsets count elements.
struct ViewValues {
    SmallVector<Value> shape, strides, limits, requested;
    Value offset, active;
};

class Lowering {
  public:
    explicit Lowering(func::FuncOp function, bool checking = false)
        : function_(function), context_(function.getContext()), builder_(context_), location_(function.getLoc()),
          checking_(checking) {
        i1_ = builder_.getI1Type();
        i16_ = builder_.getIntegerType(16);
        i32_ = builder_.getI32Type();
        i64_ = builder_.getI64Type();
    }

    ThreadKernel run() {
        ThreadKernel kernel;
        kernel.module = ModuleOp::create(location_);
        auto thread = func::FuncOp::create(location_, "carbon_kernel", FunctionType::get(context_, {}, {}));
        thread_ = thread;
        kernel.module->push_back(thread);
        builder_.setInsertionPointToStart(thread.addEntryBlock());
        kernel.memory = planMemory(function_);
        checkBindings(kernel.memory);
        memory_ = &kernel.memory;
        finalScratchUsers_ = kernel.memory.finalScratchUsers;
        auto threads = function_->getAttrOfType<IntegerAttr>("carbon.threads").getInt();
        lane_ = BuiltinOp::create(builder_, location_, i32_, "lane", 32);
        simdGroup_ = BuiltinOp::create(builder_, location_, i32_, "simd_group", threads / 32);
        size_t count = 0;
        for (auto argument : function_.getArguments()) {
            if (isa<MemRefType, ir::ViewType>(argument.getType())) {
                auto binding = count++;
                bindings_[argument] = binding;
                auto pointer = PointerType::get(context_, "device", memoryElement(argument.getType()));
                values_[argument] = {ArgumentOp::create(builder_, location_, pointer, binding)};
                if (auto view = dyn_cast<ir::ViewType>(argument.getType())) {
                    auto slot = std::to_string(binding);
                    ViewValues mapping;
                    auto fixed = view.getStrides().asArrayRef();
                    for (size_t axis = 0; axis < view.getShape().size(); ++axis) {
                        auto suffix = slot + "_" + std::to_string(axis);
                        auto extent = view.getShape()[axis];
                        mapping.shape.push_back(extent >= 0 ? u32(extent) : param(i32_, "shape" + suffix));
                        if (!fixed.empty()) {
                            mapping.strides.push_back(i64(fixed[axis]));
                            continue;
                        }
                        auto high = arith::ShLIOp::create(builder_, location_,
                                                          widen(param(i32_, "stride_hi" + suffix)), integer(i64_, 32));
                        mapping.strides.push_back(arith::OrIOp::create(builder_, location_, high,
                                                                       widen(param(i32_, "stride_lo" + suffix))));
                    }
                    mapping.limits = mapping.shape;
                    auto bytes = view.getElementType().getIntOrFloatBitWidth() / 8;
                    mapping.offset = view.getOffsetBytes() >= 0 ? i64(view.getOffsetBytes() / bytes)
                                                                : widen(param(i32_, "offset" + slot));
                    mapping.active = boolean(true);
                    views_[argument] = mapping;
                }
            } else {
                auto field = "arg" + std::to_string(argument.getArgNumber());
                auto range = function_.getArgAttrOfType<DenseI64ArrayAttr>(argument.getArgNumber(), "carbon.range");
                values_[argument] = {ParamOp::create(builder_, location_, argument.getType(), field,
                                                     range ? range[0] : 0, range ? range[1] : UINT32_MAX)};
            }
        }
        if (auto& checked = kernel.memory.checkedBindings; !checked.empty()) {
            auto words = PointerType::get(context_, "device", i32_);
            for (auto [check, at] : llvm::enumerate(checked))
                shadows_[at] = {ArgumentOp::create(builder_, location_, words, count + check), check};
            record_ = ArgumentOp::create(builder_, location_, words, count + checked.size());
            writer_ = add(add(add(mul(group(), u32(threads)), mul(simdGroup_, u32(32))), lane_), u32(1));
        }
        emitBlock(function_.getBody().front());
        func::ReturnOp::create(builder_, location_);
        return kernel;
    }

    void checkScratch() {
        auto threads = function_->getAttrOfType<IntegerAttr>("carbon.threads").getInt();
        ThreadValues values(threads);
        llvm::DenseMap<Operation*, std::optional<std::vector<ir::ScratchTouch>>> known;
        auto touches = [&](Operation* access) {
            auto [found, inserted] = known.try_emplace(access);
            if (inserted)
                found->second = scratchTouches(access, threads, values);
            return found->second;
        };
        ir::Diagnostics diagnostics(*context_);
        if (failed(ir::verifyScratch(function_, touches)))
            fail(diagnostics.text);
    }

  private:
    Value integer(Type type, uint64_t value) {
        auto width = isa<IndexType>(type) ? 64 : type.getIntOrFloatBitWidth();
        return arith::ConstantOp::create(builder_, location_, IntegerAttr::get(type, APInt(width, value)));
    }

    Value u32(uint64_t value) { return integer(i32_, uint32_t(value)); }
    Value i64(int64_t value) { return integer(i64_, uint64_t(value)); }
    Value boolean(bool value) { return integer(i1_, value); }

    Value zero(Type type) {
        if (isa<FloatType>(type))
            return arith::ConstantOp::create(builder_, location_, FloatAttr::get(type, 0.0));
        return integer(type, 0);
    }

    Value param(Type type, const std::string& field) { return ParamOp::create(builder_, location_, type, field); }

    Value group() { return BuiltinOp::create(builder_, location_, i32_, "group", int64_t(memory_->maxGrid)); }

    Value bound(Value memory) {
        auto index = binding(memory);
        auto bytes = memoryElement(ir::bindingRoot(memory).getType()).getIntOrFloatBitWidth() / 8;
        return ParamOp::create(builder_, location_, i32_, "bound" + std::to_string(index),
                               int64_t(memory_->bindingBytes[index] / bytes));
    }

    Value add(Value a, Value b) { return arith::AddIOp::create(builder_, location_, a, b); }
    Value sub(Value a, Value b) { return arith::SubIOp::create(builder_, location_, a, b); }
    Value mul(Value a, Value b) { return arith::MulIOp::create(builder_, location_, a, b); }
    Value divide(Value a, Value b) { return arith::DivUIOp::create(builder_, location_, a, b); }
    Value remainder(Value a, Value b) { return arith::RemUIOp::create(builder_, location_, a, b); }
    Value bitAnd(Value a, Value b) { return arith::AndIOp::create(builder_, location_, a, b); }
    Value bitOr(Value a, Value b) { return arith::OrIOp::create(builder_, location_, a, b); }
    Value bitXor(Value a, Value b) { return arith::XOrIOp::create(builder_, location_, a, b); }
    Value shiftLeft(Value a, Value b) { return arith::ShLIOp::create(builder_, location_, a, b); }
    Value shiftRight(Value a, Value b) { return arith::ShRUIOp::create(builder_, location_, a, b); }
    Value minimum(Value a, Value b) {
        if (isa<FloatType>(a.getType()))
            return MinimumOp::create(builder_, location_, a.getType(), a, b);
        return arith::MinUIOp::create(builder_, location_, a, b);
    }
    Value maximum(Value a, Value b) {
        if (isa<FloatType>(a.getType()))
            return MaximumOp::create(builder_, location_, a.getType(), a, b);
        return arith::MaxUIOp::create(builder_, location_, a, b);
    }
    Value select(Value c, Value a, Value b) { return arith::SelectOp::create(builder_, location_, c, a, b); }
    Value compare(arith::CmpIPredicate p, Value a, Value b) {
        return arith::CmpIOp::create(builder_, location_, p, a, b);
    }
    Value less(Value a, Value b) { return compare(arith::CmpIPredicate::ult, a, b); }
    Value equal(Value a, Value b) { return compare(arith::CmpIPredicate::eq, a, b); }
    Value nonzero(Value a) { return compare(arith::CmpIPredicate::ne, a, zero(a.getType())); }
    Value both(Value a, Value b) { return bitAnd(a, b); }
    Value bitcast(Type type, Value value) { return arith::BitcastOp::create(builder_, location_, type, value); }
    Value extend(Type type, Value value) { return arith::ExtUIOp::create(builder_, location_, type, value); }
    Value truncate(Type type, Value value) { return arith::TruncIOp::create(builder_, location_, type, value); }
    Value widen(Value value) { return value.getType() == i64_ ? value : extend(i64_, value); }
    Value word(Value value) { return value.getType() == i32_ ? value : truncate(i32_, value); }

    Value all(ArrayRef<Value> conditions) {
        Value result;
        for (auto condition : conditions)
            result = result ? both(result, condition) : condition;
        return result ? result : boolean(true);
    }

    Value product(ArrayRef<Value> values, size_t first = 0) {
        Value result = u32(1);
        for (size_t i = first; i < values.size(); ++i)
            result = mul(result, values[i]);
        return result;
    }

    bool visible(Value value) {
        auto* region = value.getParentRegion();
        return region && region->isAncestor(builder_.getInsertionBlock()->getParent());
    }

    // Runtime ModeAttr::offset/holder; the inverse keeps `preferred` on stride-zero digits.
    Value mode(ir::ModeAttr attribute, Value index, bool inverse = false, Value preferred = {}) {
        Value result = u32(0);
        int64_t divisor = attribute.size();
        for (auto [extent, stride] : attribute.digits()) {
            divisor /= extent;
            if (extent == 1)
                continue;
            if (inverse && !stride) {
                result = add(result, mul(remainder(divide(preferred, u32(divisor)), u32(extent)), u32(divisor)));
                continue;
            }
            int64_t division = inverse ? stride : divisor;
            int64_t multiplier = inverse ? divisor : stride;
            Value term = index;
            if (division != 1)
                term = divide(term, u32(division));
            if (inverse || extent != attribute.size())
                term = remainder(term, u32(extent));
            if (multiplier != 1)
                term = mul(term, u32(multiplier));
            result = add(result, term);
        }
        return result;
    }

    Value logicalIndex(ir::LayoutAttr layout, int64_t slot) {
        auto key = std::make_pair(layout.getAsOpaquePointer(), slot);
        if (auto found = indices_.find(key); found != indices_.end() && visible(found->second))
            return found->second;
        auto offset = u32(layout.getSlots().offset(slot));
        auto index = add(add(mode(layout.getGroups(), simdGroup_), mode(layout.getLanes(), lane_)), offset);
        indices_[key] = index;
        return index;
    }

    Value canonical(ir::LayoutAttr layout) {
        Value result = boolean(true);
        for (auto [mode, id] : {std::pair(layout.getGroups(), simdGroup_), std::pair(layout.getLanes(), lane_)}) {
            int64_t divisor = mode.size();
            for (auto [extent, stride] : mode.digits()) {
                divisor /= extent;
                if (extent > 1 && !stride)
                    result = both(result, equal(remainder(divide(id, u32(divisor)), u32(extent)), u32(0)));
            }
        }
        return result;
    }

    // Nonnegative strides put each slot's largest element in the last lane of the last SIMD group.
    bool padded(ir::LayoutAttr layout, int64_t slot) {
        return layout.index(layout.getGroups().size() - 1, 31, slot) >= layout.size();
    }

    Value value(Value item, size_t slot = 0) const {
        auto found = values_.find(item);
        if (found == values_.end() || slot >= found->second.size() || !found->second[slot])
            fail("value used before its slot definition");
        return found->second[slot];
    }

    size_t slotCount(Type type) const {
        auto tile = dyn_cast<ir::TileType>(type);
        return tile ? tile.getLayout().getSlots().size() : 1;
    }

    Type element(Type type) const {
        if (auto tile = dyn_cast<ir::TileType>(type))
            return tile.getElementType();
        return type;
    }

    void result(Value output, Value computed) { values_[output].push_back(computed); }

    auto slotRange(size_t count) const { return llvm::seq(size_t(0), count); }

    size_t binding(Value memory) {
        memory = ir::bindingRoot(memory);
        auto found = bindings_.find(memory);
        if (found == bindings_.end())
            fail("memory must originate in a kernel binding");
        return found->second;
    }

    Value pointer(Value memory) { return value(memory); }

    Value reinterpret(Value memory, Type elementType) {
        auto type = cast<PointerType>(memory.getType());
        if (type.getElementType() == elementType)
            return memory;
        return ReinterpretOp::create(builder_, location_, PointerType::get(context_, type.getSpace(), elementType),
                                     memory);
    }

    Value scratchIndex(Value memory, Value index) {
        auto allocation = ir::scratchAllocation(memory);
        auto view = cast<ir::ViewType>(memory.getType());
        if (llvm::is_contained(view.getShape().asArrayRef(), 0))
            return u32(0);
        Value address = index;
        if (memory != allocation.getResult()) {
            auto bytes = view.getElementType().getIntOrFloatBitWidth() / 8;
            address = u32(uint32_t(view.getOffsetBytes() / bytes));
            int64_t divisor = 1;
            for (int64_t axis = int64_t(view.getShape().size()) - 1; axis >= 0; --axis) {
                auto extent = view.getShape()[axis];
                if (extent > 1 && view.getStrides()[axis]) {
                    auto coordinate = remainder(divide(index, u32(divisor)), u32(extent));
                    address = add(address, mul(coordinate, u32(uint32_t(view.getStrides()[axis]))));
                }
                divisor *= extent;
            }
        }
        return address;
    }

    Value viewAddress(const ViewValues& mapping, ArrayRef<Value> coordinates) {
        // Valid offsets fit u32; modular arithmetic also preserves negative strides.
        Value address = word(mapping.offset);
        for (size_t axis = 0; axis < coordinates.size(); ++axis)
            address = add(address, mul(word(coordinates[axis]), word(mapping.strides[axis])));
        return address;
    }

    // Empty extents divide as 1; accesses check the flat bound first.
    SmallVector<Value> coordinates(const ViewValues& mapping, Value index) {
        SmallVector<Value> result(mapping.shape.size());
        for (size_t axis = mapping.shape.size(); axis-- > 1;) {
            auto extent = maximum(mapping.shape[axis], u32(1));
            result[axis] = remainder(index, extent);
            index = divide(index, extent);
        }
        result[0] = mapping.shape.size() == 1 ? index : remainder(index, maximum(mapping.shape[0], u32(1)));
        return result;
    }

    Value address(Value memory, Value index) {
        if (ir::scratchAllocation(memory))
            return scratchIndex(memory, index);
        auto view = dyn_cast<ir::ViewType>(memory.getType());
        if (!view)
            return index;
        if (view.isDynamic()) {
            const auto& mapping = views_.at(memory);
            return viewAddress(mapping, coordinates(mapping, index));
        }
        auto bytes = view.getElementType().getIntOrFloatBitWidth() / 8;
        Value result = u32(uint32_t(view.getOffsetBytes() / bytes));
        int64_t divisor = viewSize(view);
        if (!divisor)
            return u32(0);
        if (auto stride = ir::flatStride(view)) {
            return *stride ? add(result, mul(index, u32(uint32_t(*stride)))) : result;
        }
        for (size_t i = 0; i < view.getShape().size(); ++i) {
            divisor /= view.getShape()[i];
            if (view.getShape()[i] == 1 || view.getStrides()[i] == 0)
                continue;
            auto term = remainder(divide(index, u32(divisor)), u32(view.getShape()[i]));
            result = add(result, mul(term, u32(uint32_t(view.getStrides()[i]))));
        }
        return result;
    }

    Value limit(Value memory) {
        auto view = dyn_cast<ir::ViewType>(memory.getType());
        if (!view)
            return bound(memory);
        if (view.isDynamic())
            return product(views_.at(memory).shape);
        return u32(viewSize(view));
    }

    Value active(Value memory, ValueRange indices, Value mask, int64_t slot) {
        auto layout = cast<ir::TileType>(indices.front().getType()).getLayout();
        auto view = dyn_cast<ir::ViewType>(memory.getType());
        if (indices.size() > 1) {
            SmallVector<Value> valid{value(mask, slot)};
            if (padded(layout, slot))
                valid.push_back(less(logicalIndex(layout, slot), u32(layout.size())));
            if (view.isDynamic())
                valid.push_back(views_.at(memory).active);
            for (auto [axis, index] : llvm::enumerate(indices))
                valid.push_back(less(value(index, slot), view.isDynamic() ? views_.at(memory).limits[axis]
                                                                          : u32(view.getShape()[axis])));
            return all(valid);
        }
        auto index = value(indices.front(), slot);
        Value predicate = both(value(mask, slot), less(index, limit(memory)));
        if (padded(layout, slot))
            predicate = both(less(logicalIndex(layout, slot), u32(layout.size())), predicate);
        if (!view || !view.isDynamic())
            return predicate;
        const auto& mapping = views_.at(memory);
        SmallVector<Value> valid{predicate, mapping.active};
        for (auto [coordinate, limit] : llvm::zip(coordinates(mapping, index), mapping.limits))
            valid.push_back(less(coordinate, limit));
        return all(valid);
    }

    Value address(Value memory, ValueRange indices, int64_t slot) {
        if (indices.size() == 1)
            return address(memory, value(indices.front(), slot));
        auto view = cast<ir::ViewType>(memory.getType());
        SmallVector<Value> coordinates;
        for (auto index : indices)
            coordinates.push_back(value(index, slot));
        if (view.isDynamic())
            return viewAddress(views_.at(memory), coordinates);
        auto offset = view.getOffsetBytes() / (view.getElementType().getIntOrFloatBitWidth() / 8);
        Value result = u32(uint32_t(offset));
        for (auto [coordinate, stride] : llvm::zip(coordinates, view.getStrides().asArrayRef()))
            if (stride)
                result = add(result, mul(coordinate, u32(uint32_t(stride))));
        return result;
    }

    Value scalarActive(ir::ScalarLoadOp load) {
        auto memory = load.getBuffer();
        auto index = value(load.getIndex());
        SmallVector<Value> valid{value(load.getMask())};
        if (auto view = dyn_cast<ir::ViewType>(memory.getType())) {
            if (view.isDynamic()) {
                const auto& mapping = views_.at(memory);
                valid.push_back(mapping.active);
                valid.push_back(less(index, product(mapping.shape)));
                for (auto [coordinate, limit] : llvm::zip(coordinates(mapping, index), mapping.limits))
                    valid.push_back(less(coordinate, limit));
            } else
                valid.push_back(less(index, u32(viewSize(view))));
        } else
            valid.push_back(less(index, bound(memory)));
        return all(valid);
    }

    Value guardedLoad(Value memory, Value index, Value condition, Value fallback) {
        return GuardedLoadOp::create(builder_, location_, TypeRange{fallback.getType()}, memory, index, condition,
                                     ValueRange{fallback}).getResult(0);
    }

    GuardedStoreOp guardedStore(Value memory, Value index, Value stored, Value condition) {
        return GuardedStoreOp::create(builder_, location_, memory, index, condition, ValueRange{stored});
    }

    void barrier() { BarrierOp::create(builder_, location_); }

    void partition(ir::PartitionOp operation) {
        auto input = operation.getView(), output = operation.getResult();
        const auto parent = views_.at(input);
        auto shape = output.getType().getShape().asArrayRef();
        auto index = value(operation.getIndex());
        ViewValues grid, child;
        child.strides = parent.strides;
        for (size_t axis = 0; axis < shape.size(); ++axis) {
            auto block = u32(shape[axis]);
            auto partial = extend(i32_, nonzero(remainder(parent.shape[axis], block)));
            grid.shape.push_back(add(divide(parent.shape[axis], block), partial));
            child.shape.push_back(block);
        }
        SmallVector<Value> active{parent.active, less(index, product(grid.shape))};
        Value offset = parent.offset;
        auto tile = coordinates(grid, index);
        for (size_t axis = 0; axis < shape.size(); ++axis) {
            auto origin = mul(tile[axis], u32(shape[axis]));
            auto inside = less(origin, parent.limits[axis]);
            active.push_back(inside);
            child.limits.push_back(select(inside, minimum(u32(shape[axis]), sub(parent.limits[axis], origin)), u32(0)));
            offset = add(offset, mul(widen(origin), parent.strides[axis]));
        }
        child.active = all(active);
        child.offset = select(child.active, offset, i64(0));
        views_[output] = child;
        values_[output] = values_.lookup(input);
    }

    void subview(ir::SubviewOp operation) {
        Value input = operation.getView(), output = operation.getResult();
        ViewValues parent, child;
        if (isa<ir::ViewType>(input.getType()))
            parent = views_.at(input);
        else {
            parent.shape = parent.limits = {bound(input)};
            parent.strides = {i64(1)};
            parent.offset = i64(0);
            parent.active = boolean(true);
        }
        child.strides = parent.strides;
        SmallVector<Value> nonempty{parent.active};
        Value offset = parent.offset;
        for (size_t axis = 0; axis < parent.shape.size(); ++axis) {
            auto origin = value(operation.getOffsets()[axis]);
            auto size = value(operation.getSizes()[axis]);
            auto inside = both(parent.active, less(origin, parent.limits[axis]));
            child.shape.push_back(select(inside, minimum(size, sub(parent.limits[axis], origin)), u32(0)));
            child.requested.push_back(size);
            nonempty.push_back(nonzero(child.shape.back()));
            offset = add(offset, mul(widen(origin), parent.strides[axis]));
        }
        child.limits = child.shape;
        child.active = parent.active;
        child.offset = select(all(nonempty), offset, i64(0));
        views_[output] = child;
        values_[output] = values_.lookup(input);
    }

    void staticView(Value output, ir::ViewType type, int64_t offset) {
        ViewValues mapping;
        for (auto size : type.getShape().asArrayRef())
            mapping.shape.push_back(u32(size));
        for (auto stride : type.getStrides().asArrayRef())
            mapping.strides.push_back(i64(stride));
        mapping.limits = mapping.shape;
        mapping.offset = i64(offset);
        mapping.active = boolean(true);
        views_[output] = mapping;
    }

    // Shuffles carry bf16, predicates and storage integers as u32 to preserve their bits.
    Type carrierType(Type type) {
        return type.isBF16() || type.isInteger(1) || ir::storageInteger(type) ? i32_ : type;
    }

    Value carrierValue(Type type, Value item) {
        if (type.isInteger(1) || ir::storageInteger(type))
            return extend(i32_, item);
        if (type.isBF16())
            return extend(i32_, bitcast(i16_, item));
        return item;
    }

    Value restoreValue(Type type, Value item) {
        if (type.isInteger(1))
            return nonzero(item);
        if (ir::storageInteger(type))
            return truncate(registers(type), item);
        if (type.isBF16())
            return bitcast(type, truncate(i16_, item));
        return item;
    }

    Value storageBits(Type type, Value item) {
        if (type.isF32())
            return bitcast(i32_, item);
        if (type.isF16() || type.isBF16())
            return extend(i32_, bitcast(i16_, item));
        if (type.isInteger(1) || ir::storageInteger(type))
            return extend(i32_, item);
        return item;
    }

    Value storageValue(Type type, Value bits) {
        if (type.isF32())
            return bitcast(type, bits);
        if (type.isF16() || type.isBF16())
            return bitcast(type, truncate(i16_, bits));
        if (type.isInteger(1))
            return nonzero(bits);
        if (ir::storageInteger(type))
            return truncate(registers(type), bits);
        return bits;
    }

    Value shuffle(Value item, Value lane) {
        return ShuffleOp::create(builder_, location_, item.getType(), item, lane, false);
    }

    Value exchange(Value item, uint64_t mask) {
        return ShuffleOp::create(builder_, location_, item.getType(), item, u32(mask), true);
    }

    // `mapping` maps target elements to source elements. The result uses carrier form; the caller restores its type.
    Value gather(ir::LayoutAttr source, ir::LayoutAttr target, Type type, ArrayRef<Value> slots, int64_t slot,
                 Value index, const std::function<std::optional<int64_t>(int64_t)>& mapping) {
        std::set<int64_t> candidates;
        bool exchanged = false, uniformXor = true;
        std::optional<int64_t> xorMask;
        for (int64_t group = 0; group < target.getGroups().size(); ++group)
            for (int64_t lane = 0; lane < 32; ++lane) {
                auto destination = target.index(group, lane, slot);
                if (destination >= target.size())
                    continue;
                auto input = mapping(destination);
                if (!input)
                    continue;
                candidates.insert(source.getSlots().inverse(*input));
                auto mask = source.getLanes().holder(*input, lane) ^ lane;
                exchanged |= mask != 0;
                uniformXor &= !xorMask || *xorMask == mask;
                xorMask = mask;
            }
        auto carrier = carrierType(type);
        Value selected;
        for (auto candidate : candidates) {
            auto item = carrierValue(type, slots[candidate]);
            if (exchanged)
                item = uniformXor ? exchange(item, *xorMask)
                                  : shuffle(item, mode(source.getLanes(), index, true, lane_));
            selected = selected ? select(equal(mode(source.getSlots(), index, true), u32(candidate)), item, selected)
                                : item;
        }
        return selected ? selected : zero(carrier);
    }

    void bitConversion(ir::ConvertLayoutOp operation, const ir::ConversionPlan& plan) {
        auto output = operation.getResult();
        auto layout = output.getType().getLayout();
        auto type = output.getType().getElementType();
        auto carrier = carrierType(type);
        SmallVector<Value> slots(values_.at(operation.getValue()));
        // Bfloat selections can flush subnormals or canonicalize NaNs.
        for (auto& item : slots)
            item = carrierValue(type, item);
        for (auto [laneBit, slotBit] : plan.exchanges) {
            auto laneMask = u32(uint64_t(1) << laneBit);
            auto high = nonzero(bitAnd(lane_, laneMask));
            for (int64_t low = 0; low < int64_t(slots.size()); ++low) {
                if (low & (int64_t(1) << slotBit))
                    continue;
                auto other = low | (int64_t(1) << slotBit);
                auto chosen = select(high, slots[low], slots[other]);
                auto swapped = shuffle(chosen, bitXor(lane_, laneMask));
                slots[low] = select(high, swapped, slots[low]);
                slots[other] = select(high, slots[other], swapped);
            }
        }
        Value source = u32(0);
        bool moved = false;
        for (int64_t bit = 0; bit < 5; ++bit) {
            auto picked = bitAnd(shiftRight(lane_, u32(bit)), u32(1));
            source = bitOr(source, shiftLeft(picked, u32(plan.lanes[bit])));
            moved |= plan.lanes[bit] != bit;
        }
        if (moved)
            for (auto& item : slots)
                item = shuffle(item, source);
        for (int64_t slot = 0; slot < layout.getSlots().size(); ++slot) {
            auto chosen = select(less(logicalIndex(layout, slot), u32(layout.size())), slots[plan.slots[slot]],
                                 zero(carrier));
            result(output, restoreValue(type, chosen));
        }
    }

    void conversion(ir::ConvertLayoutOp operation) {
        auto input = operation.getValue(), output = operation.getResult();
        auto a = input.getType().getLayout(), b = output.getType().getLayout();
        if (a == b) {
            values_[output] = values_.lookup(input);
            return;
        }
        if (auto plan = ir::planBitConversion(a, b))
            return bitConversion(operation, *plan);
        auto type = output.getType().getElementType();
        for (int64_t slot = 0; slot < b.getSlots().size(); ++slot) {
            auto index = logicalIndex(b, slot);
            auto chosen = gather(a, b, type, values_.at(input), slot, index,
                                 [](int64_t i) -> std::optional<int64_t> { return i; });
            result(output, restoreValue(type, select(less(index, u32(b.size())), chosen, zero(carrierType(type)))));
        }
    }

    // Stores to scratch the slots whose position along the axis is `rootOffset` modulo `period`, then a barrier.
    void publish(ir::LayoutAttr layout, ArrayRef<Value> slots, Value scratch, int64_t inner, int64_t extent,
                 int64_t period, int64_t rootOffset) {
        for (int64_t slot = 0; slot < layout.getSlots().size(); ++slot) {
            if (!anyIndex(layout, slot, [&](int64_t i) { return i / inner % extent % period == rootOffset; }))
                continue;
            auto index = logicalIndex(layout, slot);
            auto root = remainder(remainder(divide(index, u32(inner)), u32(extent)), u32(period));
            guardedStore(pointer(scratch), scratchIndex(scratch, index), slots[slot],
                         both(less(index, u32(layout.size())), equal(root, u32(rootOffset))));
        }
        barrier();
    }

    void item(ir::ItemOp operation) {
        auto input = operation.getValue();
        auto type = input.getType().getElementType();
        if (input.getType().getLayout().getGroups().size() == 1)
            return result(operation.getResult(), restoreValue(type, shuffle(carrierValue(type, value(input)), u32(0))));
        auto scratch = operation.getScratch();
        auto memory = pointer(scratch);
        auto stored = value(input);
        if (type.isBF16()) {
            memory = reinterpret(memory, i16_);
            stored = bitcast(i16_, stored);
        } else if (type.isInteger(1))
            stored = extend(i32_, stored);
        auto index = scratchIndex(scratch, u32(0));
        guardedStore(memory, index, stored, both(equal(simdGroup_, u32(0)), equal(lane_, u32(0))));
        barrier();
        Value loaded = LoadOp::create(builder_, location_, TypeRange{stored.getType()}, memory, index).getResult(0);
        if (type.isBF16())
            loaded = bitcast(type, loaded);
        else if (type.isInteger(1))
            loaded = nonzero(loaded);
        result(operation.getResult(), loaded);
        if (!finalScratchUsers_.contains(operation))
            barrier();
    }

    Value combine(bool isMaximum, Value a, Value b) {
        if (isMaximum)
            return maximum(a, b);
        if (isa<FloatType>(a.getType()))
            return arith::AddFOp::create(builder_, location_, a, b);
        return add(a, b);
    }

    void broadcast(ir::BroadcastInDimOp operation) {
        auto input = operation.getValue(), output = operation.getResult();
        auto a = input.getType().getLayout(), b = output.getType().getLayout();
        auto axes = operation.getAxesAttr().asArrayRef();
        auto shape = a.getShape().asArrayRef(), target = b.getShape().asArrayRef();
        auto type = input.getType().getElementType();
        bool shared = ir::broadcastCrossGroup(a, b, axes);
        SmallVector<Value> slots;
        for (auto item : values_.at(input))
            slots.push_back(storageBits(type, item));
        auto scratch = operation.getScratch();
        Value memory;
        if (shared) {
            Type unit = type.isF16() || type.isBF16() ? i16_ : ir::storageInteger(type) ? registers(type) : i32_;
            memory = reinterpret(pointer(scratch), unit);
            for (int64_t slot = 0; slot < a.getSlots().size(); ++slot) {
                auto index = logicalIndex(a, slot);
                auto stored = unit == i32_ ? slots[slot] : truncate(unit, slots[slot]);
                guardedStore(memory, scratchIndex(scratch, index), stored,
                             both(less(index, u32(a.size())), canonical(a)));
            }
            barrier();
        }
        for (int64_t slot = 0; slot < b.getSlots().size(); ++slot) {
            auto index = logicalIndex(b, slot);
            Value origin = u32(0);
            for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
                int64_t stride = 1;
                for (size_t axis = axes[dimension] + 1; axis < target.size(); ++axis)
                    stride *= target[axis];
                auto position = shape[dimension] == 1 ? u32(0) :
                    remainder(divide(index, u32(stride)), u32(target[axes[dimension]]));
                origin = add(mul(origin, u32(shape[dimension])), position);
            }
            auto inside = less(index, u32(b.size()));
            Value chosen;
            if (shared) {
                auto unit = cast<PointerType>(memory.getType()).getElementType();
                chosen = guardedLoad(memory, scratchIndex(scratch, origin), inside, zero(unit));
                if (unit != i32_)
                    chosen = extend(i32_, chosen);
            } else {
                auto held = [&](int64_t i) -> std::optional<int64_t> { return ir::broadcastIndex(a, b, axes, i); };
                chosen = select(inside, gather(a, b, i32_, slots, slot, origin, held), u32(0));
            }
            result(output, storageValue(type, chosen));
        }
        if (shared)
            barrier();
    }

    template <class ReductionOp>
    void reductionTail(ReductionOp operation, ArrayRef<Value> slots, int64_t inner, int64_t extent,
                       int64_t distance, ir::ItemOp scalar) {
        auto input = operation.getValue(), output = operation.getResult();
        auto a = input.getType().getLayout(), b = output.getType().getLayout();
        auto type = output.getType().getElementType();
        auto scratch = operation.getScratch();
        publish(a, slots, scratch, inner, extent, distance, 0);
        for (int64_t slot = 0; slot < (scalar ? 1 : b.getSlots().size()); ++slot) {
            auto index = scalar ? u32(0) : logicalIndex(b, slot);
            auto origin = add(mul(divide(index, u32(inner)), u32(extent * inner)), remainder(index, u32(inner)));
            auto inside = less(index, u32(b.size()));
            SmallVector<Value> roots;
            for (int64_t offset = 0; offset < extent; offset += distance)
                roots.push_back(guardedLoad(pointer(scratch), scratchIndex(scratch, add(origin, u32(offset * inner))),
                                            inside, zero(type)));
            for (size_t step = 1; step < roots.size(); step *= 2)
                for (size_t i = 0; i + step < roots.size(); i += 2 * step)
                    roots[i] = combine(isa<ir::ReduceMaxOp>(operation), roots[i], roots[i + step]);
            result(scalar ? scalar.getResult() : Value(output), roots[0]);
        }
        if (scalar)
            deferred_.insert(scalar);
        if (!finalScratchUsers_.contains(scalar ? scalar.getOperation() : operation.getOperation()))
            barrier();
    }

    template <class ReductionOp>
    void nativeReduction(ReductionOp operation, ir::ItemOp scalar) {
        auto layout = operation.getValue().getType().getLayout();
        auto type = operation.getResult().getType().getElementType();
        auto scratch = operation.getScratch();
        auto groups = layout.getGroups().size();
        bool sum = isa<ir::ReduceSumOp>(operation);
        bool floating = sum && type.isF32();
        bool ordered = type.isF32() && !floating;
        Type carrier = floating ? type : i32_;
        StringRef kind = sum ? "sum" : "max";
        auto accumulate = [&](Value a, Value b) -> Value {
            if (floating)
                return arith::AddFOp::create(builder_, location_, a, b);
            return sum ? add(a, b) : maximum(a, b);
        };
        Value total = zero(carrier);
        for (int64_t slot = 0; slot < layout.getSlots().size(); ++slot) {
            auto index = logicalIndex(layout, slot);
            auto input = value(operation.getValue(), slot);
            if (ordered) {
                auto bits = bitcast(i32_, input);
                auto key = select(nonzero(shiftRight(bits, u32(31))), bitXor(bits, u32(0xffffffffu)),
                                  bitXor(bits, u32(0x80000000u)));
                input = select(compare(arith::CmpIPredicate::ugt, bitAnd(bits, u32(0x7fffffffu)), u32(0x7f800000u)),
                               u32(0xffffffffu), key);
            }
            input = select(less(index, u32(layout.size())), input, zero(carrier));
            total = slot ? accumulate(total, input) : input;
        }
        total = SimdOp::create(builder_, location_, carrier, total, kind);
        if (groups > 1) {
            auto memory = ordered ? reinterpret(pointer(scratch), i32_) : pointer(scratch);
            barrier();
            guardedStore(memory, scratchIndex(scratch, simdGroup_), total, equal(lane_, u32(0)));
            barrier();
            auto partial = guardedLoad(memory, scratchIndex(scratch, lane_), less(lane_, u32(groups)), zero(carrier));
            total = SimdOp::create(builder_, location_, carrier, partial, kind, true);
        }
        if (ordered) {
            auto bits = select(nonzero(shiftRight(total, u32(31))), bitXor(total, u32(0x80000000u)),
                               bitXor(total, u32(0xffffffffu)));
            total = bitcast(type, select(equal(total, u32(0xffffffffu)), u32(0x7fc00000u), bits));
        }
        if (scalar) {
            result(scalar.getResult(), total);
            deferred_.insert(scalar);
        } else {
            auto target = operation.getResult().getType().getLayout();
            for (int64_t slot = 0; slot < target.getSlots().size(); ++slot)
                result(operation.getResult(), select(less(logicalIndex(target, slot), u32(1)), total, zero(type)));
        }
        if (groups > 1 && !finalScratchUsers_.contains(scalar ? scalar.getOperation() : operation.getOperation()))
            barrier();
    }

    // After distance d, coordinate c = 0 mod 2d holds the reduction of [c, min(c + 2d, extent)).
    template <class ReductionOp>
    void reduction(ReductionOp operation) {
        auto input = operation.getValue(), output = operation.getResult();
        auto a = input.getType().getLayout(), b = output.getType().getLayout();
        auto type = output.getType().getElementType();
        auto shape = a.getShape().asArrayRef();
        auto axis = operation.getAxis();
        bool isMaximum = isa<ir::ReduceMaxOp>(operation);
        auto scalar = dyn_cast_or_null<ir::ItemOp>(operation->getNextNode());
        if (!scalar || !output.hasOneUse() || scalar.getValue() != output ||
            scalar.getScratch() != operation.getScratch())
            scalar = {};
        if (ir::nativeReduction(operation))
            return nativeReduction(operation, scalar);
        int64_t inner = 1, extent = shape[axis];
        for (size_t i = axis + 1; i < shape.size(); ++i)
            inner *= shape[i];
        bool sharedOutput = ir::reductionCrossGroup(a, b, axis, 0);
        int64_t finalShared = 0;
        if (!sharedOutput)
            for (int64_t distance = 1; distance < extent; distance *= 2)
                if (ir::reductionCrossGroup(a, a, axis, distance))
                    finalShared = distance;
        auto scratch = operation.getScratch();
        SmallVector<Value> slots(values_.at(input));
        for (int64_t distance = 1; distance < extent; distance *= 2) {
            bool shared = ir::reductionCrossGroup(a, a, axis, distance);
            auto roots = (extent + distance - 1) / distance;
            // Combine the remaining partials in one step to avoid repeated barriers.
            if (shared && roots <= (scalar ? 8 : 4) && roots * b.getSlots().size() <= 32)
                return reductionTail(operation, slots, inner, extent, distance, scalar);
            if (shared)
                publish(a, slots, scratch, inner, extent, 2 * distance, distance);
            SmallVector<Value> next;
            for (int64_t slot = 0; slot < a.getSlots().size(); ++slot) {
                auto roots = [&](bool paired) {
                    return anyIndex(a, slot, [&](int64_t i) {
                        auto coordinate = i / inner % extent;
                        return coordinate % (2 * distance) == 0 && (coordinate + distance < extent) == paired;
                    });
                };
                bool paired = roots(true), carried = roots(false);
                if (!paired) {
                    next.push_back(carried ? slots[slot] : zero(type));
                    continue;
                }
                auto index = logicalIndex(a, slot);
                auto coordinate = remainder(divide(index, u32(inner)), u32(extent));
                auto withinExtent = less(add(coordinate, u32(distance)), u32(extent));
                auto inside = all({less(index, u32(a.size())), equal(remainder(coordinate, u32(2 * distance)), u32(0)),
                                   withinExtent});
                auto partner = add(index, u32(distance * inner));
                Value right;
                if (shared) {
                    right = guardedLoad(pointer(scratch), scratchIndex(scratch, partner), inside, zero(type));
                } else {
                    auto partnerOf = [&](int64_t i) -> std::optional<int64_t> {
                        auto position = i / inner % extent;
                        if (position % (2 * distance) || position + distance >= extent)
                            return std::nullopt;
                        return i + distance * inner;
                    };
                    right = restoreValue(type, gather(a, a, type, slots, slot, partner, partnerOf));
                }
                auto combined = combine(isMaximum, slots[slot], right);
                // Only orphan roots need to preserve their incoming value.
                next.push_back(carried ? select(withinExtent, combined, slots[slot]) : combined);
            }
            if (shared && (distance != finalShared || !finalScratchUsers_.contains(operation)))
                barrier();
            slots = std::move(next);
        }
        if (sharedOutput)
            publish(a, slots, scratch, inner, extent, extent, 0);
        for (int64_t slot = 0; slot < b.getSlots().size(); ++slot) {
            auto index = logicalIndex(b, slot);
            auto origin = add(mul(divide(index, u32(inner)), u32(extent * inner)), remainder(index, u32(inner)));
            auto inside = less(index, u32(b.size()));
            Value chosen;
            if (sharedOutput)
                chosen = guardedLoad(pointer(scratch), scratchIndex(scratch, origin), inside, zero(type));
            else
                chosen = select(inside, restoreValue(type, gather(a, b, type, slots, slot, origin,
                                                                  [&](int64_t i) -> std::optional<int64_t> {
                    return i / inner * extent * inner + i % inner;
                })), zero(type));
            result(output, chosen);
        }
        if (sharedOutput && !finalScratchUsers_.contains(operation))
            barrier();
    }

    // Runs stay within one axis line: other digits must step whole runs or stay inside one axis element.
    struct Runs {
        int64_t length = 1;
        SmallVector<size_t> digits;
        SmallVector<int64_t> place, previous, end;
    };

    static Runs runs(ir::LayoutAttr layout, int64_t inner, int64_t extent, bool fast) {
        auto slots = layout.getSlots();
        auto digits = slots.digits();
        SmallVector<int64_t> places(digits.size());
        for (int64_t i = digits.size(), place = 1; i-- > 0;) {
            places[i] = place;
            place *= digits[i].first;
        }
        Runs result;
        int64_t weight = 1;
        for (bool found = fast; found;) {
            found = false;
            for (size_t i = 0; i < digits.size() && !found; ++i)
                if (digits[i].first > 1 && digits[i].second == weight * inner &&
                    !llvm::is_contained(result.digits, i)) {
                    result.digits.push_back(i);
                    weight *= digits[i].first;
                    found = true;
                }
        }
        auto whole = [&](int64_t size, int64_t stride) {
            return size == 1 || stride % (weight * inner) == 0 || stride * size <= inner;
        };
        bool aligned = extent % weight == 0;
        for (auto mode : {layout.getGroups(), layout.getLanes()})
            for (auto [size, stride] : mode.digits())
                aligned &= whole(size, stride);
        for (size_t i = 0; i < digits.size(); ++i)
            aligned &= llvm::is_contained(result.digits, i) || whole(digits[i].first, digits[i].second);
        if (!aligned)
            result.digits.clear();
        result.length = aligned ? weight : 1;
        for (int64_t slot = 0; slot < slots.size(); ++slot) {
            int64_t place = 0, scale = 1, end = slot, previous = slot;
            for (auto i : result.digits) {
                auto digit = slot / places[i] % digits[i].first;
                place += digit * scale;
                scale *= digits[i].first;
                end += (digits[i].first - 1 - digit) * places[i];
            }
            for (auto i : result.digits) {
                if (!place)
                    break;
                if (previous / places[i] % digits[i].first) {
                    previous -= places[i];
                    break;
                }
                previous += (digits[i].first - 1) * places[i];
            }
            result.place.push_back(place);
            result.previous.push_back(place ? previous : -1);
            result.end.push_back(end);
        }
        return result;
    }

    // Whether lane l holds run l of each 32-run stretch of the axis, and each SIMD group a whole stretch of one line:
    // a SIMD-group prefix over the run ends then covers the distances `length` to 16 * `length`.
    static bool lanesWalkAxis(ir::LayoutAttr layout, const Runs& run, int64_t inner, int64_t extent) {
        int64_t span = 32 * run.length;
        if (extent % span)
            return false;
        for (int64_t lane = 0; lane < 32; ++lane)
            if (layout.getLanes().offset(lane) != lane * run.length * inner)
                return false;
        auto whole = [&](int64_t size, int64_t stride) {
            return size == 1 || stride % (span * inner) == 0 || stride * size <= inner;
        };
        for (auto [size, stride] : layout.getGroups().digits())
            if (!whole(size, stride))
                return false;
        auto digits = layout.getSlots().digits();
        for (size_t i = 0; i < digits.size(); ++i)
            if (!llvm::is_contained(run.digits, i) && !whole(digits[i].first, digits[i].second))
                return false;
        return true;
    }

    Value before(ir::LayoutAttr layout, Type type, ArrayRef<Value> slots, int64_t slot, Value scratch, bool shared,
                 int64_t inner, int64_t extent, int64_t distance, Value inside) {
        auto partner = sub(logicalIndex(layout, slot), u32(distance * inner));
        if (shared)
            return guardedLoad(pointer(scratch), scratchIndex(scratch, partner), inside, zero(type));
        return restoreValue(type, gather(layout, layout, type, slots, slot, partner,
                                         [&](int64_t i) -> std::optional<int64_t> {
            if (i / inner % extent < distance)
                return std::nullopt;
            return i - distance * inner;
        }));
    }

    // Pairwise uses a layout-independent Hillis-Steele tree for reproducible f32 results.
    // Fast order scans run ends, then adds the preceding run's prefix to each element.
    template <class ScanOp>
    void scan(ScanOp operation) {
        auto layout = operation.getValue().getType().getLayout();
        auto type = operation.getResult().getType().getElementType();
        auto shape = layout.getShape().asArrayRef();
        auto axis = operation.getAxis();
        bool isMaximum = isa<ir::ScanMaxOp>(operation), fast = operation.getOrder() == "fast";
        int64_t inner = 1, extent = shape[axis];
        for (size_t i = axis + 1; i < shape.size(); ++i)
            inner *= shape[i];
        auto run = runs(layout, inner, extent, fast);
        auto scratch = operation.getScratch();
        SmallVector<Value> slots(values_.at(operation.getValue()));
        int64_t count = layout.getSlots().size();
        for (int64_t place = 1; place < run.length; ++place)
            for (int64_t slot = 0; slot < count; ++slot)
                if (run.place[slot] == place)
                    slots[slot] = combine(isMaximum, slots[run.previous[slot]], slots[slot]);
        bool hardware = fast && !isMaximum && lanesWalkAxis(layout, run, inner, extent);
        int64_t unit = run.length;
        Value lastLane;
        if (hardware) {
            for (int64_t slot = 0; slot < count; ++slot) {
                if (run.end[slot] != slot)
                    continue;
                auto exclusive = SimdOp::create(builder_, location_, type, slots[slot], "prefix_exclusive_sum");
                for (int64_t member = 0; member < count; ++member)
                    if (run.end[member] == slot && member != slot)
                        slots[member] = combine(false, exclusive, slots[member]);
                slots[slot] = SimdOp::create(builder_, location_, type, slots[slot], "prefix_inclusive_sum");
            }
            unit = 32 * run.length;
            lastLane = equal(lane_, u32(31));
        }
        SmallVector<std::pair<int64_t, bool>> plan;
        for (int64_t distance = unit; distance < extent; distance *= 2)
            plan.push_back({distance, false});
        if (unit > 1 && unit < extent)
            plan.push_back({unit, true});
        int64_t lastShared = -1;
        for (size_t step = 0; step < plan.size(); ++step)
            if (ir::scanCrossGroup(layout, axis, plan[step].first))
                lastShared = step;
        for (size_t step = 0; step < plan.size(); ++step) {
            auto [distance, shift] = plan[step];
            bool shared = ir::scanCrossGroup(layout, axis, distance);
            if (shared)
                publish(layout, slots, scratch, inner, extent, 1, 0);
            SmallVector<Value> next(slots);
            for (int64_t slot = 0; slot < count; ++slot) {
                if (run.end[slot] != slot ||
                    !anyIndex(layout, slot, [&](int64_t i) { return i / inner % extent >= distance; }))
                    continue;
                auto index = logicalIndex(layout, slot);
                auto coordinate = remainder(divide(index, u32(inner)), u32(extent));
                auto inside = both(less(index, u32(layout.size())), less(u32(distance - 1), coordinate));
                auto earlier = before(layout, type, slots, slot, scratch, shared, inner, extent, distance, inside);
                if (!shift) {
                    auto active = hardware ? both(inside, lastLane) : inside;
                    next[slot] = select(active, combine(isMaximum, slots[slot], earlier), slots[slot]);
                    continue;
                }
                if (hardware)
                    earlier = restoreValue(type, shuffle(carrierValue(type, earlier), u32(31)));
                for (int64_t member = 0; member < count; ++member) {
                    if (run.end[member] != slot || (member == slot && !hardware))
                        continue;
                    auto active = member == slot ? both(inside, bitXor(lastLane, boolean(true))) : inside;
                    next[member] = select(active, combine(isMaximum, earlier, slots[member]), slots[member]);
                }
            }
            if (shared && (int64_t(step) != lastShared || !finalScratchUsers_.contains(operation)))
                barrier();
            slots = std::move(next);
        }
        for (auto item : slots)
            result(operation.getResult(), item);
    }

    SmallVector<Value> flatten(ValueRange items) {
        SmallVector<Value> flat;
        for (auto item : items)
            for (auto slot : values_.at(item))
                flat.push_back(slot);
        return flat;
    }

    void assign(ValueRange outputs, ValueRange flat) {
        size_t next = 0;
        for (auto output : outputs) {
            auto& slots = values_[output];
            slots.clear();
            for (size_t slot = 0; slot < slotCount(output.getType()); ++slot)
                slots.push_back(flat[next++]);
        }
    }

    static std::optional<uint64_t> known(Value index) {
        if (auto cast = index.getDefiningOp<arith::IndexCastUIOp>())
            index = cast.getIn();
        APInt bits;
        if (matchPattern(index, m_ConstantInt(&bits)))
            return bits.getZExtValue();
        return std::nullopt;
    }

    // Count trips to prevent a u32 induction variable wrapping past a runtime upper bound.
    void loop(scf::ForOp operation) {
        auto initial = flatten(operation.getInitArgs());
        auto first = known(operation.getLowerBound()), upper = known(operation.getUpperBound());
        auto step = known(operation.getStep());
        bool direct = step && (*step == 1 || (upper && *upper + *step - 1 <= UINT32_MAX));
        auto bound = [&](Value item) {
            auto converted = value(item);
            return step && !converted.getType().isInteger(32)
                ? arith::IndexCastUIOp::create(builder_, location_, i32_, converted).getResult() : converted;
        };
        auto lower = bound(operation.getLowerBound()), upperValue = bound(operation.getUpperBound());
        auto stepValue = bound(operation.getStep());
        if (direct && first && upper && *upper > *first)
            upperValue = u32(*first + (*upper - 1 - *first) / *step * *step + 1);
        Value start = lower;
        if (step && !direct) {
            auto trips = add(divide(sub(sub(upperValue, lower), u32(1)), stepValue), u32(1));
            upperValue = select(less(lower, upperValue), trips, u32(0));
            lower = u32(0);
            stepValue = u32(1);
        }
        auto loop = scf::ForOp::create(builder_, location_, lower, upperValue, stepValue, initial);
        OpBuilder::InsertionGuard guard(builder_);
        if (loop.getBody()->mightHaveTerminator())
            loop.getBody()->getTerminator()->erase();
        builder_.setInsertionPointToEnd(loop.getBody());
        Value index = loop.getInductionVar();
        if (step && !direct)
            index = add(start, mul(index, u32(uint32_t(*step))));
        values_[operation.getInductionVar()] = {index};
        assign(operation.getRegionIterArgs(), loop.getRegionIterArgs());
        emitBlock(*operation.getBody());
        scf::YieldOp::create(builder_, location_, flatten(operation.getBody()->getTerminator()->getOperands()));
        assign(operation.getResults(), loop.getResults());
    }

    void branch(scf::IfOp operation) {
        SmallVector<Type> types;
        for (auto output : operation.getResults())
            for (size_t slot = 0; slot < slotCount(output.getType()); ++slot)
                types.push_back(registers(element(output.getType())));
        bool otherwise = !operation.getElseRegion().empty();
        auto conditional = scf::IfOp::create(builder_, location_, types, value(operation.getCondition()), otherwise);
        for (auto [source, target] : llvm::zip(operation->getRegions(), conditional->getRegions())) {
            if (source.empty())
                continue;
            OpBuilder::InsertionGuard guard(builder_);
            auto& block = target.front();
            if (block.mightHaveTerminator())
                block.getTerminator()->erase();
            builder_.setInsertionPointToEnd(&block);
            emitBlock(source.front());
            scf::YieldOp::create(builder_, location_, flatten(source.front().getTerminator()->getOperands()));
        }
        assign(operation.getResults(), conditional.getResults());
    }

    ir::StoreOp directStore(ir::TensorMatmulOp operation) {
        auto constant = [](Value tile, auto pattern) {
            auto broadcast = tile.getDefiningOp<ir::BroadcastOp>();
            return broadcast && matchPattern(broadcast.getValue(), pattern);
        };
        Value stored = operation.getResult();
        if (function_->hasAttr("carbon.check_writes") || checking_ || !stored.hasOneUse() ||
            !constant(operation.getAccumulator(), m_PosZeroFloat()))
            return {};
        auto* user = *stored.getUsers().begin();
        if (isa<ir::CastOp>(user) && user->getResult(0).hasOneUse()) {
            stored = user->getResult(0);
            user = *stored.getUsers().begin();
        }
        auto store = dyn_cast<ir::StoreOp>(user);
        if (!store || store.getValues() != stored || store.getIndices().size() != 1 ||
            store->getBlock() != operation->getBlock() || !constant(store.getMask(), m_One()) ||
            !values_.count(store.getBuffer()))
            return {};
        auto layout = cast<ir::TileType>(stored.getType()).getLayout();
        auto index = store.getIndices().front().getDefiningOp<ir::IotaOp>();
        auto view = dyn_cast<ir::ViewType>(store.getBuffer().getType());
        if (!index || cast<ir::TileType>(index.getType()).getLayout() != layout || layout.capacity() != layout.size() ||
            !view || view.getShape().asArrayRef() != layout.getShape().asArrayRef())
            return {};
        for (auto* between = operation->getNextNode(); between != store.getOperation();
             between = between->getNextNode())
            if (!isPure(between))
                return {};
        return store;
    }

    // Tensor geometry is column-first: column-major views are transposed row-major tensors.
    // Results follow the tensor op's register order unless directStore writes a row-major destination.
    void tensorMatmul(ir::TensorMatmulOp operation) {
        SmallVector<Value> tensors;
        SmallVector<bool> transposed;
        auto describe = [&](Value memory) {
            auto view = cast<ir::ViewType>(memory.getType());
            auto strides = (view.isDynamic() ? cast<ir::ViewType>(ir::viewRoot(memory).getType()) : view).getStrides();
            bool columnMajor = strides[1] != 1;
            Value rows = u32(view.getShape()[0]), columns = u32(view.getShape()[1]);
            if (view.isDynamic()) {
                const auto& mapping = views_.at(memory);
                rows = select(mapping.active, mapping.limits[0], u32(0));
                columns = select(mapping.active, mapping.limits[1], u32(0));
            }
            if (columnMajor)
                std::swap(rows, columns);
            auto start = pointer(memory);
            auto offset = address(memory, u32(0));
            auto stride = view.isDynamic() ? views_.at(memory).strides[columnMajor ? 1 : 0]
                                           : i64(columnMajor ? strides[1] : strides[0]);
            tensors.append({start, offset, columns, rows, stride});
            transposed.push_back(columnMajor);
        };
        describe(operation.getLeft());
        describe(operation.getRight());
        auto shape = operation.getAccumulator().getType().getLayout().getShape();
        auto threads = function_->getAttrOfType<IntegerAttr>("carbon.threads").getInt();
        if (auto store = directStore(operation)) {
            describe(store.getBuffer());
            if (!transposed.back()) {
                TensorMatmulOp::create(builder_, location_, TypeRange{}, tensors, ValueRange{}, shape[0], shape[1],
                                       threads / 32, operation.getRelaxed(), transposed[0], transposed[1]);
                for (auto* user : operation.getResult().getUsers())
                    deferred_.insert(user);
                deferred_.insert(store);
                return;
            }
            tensors.resize(2 * tensorOperands);
            transposed.pop_back();
        }
        const auto& accumulator = values_.at(operation.getAccumulator());
        auto product = TensorMatmulOp::create(builder_, location_, ValueRange(accumulator).getTypes(), tensors,
                                              accumulator, shape[0], shape[1], threads / 32, operation.getRelaxed(),
                                              transposed[0], transposed[1]);
        for (auto item : product.getResults())
            result(operation.getResult(), item);
    }

    void recordRegisters(ir::MmaOp operation) {
        if (operation.getUnit() != "simdgroup")
            return;
        for (auto operand : {operation.getLeft(), operation.getRight(), operation.getAccumulator()}) {
            auto layout = operand.getType().getLayout();
            SmallVector<int64_t> elements;
            for (int64_t lane = 0; lane < 32; ++lane)
                for (int64_t slot = 0; slot < 2; ++slot)
                    elements.push_back(layout.index(0, lane, slot));
            if (auto name = registersName(operand.getType().getElementType()); !thread_->hasAttr(name))
                thread_->setAttr(name, DenseI64ArrayAttr::get(context_, elements));
        }
    }

    void mma(ir::MmaOp operation) {
        recordRegisters(operation);
        auto shape = operation.getShape();
        const auto& accumulator = values_.at(operation.getAccumulator());
        auto product = MmaOp::create(builder_, location_, ValueRange(accumulator).getTypes(),
                                     values_.at(operation.getLeft()), values_.at(operation.getRight()), accumulator,
                                     operation.getUnit(), shape[0], shape[1], shape[2], operation.getRelaxed());
        for (auto item : product.getResults())
            result(operation.getResult(), item);
    }

    // `a opcode b`: unsigned for integers, ordered for f16 and bf16, and for f32 on ordered integer keys of the bits,
    // which keep the subnormals that native f32 comparisons flush; NaNs are unordered and -0 equals +0.
    Value comparison(StringRef opcode, Value a, Value b) {
        auto type = a.getType();
        auto unsignedPredicate = [&](StringRef op) {
            return op == "<" ? arith::CmpIPredicate::ult : op == "<=" ? arith::CmpIPredicate::ule :
                   op == ">" ? arith::CmpIPredicate::ugt : op == ">=" ? arith::CmpIPredicate::uge :
                   op == "==" ? arith::CmpIPredicate::eq : arith::CmpIPredicate::ne;
        };
        if (isa<FloatType>(type) && !type.isF32()) {
            auto predicate = opcode == "<" ? arith::CmpFPredicate::OLT : opcode == "<=" ? arith::CmpFPredicate::OLE :
                             opcode == ">" ? arith::CmpFPredicate::OGT : opcode == ">=" ? arith::CmpFPredicate::OGE :
                             opcode == "==" ? arith::CmpFPredicate::OEQ : arith::CmpFPredicate::UNE;
            return arith::CmpFOp::create(builder_, location_, predicate, a, b);
        }
        if (!type.isF32())
            return compare(unsignedPredicate(opcode), a, b);
        auto bitsA = bitcast(i32_, a), bitsB = bitcast(i32_, b);
        auto magnitudeA = bitAnd(bitsA, u32(0x7fffffffu)), magnitudeB = bitAnd(bitsB, u32(0x7fffffffu));
        auto ordered = both(compare(arith::CmpIPredicate::ule, magnitudeA, u32(0x7f800000u)),
                            compare(arith::CmpIPredicate::ule, magnitudeB, u32(0x7f800000u)));
        if (opcode == "==" || opcode == "!=") {
            auto same = bitOr(equal(bitsA, bitsB), equal(bitOr(magnitudeA, magnitudeB), u32(0)));
            auto result = both(ordered, same);
            return opcode == "==" ? result : bitXor(result, boolean(true));
        }
        auto key = [&](Value bits, Value magnitude) {
            auto signedKey = select(nonzero(shiftRight(bits, u32(31))), bitXor(bits, u32(0xffffffffu)),
                                    bitXor(bits, u32(0x80000000u)));
            return select(equal(magnitude, u32(0)), u32(0x80000000u), signedKey);
        };
        return both(ordered, compare(unsignedPredicate(opcode), key(bitsA, magnitudeA), key(bitsB, magnitudeB)));
    }

    Value arithmetic(StringRef opcode, Value a, Value b) {
        if (isa<FloatType>(a.getType())) {
            if (opcode == "add") return arith::AddFOp::create(builder_, location_, a, b);
            if (opcode == "sub") return arith::SubFOp::create(builder_, location_, a, b);
            if (opcode == "mul") return arith::MulFOp::create(builder_, location_, a, b);
            if (opcode == "truediv") return arith::DivFOp::create(builder_, location_, a, b);
            fail("unsupported floating-point map opcode " + opcode.str());
        }
        if (opcode == "add") return add(a, b);
        if (opcode == "sub") return sub(a, b);
        if (opcode == "mul") return mul(a, b);
        if (opcode == "floordiv" || opcode == "truediv") return divide(a, b);
        if (opcode == "mod") return remainder(a, b);
        fail("unsupported integer map opcode " + opcode.str());
    }

    // Clipping hides adjacency when flattening a subview. Proven full extents expose it to vectorization.
    std::optional<std::pair<Value, ViewValues>> regular(Value memory, ValueRange indices) {
        auto view = dyn_cast<ir::ViewType>(memory.getType());
        if (indices.size() != 1 || !view || !view.isDynamic())
            return std::nullopt;
        auto window = views_.at(memory);
        if (window.shape.size() < 2 || window.requested.size() != window.shape.size())
            return std::nullopt;
        auto run = runLength(cast<ir::TileType>(indices.front().getType()).getLayout(), view.getElementType());
        APInt unit;
        bool contiguous = matchPattern(window.strides.back(), m_ConstantInt(&unit)) && unit == 1;
        SmallVector<Value> whole;
        for (size_t axis = 1; axis < window.shape.size(); ++axis) {
            APInt extent;
            Value regular;
            if (matchPattern(window.requested[axis], m_ConstantInt(&extent))) {
                regular = u32(uint32_t(extent.getZExtValue()));
                whole.push_back(equal(window.shape[axis], regular));
            } else if (run > 1 && contiguous) {
                auto multiple = bitAnd(window.shape[axis], u32(~uint32_t(run - 1)));
                whole.push_back(both(nonzero(window.shape[axis]), equal(multiple, window.shape[axis])));
                regular = maximum(multiple, u32(uint32_t(run)));
            } else {
                return std::nullopt;
            }
            window.shape[axis] = window.limits[axis] = regular;
        }
        return std::make_pair(all(whole), window);
    }

    static int64_t runLength(ir::LayoutAttr layout, Type element) {
        auto digits = layout.getSlots().digits();
        while (!digits.empty() && digits.back().first == 1)
            digits.pop_back();
        if (digits.empty() || digits.back().second != 1)
            return 1;
        int64_t run = 1;
        auto most = element.isInteger(8) ? 16 : 4;
        while (run < most && digits.back().first % (2 * run) == 0)
            run *= 2;
        return run;
    }

    SmallVector<Value> versioned(Value memory, Value condition, const ViewValues& window, TypeRange types,
                                 llvm::function_ref<SmallVector<Value>()> emit) {
        auto conditional = scf::IfOp::create(builder_, location_, types, condition, true);
        auto general = views_.at(memory);
        for (auto [index, region] : llvm::enumerate(conditional->getRegions())) {
            OpBuilder::InsertionGuard guard(builder_);
            auto& block = region.front();
            if (block.mightHaveTerminator())
                block.getTerminator()->erase();
            builder_.setInsertionPointToEnd(&block);
            views_[memory] = index ? general : window;
            scf::YieldOp::create(builder_, location_, emit());
        }
        views_[memory] = general;
        return {conditional.result_begin(), conditional.result_end()};
    }

    template <class Access>
    void recordScratch(Operation* operation, Access access) {
        if (!access)
            fail("internal compiler error: a scratch access is not one guarded access per slot");
        scratchAccesses_[operation].push_back({access.getIndex(), access.getCondition()});
    }

    // Unknown conditions make touches optional; unknown indices prevent cell-level checking.
    std::optional<std::vector<ir::ScratchTouch>> scratchTouches(Operation* access, int64_t threads,
                                                                ThreadValues& values) {
        std::vector<ir::ScratchTouch> touches;
        auto slots = scratchAccesses_.lookup(access);
        for (int64_t thread = 0; thread < threads; ++thread) {
            for (auto [index, condition] : slots) {
                auto active = values.at(condition, thread);
                if (active == uint64_t(0))
                    continue;
                auto cell = values.at(index, thread);
                if (!cell)
                    return std::nullopt;
                touches.push_back({int(thread), int64_t(*cell), active.has_value()});
            }
        }
        return touches;
    }

    void load(ir::LoadOp operation) {
        auto memory = operation.getBuffer();
        auto emit = [&] {
            SmallVector<Value> slots;
            for (auto slot : slotRange(slotCount(operation.getResult().getType())))
                slots.push_back(guardedLoad(pointer(memory), address(memory, operation.getIndices(), slot),
                                            active(memory, operation.getIndices(), operation.getMask(), slot),
                                            value(operation.getFallback(), slot)));
            return slots;
        };
        SmallVector<Type> types;
        for (auto slot : slotRange(slotCount(operation.getResult().getType())))
            types.push_back(value(operation.getFallback(), slot).getType());
        auto window = regular(memory, operation.getIndices());
        auto loaded = window ? versioned(memory, window->first, window->second, types, emit) : emit();
        for (auto item : loaded) {
            if (ir::scratchAllocation(memory))
                recordScratch(operation, item.getDefiningOp<GuardedLoadOp>());
            result(operation.getResult(), item);
        }
    }

    // Each store marks its element with the writer's id; a different id already there means two threads wrote it.
    void recordWrites(ir::StoreOp operation, int64_t slots) {
        auto memory = operation.getBuffer();
        auto found = bindings_.find(ir::bindingRoot(memory));
        if (found == bindings_.end() || !shadows_.count(found->second))
            return;
        auto [shadow, check] = shadows_.at(found->second);
        auto owner = canonical(cast<ir::TileType>(operation.getIndices().front().getType()).getLayout());
        for (auto slot : slotRange(slots)) {
            auto condition = both(active(memory, operation.getIndices(), operation.getMask(), slot), owner);
            auto element = word(address(memory, operation.getIndices(), slot));
            Value previous = AtomicOp::create(builder_, location_, i32_, shadow, element, writer_, condition, "max");
            auto conflict = all({condition, nonzero(previous), compare(arith::CmpIPredicate::ne, previous, writer_)});
            AtomicOp::create(builder_, location_, i32_, record_, u32(0), u32(1), conflict, "add");
            AtomicOp::create(builder_, location_, i32_, record_, u32(1 + check), element, conflict, "min");
        }
    }

    void store(ir::StoreOp operation) {
        recordWrites(operation, slotCount(operation.getValues().getType()));
        auto memory = operation.getBuffer();
        bool bits = memoryElement(memory.getType()).isBF16();
        auto target = bits ? reinterpret(pointer(memory), i16_) : pointer(memory);
        auto owner = canonical(operation.getValues().getType().getLayout());
        auto emit = [&] {
            for (auto slot : slotRange(slotCount(operation.getValues().getType()))) {
                auto item = value(operation.getValues(), slot);
                auto stored = guardedStore(
                    target, address(memory, operation.getIndices(), slot), bits ? bitcast(i16_, item) : item,
                    both(active(memory, operation.getIndices(), operation.getMask(), slot), owner));
                if (ir::scratchAllocation(memory))
                    recordScratch(operation, stored);
            }
            return SmallVector<Value>();
        };
        if (auto window = regular(memory, operation.getIndices()))
            versioned(memory, window->first, window->second, {}, emit);
        else
            emit();
    }

    void convert(ir::CastOp operation) {
        auto input = operation.getValue(), output = operation.getResult();
        if (input.getType() == output.getType()) {
            values_[output] = values_.lookup(input);
            return;
        }
        auto from = element(input.getType()), type = element(output.getType());
        auto f32 = builder_.getF32Type();
        for (auto slot : slotRange(slotCount(output.getType()))) {
            Value source = value(input, slot);
            if (ir::storageInteger(type)) {
                result(output, truncate(registers(type), source));
                continue;
            }
            if (ir::storageInteger(from)) {
                bool sign = from.isSignedInteger();
                Value word = sign ? arith::ExtSIOp::create(builder_, location_, i32_, source).getResult()
                                  : extend(i32_, source);
                if (type.isF32())
                    word = sign ? arith::SIToFPOp::create(builder_, location_, f32, word).getResult()
                                : ConvertOp::create(builder_, location_, f32, word).getResult();
                result(output, word);
                continue;
            }
            if (!source.getType().isF32())
                source = ConvertOp::create(builder_, location_, f32, source);
            if (!type.isBF16()) {
                result(output,
                       type.isF32() ? source : ConvertOp::create(builder_, location_, type, source).getResult());
                continue;
            }
            // Native bf16 narrowing can flush subnormals even with fast math disabled.
            auto bits = bitcast(i32_, source);
            auto rounded = shiftRight(add(add(bits, u32(0x7fffu)), bitAnd(shiftRight(bits, u32(16)), u32(1))), u32(16));
            auto nan = compare(arith::CmpIPredicate::ugt, bitAnd(bits, u32(0x7fffffffu)), u32(0x7f800000u));
            auto word = select(nan, bitOr(shiftRight(bits, u32(16)), u32(0x40u)), rounded);
            result(output, bitcast(type, truncate(i16_, word)));
        }
    }

    void elementwise(Operation& operation) {
        auto output = operation.getResult(0);
        auto type = element(output.getType());
        for (auto slot : slotRange(slotCount(output.getType()))) {
            Value computed;
            if (auto division = dyn_cast<ir::IntegerDivOp>(operation))
                computed = GuardedIntegerOp::create(builder_, location_, i32_, value(division.getLeft(), slot),
                                                    value(division.getRight(), slot), division.getOpcode());
            else if (auto bits = dyn_cast<ir::BitwiseOp>(operation)) {
                auto a = value(bits.getLeft(), slot), b = value(bits.getRight(), slot);
                auto opcode = bits.getOpcode();
                if (opcode == "shl" || opcode == "shr")
                    computed = GuardedIntegerOp::create(builder_, location_, i32_, a, b, opcode);
                else
                    computed = opcode == "and" ? bitAnd(a, b) : opcode == "or" ? bitOr(a, b) : bitXor(a, b);
            } else if (auto bits = dyn_cast<ir::BitcastOp>(operation))
                computed = bitcast(type, value(bits.getValue(), slot));
            else if (auto math = dyn_cast<ir::MathOp>(operation))
                computed = MathOp::create(builder_, location_, type, value(math.getValue(), slot), math.getOpcode());
            else if (auto maximumOp = dyn_cast<ir::MaximumOp>(operation))
                computed = maximum(value(maximumOp.getLeft(), slot), value(maximumOp.getRight(), slot));
            else if (auto minimumOp = dyn_cast<ir::MinimumOp>(operation))
                computed = minimum(value(minimumOp.getLeft(), slot), value(minimumOp.getRight(), slot));
            else if (auto fma = dyn_cast<ir::FmaOp>(operation))
                computed = FmaOp::create(builder_, location_, type, value(fma.getA(), slot), value(fma.getB(), slot),
                                         value(fma.getC(), slot));
            else {
                auto choice = llvm::cast<ir::SelectOp>(operation);
                computed = storageValue(type, select(value(choice.getCondition(), slot),
                                                     storageBits(type, value(choice.getLeft(), slot)),
                                                     storageBits(type, value(choice.getRight(), slot))));
            }
            result(output, computed);
        }
    }

    void map(ir::MapOp operation) {
        static const std::map<std::string, std::string> comparisons = {
            {"lt", "<"}, {"le", "<="}, {"gt", ">"}, {"ge", ">="}, {"eq", "=="}, {"ne", "!="}};
        auto opcode = operation.getOpcode();
        for (auto slot : slotRange(slotCount(operation.getResult().getType()))) {
            auto a = value(operation.getInputs()[0], slot);
            Value computed;
            if (opcode == "neg")
                computed = isa<FloatType>(a.getType()) ? arith::NegFOp::create(builder_, location_, a).getResult()
                                                       : sub(zero(a.getType()), a);
            else if (operation.getResult().getType().getElementType().isInteger(1))
                computed = comparison(comparisons.at(opcode.str()), a, value(operation.getInputs()[1], slot));
            else
                computed = arithmetic(opcode, a, value(operation.getInputs()[1], slot));
            result(operation.getResult(), computed);
        }
    }

    void emitBlock(Block& block) {
        for (auto& operation : block.without_terminator())
            emit(operation);
    }

    void emit(Operation& operation) {
        location_ = operation.getLoc();
        if (deferred_.contains(&operation))
            return;
        if (auto iteration = dyn_cast<scf::ForOp>(operation))
            return loop(iteration);
        if (auto conditional = dyn_cast<scf::IfOp>(operation))
            return branch(conditional);
        if (auto conversion = dyn_cast<arith::IndexCastUIOp>(operation)) {
            auto input = value(conversion.getIn());
            auto type = conversion.getType();
            return result(conversion.getResult(), input.getType() == type ? input :
                          arith::IndexCastUIOp::create(builder_, location_, type, input).getResult());
        }
        if (isa<func::ReturnOp>(operation))
            return;
        if (auto constant = dyn_cast<arith::ConstantOp>(operation))
            return result(constant.getResult(), arith::ConstantOp::create(builder_, location_, constant.getValue()));
        if (isa<ir::ProgramIdOp>(operation))
            return result(operation.getResult(0), group());
        if (isa<ir::SimdIdOp>(operation))
            return result(operation.getResult(0), simdGroup_);
        if (auto matmul = dyn_cast<ir::TensorMatmulOp>(operation))
            return tensorMatmul(matmul);
        if (auto product = dyn_cast<ir::MmaOp>(operation))
            return mma(product);
        if (auto allocation = dyn_cast<ir::ScratchOp>(operation)) {
            auto type = allocation.getResult().getType();
            auto offset = memory_->scratchOffsets.at(allocation.getResult());
            auto pointerType = PointerType::get(context_, "threadgroup", registers(type.getElementType()));
            values_[allocation.getResult()] = {ScratchOp::create(builder_, location_, pointerType, offset)};
            staticView(allocation.getResult(), type, 0);
            return;
        }
        if (isa<ir::BarrierOp>(operation))
            return barrier();
        if (auto sum = dyn_cast<ir::ReduceSumOp>(operation))
            return reduction(sum);
        if (auto maximumOp = dyn_cast<ir::ReduceMaxOp>(operation))
            return reduction(maximumOp);
        if (auto sum = dyn_cast<ir::ScanSumOp>(operation))
            return scan(sum);
        if (auto maximumOp = dyn_cast<ir::ScanMaxOp>(operation))
            return scan(maximumOp);
        if (auto expansion = dyn_cast<ir::BroadcastInDimOp>(operation))
            return broadcast(expansion);
        if (auto scalar = dyn_cast<ir::ItemOp>(operation))
            return item(scalar);
        if (auto view = dyn_cast<ir::MakeViewOp>(operation)) {
            values_[view.getResult()] = values_.lookup(view.getBuffer());
            auto type = view.getResult().getType();
            staticView(view.getResult(), type,
                       type.getOffsetBytes() / (type.getElementType().getIntOrFloatBitWidth() / 8));
            return;
        }
        if (auto dimension = dyn_cast<ir::DimOp>(operation)) {
            auto memory = dimension.getView();
            return result(dimension.getResult(), isa<ir::ViewType>(memory.getType())
                ? views_.at(memory).shape[dimension.getAxis()] : bound(memory));
        }
        if (auto tile = dyn_cast<ir::PartitionOp>(operation))
            return partition(tile);
        if (auto region = dyn_cast<ir::SubviewOp>(operation))
            return subview(region);
        if (auto convert = dyn_cast<ir::ConvertLayoutOp>(operation))
            return conversion(convert);
        if (isa<ir::ReshapeOp, ir::TransposeOp>(operation)) {
            values_[operation.getResult(0)] = values_.lookup(operation.getOperand(0));
            return;
        }
        if (auto stack = dyn_cast<ir::StackOp>(operation)) {
            auto& slots = values_[stack.getResult()];
            for (auto input : stack.getInputs()) {
                auto source = values_.lookup(input);
                slots.insert(slots.end(), source.begin(), source.end());
            }
            return;
        }
        if (auto conversion = dyn_cast<ir::CastOp>(operation))
            return convert(conversion);
        if (auto iota = dyn_cast<ir::IotaOp>(operation)) {
            auto layout = cast<ir::TileType>(iota.getResult().getType()).getLayout();
            for (auto slot : slotRange(layout.getSlots().size()))
                result(iota.getResult(), logicalIndex(layout, slot));
            return;
        }
        if (auto broadcastOp = dyn_cast<ir::BroadcastOp>(operation)) {
            values_[broadcastOp.getResult()] = SmallVector<Value>(
                broadcastOp.getResult().getType().getLayout().getSlots().size(), value(broadcastOp.getValue()));
            return;
        }
        if (auto loadOp = dyn_cast<ir::LoadOp>(operation))
            return load(loadOp);
        if (auto loadOp = dyn_cast<ir::ScalarLoadOp>(operation)) {
            auto memory = loadOp.getBuffer();
            return result(loadOp.getResult(), guardedLoad(pointer(memory), address(memory, value(loadOp.getIndex())),
                                                          scalarActive(loadOp), value(loadOp.getFallback())));
        }
        if (auto atomic = dyn_cast<ir::AtomicOp>(operation)) {
            auto memory = atomic.getBuffer();
            auto element = atomic.getResult().getType().getElementType();
            for (auto slot : slotRange(slotCount(atomic.getResult().getType())))
                result(atomic.getResult(), AtomicOp::create(builder_, location_, element, pointer(memory),
                    address(memory, value(atomic.getIndices(), slot)), value(atomic.getValues(), slot),
                    both(active(memory, atomic.getIndices(), atomic.getMask(), slot),
                         canonical(atomic.getIndices().getType().getLayout())), atomic.getOpcode()));
            return;
        }
        if (auto atomic = dyn_cast<ir::AtomicStoreOp>(operation)) {
            auto memory = atomic.getBuffer();
            for (auto slot : slotRange(slotCount(atomic.getIndices().getType())))
                AtomicStoreOp::create(builder_, location_, pointer(memory),
                                      address(memory, value(atomic.getIndices(), slot)),
                                      value(atomic.getValues(), slot),
                                      both(active(memory, atomic.getIndices(), atomic.getMask(), slot),
                                           canonical(atomic.getIndices().getType().getLayout())));
            return;
        }
        if (auto storeOp = dyn_cast<ir::StoreOp>(operation))
            return store(storeOp);
        if (isa<ir::MathOp, ir::MaximumOp, ir::MinimumOp, ir::FmaOp, ir::SelectOp, ir::BitwiseOp, ir::BitcastOp,
                ir::IntegerDivOp>(operation))
            return elementwise(operation);
        if (auto mapOp = dyn_cast<ir::MapOp>(operation))
            return map(mapOp);
        if (auto compareOp = dyn_cast<arith::CmpFOp>(operation)) {
            StringRef opcode;
            switch (compareOp.getPredicate()) {
            case arith::CmpFPredicate::OLT: opcode = "<"; break;
            case arith::CmpFPredicate::OLE: opcode = "<="; break;
            case arith::CmpFPredicate::OGT: opcode = ">"; break;
            case arith::CmpFPredicate::OGE: opcode = ">="; break;
            case arith::CmpFPredicate::OEQ: opcode = "=="; break;
            case arith::CmpFPredicate::UNE: opcode = "!="; break;
            default: fail("unsupported floating-point comparison predicate");
            }
            return result(compareOp.getResult(),
                          comparison(opcode, value(compareOp.getLhs()), value(compareOp.getRhs())));
        }
        if (isa<arith::AddIOp, arith::AddFOp, arith::SubIOp, arith::SubFOp, arith::MulIOp, arith::MulFOp,
                arith::DivUIOp, arith::DivFOp, arith::RemUIOp, arith::CmpIOp, arith::NegFOp>(operation)) {
            IRMapping mapping;
            for (auto operand : operation.getOperands())
                mapping.map(operand, value(operand));
            return result(operation.getResult(0), builder_.clone(operation, mapping)->getResult(0));
        }
        fail("unsupported operation in Metal lowering: " + operation.getName().getStringRef().str());
    }

    func::FuncOp function_, thread_;
    MLIRContext* context_;
    OpBuilder builder_;
    Location location_;
    // Lowered only to check scratch: every store stays one access per slot, so the cells it writes are recorded.
    bool checking_;
    Type i1_, i16_, i32_, i64_;
    Value lane_, simdGroup_;
    // Write-check shadows hold grid thread index + 1 per element. record[0] counts conflicts;
    // record[1 + i] holds checked binding i's lowest conflicting element.
    llvm::DenseMap<size_t, std::pair<Value, size_t>> shadows_;
    Value record_, writer_;
    MemoryPlan* memory_ = nullptr;
    llvm::DenseMap<Value, SmallVector<Value>> values_;
    llvm::DenseMap<Value, size_t> bindings_;
    llvm::DenseMap<Value, ViewValues> views_;
    llvm::DenseMap<Operation*, SmallVector<std::pair<Value, Value>>> scratchAccesses_;
    std::map<std::pair<const void*, int64_t>, Value> indices_;
    llvm::DenseSet<Operation*> finalScratchUsers_;
    llvm::DenseSet<Operation*> deferred_;
};

}

std::string registersName(Type element) {
    std::string name = "msl.simdgroup.";
    llvm::raw_string_ostream stream(name);
    element.print(stream);
    return name;
}

static void materializeConversions(func::FuncOp function) {
    function.walk([](ir::ConvertLayoutOp conversion) {
        auto scratch = conversion.getScratch();
        if (!scratch)
            return;
        OpBuilder builder(conversion);
        auto location = conversion.getLoc();
        auto input = conversion.getValue(), output = conversion.getResult();
        auto index = [&](ir::LayoutAttr layout) {
            return ir::IotaOp::create(builder, location,
                                      ir::TileType::get(builder.getContext(), builder.getI32Type(), layout));
        };
        auto full = [&](ir::LayoutAttr layout, Type element, TypedAttr value) {
            Value scalar = arith::ConstantOp::create(builder, location, value);
            if (scalar.getType() != element)
                scalar = ir::CastOp::create(builder, location, element, scalar);
            return ir::BroadcastOp::create(builder, location,
                                           ir::TileType::get(builder.getContext(), element, layout), scalar);
        };
        auto a = input.getType().getLayout(), b = output.getType().getLayout();
        auto before = index(a);
        auto active = full(a, builder.getI1Type(), builder.getBoolAttr(true));
        ir::StoreOp::create(builder, location, scratch, input, active, ValueRange{before});
        ir::BarrierOp::create(builder, location);
        auto after = index(b);
        active = full(b, builder.getI1Type(), builder.getBoolAttr(true));
        auto element = output.getType().getElementType();
        Type seed = isa<FloatType>(element) ? Type(builder.getF32Type()) : Type(builder.getI32Type());
        auto zero = full(b, element, builder.getZeroAttr(seed));
        auto loaded = ir::LoadOp::create(builder, location, output.getType(), scratch, active, zero, ValueRange{after});
        ir::BarrierOp::create(builder, location);
        conversion.replaceAllUsesWith(loaded.getResult());
        conversion.erase();
    });
}

ThreadKernel lowerToThreads(func::FuncOp function) {
    function.getContext()->loadDialect<MslDialect, arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
    materializeConversions(function);
    auto kernel = Lowering(function).run();
    auto capacity = function->getAttrOfType<IntegerAttr>("carbon.threadgroup_memory");
    if (capacity && kernel.memory.scratchBytes > uint64_t(capacity.getInt()))
        fail("kernel needs " + std::to_string(kernel.memory.scratchBytes) + " bytes of threadgroup memory; device allows " +
             std::to_string(capacity.getInt()));
    return kernel;
}

void checkScratch(func::FuncOp function) {
    function.getContext()->loadDialect<MslDialect, arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
    materializeConversions(function);
    Lowering lowering(function, true);
    auto kernel = lowering.run();
    lowering.checkScratch();
}

}

namespace carbon {

void validateMetal(func::FuncOp function) { msl::checkBindings(planMemory(function)); }

KernelSource lowerMetal(func::FuncOp function) {
    auto kernel = msl::lowerToThreads(function);
    msl::optimizeThreads(*kernel.module, kernel.memory);
    msl::verifyConvergence(*kernel.module, kernel.memory);
    KernelSource output;
    llvm::raw_string_ostream threadIR(output.threadIR);
    kernel.module->print(threadIR);
    output.source = msl::printMetal(*kernel.module, function, kernel.memory);
    output.memory = std::move(kernel.memory);
    return output;
}

}
