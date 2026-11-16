#include "tile/analysis.h"
#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <set>

namespace carbon::ir {
namespace {

// `value` with its bit i moved to position bits[i].
int64_t permute(int64_t value, const std::vector<int64_t>& bits) {
    int64_t result = 0;
    for (size_t bit = 0; bit < bits.size(); ++bit)
        result |= ((value >> bit) & 1) << bits[bit];
    return result;
}

}

// A plan requires a uniform lane/slot bit permutation with no movement between SIMD groups.
// Reject it unless it saves shuffles over gathering slots directly.
std::optional<ConversionPlan> planBitConversion(LayoutAttr source, LayoutAttr target) {
    if (source.replicated() || target.replicated())
        return std::nullopt;
    auto slots = target.getSlots().size();
    if (source.getSlots().size() != slots || source.getGroups().size() != target.getGroups().size())
        return std::nullopt;
    auto atomSlots = slots & -slots;
    std::vector<int64_t> bits;
    for (int64_t position = 1; position < atomSlots * 32; position *= 2) {
        auto index = target.index(0, position % 32, position / 32);
        auto owner = source.getLanes().inverse(index) + 32 * source.getSlots().inverse(index);
        if (!owner || (owner & (owner - 1)) || owner >= atomSlots * 32 || source.groupOwner(index))
            return std::nullopt;
        int64_t bit = 0;
        while ((int64_t(1) << bit) != owner)
            ++bit;
        if (std::find(bits.begin(), bits.end(), bit) != bits.end())
            return std::nullopt;
        bits.push_back(bit);
    }

    int64_t directShuffles = 0;
    for (int64_t slot = 0; slot < slots; ++slot) {
        std::set<int64_t> candidates;
        bool shuffle = false;
        for (int64_t group = 0; group < target.getGroups().size(); ++group)
            for (int64_t lane = 0; lane < 32; ++lane) {
                auto index = target.index(group, lane, slot);
                auto ownerSlot = source.getSlots().inverse(index), ownerLane = source.getLanes().inverse(index);
                auto expected = permute(lane + 32 * (slot % atomSlots), bits) + 32 * (slot / atomSlots * atomSlots);
                if (source.groupOwner(index) != group || ownerLane + 32 * ownerSlot != expected)
                    return std::nullopt;
                if (index < target.size()) {
                    candidates.insert(ownerSlot);
                    shuffle |= ownerLane != lane;
                }
            }
        if (shuffle)
            directShuffles += candidates.size();
    }

    ConversionPlan plan;
    std::vector<int64_t> current(bits.size());
    std::iota(current.begin(), current.end(), 0);
    std::vector<int64_t> laneBits, slotBits;
    for (int64_t bit = 0; bit < 5; ++bit) {
        if (std::find(bits.begin(), bits.begin() + 5, bit) == bits.begin() + 5)
            laneBits.push_back(bit);
        if (bits[bit] >= 5)
            slotBits.push_back(bits[bit]);
    }
    std::sort(slotBits.begin(), slotBits.end());
    for (size_t i = 0; i < laneBits.size(); ++i) {
        plan.exchanges.emplace_back(laneBits[i], slotBits[i] - 5);
        std::swap(current[laneBits[i]], current[slotBits[i]]);
    }
    bool laneShuffle = false;
    std::vector<int64_t> slotPermutation;
    for (size_t bit = 0; bit < bits.size(); ++bit) {
        auto position = std::find(current.begin(), current.end(), bits[bit]) - current.begin();
        if (bit < 5) {
            plan.lanes[bit] = position;
            laneShuffle |= position != int64_t(bit);
        } else
            slotPermutation.push_back(position - 5);
    }
    for (int64_t slot = 0; slot < slots; ++slot)
        plan.slots.push_back(slot / atomSlots * atomSlots + permute(slot % atomSlots, slotPermutation));
    auto shuffles = plan.exchanges.size() * (slots / 2) + (laneShuffle ? slots : 0);
    if (shuffles >= uint64_t(directShuffles))
        return std::nullopt;
    auto stages = plan.exchanges.size() + size_t(laneShuffle);
    // Keep two-register routes shallow when factoring only saves one shuffle.
    if (slots == 2 && stages > 1)
        return std::nullopt;
    return plan;
}

}

namespace carbon::ir {

using namespace mlir;

namespace {

Attribute integer(MLIRContext* context, int64_t value) {
    return IntegerAttr::get(IntegerType::get(context, 64), value);
}

int64_t product(ArrayRef<int64_t> values) {
    return std::accumulate(values.begin(), values.end(), int64_t(1), std::multiplies<int64_t>());
}

std::optional<std::pair<Attribute, Attribute>> remapTree(Attribute shape, Attribute stride,
                                                         ArrayRef<int64_t> boundaries,
                                                         llvm::function_ref<int64_t(int64_t)> transform) {
    auto* context = shape.getContext();
    if (auto extents = dyn_cast<ArrayAttr>(shape)) {
        auto steps = cast<ArrayAttr>(stride);
        SmallVector<Attribute> sizes, strides;
        for (auto [extent, step] : llvm::zip(extents, steps)) {
            auto pair = remapTree(extent, step, boundaries, transform);
            if (!pair)
                return std::nullopt;
            sizes.push_back(pair->first);
            strides.push_back(pair->second);
        }
        return std::make_pair(ArrayAttr::get(context, sizes), ArrayAttr::get(context, strides));
    }
    auto extent = cast<IntegerAttr>(shape).getInt(), step = cast<IntegerAttr>(stride).getInt();
    if (extent == 1)
        return std::make_pair(integer(context, 1), integer(context, 0));
    if (!step)
        return std::make_pair(shape, stride);
    SmallVector<int64_t> points{step};
    for (auto point : boundaries)
        if (step < point && point < step * extent)
            points.push_back(point);
    points.push_back(step * extent);
    SmallVector<Attribute> sizes, strides;
    for (size_t at = points.size() - 1; at > 0; --at) {
        if (points[at] % points[at - 1])
            return std::nullopt;
        sizes.push_back(integer(context, points[at] / points[at - 1]));
        strides.push_back(integer(context, transform(points[at - 1])));
    }
    if (sizes.size() == 1)
        return std::make_pair(sizes[0], strides[0]);
    return std::make_pair(ArrayAttr::get(context, sizes), ArrayAttr::get(context, strides));
}

LayoutAttr layout(MLIRContext* context, ArrayRef<int64_t> shape, ModeAttr groups, ModeAttr lanes, ModeAttr slots) {
    if (!groups || !lanes || !slots)
        return {};
    return LayoutAttr::get(context, DenseI64ArrayAttr::get(context, shape), groups, lanes, slots);
}

}

ModeAttr ModeAttr::of(MLIRContext* context, int64_t extent, int64_t stride) {
    return get(context, integer(context, extent), integer(context, stride));
}

ModeAttr ModeAttr::remap(ArrayRef<int64_t> boundaries, llvm::function_ref<int64_t(int64_t)> transform) const {
    auto pair = remapTree(getShape(), getStride(), boundaries, transform);
    return pair ? get(getContext(), pair->first, pair->second) : ModeAttr();
}

LayoutAttr LayoutAttr::striped(MLIRContext* context, ArrayRef<int64_t> shape, int64_t threads) {
    auto slots = (product(shape) + threads - 1) / threads;
    return layout(context, shape, ModeAttr::of(context, threads / 32, 32), ModeAttr::of(context, 32, 1),
                  ModeAttr::of(context, slots, threads));
}

LayoutAttr LayoutAttr::blocked(MLIRContext* context, ArrayRef<int64_t> shape, int64_t threads) {
    auto slots = (product(shape) + threads - 1) / threads;
    return layout(context, shape, ModeAttr::of(context, threads / 32, 32 * slots), ModeAttr::of(context, 32, slots),
                  ModeAttr::of(context, slots, 1));
}

LayoutAttr LayoutAttr::reshape(ArrayRef<int64_t> shape) const {
    if (product(shape) != size())
        return {};
    return layout(getContext(), shape, getGroups(), getLanes(), getSlots());
}

LayoutAttr LayoutAttr::stack(int64_t count, int64_t axis) const {
    auto shape = getShape().asArrayRef();
    SmallVector<int64_t> result(shape);
    result.insert(result.begin() + axis, count);
    if (count == 1)
        return layout(getContext(), result, getGroups(), getLanes(), getSlots());
    auto inner = product(shape.drop_front(axis));
    auto move = [&](int64_t step) { return step >= inner ? step * count : step; };
    auto slots = getSlots().remap({inner}, move);
    if (!slots)
        return {};
    auto* context = getContext();
    slots = ModeAttr::get(context, ArrayAttr::get(context, {integer(context, count), slots.getShape()}),
                          ArrayAttr::get(context, {integer(context, inner), slots.getStride()}));
    return layout(context, result, getGroups().remap({inner}, move), getLanes().remap({inner}, move), slots);
}

LayoutAttr LayoutAttr::transpose(ArrayRef<int64_t> axes) const {
    auto shape = getShape().asArrayRef();
    SmallVector<int64_t> result, strides, target(shape.size()), boundaries;
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        result.push_back(shape[axes[axis]]);
        strides.push_back(product(shape.drop_front(axis + 1)));
    }
    for (size_t axis = 0; axis < shape.size(); ++axis)
        target[axis] = product(ArrayRef<int64_t>(result).drop_front(axis + 1));
    auto transform = [&](int64_t step) {
        for (size_t axis = 0; axis < shape.size(); ++axis)
            if (shape[axis] > 1 && step >= strides[axis])
                return step / strides[axis] * target[std::find(axes.begin(), axes.end(), axis) - axes.begin()];
        return step;
    };
    boundaries.assign(strides.begin(), strides.end());
    llvm::sort(boundaries);
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
    return layout(getContext(), result, getGroups().remap(boundaries, transform),
                  getLanes().remap(boundaries, transform), getSlots().remap(boundaries, transform));
}

// Split digits at axis boundaries; slots on the reduced axis collapse locally.
// Renumber its lane/group digits last so roots keep the line-start owner and other threads hold padding.
LayoutAttr LayoutAttr::reduced(int64_t axis) const {
    auto shape = getShape().asArrayRef();
    auto* context = getContext();
    if (shape.size() == 1)
        return striped(context, {}, threads());
    auto inner = product(shape.drop_front(axis + 1)), outer = inner * shape[axis];
    SmallVector<int64_t> boundaries{inner};
    if (outer != inner)
        boundaries.push_back(outer);
    auto same = [](int64_t step) { return step; };
    auto groups = getGroups().remap(boundaries, same), lanes = getLanes().remap(boundaries, same);
    auto slots = getSlots().remap(boundaries, same);
    if (!groups || !lanes || !slots)
        return {};
    std::function<Attribute(Attribute, Attribute, bool)> retained = [&](Attribute size, Attribute step, bool extent) {
        if (auto sizes = dyn_cast<ArrayAttr>(size)) {
            SmallVector<Attribute> parts;
            for (auto [a, b] : llvm::zip(sizes, cast<ArrayAttr>(step)))
                parts.push_back(retained(a, b, extent));
            return Attribute(ArrayAttr::get(context, parts));
        }
        auto stride = cast<IntegerAttr>(step).getInt();
        bool dropped = inner <= stride && stride < outer;
        return dropped ? integer(context, extent ? 1 : 0) : (extent ? size : step);
    };
    slots = ModeAttr::get(context, retained(slots.getShape(), slots.getStride(), true),
                          retained(slots.getShape(), slots.getStride(), false));
    SmallVector<std::pair<int64_t, int64_t>> digits;
    for (auto mode : {groups, lanes, slots})
        for (auto [extent, stride] : mode.digits())
            if (extent > 1 && stride)
                digits.emplace_back(stride, extent);
    llvm::sort(digits, [&](auto a, auto b) {
        bool x = inner <= a.first && a.first < outer, y = inner <= b.first && b.first < outer;
        return std::make_pair(x, a.first) < std::make_pair(y, b.first);
    });
    std::map<int64_t, int64_t> strides;
    int64_t capacity = 1;
    for (auto [stride, extent] : digits) {
        strides[stride] = capacity;
        capacity *= extent;
    }
    auto renumber = [&](int64_t step) { return strides.at(step); };
    SmallVector<int64_t> result(shape);
    result.erase(result.begin() + axis);
    return layout(context, result, groups.remap({}, renumber), lanes.remap({}, renumber), slots.remap({}, renumber));
}

LayoutAttr LayoutAttr::acrossGroups(int64_t groups) const {
    if (getGroups().size() != 1)
        return {};
    return layout(getContext(), getShape().asArrayRef(), ModeAttr::of(getContext(), groups, 0), getLanes(), getSlots());
}

}
