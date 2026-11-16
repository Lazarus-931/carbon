#include "tile/analysis.h"
#include "tile/dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include <functional>
#include <map>
#include <numeric>
#include <set>

using namespace mlir;

namespace carbon::ir {

Value viewParent(Value value) {
    if (auto partition = value.getDefiningOp<PartitionOp>())
        return partition.getView();
    if (auto subview = value.getDefiningOp<SubviewOp>())
        return subview.getView();
    return {};
}

Value viewRoot(Value value) {
    while (auto parent = viewParent(value))
        value = parent;
    return value;
}

Value bindingRoot(Value value) {
    value = viewRoot(value);
    if (auto view = value.getDefiningOp<MakeViewOp>())
        value = view.getBuffer();
    return value;
}

std::optional<int64_t> flatStride(ViewType view) {
    auto shape = view.getShape().asArrayRef(), strides = view.getStrides().asArrayRef();
    std::optional<int64_t> stride;
    int64_t inner = 1;
    for (size_t axis = shape.size(); axis-- > 0;) {
        if (shape[axis] <= 1)
            continue;
        if (!stride)
            stride = strides[axis];
        else if (__int128(*stride) * inner != strides[axis])
            return std::nullopt;
        inner *= shape[axis];
    }
    return stride.value_or(0);
}

ScratchOp scratchAllocation(Value value) {
    while (auto view = value.getDefiningOp<MakeViewOp>())
        value = view.getBuffer();
    return value.getDefiningOp<ScratchOp>();
}

static void scratchLiveness(Block& block, llvm::DenseSet<Value>& live,
                            llvm::DenseSet<Operation*>& final, bool repeated) {
    for (auto& operation : llvm::reverse(block)) {
        if (operation.getNumRegions()) {
            auto incoming = live;
            for (auto& region : operation.getRegions()) {
                for (auto& nested : region) {
                    auto branch = live;
                    scratchLiveness(nested, branch, final, repeated || !isa<scf::IfOp>(operation));
                    incoming.insert(branch.begin(), branch.end());
                }
            }
            live = std::move(incoming);
        }
        Value scratch;
        if (auto item = dyn_cast<ItemOp>(operation))
            scratch = item.getScratch();
        else if (isa<ReduceSumOp, ReduceMaxOp, ScanSumOp, ScanMaxOp>(operation) && operation.getNumOperands() == 2)
            scratch = operation.getOperand(1);
        if (scratch && !repeated) {
            auto allocation = scratchAllocation(scratch);
            if (allocation && !live.contains(allocation.getResult()))
                final.insert(&operation);
        }
        for (auto operand : operation.getOperands())
            if (auto allocation = scratchAllocation(operand))
                live.insert(allocation.getResult());
    }
}

llvm::DenseSet<Operation*> finalScratchUsers(func::FuncOp function) {
    llvm::DenseSet<Value> live;
    llvm::DenseSet<Operation*> final;
    scratchLiveness(function.getBody().front(), live, final, false);
    return final;
}

bool collectiveUsesScratch(Operation& operation) {
    if (auto item = dyn_cast<ItemOp>(operation))
        return item.getValue().getType().getLayout().getGroups().size() > 1;
    if (!isa<ReduceSumOp, ReduceMaxOp, ScanSumOp, ScanMaxOp, BroadcastInDimOp>(operation) ||
        operation.getNumOperands() != 2)
        return false;
    auto layout = cast<TileType>(operation.getOperand(0).getType()).getLayout();
    auto target = cast<TileType>(operation.getResult(0).getType()).getLayout();
    if (isa<BroadcastInDimOp>(operation))
        return broadcastCrossGroup(layout, target, operation.getAttrOfType<DenseI64ArrayAttr>("axes").asArrayRef());
    if (isa<ScanSumOp, ScanMaxOp>(operation))
        return scanNeedsScratch(layout, operation.getAttrOfType<IntegerAttr>("axis").getInt());
    return reductionUsesScratch(&operation);
}

namespace {

using Pending = llvm::DenseSet<Value>;

void pendingBlock(Block& block, Pending& state, const llvm::DenseSet<Operation*>& final,
                  llvm::DenseMap<Operation*, Pending>& before) {
    for (auto& operation : block) {
        before[&operation] = state;
        if (isa<BarrierOp>(operation)) {
            state.clear();
        } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
            auto left = state, right = state;
            pendingBlock(branch.getThenRegion().front(), left, final, before);
            if (!branch.getElseRegion().empty())
                pendingBlock(branch.getElseRegion().front(), right, final, before);
            left.insert(right.begin(), right.end());
            state = std::move(left);
        } else if (auto loop = dyn_cast<scf::ForOp>(operation)) {
            auto incoming = state;
            while (true) {
                auto iteration = incoming;
                pendingBlock(*loop.getBody(), iteration, final, before);
                auto next = state;
                next.insert(iteration.begin(), iteration.end());
                if (next.size() == incoming.size()) {
                    state = std::move(next);
                    break;
                }
                incoming = std::move(next);
            }
        } else if (collectiveUsesScratch(operation)) {
            auto item = dyn_cast<ItemOp>(operation);
            auto allocation = scratchAllocation(item ? item.getScratch() : operation.getOperand(1));
            state.clear();
            if (allocation)
                state.insert(allocation.getResult());
            if (isa<BroadcastInDimOp>(operation) || !final.contains(&operation))
                state.clear();
        } else if (isa<LoadOp, StoreOp, TensorMatmulOp>(operation)) {
            for (auto operand : operation.getOperands())
                if (auto allocation = scratchAllocation(operand))
                    state.insert(allocation.getResult());
        }
    }
}

}

llvm::DenseMap<Operation*, llvm::DenseSet<Value>> pendingScratch(func::FuncOp function,
                                                                   const llvm::DenseSet<Operation*>& finalUsers) {
    llvm::DenseMap<Operation*, Pending> before;
    Pending state;
    pendingBlock(function.getBody().front(), state, finalUsers, before);
    return before;
}

namespace {

// A digit of a SIMD-group mode: the group number's digit at `place` takes `extent` values and is an element index
// over `stride` modulo `extent`, or any of them when `stride` is 0 because the element repeats across it.
struct Piece {
    int64_t place, extent, stride;
};

std::vector<Piece> groupPieces(ModeAttr mode) {
    std::vector<Piece> pieces;
    int64_t place = mode.size();
    for (auto [extent, stride] : mode.digits()) {
        place /= extent;
        if (extent > 1)
            pieces.push_back({place, extent, stride});
    }
    return pieces;
}

// Pieces split where an element index in `cuts`, ascending, falls inside them; none when a cut does not divide one.
std::optional<std::vector<Piece>> splitElements(ArrayRef<Piece> pieces, ArrayRef<int64_t> cuts) {
    std::vector<Piece> result;
    for (auto piece : pieces) {
        for (auto cut : cuts) {
            if (!piece.stride || cut <= piece.stride || cut >= piece.stride * piece.extent)
                continue;
            if (cut % piece.stride || piece.extent % (cut / piece.stride))
                return std::nullopt;
            auto low = cut / piece.stride;
            result.push_back({piece.place, low, piece.stride});
            piece = {piece.place * low, piece.extent / low, cut};
        }
        result.push_back(piece);
    }
    return result;
}

// Pieces split where a place in `cuts` falls inside them, keyed by place; none when a cut does not divide one.
std::optional<std::map<int64_t, Piece>> splitPlaces(ArrayRef<Piece> pieces, const std::set<int64_t>& cuts) {
    std::map<int64_t, Piece> result;
    for (auto piece : pieces) {
        for (auto cut : cuts) {
            if (cut <= piece.place || cut >= piece.place * piece.extent)
                continue;
            if (cut % piece.place || piece.extent % (cut / piece.place))
                return std::nullopt;
            auto low = cut / piece.place;
            result[piece.place] = {piece.place, low, piece.stride};
            piece = {cut, piece.extent / low, piece.stride * low};
        }
        result[piece.place] = piece;
    }
    return result;
}

// Whether an element below `size` that `target` gives some SIMD group has no copy in that group in `source`, both
// modes' pieces reading the same element index. Group numbers agree digit by digit or not at all; a stride at or
// past `size` reads 0 for every element, and digits that do not line up count as moving.
bool movesBetweenGroups(ArrayRef<Piece> source, ArrayRef<Piece> target, int64_t size) {
    std::set<int64_t> cuts;
    for (auto pieces : {source, target})
        for (auto piece : pieces)
            cuts.insert({piece.place, piece.place * piece.extent});
    auto ours = splitPlaces(source, cuts), theirs = splitPlaces(target, cuts);
    if (!ours || !theirs)
        return true;
    auto read = [&](int64_t stride) { return std::min(stride, size); };
    for (auto [place, piece] : *ours) {
        if (!piece.stride)
            continue;
        auto other = theirs->find(place);
        if (other == theirs->end() || !other->second.stride || read(other->second.stride) != read(piece.stride))
            return true;
    }
    return false;
}

struct Line {
    int64_t outer, extent, inner;
};

Line line(LayoutAttr layout, int64_t axis) {
    auto shape = layout.getShape().asArrayRef();
    Line result{1, shape[axis], 1};
    for (int64_t at = 0; at < axis; ++at)
        result.outer *= shape[at];
    for (int64_t at = axis + 1; at < int64_t(shape.size()); ++at)
        result.inner *= shape[at];
    return result;
}

// Group numbers as owners read them: a repeated digit reads 0.
std::vector<Piece> owners(std::vector<Piece> pieces, int64_t size) {
    for (auto& piece : pieces)
        if (!piece.stride)
            piece.stride = size;
    return pieces;
}

// Whether floor(q / weight) mod extent differs between q and q + distance for a q = o * length + c with o below
// `outer`, c a multiple of `step` and c + distance < length: the digit moves by distance / weight, plus one where
// the remainder carries, and q mod weight repeats in both o and c.
bool crosses(int64_t weight, int64_t extent, int64_t distance, int64_t step, int64_t length, int64_t outer) {
    auto quotient = distance / weight, rest = distance % weight;
    if (quotient % extent)
        return true;
    for (int64_t o = 0, start = 0; rest && o < outer; ++o) {
        for (int64_t c = 0, residue = start; c + distance < length; c += step) {
            if (residue + rest >= weight)
                return true;
            residue = (residue + step) % weight;
            if (residue == start)
                break;
        }
        start = (start + length) % weight;
        if (!start)
            break;
    }
    return false;
}

// The source's group digits at the first element of each reduction line, as digits of the reduced element index;
// none where a digit's value there is not a digit of the reduced index. A line start o * inner * extent + i reduces
// to o * inner + i, so a digit keeps its stride where the difference is a multiple of its span, and a digit whose
// stride passes every line start reads 0.
std::optional<std::vector<Piece>> lineStarts(LayoutAttr source, int64_t axis, int64_t size) {
    auto [outer, extent, inner] = line(source, axis);
    auto kept = [&](Piece piece) { return outer == 1 || inner * (extent - 1) % (piece.stride * piece.extent) == 0; };
    auto end = ((outer - 1) * extent + 1) * inner;
    std::vector<Piece> result;
    for (auto piece : groupPieces(source.getGroups())) {
        if (!piece.stride || piece.stride >= end) {
            result.push_back({piece.place, piece.extent, size});
            continue;
        }
        if (kept(piece)) {
            result.push_back(piece);
            continue;
        }
        auto parts = splitElements({piece}, {inner, inner * extent});
        if (!parts)
            return std::nullopt;
        for (auto part : *parts) {
            if (!kept(part)) {
                if (part.stride >= inner * extent && part.stride % (inner * extent) == 0)
                    part.stride /= extent;
                else if (part.stride >= inner && part.stride % inner == 0 &&
                         extent % (part.stride / inner * part.extent) == 0)
                    part.stride = size;
                else
                    return std::nullopt;
            }
            result.push_back(part);
        }
    }
    return result;
}

// The source's group digits as digits of the target's element index when source axis i lands on target axis
// axes[i]. Source axes that land side by side form one run; none where a digit reads more than one run.
std::optional<std::vector<Piece>> broadcastPieces(LayoutAttr source, LayoutAttr target, ArrayRef<int64_t> axes) {
    auto a = source.getShape().asArrayRef(), b = target.getShape().asArrayRef();
    SmallVector<int64_t> to(b.size() + 1, 1);
    for (size_t axis = b.size(); axis-- > 0;)
        to[axis] = to[axis + 1] * b[axis];
    // Source axes that land side by side in the target: their source stride, joint extent and target stride.
    struct Run {
        int64_t stride, extent, image;
    };
    SmallVector<Run> runs;
    SmallVector<int64_t> cuts;
    int64_t stride = 1;
    for (size_t axis = a.size(); axis-- > 0;) {
        if (a[axis] == 1)
            continue;
        auto image = to[axes[axis] + 1];
        if (!runs.empty() && runs.back().image * runs.back().extent == image) {
            runs.back().extent *= a[axis];
        } else {
            runs.push_back({stride, a[axis], image});
            cuts.push_back(stride);
        }
        stride *= a[axis];
    }
    auto pieces = splitElements(groupPieces(source.getGroups()), cuts);
    if (!pieces)
        return std::nullopt;
    for (auto& piece : *pieces) {
        if (!piece.stride)
            continue;
        if (piece.stride >= source.size()) {
            piece.stride = target.size();
            continue;
        }
        auto run = runs.begin();
        while (run->stride * run->extent <= piece.stride)
            ++run;
        auto step = piece.stride / run->stride;
        bool alone = run + 1 == runs.end() && run->image * run->extent == target.size();
        if (piece.stride % run->stride || (run->extent % (step * piece.extent) && !alone))
            return std::nullopt;
        piece.stride = step * run->image;
    }
    return pieces;
}

// Whether the element at coordinate c along `axis` and the one at c + distance sit in different SIMD groups, for
// some c that is a multiple of `step` with c + distance inside the axis: reduction trees pair c = 0 mod 2 * distance,
// scans pair every c.
bool pairsCrossGroup(LayoutAttr source, int64_t axis, int64_t distance, int64_t step) {
    auto [outer, extent, inner] = line(source, axis);
    auto last = (extent - distance - 1) / step * step;
    auto end = ((outer - 1) * extent + last + distance + 1) * inner;
    for (auto piece : groupPieces(source.getGroups())) {
        if (!piece.stride || piece.stride >= end || distance * inner % (piece.stride * piece.extent) == 0)
            continue;
        auto parts = splitElements({piece}, {inner});
        if (!parts)
            return true;
        for (auto part : *parts)
            if (part.stride < inner ? distance * inner % (part.stride * part.extent) != 0
                                    : part.stride % inner ||
                                          crosses(part.stride / inner, part.extent, distance, step, extent, outer))
                return true;
    }
    return false;
}

}

bool conversionCrossGroup(LayoutAttr source, LayoutAttr target) {
    return movesBetweenGroups(groupPieces(source.getGroups()), groupPieces(target.getGroups()), target.size());
}

bool conversionMovesBetweenThreads(LayoutAttr source, LayoutAttr target) {
    if (source == target)
        return false;
    return source.replicated() || conversionCrossGroup(source, target) ||
           movesBetweenGroups(groupPieces(source.getLanes()), groupPieces(target.getLanes()), target.size());
}

bool reductionCrossGroup(LayoutAttr source, LayoutAttr target, int64_t axis, int64_t distance) {
    if (!distance) {
        auto starts = lineStarts(source, axis, target.size());
        return !starts ||
               movesBetweenGroups(*starts, owners(groupPieces(target.getGroups()), target.size()), target.size());
    }
    return pairsCrossGroup(source, axis, distance, 2 * distance);
}

bool scanCrossGroup(LayoutAttr layout, int64_t axis, int64_t distance) {
    return pairsCrossGroup(layout, axis, distance, 1);
}

bool scanNeedsScratch(LayoutAttr layout, int64_t axis) {
    for (int64_t distance = 1; distance < layout.getShape()[axis]; distance *= 2)
        if (scanCrossGroup(layout, axis, distance))
            return true;
    return false;
}

bool reductionNeedsScratch(LayoutAttr source, LayoutAttr target, int64_t axis) {
    if (reductionCrossGroup(source, target, axis, 0))
        return true;
    for (int64_t distance = 1; distance < source.getShape()[axis]; distance *= 2)
        if (reductionCrossGroup(source, source, axis, distance))
            return true;
    return false;
}

bool nativeReduction(Operation* operation) {
    if (!isa<ReduceSumOp, ReduceMaxOp>(operation))
        return false;
    auto source = cast<TileType>(operation->getOperand(0).getType()).getLayout();
    auto result = cast<TileType>(operation->getResult(0).getType());
    auto type = result.getElementType();
    bool maximum = isa<ReduceMaxOp>(operation);
    bool exact = type.isInteger(32) || (maximum && source.size() > 1);
    bool fast = !maximum && type.isF32() && operation->getAttrOfType<StringAttr>("order").getValue() == "fast";
    auto axis = operation->getAttrOfType<IntegerAttr>("axis").getInt();
    return (exact || fast) && result.getLayout().size() == 1 && source.size() >= source.getGroups().size() &&
           (source.getGroups().size() == 1 || reductionNeedsScratch(source, result.getLayout(), axis));
}

bool reductionUsesScratch(Operation* operation) {
    auto layout = cast<TileType>(operation->getOperand(0).getType()).getLayout();
    auto target = cast<TileType>(operation->getResult(0).getType()).getLayout();
    return reductionNeedsScratch(layout, target, operation->getAttrOfType<IntegerAttr>("axis").getInt());
}

int64_t broadcastIndex(LayoutAttr source, LayoutAttr target, ArrayRef<int64_t> axes, int64_t index) {
    auto a = source.getShape().asArrayRef(), b = target.getShape().asArrayRef();
    int64_t result = 0;
    for (size_t dimension = 0; dimension < a.size(); ++dimension) {
        int64_t stride = 1;
        for (size_t axis = axes[dimension] + 1; axis < b.size(); ++axis)
            stride *= b[axis];
        result = result * a[dimension] + (a[dimension] == 1 ? 0 : index / stride % b[axes[dimension]]);
    }
    return result;
}

bool broadcastCrossGroup(LayoutAttr source, LayoutAttr target, ArrayRef<int64_t> axes) {
    auto pieces = broadcastPieces(source, target, axes);
    return !pieces || movesBetweenGroups(*pieces, groupPieces(target.getGroups()), target.size());
}

namespace {

constexpr int noOwner = -1, manyOwners = -2;

struct Cell {
    bool initialized = false;
    int reader = noOwner, writer = noOwner;
    bool operator==(const Cell& other) const {
        return initialized == other.initialized && reader == other.reader && writer == other.writer;
    }
};

using State = llvm::DenseMap<Value, std::vector<Cell>>;

int mergeOwners(int a, int b) {
    if (a == noOwner) return b;
    if (b == noOwner || a == b) return a;
    return manyOwners;
}

State merge(State left, const State& right) {
    for (auto& [allocation, cells] : left) {
        const auto& other = right.at(allocation);
        for (size_t i = 0; i < cells.size(); ++i) {
            cells[i].initialized &= other[i].initialized;
            cells[i].reader = mergeOwners(cells[i].reader, other[i].reader);
            cells[i].writer = mergeOwners(cells[i].writer, other[i].writer);
        }
    }
    return left;
}

bool sameState(const State& left, const State& right) {
    for (const auto& [allocation, cells] : left)
        if (cells != right.at(allocation))
            return false;
    return true;
}

LogicalResult touch(Operation& operation, Cell& cell, int owner, bool write, bool definite) {
    auto conflicts = [&](int other) { return other != noOwner && (other == manyOwners || other != owner); };
    if (conflicts(cell.writer) || (write && conflicts(cell.reader)))
        return operation.emitOpError("scratch ownership changes require a threadgroup barrier");
    if (!write && !cell.initialized)
        return operation.emitOpError("scratch read may access uninitialized storage");
    if (write) {
        cell.initialized |= definite;
        cell.writer = owner;
    } else
        cell.reader = mergeOwners(cell.reader, owner);
    return success();
}

int64_t logicalSize(ViewType view) {
    if (llvm::is_contained(view.getShape().asArrayRef(), 0))
        return 0;
    int64_t size = 1;
    for (auto extent : view.getShape().asArrayRef())
        size *= extent;
    return size;
}

int64_t elementAddress(ViewType view, int64_t index) {
    int64_t address = view.getOffsetBytes() / (view.getElementType().getIntOrFloatBitWidth() / 8);
    for (int64_t axis = int64_t(view.getShape().size()) - 1; axis >= 0; --axis) {
        address += (index % view.getShape()[axis]) * view.getStrides()[axis];
        index /= view.getShape()[axis];
    }
    return address;
}

// Each cell `view` reaches, once: an axis of stride 0 repeats cells, and a view naming more elements than its
// allocation holds overlaps itself, so it reaches at most the cells between its lowest and highest address.
LogicalResult reachable(ViewType view, int64_t cells, llvm::function_ref<LogicalResult(int64_t)> visit) {
    if (!logicalSize(view))
        return success();
    SmallVector<std::pair<int64_t, int64_t>> axes;
    int64_t distinct = 1, low = elementAddress(view, 0), high = low;
    for (auto [extent, stride] : llvm::zip(view.getShape().asArrayRef(), view.getStrides().asArrayRef())) {
        if (!stride || extent == 1)
            continue;
        axes.emplace_back(extent, stride);
        distinct = std::min(distinct * extent, cells + 1);
        (stride < 0 ? low : high) += (extent - 1) * stride;
    }
    if (distinct > cells) {
        for (auto address = std::max<int64_t>(low, 0); address <= std::min(high, cells - 1); ++address)
            if (failed(visit(address)))
                return failure();
        return success();
    }
    SmallVector<int64_t> digits(axes.size());
    for (int64_t count = 0; count < distinct; ++count) {
        int64_t address = elementAddress(view, 0);
        for (auto [digit, axis] : llvm::zip(digits, axes))
            address += digit * axis.second;
        if (failed(visit(address)))
            return failure();
        for (size_t at = axes.size(); at-- > 0 && ++digits[at] == axes[at].first;)
            digits[at] = 0;
    }
    return success();
}

// The cells a scratch load or store touches, each by the thread that touches it, as the per-thread IR computes them.
// Stores must name distinct cells that `touches` pins down; a read it cannot pin down reads the whole allocation.
LogicalResult memory(Operation& operation, Value buffer, bool write, State& state, ScratchTouches touches) {
    auto type = dyn_cast<ViewType>(buffer.getType());
    if (!type || type.getAddressSpace() != "threadgroup")
        return success();
    auto allocation = scratchAllocation(buffer);
    auto found = allocation ? state.find(allocation.getResult()) : state.end();
    if (found == state.end())
        return operation.emitOpError("scratch accesses require a kernel-scope allocation");
    auto& cells = found->second;
    auto accesses = touches(&operation);
    std::set<int64_t> seen;
    if (write && (!accesses || llvm::any_of(*accesses, [&](const ScratchTouch& access) {
            return !seen.insert(access.cell).second;
        })))
        return operation.emitOpError("scratch stores require unique indices computed from the lane, SIMD group and "
                                     "constants");
    if (!accesses)
        return reachable(type, cells.size(), [&](int64_t address) {
            return touch(operation, cells[address], manyOwners, false, false);
        });
    for (const auto& access : *accesses)
        if (access.cell < int64_t(cells.size()) &&
            failed(touch(operation, cells[access.cell], access.thread, write, access.definite)))
            return failure();
    return success();
}

LogicalResult block(Block& body, State& state, ScratchTouches touches) {
    for (auto& operation : body) {
        if (isa<BarrierOp>(operation)) {
            for (auto& [allocation, cells] : state)
                for (auto& cell : cells)
                    cell.reader = cell.writer = noOwner;
        } else if (auto matmul = dyn_cast<TensorMatmulOp>(operation)) {
            for (auto memory : {matmul.getLeft(), matmul.getRight()}) {
                auto view = cast<ViewType>(memory.getType());
                auto allocation = scratchAllocation(memory);
                auto found = allocation ? state.find(allocation.getResult()) : state.end();
                if (view.getAddressSpace() != "threadgroup" || found == state.end())
                    continue;
                if (failed(reachable(view, found->second.size(), [&](int64_t address) {
                        return touch(operation, found->second[address], manyOwners, false, false);
                    })))
                    return failure();
            }
        } else if (auto load = dyn_cast<LoadOp>(operation)) {
            if (failed(memory(operation, load.getBuffer(), false, state, touches)))
                return failure();
        } else if (auto store = dyn_cast<StoreOp>(operation)) {
            if (failed(memory(operation, store.getBuffer(), true, state, touches)))
                return failure();
        } else if (auto item = dyn_cast<ItemOp>(operation)) {
            if (item.getValue().getType().getLayout().getGroups().size() > 1) {
                auto scratch = item.getScratch();
                auto allocation = scratchAllocation(scratch);
                auto found = allocation ? state.find(allocation.getResult()) : state.end();
                if (found == state.end())
                    return item.emitOpError("item workspace must be a kernel-scope allocation");
                auto& cell = found->second[elementAddress(scratch.getType(), 0)];
                if (failed(touch(operation, cell, 0, true, true)))
                    return failure();
                cell = Cell{};
            }
        } else if (isa<ReduceSumOp, ReduceMaxOp, ScanSumOp, ScanMaxOp, BroadcastInDimOp>(operation)) {
            if (operation.getNumOperands() == 2) {
                auto scratch = operation.getOperand(1);
                auto allocation = scratchAllocation(scratch);
                auto found = allocation ? state.find(allocation.getResult()) : state.end();
                if (found == state.end())
                    return operation.emitOpError("collective workspace must be a kernel-scope allocation");
                auto layout = cast<TileType>(operation.getOperand(0).getType()).getLayout();
                auto target = cast<TileType>(operation.getResult(0).getType()).getLayout();
                bool used = collectiveUsesScratch(operation);
                auto view = cast<ViewType>(scratch.getType());
                if (used && nativeReduction(&operation)) {
                    for (auto& [allocation, cells] : state)
                        for (auto& cell : cells)
                            cell.reader = cell.writer = noOwner;
                    for (int64_t group = 0; group < layout.getGroups().size(); ++group) {
                        auto& cell = found->second[elementAddress(view, group)];
                        if (failed(touch(operation, cell, group * 32, true, true)))
                            return failure();
                        cell = Cell{};
                    }
                    continue;
                }
                for (int64_t i = 0; i < layout.size(); ++i) {
                    auto& cell = found->second[elementAddress(view, i)];
                    if (used) {
                        int owner = layout.getGroups().inverse(i) * 32 + layout.getLanes().inverse(i);
                        if (failed(touch(operation, cell, owner, true, true)))
                            return failure();
                        cell = Cell{};
                    } else
                        cell.initialized = false;
                }
            }
        } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
            auto left = state, right = state;
            if (failed(block(branch.getThenRegion().front(), left, touches)) ||
                (!branch.getElseRegion().empty() && failed(block(branch.getElseRegion().front(), right, touches))))
                return failure();
            state = merge(std::move(left), right);
        } else if (auto loop = dyn_cast<scf::ForOp>(operation)) {
            auto incoming = state;
            while (true) {
                auto iteration = incoming;
                if (failed(block(*loop.getBody(), iteration, touches)))
                    return failure();
                auto next = merge(state, iteration);
                if (sameState(incoming, next)) {
                    state = std::move(next);
                    break;
                }
                incoming = std::move(next);
            }
        }
    }
    return success();
}

}

LogicalResult verifyScratch(func::FuncOp function, ScratchTouches touches) {
    State state;
    for (auto allocation : function.getOps<ScratchOp>()) {
        auto type = allocation.getResult().getType();
        state[allocation.getResult()] = std::vector<Cell>(type.getExtentBytes() /
                                                          (type.getElementType().getIntOrFloatBitWidth() / 8));
    }
    return block(function.getBody().front(), state, touches);
}

}
