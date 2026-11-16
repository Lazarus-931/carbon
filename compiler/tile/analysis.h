#pragma once

#include "tile/dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace carbon::ir {

/// The single element stride of a view whose axes nest without gaps; 0 for one element.
std::optional<int64_t> flatStride(ViewType view);

/// Lane/slot bit permutations within each SIMD group.
struct ConversionPlan {
    /// (lane bit, slot bit) pairs.
    std::vector<std::pair<int64_t, int64_t>> exchanges;
    /// After the exchanges, one shuffle per slot: bit `i` of a lane's index is bit `lanes[i]` of the lane it reads.
    std::array<int64_t, 5> lanes;
    /// After those shuffles, slot `i` of the target holds slot `slots[i]`.
    std::vector<int64_t> slots;
};

/// The plan that converts `source` to `target` in registers; none when the conversion is not a bit permutation.
std::optional<ConversionPlan> planBitConversion(LayoutAttr source, LayoutAttr target);

/// The view a partition or subview is taken from; null for any other value.
mlir::Value viewParent(mlir::Value value);
/// `value` without its partitions and subviews.
mlir::Value viewRoot(mlir::Value value);
/// `value` without its partitions, subviews and view: the kernel argument or scratch allocation it reads.
mlir::Value bindingRoot(mlir::Value value);
/// The scratch allocation `value` is, or is a view of; partitions and subviews are not looked through.
ScratchOp scratchAllocation(mlir::Value value);
/// Whether scalar `value` is equal in every thread of the threadgroup or, with `withinSimd`, of each SIMD group;
/// false for tiles and buffers.
bool groupUniform(mlir::Value value, bool withinSimd = false);

/// Whether a reduction along `axis` moves values between SIMD groups: for `distance` 0, when each line's result moves
/// to its owner in `target`; otherwise, in the tree step that combines elements `distance` apart.
bool reductionCrossGroup(LayoutAttr source, LayoutAttr target, int64_t axis, int64_t distance);
bool reductionNeedsScratch(LayoutAttr source, LayoutAttr target, int64_t axis);
bool reductionUsesScratch(mlir::Operation* operation);
bool collectiveUsesScratch(mlir::Operation& operation);
bool nativeReduction(mlir::Operation* operation);
/// Whether some element of a scan along `axis` and the element `distance` before it sit in different SIMD groups.
bool scanCrossGroup(LayoutAttr layout, int64_t axis, int64_t distance);
bool scanNeedsScratch(LayoutAttr layout, int64_t axis);
/// The element of `source` that element `index` of `target` repeats, where source axis `i` is target axis `axes[i]`.
int64_t broadcastIndex(LayoutAttr source, LayoutAttr target, llvm::ArrayRef<int64_t> axes, int64_t index);
bool broadcastCrossGroup(LayoutAttr source, LayoutAttr target, llvm::ArrayRef<int64_t> axes);
bool conversionCrossGroup(LayoutAttr source, LayoutAttr target);
/// Conservatively true for replicated sources unless the layouts are equal.
bool conversionMovesBetweenThreads(LayoutAttr source, LayoutAttr target);

/// The items, reductions and scans outside loops that are the last users of their scratch allocation; lowering omits
/// their trailing barrier. Broadcasts always keep theirs.
llvm::DenseSet<mlir::Operation*> finalScratchUsers(mlir::func::FuncOp function);
/// For each operation, the scratch allocations accessed since the last barrier every thread executed, as lowered:
/// a collective that is its allocation's final user omits its trailing barrier, so its reads stay in flight.
llvm::DenseMap<mlir::Operation*, llvm::DenseSet<mlir::Value>> pendingScratch(
    mlir::func::FuncOp function, const llvm::DenseSet<mlir::Operation*>& finalUsers);
/// `thread` is SIMD group * 32 + lane; `cell` is relative to the scratch allocation.
struct ScratchTouch {
    int thread;
    int64_t cell;
    bool definite;
};
/// Every thread's accesses for one scratch load or store; none when one depends on more than the lane, SIMD group and
/// constants.
using ScratchTouches = llvm::function_ref<std::optional<std::vector<ScratchTouch>>(mlir::Operation*)>;
/// Check that scratch is written before it is read and that its cells change thread only across a barrier, taking
/// each load's and store's cells from `touches`; a store must name each cell once.
mlir::LogicalResult verifyScratch(mlir::func::FuncOp function, ScratchTouches touches);
mlir::LogicalResult verifyContracts(mlir::func::FuncOp function, unsigned index);

}
