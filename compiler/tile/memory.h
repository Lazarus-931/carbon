#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <cstdint>
#include <string>
#include <vector>

namespace carbon {

/// First and last use are positions in a preorder walk of the kernel; -1 means unused.
struct ScratchRegion {
    uint64_t offset = 0, bytes = 0;
    int64_t first = -1, last = -1;
    std::string storage, location;
};

struct MemoryPlan {
    std::vector<std::string> access;
    /// The bytes each binding must have: its `carbon.min_bytes`, raised to the extent of each static view of it.
    std::vector<uint64_t> bindingBytes;
    /// Bytes each binding's offset must be a multiple of; vector accesses raise it to their width.
    std::vector<uint64_t> bindingAlignment;
    uint64_t maxGrid = UINT32_MAX;
    /// Under carbon.check_writes, the bindings whose stores record their writer, in shadow-buffer order.
    std::vector<size_t> checkedBindings;
    uint64_t scratchBytes = 0;
    /// The byte offset of each scratch allocation; allocations whose lifetimes are fenced apart share bytes.
    llvm::DenseMap<mlir::Value, uint64_t> scratchOffsets;
    std::vector<ScratchRegion> scratchLayout;
    /// The collectives that omit their trailing barrier; see `ir::finalScratchUsers`.
    llvm::DenseSet<mlir::Operation*> finalScratchUsers;
};

MemoryPlan planMemory(mlir::func::FuncOp function);

}
