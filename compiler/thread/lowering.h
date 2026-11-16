#pragma once

#include "tile/memory.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include <string>

namespace carbon {

struct KernelSource {
    std::string source;
    std::string threadIR;
    MemoryPlan memory;
};

/// Check a tile kernel's bindings and memory plan for Metal without lowering it.
void validateMetal(mlir::func::FuncOp function);

KernelSource lowerMetal(mlir::func::FuncOp function);

}

namespace carbon::msl {

struct ThreadKernel {
    mlir::OwningOpRef<mlir::ModuleOp> module;
    MemoryPlan memory;
};

ThreadKernel lowerToThreads(mlir::func::FuncOp function);

/// Check a verified, memory-planned tile kernel's scratch ownership on the cells each thread computes when lowered.
void checkScratch(mlir::func::FuncOp function);

/// Function attribute mapping each lane's two matrix slots to elements of an 8x8 block.
std::string registersName(mlir::Type element);

/// Floating-point operations are never folded; bfloat selects move bits.
void optimizeThreads(mlir::ModuleOp module, MemoryPlan& memory);

/// Reject a SIMD-group collective under lane-divergent control flow, or a barrier under control flow that is not
/// uniform across the threadgroup. Loads at a uniform address are uniform only from read-only bindings or scratch.
void verifyConvergence(mlir::ModuleOp module, const MemoryPlan& memory);

std::string printMetal(mlir::ModuleOp module, mlir::func::FuncOp tile, const MemoryPlan& memory);

}
