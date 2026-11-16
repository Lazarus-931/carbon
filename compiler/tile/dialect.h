#pragma once

#include <utility>
#include <vector>
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "CarbonDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "CarbonAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "CarbonTypes.h.inc"
#define GET_OP_CLASSES
#include "CarbonOps.h.inc"

namespace carbon::ir {
/// Whether `type` is an 8- or 16-bit storage integer, ui8, si8, ui16 or si16: tiles load, store, select and cast
/// them, and compute only after a cast.
inline bool storageInteger(mlir::Type type) {
    auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
    return integer && !integer.isSignless() && (integer.getWidth() == 8 || integer.getWidth() == 16);
}

/// Check kernel-wide invariants that detached operations cannot establish.
mlir::LogicalResult verifyLayoutKernels(mlir::ModuleOp module);
}
