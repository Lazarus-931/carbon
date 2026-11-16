#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "ThreadDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ThreadTypes.h.inc"
#define GET_OP_CLASSES
#include "ThreadOps.h.inc"

#include <cstddef>

namespace carbon::msl {
/// Operands that describe one tensor of a tensor_matmul: pointer, element offset, columns, rows, leading stride.
inline constexpr size_t tensorOperands = 5;
}
