#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include <string>

namespace carbon::ir {

/// The process's MLIR context: layouts made from Python live here, so every kernel built from them can use them.
mlir::MLIRContext& context();

/// Every diagnostic the context reports while alive, one per line, as `location: message` when `located`.
struct Diagnostics {
    explicit Diagnostics(mlir::MLIRContext& context, bool located = true);
    std::string text;
    mlir::ScopedDiagnosticHandler handler;
};

mlir::OwningOpRef<mlir::ModuleOp> parse(const std::string& input, mlir::MLIRContext& context);
void check(mlir::ModuleOp module);
void optimize(mlir::ModuleOp module);
/// The module as MLIR text, with locations.
std::string print(mlir::ModuleOp module);
mlir::func::FuncOp kernel(mlir::ModuleOp module);

}
