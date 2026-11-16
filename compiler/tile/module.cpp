#include "tile/module.h"
#include "tile/dialect.h"
#include "tile/analysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <stdexcept>

using namespace mlir;

namespace carbon::ir {

MLIRContext& context() {
    static auto* shared = [] {
        auto* context = new MLIRContext(MLIRContext::Threading::DISABLED);
        context->loadDialect<ir::CarbonDialect, arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
        return context;
    }();
    return *shared;
}

Diagnostics::Diagnostics(MLIRContext& context, bool located)
    : handler(&context, [this, located](Diagnostic& diagnostic) {
          llvm::raw_string_ostream stream(text);
          if (located) {
              diagnostic.getLocation().print(stream);
              stream << ": ";
          }
          diagnostic.print(stream);
          stream << '\n';
          return success();
      }) {}

namespace {

bool elementwise(Operation* operation) {
    return isa<MapOp, CastOp, MathOp, BitwiseOp, BitcastOp, IntegerDivOp, MaximumOp, MinimumOp, FmaOp,
               SelectOp, BroadcastOp, IotaOp>(operation);
}

bool propagateLayout(ConvertLayoutOp conversion) {
    auto target = conversion.getResult().getType().getLayout();
    auto removable = [&](ConvertLayoutOp operation) {
        auto scratch = operation.getScratch();
        return !scratch || (!target.replicated() && scratch.getDefiningOp<ScratchOp>() &&
                            llvm::all_of(scratch.getUsers(), [](Operation* user) { return isa<ConvertLayoutOp>(user); }));
    };
    if (!removable(conversion))
        return false;
    IRMapping mapping;
    llvm::SetVector<Operation*> chain;
    bool cancels = false;
    std::function<bool(Value)> visit = [&](Value value) {
        auto tile = dyn_cast<TileType>(value.getType());
        if (!tile || tile.getLayout() == target || mapping.contains(value))
            return true;
        auto* producer = value.getDefiningOp();
        if (!producer || producer->getBlock() != conversion->getBlock())
            return false;
        if (auto source = dyn_cast<ConvertLayoutOp>(producer)) {
            if (source.getValue().getType().getLayout() != target || !removable(source))
                return false;
            mapping.map(value, source.getValue());
            chain.insert(producer);
            cancels = true;
            return true;
        }
        if (!elementwise(producer) || producer->getNumResults() != 1)
            return false;
        // Scratch selects the canonical replica, which can belong to a different SIMD group.
        if (auto broadcast = dyn_cast<BroadcastOp>(producer);
            broadcast && !groupUniform(broadcast.getValue(),
                                       !conversion.getScratch() && !conversionCrossGroup(tile.getLayout(), target)))
            return false;
        if (chain.contains(producer))
            return true;
        for (auto operand : producer->getOperands())
            if (!visit(operand))
                return false;
        chain.insert(producer);
        return true;
    };
    if (!visit(conversion.getValue()))
        return false;
    if (!cancels && conversion.getValue().getType().getLayout() != target)
        return false;
    for (auto* operation : chain)
        if (!isa<ConvertLayoutOp, BroadcastOp, IotaOp>(operation))
            for (auto* user : operation->getUsers())
                if (user != conversion && !chain.contains(user))
                    return false;
    OpBuilder builder(conversion);
    for (auto* operation : chain) {
        if (isa<ConvertLayoutOp>(operation))
            continue;
        auto* clone = builder.clone(*operation, mapping);
        auto type = cast<TileType>(clone->getResult(0).getType());
        clone->getResult(0).setType(TileType::get(builder.getContext(), type.getElementType(), target));
    }
    llvm::SetVector<Operation*> workspaces;
    auto erase = [&](Operation* operation) {
        if (auto conversion = dyn_cast<ConvertLayoutOp>(operation); conversion && conversion.getScratch()) {
            workspaces.insert(conversion.getScratch().getDefiningOp());
            builder.setInsertionPoint(operation);
            BarrierOp::create(builder, operation->getLoc());
        }
        operation->erase();
    };
    conversion.replaceAllUsesWith(mapping.lookupOrDefault(conversion.getValue()));
    erase(conversion);
    for (auto* operation : llvm::reverse(chain))
        if (operation->use_empty())
            erase(operation);
    for (auto* workspace : workspaces)
        if (workspace->use_empty())
            workspace->erase();
    return true;
}

void propagateLayouts(ModuleOp module) {
    while (module.walk([&](ConvertLayoutOp conversion) {
               return propagateLayout(conversion) ? WalkResult::interrupt() : WalkResult::advance();
           }).wasInterrupted()) {
    }
}

void verified(ModuleOp module, Diagnostics& diagnostics) {
    if (failed(mlir::verify(module)) || failed(ir::verifyLayoutKernels(module)))
        throw std::invalid_argument(diagnostics.text);
    auto functions = module.getOps<func::FuncOp>();
    if (!llvm::hasSingleElement(functions) || !llvm::hasSingleElement(module.getBody()->getOperations()))
        throw std::invalid_argument("expected exactly one kernel function");
}

}

OwningOpRef<ModuleOp> parse(const std::string& input, MLIRContext& context) {
    Diagnostics diagnostics(context);
    auto module = parseSourceString<ModuleOp>(input, &context);
    if (!module)
        throw std::invalid_argument(diagnostics.text);
    verified(*module, diagnostics);
    return module;
}

void check(ModuleOp module) {
    Diagnostics diagnostics(*module.getContext());
    verified(module, diagnostics);
}

void optimize(ModuleOp module) {
    propagateLayouts(module);
    PassManager passes(module.getContext());
    passes.addPass(createCSEPass());
    if (failed(passes.run(module)) || failed(mlir::verify(module)) || failed(ir::verifyLayoutKernels(module)))
        throw std::invalid_argument("IR verification failed after optimization");
}

std::string print(ModuleOp module) {
    std::string result;
    llvm::raw_string_ostream stream(result);
    module.print(stream, OpPrintingFlags().enableDebugInfo());
    return result;
}

func::FuncOp kernel(ModuleOp module) { return *module.getOps<func::FuncOp>().begin(); }

}
