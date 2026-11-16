#include "thread/dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ThreadDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "ThreadTypes.cpp.inc"
#define GET_OP_CLASSES
#include "ThreadOps.cpp.inc"

namespace carbon::msl {

namespace {

llvm::APInt word(uint64_t value) { return llvm::APInt(32, value); }

mlir::ConstantIntRanges words(uint64_t low, uint64_t high) {
    return mlir::ConstantIntRanges::fromUnsigned(word(low), word(high));
}

}

void BuiltinOp::inferResultRanges(llvm::ArrayRef<mlir::ConstantIntRanges>, mlir::SetIntRangeFn set) {
    set(getResult(), words(0, uint64_t(getBound()) - 1));
}

void ParamOp::inferResultRanges(llvm::ArrayRef<mlir::ConstantIntRanges>, mlir::SetIntRangeFn set) {
    if (getResult().getType().isInteger(32))
        set(getResult(), words(uint64_t(getMinimum()), uint64_t(getMaximum())));
}

// Bounds from the operands' bounds; a zero divisor or a shift of 32 or more gives 0, so every lower bound is 0.
void GuardedIntegerOp::inferResultRanges(llvm::ArrayRef<mlir::ConstantIntRanges> inputs, mlir::SetIntRangeFn set) {
    uint64_t left = inputs[0].umax().getZExtValue(), low = inputs[1].umin().getZExtValue();
    uint64_t right = inputs[1].umax().getZExtValue();
    auto opcode = getOpcode();
    if (opcode == "floordiv")
        return set(getResult(), words(0, low ? left / low : left));
    if (opcode == "mod")
        return set(getResult(), words(0, right ? std::min(left, right - 1) : 0));
    if (opcode == "shr")
        return set(getResult(), words(0, low < 32 ? left >> low : 0));
    set(getResult(), right < 32 && (left << right) <= UINT32_MAX ? words(0, left << right) : words(0, UINT32_MAX));
}

void GuardedLoadOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext*) {
    patterns.add(+[](GuardedLoadOp load, mlir::PatternRewriter& rewriter) -> mlir::LogicalResult {
        llvm::APInt bits;
        if (!mlir::matchPattern(load.getCondition(), mlir::m_ConstantInt(&bits)))
            return mlir::failure();
        if (bits.isZero())
            rewriter.replaceOp(load, load.getFallbacks());
        else
            rewriter.replaceOpWithNewOp<LoadOp>(load, load.getResultTypes(), load.getPointer(), load.getIndex());
        return mlir::success();
    });
}

void GuardedStoreOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext*) {
    patterns.add(+[](GuardedStoreOp store, mlir::PatternRewriter& rewriter) -> mlir::LogicalResult {
        llvm::APInt bits;
        if (!mlir::matchPattern(store.getCondition(), mlir::m_ConstantInt(&bits)))
            return mlir::failure();
        if (bits.isZero())
            rewriter.eraseOp(store);
        else
            rewriter.replaceOpWithNewOp<StoreOp>(store, store.getPointer(), store.getIndex(), store.getValues());
        return mlir::success();
    });
}

void MslDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "ThreadTypes.cpp.inc"
        >();
    addOperations<
#define GET_OP_LIST
#include "ThreadOps.cpp.inc"
        >();
}

}
