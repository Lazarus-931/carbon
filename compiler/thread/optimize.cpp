
#include "thread/analysis.h"
#include "thread/dialect.h"
#include "thread/lowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include <algorithm>
#include <map>
#include <stdexcept>
#include <tuple>

using namespace mlir;

namespace carbon::msl {
namespace {

bool integral(Operation* operation) {
    auto integer = [](Type type) { return type.isIntOrIndex(); };
    return isa<arith::ArithDialect>(operation->getDialect()) && llvm::all_of(operation->getOperandTypes(), integer) &&
           llvm::all_of(operation->getResultTypes(), integer);
}

// The leaves of an AndI tree, each once, in order; returns how many it met, repeats included.
size_t conjuncts(Value condition, SmallVector<Value>& leaves) {
    if (auto conjunction = condition.getDefiningOp<arith::AndIOp>())
        return conjuncts(conjunction.getLhs(), leaves) + conjuncts(conjunction.getRhs(), leaves);
    if (!llvm::is_contained(leaves, condition))
        leaves.push_back(condition);
    return 1;
}

Value root(Value pointer) {
    while (auto cast = pointer.getDefiningOp<ReinterpretOp>())
        pointer = cast.getPointer();
    return pointer;
}

// Merge unsigned x < a && x < b into x < min(a, b), sharing masks with bounds checks.
struct MergeBounds : OpRewritePattern<arith::AndIOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::AndIOp operation, PatternRewriter& rewriter) const override {
        if (!operation.getType().isInteger(1) ||
            llvm::any_of(operation->getUsers(), [](Operation* user) { return isa<arith::AndIOp>(user); }))
            return failure();
        SmallVector<Value> leaves;
        auto occurrences = conjuncts(operation.getResult(), leaves);
        llvm::MapVector<Value, SmallVector<arith::CmpIOp>> limits;
        SmallVector<Value> others;
        for (auto leaf : leaves) {
            auto compare = leaf.getDefiningOp<arith::CmpIOp>();
            if (compare && compare.getPredicate() == arith::CmpIPredicate::ult)
                limits[compare.getLhs()].push_back(compare);
            else
                others.push_back(leaf);
        }
        if (occurrences == leaves.size() && others.size() + limits.size() == leaves.size())
            return failure();
        auto location = operation.getLoc();
        for (auto& [index, compares] : limits) {
            Value limit = compares[0].getRhs();
            for (auto compare : ArrayRef<arith::CmpIOp>(compares).drop_front())
                limit = arith::MinUIOp::create(rewriter, location, limit, compare.getRhs());
            others.push_back(compares.size() == 1 ? compares[0].getResult()
                                                  : arith::CmpIOp::create(rewriter, location, arith::CmpIPredicate::ult,
                                                                          index, limit).getResult());
        }
        Value every = others[0];
        for (auto predicate : ArrayRef<Value>(others).drop_front())
            every = arith::AndIOp::create(rewriter, location, every, predicate);
        rewriter.replaceOp(operation, every);
        return success();
    }
};

void eraseDeadCode(ModuleOp module) {
    bool erased = true;
    while (erased) {
        erased = false;
        SmallVector<Operation*> dead;
        module.walk([&](Operation* operation) {
            if (isOpTriviallyDead(operation))
                dead.push_back(operation);
        });
        for (auto* operation : dead) {
            operation->erase();
            erased = true;
        }
    }
}

enum class Uniformity { Threadgroup, Simd, Varying };

Uniformity join(Uniformity a, Uniformity b) { return std::max(a, b); }

class Convergence {
  public:
    explicit Convergence(const MemoryPlan& memory, Facts* facts = nullptr) : memory_(memory), facts_(facts) {}

    void run(func::FuncOp function, bool report = true) {
        for (int pass = 0; pass < 2; ++pass)
            visit(function.getBody(), Uniformity::Threadgroup, report && pass == 1);
    }

    Uniformity level(Value value) const {
        auto found = levels_.find(value);
        return found == levels_.end() ? Uniformity::Threadgroup : found->second;
    }

  private:
    void set(Value value, Uniformity uniformity) {
        auto [found, inserted] = levels_.try_emplace(value, uniformity);
        auto joined = inserted ? uniformity : join(found->second, uniformity);
        if (inserted || joined != found->second) {
            found->second = joined;
            ++updates_;
        }
    }

    Uniformity formed(Value value, Uniformity produced) {
        if (!facts_ || produced == Uniformity::Threadgroup || !value.getType().isIntOrIndex())
            return produced;
        auto id = facts_->form(value);
        if (!id)
            return produced;
        SmallVector<Value> leaves;
        facts_->leaves(*id, leaves);
        auto uniformity = Uniformity::Threadgroup;
        for (auto leaf : leaves) {
            if (leaf == value)
                return produced;
            uniformity = join(uniformity, level(leaf));
        }
        return std::min(produced, uniformity);
    }

    Uniformity operands(Operation* operation) {
        auto result = Uniformity::Threadgroup;
        for (auto operand : operation->getOperands())
            result = join(result, level(operand));
        return result;
    }

    // Only read-only device bindings and barrier-validated scratch yield uniform loads.
    bool readOnly(Value pointer) {
        pointer = root(pointer);
        if (pointer.getDefiningOp<ScratchOp>())
            return true;
        auto argument = pointer.getDefiningOp<ArgumentOp>();
        return argument && memory_.access[argument.getBinding()] == "read";
    }

    void require(Operation* operation, Uniformity control, Uniformity needed, bool report) {
        if (report && control > needed)
            throw std::invalid_argument(std::string("internal compiler error: ") +
                (needed == Uniformity::Threadgroup ? "barrier" : "SIMD-group collective") + " '" +
                operation->getName().getStringRef().str() + "' under control flow that not every participant takes");
    }

    void visit(Region& region, Uniformity control, bool report) {
        for (auto& operation : region.front())
            visit(&operation, control, report);
    }

    void visit(Operation* operation, Uniformity control, bool report) {
        if (auto loop = dyn_cast<scf::ForOp>(operation)) {
            auto bounds = join(control, join(level(loop.getLowerBound()),
                                             join(level(loop.getUpperBound()), level(loop.getStep()))));
            set(loop.getInductionVar(), bounds);
            auto round = [&](bool last) {
                for (auto [argument, initial] : llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs()))
                    set(argument, join(bounds, level(initial)));
                visit(loop.getRegion(), bounds, report && last);
                for (auto [argument, yielded] :
                     llvm::zip(loop.getRegionIterArgs(), loop.getBody()->getTerminator()->getOperands()))
                    set(argument, level(yielded));
            };
            for (auto before = updates_ + 1; before != updates_;) {
                before = updates_;
                round(false);
            }
            round(true);
            for (auto [output, argument] : llvm::zip(loop.getResults(), loop.getRegionIterArgs()))
                set(output, level(argument));
            return;
        }
        if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
            auto branch = join(control, level(conditional.getCondition()));
            for (auto& region : conditional->getRegions()) {
                if (region.empty())
                    continue;
                visit(region, branch, report);
                for (auto [output, yielded] :
                     llvm::zip(conditional.getResults(), region.front().getTerminator()->getOperands()))
                    set(output, join(branch, level(yielded)));
            }
            return;
        }
        if (isa<BarrierOp>(operation))
            return require(operation, control, Uniformity::Threadgroup, report);
        if (isa<MmaOp, MatrixLoadOp>(operation))
            require(operation, control, Uniformity::Simd, report);
        if (isa<TensorMatmulOp>(operation))
            require(operation, control, Uniformity::Threadgroup, report);
        auto inputs = operands(operation);
        Uniformity produced = inputs;
        if (auto builtin = dyn_cast<BuiltinOp>(operation))
            produced = builtin.getName() == "lane" ? Uniformity::Varying :
                       builtin.getName() == "simd_group" ? Uniformity::Simd : Uniformity::Threadgroup;
        else if (auto shuffle = dyn_cast<ShuffleOp>(operation)) {
            require(operation, control, Uniformity::Simd, report);
            produced = level(shuffle.getValue());
        } else if (auto simd = dyn_cast<SimdOp>(operation)) {
            require(operation, control, Uniformity::Simd, report);
            bool prefix = simd.getKind().starts_with("prefix");
            produced = prefix ? Uniformity::Varying :
                       simd.getUniform() ? Uniformity::Threadgroup : std::min(level(simd.getValue()), Uniformity::Simd);
        } else if (auto load = dyn_cast<LoadOp>(operation))
            produced = readOnly(load.getPointer()) ? inputs : Uniformity::Varying;
        else if (auto load = dyn_cast<GuardedLoadOp>(operation))
            produced = readOnly(load.getPointer()) ? inputs : Uniformity::Varying;
        else if (isa<AtomicOp, MmaOp, TensorMatmulOp, MatrixLoadOp>(operation))
            produced = Uniformity::Varying;
        for (auto output : operation->getResults())
            set(output, join(control, formed(output, produced)));
    }

    const MemoryPlan& memory_;
    Facts* facts_;
    llvm::DenseMap<Value, Uniformity> levels_;
    uint64_t updates_ = 0;
};

void inlineThen(scf::IfOp conditional) {
    auto& block = conditional.getThenRegion().front();
    auto* yield = block.getTerminator();
    conditional->getBlock()->getOperations().splice(Block::iterator(conditional), block.getOperations(),
                                                    block.begin(), Block::iterator(yield));
    conditional.replaceAllUsesWith(yield->getOperands());
    conditional.erase();
}

void unswitchLoops(func::FuncOp function, const MemoryPlan& memory) {
    Convergence convergence(memory);
    convergence.run(function, false);
    SmallVector<scf::ForOp> loops;
    function.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
    llvm::DenseSet<Operation*> unswitched;
    for (auto loop : loops) {
        if (llvm::any_of(unswitched, [&](Operation* outer) { return outer->isProperAncestor(loop); }))
            continue;
        SmallVector<Operation*> guards, loads;
        llvm::SetVector<Value> conditions;
        auto needed = Uniformity::Varying;
        loop.walk([&](Operation* operation) {
            if (isa<BarrierOp, TensorMatmulOp>(operation))
                needed = Uniformity::Threadgroup;
            else if (isa<MmaOp, MatrixLoadOp, ShuffleOp, SimdOp>(operation))
                needed = std::min(needed, Uniformity::Simd);
            if (auto conditional = dyn_cast<scf::IfOp>(operation);
                conditional && loop.isDefinedOutsideOfLoop(conditional.getCondition())) {
                guards.push_back(conditional);
                conditions.insert(conditional.getCondition());
            }
            if (auto load = dyn_cast<GuardedLoadOp>(operation);
                load && loop.isDefinedOutsideOfLoop(load.getCondition()) &&
                convergence.level(load.getCondition()) == Uniformity::Varying)
                loads.push_back(load);
        });
        if (llvm::any_of(conditions, [&](Value condition) { return convergence.level(condition) > needed; }))
            continue;
        bool lanes = needed == Uniformity::Simd && !loads.empty();
        if (lanes)
            for (auto* load : loads) {
                guards.push_back(load);
                conditions.insert(cast<GuardedLoadOp>(load).getCondition());
            }
        if (guards.empty())
            continue;
        OpBuilder builder(loop);
        Value all;
        for (auto condition : conditions)
            all = all ? arith::AndIOp::create(builder, loop.getLoc(), all, condition).getResult() : condition;
        if (lanes)
            all = SimdOp::create(builder, loop.getLoc(), all.getType(), all, "all");
        auto branch = scf::IfOp::create(builder, loop.getLoc(), loop.getResultTypes(), all, true);
        for (auto* block : {&branch.getThenRegion().front(), &branch.getElseRegion().front()})
            if (block->mightHaveTerminator())
                block->getTerminator()->erase();
        builder.setInsertionPointToEnd(&branch.getThenRegion().front());
        IRMapping mapping;
        auto fast = builder.clone(*loop, mapping);
        for (auto* guard : guards) {
            auto* copy = mapping.lookup(guard);
            if (auto conditional = dyn_cast<scf::IfOp>(copy)) {
                inlineThen(conditional);
                continue;
            }
            auto load = cast<GuardedLoadOp>(copy);
            OpBuilder replacing(load);
            load.replaceAllUsesWith(
                LoadOp::create(replacing, load.getLoc(), load.getResultTypes(), load.getPointer(), load.getIndex()));
            load.erase();
        }
        scf::YieldOp::create(builder, loop.getLoc(), fast->getResults());
        loop->replaceAllUsesWith(branch.getResults());
        auto& otherwise = branch.getElseRegion().front();
        loop->moveBefore(&otherwise, otherwise.end());
        builder.setInsertionPointToEnd(&otherwise);
        scf::YieldOp::create(builder, loop.getLoc(), loop->getResults());
        unswitched.insert(branch);
    }
}

void simplify(ModuleOp module, bool merge = false) {
    auto* context = module.getContext();
    IRRewriter rewriter(context);
    {
        DominanceInfo dominance(module);
        eliminateCommonSubExpressions(rewriter, dominance, module);
    }
    RewritePatternSet patterns(context);
    for (auto name : context->getRegisteredOperations())
        if (name.getDialectNamespace() == "arith" || name.getStringRef() == "scf.if")
            name.getCanonicalizationPatterns(patterns, context);
    GuardedLoadOp::getCanonicalizationPatterns(patterns, context);
    GuardedStoreOp::getCanonicalizationPatterns(patterns, context);
    if (merge)
        patterns.add<MergeBounds>(context);
    FrozenRewritePatternSet frozen(std::move(patterns));
    SmallVector<Operation*> candidates;
    module.walk([&](Operation* operation) {
        if (integral(operation) || isa<scf::IfOp, GuardedLoadOp, GuardedStoreOp>(operation))
            candidates.push_back(operation);
    });
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
    (void)applyOpPatternsGreedily(candidates, frozen, config);
    module.walk([](scf::ForOp loop) { moveLoopInvariantCode(loop); });
    {
        DominanceInfo dominance(module);
        eliminateCommonSubExpressions(rewriter, dominance, module);
    }
    eraseDeadCode(module);
}

// Patterns that rewrite the IR must forget the facts of what they erase.
struct Forget : RewriterBase::Listener {
    explicit Forget(DataFlowSolver& solver) : solver(solver) {}
    void notifyOperationErased(Operation* operation) override {
        solver.eraseState(solver.getProgramPointAfter(operation));
        for (auto result : operation->getResults())
            solver.eraseState(result);
    }
    DataFlowSolver& solver;
};

void dropImplied(func::FuncOp function, Facts& facts) {
    auto below = [](Value predicate) {
        auto compare = predicate.getDefiningOp<arith::CmpIOp>();
        return compare && compare.getPredicate() == arith::CmpIPredicate::ult ? compare : arith::CmpIOp();
    };
    SmallVector<OpOperand*> conditions;
    function.walk([&](Operation* operation) {
        if (isa<GuardedLoadOp, GuardedStoreOp, AtomicOp, AtomicStoreOp>(operation))
            for (auto& operand : operation->getOpOperands())
                if (operand.get().getType().isInteger(1) && operand.get().getDefiningOp<arith::AndIOp>())
                    conditions.push_back(&operand);
    });
    for (auto* operand : conditions) {
        SmallVector<Value> leaves;
        conjuncts(operand->get(), leaves);
        bool changed = false;
        for (size_t at = 0; at < leaves.size();) {
            auto check = below(leaves[at]);
            auto least = check ? facts.range(check.getRhs()).first : 0;
            SmallVector<std::pair<Value, uint64_t>> limits;
            for (auto [other, leaf] : llvm::enumerate(leaves))
                if (auto assumption = below(leaf); assumption && other != at)
                    if (auto top = facts.range(assumption.getRhs()).second)
                        limits.emplace_back(assumption.getLhs(), top - 1);
            auto top = check && least ? facts.bound(check.getLhs(), limits) : std::nullopt;
            if (top && *top < least) {
                leaves.erase(leaves.begin() + at);
                changed = true;
            } else
                ++at;
        }
        if (!changed)
            continue;
        OpBuilder builder(operand->getOwner());
        auto location = operand->getOwner()->getLoc();
        Value every;
        for (auto leaf : leaves)
            every = every ? arith::AndIOp::create(builder, location, every, leaf).getResult() : leaf;
        operand->set(every ? every : arith::ConstantOp::create(builder, location, builder.getBoolAttr(true)));
    }
}

void foldGuards(func::FuncOp function) {
    Facts facts(function);
    SmallVector<GuardedIntegerOp> integers;
    function.walk([&](GuardedIntegerOp operation) {
        auto [low, high] = facts.range(operation.getRhs());
        auto opcode = operation.getOpcode();
        if (((opcode == "floordiv" || opcode == "mod") && low > 0) ||
            ((opcode == "shl" || opcode == "shr") && high < 32))
            integers.push_back(operation);
    });
    for (auto operation : integers) {
        OpBuilder builder(operation);
        auto a = operation.getLhs(), b = operation.getRhs();
        auto opcode = operation.getOpcode();
        Value plain = opcode == "floordiv" ? arith::DivUIOp::create(builder, operation.getLoc(), a, b).getResult()
                    : opcode == "mod" ? arith::RemUIOp::create(builder, operation.getLoc(), a, b).getResult()
                    : opcode == "shl" ? arith::ShLIOp::create(builder, operation.getLoc(), a, b).getResult()
                                      : arith::ShRUIOp::create(builder, operation.getLoc(), a, b).getResult();
        operation.replaceAllUsesWith(plain);
        facts.solver().eraseState(operation.getResult());
        operation.erase();
    }
    RewritePatternSet patterns(function.getContext());
    arith::populateIntRangeOptimizationsPatterns(patterns, facts.solver());
    FrozenRewritePatternSet frozen(std::move(patterns));
    SmallVector<Operation*> candidates;
    function.walk([&](Operation* operation) {
        if (integral(operation) || isa<BuiltinOp, ParamOp>(operation))
            candidates.push_back(operation);
    });
    Forget listener(facts.solver());
    GreedyRewriteConfig config;
    config.setListener(&listener);
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    (void)applyOpPatternsGreedily(candidates, frozen, config);
    dropImplied(function, facts);
}

void numberValues(func::FuncOp function) {
    Facts facts(function);
    DominanceInfo dominance(function);
    std::map<int, SmallVector<Value>> seen;
    function.walk<WalkOrder::PreOrder>([&](Operation* operation) {
        if (operation->getNumResults() != 1 || !integral(operation) || isa<arith::ConstantOp>(operation))
            return;
        auto value = operation->getResult(0);
        auto id = facts.form(value);
        if (!id)
            return;
        auto& values = seen[*id];
        for (auto earlier : values)
            if (earlier.getType() == value.getType() && dominance.properlyDominates(earlier, operation)) {
                value.replaceAllUsesWith(earlier);
                return;
            }
        values.push_back(value);
    });
}

void hoistReadOnlyLoads(scf::ForOp loop, const MemoryPlan& memory) {
    SmallVector<Operation*> loads;
    SmallVector<Type> types;
    for (auto& operation : *loop.getBody()) {
        if (!isa<LoadOp, GuardedLoadOp>(operation) ||
            !llvm::all_of(operation.getOperands(), [&](Value value) { return loop.isDefinedOutsideOfLoop(value); }))
            continue;
        auto argument = root(operation.getOperand(0)).getDefiningOp<ArgumentOp>();
        if (!argument || memory.access[argument.getBinding()] != "read")
            continue;
        loads.push_back(&operation);
        llvm::append_range(types, operation.getResultTypes());
    }
    if (loads.empty())
        return;
    APInt lower, upper;
    if (matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) &&
        matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) && lower.ult(upper)) {
        for (auto* load : loads)
            load->moveBefore(loop);
    } else {
        OpBuilder builder(loop);
        auto active = arith::CmpIOp::create(builder, loop.getLoc(), arith::CmpIPredicate::ult,
                                           loop.getLowerBound(), loop.getUpperBound());
        auto branch = scf::IfOp::create(builder, loop.getLoc(), types, active, true);
        builder.setInsertionPointToStart(&branch.getThenRegion().front());
        SmallVector<Value> values;
        for (auto* load : loads)
            llvm::append_range(values, builder.clone(*load)->getResults());
        scf::YieldOp::create(builder, loop.getLoc(), values);
        builder.setInsertionPointToStart(&branch.getElseRegion().front());
        values.clear();
        for (auto type : types)
            values.push_back(arith::ConstantOp::create(builder, loop.getLoc(), builder.getZeroAttr(type)));
        scf::YieldOp::create(builder, loop.getLoc(), values);
        size_t offset = 0;
        for (auto* load : loads) {
            load->replaceAllUsesWith(branch.getResults().slice(offset, load->getNumResults()));
            offset += load->getNumResults();
            load->erase();
        }
    }
    moveLoopInvariantCode(loop);
}

bool touches(Operation* operation, Value base, bool write, bool read) {
    bool found = false;
    operation->walk([&](Operation* nested) {
        if (isa<BarrierOp>(nested)) {
            found = true;
            return;
        }
        bool writes = isa<StoreOp, GuardedStoreOp, AtomicOp, AtomicStoreOp, TensorMatmulOp>(nested);
        bool reads = isa<LoadOp, GuardedLoadOp, AtomicOp, TensorMatmulOp, MatrixLoadOp>(nested);
        if (!((write && writes) || (read && reads)))
            return;
        for (auto operand : nested->getOperands())
            if (isa<PointerType>(operand.getType()) && root(operand) == base)
                found = true;
    });
    return found;
}

bool simdUniform(Operation* operation, const Convergence& convergence) {
    for (auto* parent = operation->getParentOp(); !isa<func::FuncOp>(parent); parent = parent->getParentOp()) {
        SmallVector<Value> conditions;
        if (auto conditional = dyn_cast<scf::IfOp>(parent))
            conditions.push_back(conditional.getCondition());
        if (auto loop = dyn_cast<scf::ForOp>(parent))
            conditions.append({loop.getLowerBound(), loop.getUpperBound(), loop.getStep()});
        if (llvm::any_of(conditions, [&](Value value) { return convergence.level(value) == Uniformity::Varying; }))
            return false;
    }
    return true;
}

struct MatrixBlock {
    int64_t stride;
    bool transposed;
    std::array<int, 2> slots;
    uint32_t corner;
};

// How two loads per lane read one 8x8 block whose register map is `map`: the element at row r and column c of the
// block sits at corner + r * stride + c (c * stride + r when transposed), and each load supplies one slot of the map.
std::optional<MatrixBlock> matchBlock(ArrayRef<std::array<uint64_t, 32>> offsets, ArrayRef<int64_t> map) {
    for (bool transposed : {false, true})
        for (int first = 0; first < 2; ++first) {
            std::array<int, 2> slots{first, 1 - first};
            auto element = [&](int member, int lane) { return map[lane * 2 + slots[member]]; };
            auto major = [&](int64_t at) { return transposed ? at % 8 : at / 8; };
            auto minor = [&](int64_t at) { return transposed ? at / 8 : at % 8; };
            int lane = 1;
            while (lane < 32 && major(element(0, lane)) == major(element(0, 0)))
                ++lane;
            if (lane == 32)
                continue;
            auto rise = int64_t(int32_t(uint32_t(offsets[0][lane] - offsets[0][0]))) -
                        (minor(element(0, lane)) - minor(element(0, 0)));
            auto steps = major(element(0, lane)) - major(element(0, 0));
            if (rise % steps || rise / steps < 8 || rise / steps > INT32_MAX)
                continue;
            auto stride = rise / steps;
            auto corner = uint32_t(offsets[0][0] - (stride * major(element(0, 0)) + minor(element(0, 0))));
            bool block = true;
            for (int member = 0; member < 2; ++member)
                for (int at = 0; at < 32; ++at)
                    block &= uint32_t(offsets[member][at]) ==
                             uint32_t(corner + stride * major(element(member, at)) + minor(element(member, at)));
            if (block)
                return MatrixBlock{stride, transposed, slots, corner};
        }
    return std::nullopt;
}

void matrixLoads(func::FuncOp function, const MemoryPlan& memory) {
    Value lane;
    function.walk([&](BuiltinOp builtin) {
        if (builtin.getName() == "lane")
            lane = builtin.getResult();
    });
    if (!lane)
        return;
    Facts facts(function);
    Convergence convergence(memory);
    convergence.run(function, false);
    struct Member {
        LoadOp load;
        int shared;
        std::array<uint64_t, 32> offsets;
    };
    std::map<std::tuple<Block*, void*, int>, std::vector<Member>> groups;
    function.walk([&](LoadOp load) {
        auto pointer = cast<PointerType>(load.getPointer().getType());
        if (load.getNumResults() != 1 || pointer.getSpace() != "device" || !load.getIndex().getType().isInteger(32) ||
            !function->hasAttr(registersName(pointer.getElementType())) || !load->hasOneUse() ||
            !isa<MmaOp>(*load->getUsers().begin()) || !simdUniform(load, convergence))
            return;
        auto split = facts.lanes(load.getIndex(), lane);
        SmallVector<Value> shared;
        if (split)
            facts.leaves(split->first, shared);
        if (!split ||
            llvm::any_of(shared, [&](Value value) { return convergence.level(value) == Uniformity::Varying; }))
            return;
        groups[{load->getBlock(), load.getPointer().getAsOpaquePointer(), split->first}].push_back(
            {load, split->first, split->second});
    });
    auto fuse = [&](Member& a, Member& b, const MatrixBlock& block, Type element, Value& shared) {
        for (auto* between = a.load->getNextNode(); between != b.load; between = between->getNextNode())
            if (touches(between, root(a.load.getPointer()), true, false))
                return false;
        // The corner is rebuilt below by subtracting one offset per set bit of the lane, so each lane's offset must
        // be the sum of its bits' offsets.
        const auto& offsets = a.offsets;
        auto bit = [&](int index) { return uint32_t(offsets[1 << index] - offsets[0]); };
        for (int at = 0; at < 32; ++at) {
            uint32_t sum = 0;
            for (int index = 0; index < 5; ++index)
                sum += ((at >> index) & 1) * bit(index);
            if (uint32_t(offsets[at] - offsets[0]) != sum)
                return false;
        }
        OpBuilder builder(a.load);
        auto location = a.load.getLoc();
        auto u32 = [&](uint32_t value) {
            return arith::ConstantOp::create(builder, location, builder.getI32IntegerAttr(int32_t(value))).getResult();
        };
        if (!shared)
            shared = facts.materialize(a.shared, builder, location);
        Value corner;
        if (shared) {
            corner = block.corner ? arith::AddIOp::create(builder, location, shared, u32(block.corner)).getResult()
                                  : shared;
        } else {
            corner = arith::SubIOp::create(builder, location, a.load.getIndex(),
                                           u32(uint32_t(offsets[0]) - block.corner));
            for (int index = 0; index < 5; ++index)
                if (bit(index)) {
                    auto shifted = arith::ShRUIOp::create(builder, location, lane, u32(index));
                    auto set = arith::AndIOp::create(builder, location, shifted, u32(1));
                    corner = arith::SubIOp::create(builder, location, corner,
                                                   arith::MulIOp::create(builder, location, set, u32(bit(index))));
                }
        }
        auto matrix = MatrixLoadOp::create(builder, location, TypeRange{element, element}, a.load.getPointer(), corner,
                                           block.stride, block.transposed);
        for (auto [member, slot] : {std::pair(&a, block.slots[0]), std::pair(&b, block.slots[1])}) {
            member->load.getResult(0).replaceAllUsesWith(matrix.getResult(slot));
            member->load.erase();
        }
        return true;
    };
    for (auto& [key, members] : groups) {
        auto element = cast<PointerType>(members[0].load.getPointer().getType()).getElementType();
        auto map = function->getAttrOfType<DenseI64ArrayAttr>(registersName(element)).asArrayRef();
        std::vector<bool> used(members.size());
        Value shared;
        for (size_t first = 0; first < members.size(); ++first)
            for (size_t second = first + 1; second < members.size() && !used[first]; ++second)
                if (auto block = matchBlock({members[first].offsets, members[second].offsets}, map);
                    block && !used[second] && fuse(members[first], members[second], *block, element, shared))
                    used[first] = used[second] = true;
    }
}

struct Access {
    Operation* operation;
    Value pointer, index, condition, item;
    size_t position;
};

std::optional<Access> access(Operation* operation, size_t position) {
    if (auto load = dyn_cast<LoadOp>(operation); load && load.getNumResults() == 1)
        return Access{operation, load.getPointer(), load.getIndex(), {}, {}, position};
    if (auto load = dyn_cast<GuardedLoadOp>(operation); load && load.getNumResults() == 1)
        return Access{operation, load.getPointer(), load.getIndex(), load.getCondition(), load.getFallbacks()[0],
                      position};
    if (auto store = dyn_cast<StoreOp>(operation); store && store.getValues().size() == 1)
        return Access{operation, store.getPointer(), store.getIndex(), {}, store.getValues()[0], position};
    if (auto store = dyn_cast<GuardedStoreOp>(operation); store && store.getValues().size() == 1)
        return Access{operation, store.getPointer(), store.getIndex(), store.getCondition(), store.getValues()[0],
                      position};
    return std::nullopt;
}

class Vectorizer {
  public:
    Vectorizer(func::FuncOp function, MemoryPlan& memory) : facts_(function), memory_(memory) {}

    void run(func::FuncOp function) {
        SmallVector<Block*> blocks;
        function.walk([&](Block* block) { blocks.push_back(block); });
        for (auto* block : blocks)
            vectorize(*block);
        for (auto* operation : fused_)
            operation->erase();
    }

  private:
    void vectorize(Block& block) {
        order_.clear();
        std::map<std::tuple<bool, void*, int>, std::map<uint64_t, Access>> groups;
        for (auto& operation : block) {
            order_.push_back(&operation);
            auto found = access(&operation, order_.size() - 1);
            auto parts = found ? facts_.split(found->index) : std::nullopt;
            if (!parts)
                continue;
            bool load = isa<LoadOp, GuardedLoadOp>(operation);
            groups[{load, found->pointer.getAsOpaquePointer(), parts->first}].try_emplace(parts->second, *found);
        }
        for (auto& [key, members] : groups) {
            auto [load, pointer, part] = key;
            auto element = cast<PointerType>(Value::getFromOpaquePointer(pointer).getType()).getElementType();
            auto bytes = element.getIntOrFloatBitWidth() / 8;
            for (int64_t width = 16 / bytes; width >= 2; width /= 2) {
                if (facts_.divides(part) < uint64_t(bytes == 2 && width > 4 ? 2 * width : width))
                    continue;
                for (auto& [offset, first] : members) {
                    if (offset % width || !first.operation)
                        continue;
                    SmallVector<Access*> run;
                    for (int64_t k = 0; k < width; ++k) {
                        auto found = members.find(offset + k);
                        if (found == members.end() || !found->second.operation)
                            break;
                        run.push_back(&found->second);
                    }
                    if (int64_t(run.size()) == width && fuse(run, load, bytes * width))
                        for (auto* member : run)
                            member->operation = nullptr;
                }
            }
        }
    }

    // Every member's checks at once. Checks x + c < B that differ only in c, from a base aligned to the run, cannot
    // wrap within it, so the largest c stands for all of them.
    Value together(OpBuilder& builder, Location location, ArrayRef<Value> conditions, int64_t width) {
        SmallVector<Value> leaves;
        for (auto condition : conditions)
            conjuncts(condition, leaves);
        std::map<std::pair<void*, int>, std::pair<uint64_t, uint64_t>> spans;
        auto key = [&](Value leaf) -> std::optional<std::tuple<std::pair<void*, int>, uint64_t>> {
            auto compare = leaf.getDefiningOp<arith::CmpIOp>();
            auto parts = compare && compare.getPredicate() == arith::CmpIPredicate::ult
                             ? facts_.split(compare.getLhs()) : std::nullopt;
            if (!parts || facts_.divides(parts->first) < uint64_t(width))
                return std::nullopt;
            return std::make_tuple(std::make_pair(compare.getRhs().getAsOpaquePointer(), parts->first), parts->second);
        };
        for (auto leaf : leaves)
            if (auto found = key(leaf)) {
                auto [span, inserted] =
                    spans.try_emplace(std::get<0>(*found), std::get<1>(*found), std::get<1>(*found));
                span->second.first = std::min(span->second.first, std::get<1>(*found));
                span->second.second = std::max(span->second.second, std::get<1>(*found));
            }
        Value every;
        for (auto leaf : leaves) {
            if (auto found = key(leaf)) {
                auto [low, high] = spans.at(std::get<0>(*found));
                if (low % width == 0 && high - low < uint64_t(width) && std::get<1>(*found) != high)
                    continue;
            }
            every = every ? arith::AndIOp::create(builder, location, every, leaf).getResult() : leaf;
        }
        return every;
    }

    // Fuse at the last member only if intervening accesses cannot alias and no load result is used early.
    // Mixed guards need a vector/scalar branch; common guards remain outside it.
    bool fuse(ArrayRef<Access*> run, bool load, int64_t bytes) {
        auto [first, last] = std::minmax_element(run.begin(), run.end(), [](Access* a, Access* b) {
            return a->position < b->position;
        });
        auto base = root((*first)->pointer);
        auto form = facts_.split(run[0]->index);
        uint64_t low = ~uint64_t(0), high = 0;
        for (auto* member : run) {
            auto offset = facts_.split(member->index)->second;
            low = std::min(low, offset);
            high = std::max(high, offset);
        }
        // An access between the members to another element of the same address form cannot overlap the run.
        auto disjoint = [&](size_t at) {
            auto other = access(order_[at], at);
            auto parts = other ? facts_.split(other->index) : std::nullopt;
            return parts && other->pointer.getType() == run[0]->pointer.getType() && root(other->pointer) == base &&
                   parts->first == form->first && (parts->second < low || parts->second > high);
        };
        for (size_t at = (*first)->position + 1; at < (*last)->position; ++at)
            if (llvm::none_of(run, [&](Access* member) { return member->position == at; }) &&
                touches(order_[at], base, true, !load) && !disjoint(at))
                return false;
        auto* insertion = (*last)->operation;
        if (load)
            for (auto* member : run)
                for (auto* user : member->operation->getUsers()) {
                    auto* owner = insertion->getBlock()->findAncestorOpInBlock(*user);
                    if (owner && owner->isBeforeInBlock(insertion))
                        return false;
                }
        if (auto argument = base.getDefiningOp<ArgumentOp>();
            argument && argument.getBinding() < memory_.bindingAlignment.size()) {
            auto& alignment = memory_.bindingAlignment[argument.getBinding()];
            alignment = std::max<uint64_t>(alignment, bytes);
        }
        OpBuilder builder(insertion);
        auto location = insertion->getLoc();
        auto pointer = run[0]->pointer, index = run[0]->index;
        SmallVector<Value> conditions, items;
        for (auto* member : run) {
            if (member->condition && !llvm::is_contained(conditions, member->condition))
                conditions.push_back(member->condition);
            items.push_back(member->item);
        }
        bool partial = conditions.size() > 1 ||
                       (conditions.size() == 1 && llvm::any_of(run, [](Access* member) { return !member->condition; }));
        SmallVector<Type> types(run.size(), cast<PointerType>(pointer.getType()).getElementType());
        auto plain = [&](OpBuilder& at) -> ValueRange {
            if (load)
                return LoadOp::create(at, location, types, pointer, index).getResults();
            StoreOp::create(at, location, pointer, index, items);
            return {};
        };
        SmallVector<Value> results;
        if (conditions.empty())
            llvm::append_range(results, plain(builder));
        else if (!partial) {
            if (load)
                llvm::append_range(results, GuardedLoadOp::create(builder, location, types, pointer, index,
                                                                  conditions[0], items).getResults());
            else
                GuardedStoreOp::create(builder, location, pointer, index, conditions[0], items);
        } else {
            auto every = together(builder, location, conditions, run.size());
            auto branch = scf::IfOp::create(builder, location, load ? TypeRange(types) : TypeRange{}, every, true);
            OpBuilder then = OpBuilder::atBlockEnd(&branch.getThenRegion().front());
            OpBuilder otherwise = OpBuilder::atBlockEnd(&branch.getElseRegion().front());
            if (load) {
                then.setInsertionPointToStart(&branch.getThenRegion().front());
                otherwise.setInsertionPointToStart(&branch.getElseRegion().front());
                scf::YieldOp::create(then, location, plain(then));
                SmallVector<Value> scalars;
                for (auto* member : run)
                    scalars.push_back(otherwise.clone(*member->operation)->getResult(0));
                scf::YieldOp::create(otherwise, location, scalars);
                llvm::append_range(results, branch.getResults());
            } else {
                then.setInsertionPoint(branch.getThenRegion().front().getTerminator());
                otherwise.setInsertionPoint(branch.getElseRegion().front().getTerminator());
                plain(then);
                for (auto* member : run)
                    otherwise.clone(*member->operation);
            }
        }
        for (auto [member, result] : llvm::zip(run, results))
            member->operation->getResult(0).replaceAllUsesWith(result);
        for (auto* member : run)
            fused_.push_back(member->operation);
        return true;
    }

    Facts facts_;
    MemoryPlan& memory_;
    std::vector<Operation*> order_, fused_;
};

}

void verifyConvergence(ModuleOp module, const MemoryPlan& memory) {
    auto function = *module.getOps<func::FuncOp>().begin();
    Facts facts(function);
    Convergence(memory, &facts).run(function);
}

void optimizeThreads(ModuleOp module, MemoryPlan& memory) {
    auto function = *module.getOps<func::FuncOp>().begin();
    simplify(module);
    foldGuards(function);
    simplify(module, true);
    numberValues(function);
    simplify(module, true);
    matrixLoads(function, memory);
    function.walk([&](scf::ForOp loop) {
        if (auto count = loop.getStaticTripCount(); count && count->ule(1))
            return;
        if (loop.walk([](BarrierOp) { return WalkResult::interrupt(); }).wasInterrupted())
            hoistReadOnlyLoads(loop, memory);
    });
    Vectorizer(function, memory).run(function);
    simplify(module, true);
    function.walk([&](scf::ForOp loop) { hoistReadOnlyLoads(loop, memory); });
    unswitchLoops(function, memory);
    eraseDeadCode(module);
}

}
