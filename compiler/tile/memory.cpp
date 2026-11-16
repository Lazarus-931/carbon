#include "tile/memory.h"
#include "tile/analysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

using namespace mlir;

namespace carbon {
namespace {

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument(message); }

enum class Access { Read, Write, Atomic };

class Planner {
  public:
    explicit Planner(func::FuncOp function) : function_(function) {}

    MemoryPlan run() {
        if (auto bound = function_->getAttrOfType<IntegerAttr>("carbon.max_grid"))
            plan_.maxGrid = bound.getInt();
        bool scalars = false;
        for (auto argument : function_.getArguments()) {
            if (isa<MemRefType, ir::ViewType>(argument.getType())) {
                if (scalars)
                    fail("buffers must precede runtime scalars");
                bindings_[argument] = accesses_.size();
                accesses_.emplace_back();
                auto minimum = function_.getArgAttrOfType<IntegerAttr>(argument.getArgNumber(), "carbon.min_bytes");
                plan_.bindingBytes.push_back(minimum ? minimum.getInt() : 0);
                // Metal binds buffers at four-byte aligned offsets; vector accesses raise this.
                plan_.bindingAlignment.push_back(4);
            } else
                scalars = true;
        }
        function_.walk<WalkOrder::PreOrder>([&](Operation* operation) {
            if (auto view = dyn_cast<ir::MakeViewOp>(operation)) {
                if (view.getResult().getType().getAddressSpace() == "device") {
                    auto& bytes = plan_.bindingBytes[binding(view.getBuffer())];
                    bytes = std::max(bytes, uint64_t(view.getResult().getType().getExtentBytes()));
                }
            } else if (auto load = dyn_cast<ir::LoadOp>(operation))
                record(load.getBuffer(), Access::Read);
            else if (auto load = dyn_cast<ir::ScalarLoadOp>(operation))
                record(load.getBuffer(), Access::Read);
            else if (auto store = dyn_cast<ir::StoreOp>(operation))
                record(store.getBuffer(), Access::Write);
            else if (isa<ir::AtomicOp, ir::AtomicStoreOp>(operation))
                record(operation->getOperand(0), Access::Atomic);
            else if (auto matmul = dyn_cast<ir::TensorMatmulOp>(operation)) {
                record(matmul.getLeft(), Access::Read);
                record(matmul.getRight(), Access::Read);
            }
        });
        for (const auto& kinds : accesses_) {
            auto has = [&](Access kind) { return llvm::is_contained(kinds, kind); };
            bool atomic = has(Access::Atomic), writes = has(Access::Write), reads = has(Access::Read);
            plan_.access.push_back(atomic && !writes && !reads ? "atomic" : atomic || (writes && reads) ? "read_write"
                                   : writes ? "write" : "read");
            if (function_->hasAttr("carbon.check_writes") && writes)
                plan_.checkedBindings.push_back(plan_.access.size() - 1);
        }
        placeScratch();
        return plan_;
    }

  private:
    struct Lifetime {
        int64_t first = -1, last = -1;
        Operation* entry = nullptr;
    };

    // Threads on either side of a divergent branch run concurrently, so both sides are live at once.
    static bool divergent(Operation* operation) {
        auto branch = dyn_cast<scf::IfOp>(operation);
        return branch && !ir::groupUniform(branch.getCondition());
    }

    llvm::DenseMap<Value, Lifetime> lifetimes(
        const llvm::DenseMap<Operation*, llvm::DenseSet<Value>>& pending) {
        llvm::DenseMap<Operation*, int64_t> position, end;
        int64_t next = 0;
        function_.walk<WalkOrder::PreOrder>([&](Operation* operation) { position[operation] = next++; });
        function_.walk<WalkOrder::PostOrder>([&](Operation* operation) {
            int64_t last = position[operation];
            for (auto& region : operation->getRegions())
                for (auto& block : region)
                    for (auto& nested : block)
                        last = std::max(last, end[&nested]);
            end[operation] = last;
        });
        llvm::DenseMap<Value, SmallVector<Operation*>> uses;
        function_.walk([&](Operation* operation) {
            if (isa<ir::ScratchOp, ir::MakeViewOp, ir::PartitionOp, ir::SubviewOp, ir::DimOp>(operation))
                return;
            for (auto operand : operation->getOperands())
                if (auto allocation = ir::scratchAllocation(operand))
                    uses[allocation.getResult()].push_back(operation);
        });
        llvm::DenseMap<Value, Lifetime> result;
        for (auto& [allocation, accesses] : uses) {
            llvm::DenseSet<Operation*> localLoops;
            // Enclosed uses start uninitialized on the first iteration, which scratch verification checks.
            // The pending fixed point must also prove every previous iteration's accesses completed.
            for (auto* parent = accesses.front()->getParentOp(); !isa<func::FuncOp>(parent);
                 parent = parent->getParentOp()) {
                auto loop = dyn_cast<scf::ForOp>(parent);
                if (!loop || !ir::groupUniform(loop.getLowerBound()) ||
                    !ir::groupUniform(loop.getUpperBound()) || !ir::groupUniform(loop.getStep()))
                    continue;
                auto incoming = pending.find(&loop.getBody()->front());
                if (incoming != pending.end() && !incoming->second.contains(allocation) &&
                    llvm::all_of(accesses, [&](Operation* access) { return loop->isProperAncestor(access); }))
                    localLoops.insert(parent);
            }
            for (auto* operation : accesses) {
                int64_t first = position[operation], last = first;
                Operation* entry = operation;
                for (auto* parent = operation->getParentOp(); !isa<func::FuncOp>(parent);
                     parent = parent->getParentOp())
                    if ((isa<scf::ForOp>(parent) && !localLoops.contains(parent)) || divergent(parent)) {
                        first = position[parent];
                        last = std::max(last, end[parent]);
                        entry = parent;
                    }
                auto& lifetime = result[allocation];
                if (lifetime.first < 0 || first < lifetime.first) {
                    lifetime.first = first;
                    lifetime.entry = entry;
                }
                lifetime.last = std::max(lifetime.last, last);
            }
        }
        return result;
    }

    struct Region {
        ir::ScratchOp scratch;
        uint64_t bytes = 0, offset = 0;
        Lifetime life;
    };

    static uint64_t scratchBytes(ir::ScratchOp scratch) {
        uint64_t bytes = 0;
        for (auto* user : scratch.getResult().getUsers()) {
            if (isa<ir::ItemOp, ir::ReduceSumOp, ir::ReduceMaxOp, ir::ScanSumOp, ir::ScanMaxOp,
                    ir::BroadcastInDimOp>(user) && !ir::collectiveUsesScratch(*user))
                continue;
            if (isa<ir::ItemOp>(user)) {
                bytes = std::max(bytes, uint64_t(4));
                continue;
            }
            if (!ir::nativeReduction(user))
                return scratch.allocationBytes();
            auto layout = cast<ir::TileType>(user->getOperand(0).getType()).getLayout();
            bytes = std::max(bytes, uint64_t(layout.getGroups().size() * 4));
        }
        return bytes;
    }

    // Try multiple placement orders so long-lived regions do not fragment reusable space.
    void placeScratch() {
        std::vector<Region> regions;
        plan_.finalScratchUsers = ir::finalScratchUsers(function_);
        auto pending = ir::pendingScratch(function_, plan_.finalScratchUsers);
        auto lives = lifetimes(pending);
        function_.walk([&](ir::ScratchOp scratch) {
            // Align to Metal's 16-byte threadgroup-memory units and maximum vector-access width.
            regions.push_back({scratch, (scratchBytes(scratch) + 15) / 16 * 16, 0,
                               lives.lookup(scratch.getResult())});
        });
        if (regions.empty())
            return;
        auto inFlight = [&](Region& earlier, Region& later) {
            return pending[later.life.entry].contains(earlier.scratch.getResult());
        };
        auto disjoint = [&](Region& a, Region& b) {
            if (a.life.first < 0 || b.life.first < 0)
                return true;
            if (a.life.last < b.life.first)
                return !inFlight(a, b);
            if (b.life.last < a.life.first)
                return !inFlight(b, a);
            return false;
        };
        std::vector<std::vector<bool>> conflicts(regions.size(), std::vector<bool>(regions.size()));
        for (size_t a = 0; a < regions.size(); ++a)
            for (size_t b = 0; b < a; ++b)
                conflicts[a][b] = conflicts[b][a] = !disjoint(regions[a], regions[b]);
        std::vector<size_t> order(regions.size());
        std::iota(order.begin(), order.end(), 0);
        uint64_t best = UINT64_MAX;
        auto pack = [&] {
            std::vector<uint64_t> offsets(regions.size());
            uint64_t bytes = 0;
            for (size_t at = 0; at < order.size(); ++at) {
                auto index = order[at];
                std::vector<std::pair<uint64_t, uint64_t>> occupied;
                for (size_t before = 0; before < at; ++before) {
                    auto other = order[before];
                    if (conflicts[index][other])
                        occupied.emplace_back(offsets[other], offsets[other] + regions[other].bytes);
                }
                std::sort(occupied.begin(), occupied.end());
                uint64_t offset = 0;
                for (auto [begin, end] : occupied) {
                    if (offset + regions[index].bytes <= begin)
                        break;
                    offset = std::max(offset, end);
                }
                offsets[index] = offset;
                bytes = std::max(bytes, offset + regions[index].bytes);
            }
            if (bytes < best) {
                best = bytes;
                for (size_t at = 0; at < regions.size(); ++at)
                    regions[at].offset = offsets[at];
            }
        };
        pack();
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return regions[a].bytes > regions[b].bytes;
        });
        pack();
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return regions[a].life.first < regions[b].life.first;
        });
        pack();
        for (auto& region : regions) {
            plan_.scratchOffsets[region.scratch.getResult()] = region.offset;
            plan_.scratchLayout.push_back({region.offset, region.bytes, region.life.first, region.life.last,
                                           storage(region.scratch), location(region.scratch)});
            plan_.scratchBytes = std::max(plan_.scratchBytes, region.offset + region.bytes);
        }
    }

    static std::string location(Operation* operation) {
        std::string text;
        llvm::raw_string_ostream stream(text);
        operation->getLoc().print(stream);
        return text;
    }

    static std::string storage(ir::ScratchOp scratch) {
        auto type = cast<ir::ViewType>(scratch.getResult().getType());
        std::string text;
        llvm::raw_string_ostream stream(text);
        stream << type.getElementType() << "[";
        for (auto [axis, extent] : llvm::enumerate(type.getShape().asArrayRef()))
            stream << (axis ? "," : "") << extent;
        stream << "]";
        return text;
    }

    size_t binding(Value memory) {
        memory = ir::bindingRoot(memory);
        auto found = bindings_.find(memory);
        if (found == bindings_.end())
            fail("memory must originate in a kernel binding");
        return found->second;
    }

    void record(Value memory, Access kind) {
        auto view = dyn_cast<ir::ViewType>(memory.getType());
        if (!view || view.getAddressSpace() == "device")
            accesses_[binding(memory)].push_back(kind);
    }

    func::FuncOp function_;
    MemoryPlan plan_;
    llvm::DenseMap<Value, size_t> bindings_;
    std::vector<std::vector<Access>> accesses_;
};

}

MemoryPlan planMemory(func::FuncOp function) { return Planner(function).run(); }

}
