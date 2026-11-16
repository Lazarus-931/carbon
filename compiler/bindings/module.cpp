#include "thread/lowering.h"
#include "tile/dialect.h"
#include "tile/module.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Config/llvm-config.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>

namespace nb = nanobind;
using namespace mlir;

namespace carbon {

ModuleOp stagedModule(nb::handle staging);

static std::string verifyIR(const std::string& input) {
    auto module = ir::parse(input, ir::context());
    ir::optimize(*module);
    return ir::print(*module);
}

// Check scratch ownership on a lowered copy before cleanup can erase unused accesses.
static OwningOpRef<ModuleOp> checked(nb::handle staging, int64_t threadgroupMemory) {
    OwningOpRef<ModuleOp> module = stagedModule(staging).clone();
    ir::kernel(*module)->setAttr("carbon.threadgroup_memory",
                                 IntegerAttr::get(IntegerType::get(&ir::context(), 64), threadgroupMemory));
    ir::check(*module);
    if (!ir::kernel(*module).getOps<ir::ScratchOp>().empty()) {
        OwningOpRef<ModuleOp> lowered = module->clone();
        validateMetal(ir::kernel(*lowered));
        msl::checkScratch(ir::kernel(*lowered));
    }
    return module;
}

static std::string canonical(nb::handle staging, int64_t threadgroupMemory) {
    auto module = checked(staging, threadgroupMemory);
    ir::optimize(*module);
    return ir::print(*module);
}

static nb::dict compile(nb::handle staging, int64_t threadgroupMemory, bool checkWrites) {
    auto module = checked(staging, threadgroupMemory);
    if (checkWrites)
        ir::kernel(*module)->setAttr("carbon.check_writes", UnitAttr::get(&ir::context()));
    validateMetal(ir::kernel(*module));
    ir::optimize(*module);
    auto kernel = lowerMetal(ir::kernel(*module));
    nb::dict output;
    output["ir"] = ir::print(*module);
    output["source"] = kernel.source;
    output["thread_ir"] = kernel.threadIR;
    output["access"] = kernel.memory.access;
    output["binding_bytes"] = kernel.memory.bindingBytes;
    output["binding_alignment"] = kernel.memory.bindingAlignment;
    output["max_grid"] = kernel.memory.maxGrid;
    output["checked_bindings"] = kernel.memory.checkedBindings;
    output["scratch_bytes"] = kernel.memory.scratchBytes;
    nb::list layout;
    for (const auto& region : kernel.memory.scratchLayout) {
        nb::dict entry;
        entry["offset"] = region.offset;
        entry["bytes"] = region.bytes;
        entry["storage"] = region.storage;
        entry["first"] = region.first;
        entry["last"] = region.last;
        entry["location"] = region.location;
        layout.append(entry);
    }
    output["scratch_layout"] = layout;
    output["llvm_version"] = LLVM_VERSION_STRING;
    return output;
}

void bindLayouts(nb::module_& module);
void bindBuilder(nb::module_& module);

}

NB_MODULE(_compiler, module) {
    carbon::bindLayouts(module);
    carbon::bindBuilder(module);
    module.def("verify", &carbon::verifyIR, nb::arg("ir"), "Parse and verify kernel IR text, and print it cleaned up.");
    module.def("canonical", &carbon::canonical, nb::arg("staging"), nb::arg("threadgroup_memory"),
               "The staged kernel, checked and cleaned up, as text.");
    module.def("compile", &carbon::compile, nb::arg("staging"), nb::arg("threadgroup_memory"),
               nb::arg("check_writes") = false,
               "Check, plan and lower a staged kernel: its IR, per-thread IR, Metal source and memory plan.");
    module.attr("llvm_version") = LLVM_VERSION_STRING;
    module.attr("abi_version") = 58;
    module.attr("toolchain_id") = CARBON_TOOLCHAIN_ID;
    module.attr("build_id") = CARBON_BUILD_ID;
}
