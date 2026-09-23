// Inserts the work-group barriers that shared-memory reuse needs.
//
// The core `ModuleMembarAnalysis` is target-independent: it walks the shared
// memory allocation, finds the read/write pairs that alias, and drops a
// `ttg.barrier local` between them (RAW: a write a later read depends on; WAR:
// a read whose buffer is written again). Every backend wraps it in its own
// pass because the analysis needs the backend's allocation — NVIDIA and AMD
// pass vendor-specific scratch-size functions and filters, while this target
// has neither: `ModuleAllocation`'s default scratch-size function already
// covers the operations this backend lowers (reduce, and layout conversions
// once they exist), and there are no vendor ops to filter out.
//
// Ordering matters and is asserted by the pipeline: this pass must run after
// `allocate-shared-memory` (the analysis needs offsets and sizes) and before
// the TritonGPU-to-LLVM conversion (which is where `ttg.barrier` becomes the
// hardware instruction).
#include "Target/TritonVentusToLLVM.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"

namespace mlir::triton::ventus {
namespace {

struct VentusMembar
    : public PassWrapper<VentusMembar, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VentusMembar)

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ModuleAllocation allocation(mod);
    ModuleMembarAnalysis membarPass(allocation);
    membarPass.run();
  }

  StringRef getName() const override { return "ventus-membar"; }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createVentusMembarPass() {
  return std::make_unique<VentusMembar>();
}

} // namespace mlir::triton::ventus
