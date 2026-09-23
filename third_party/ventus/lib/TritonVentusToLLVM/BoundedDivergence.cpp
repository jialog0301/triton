// P3 producer-side bounded-divergence check.
//
// The pinned Ventus toolchain documents the *validated* provider bound as "at
// most two nested divergent regions, including at most two nested divergent
// loops" (VentusLLVMInputContract.md, "Scalar-SIMT And Control Flow"). Deeper
// nesting is "rejected by the producer checker until a larger bound has
// mandatory Spike/RTL evidence" -- this is that checker.
//
// The pass is deliberately conservative: it assumes every region is divergent.
// That is the honest reading of the contract for arbitrary provider IR (any
// `scf.if`/`scf.for` may be per-lane), and erring toward rejection keeps an
// unvalidated shape from reaching the hardware, whose SIMT-stack depth and
// overflow behaviour are explicitly not a general contract.
//
// It runs on TritonGPU IR, before `scf-to-cf`, while the regions are still
// explicitly `scf.if`/`scf.for`/... . Rejection is a hard error through the
// pass manager (the module is left untouched and the pass fails), so a caller
// sees `PassManager::run failed` alongside the emitted diagnostic naming the
// bound.
#include "Target/TritonVentusToLLVM.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::triton::ventus {
namespace {

// The validated provider bound (VentusLLVMInputContract.md): two nested
// divergent regions. A third level is a producer error.
constexpr unsigned kMaxDivergentNesting = 2;

bool isDivergentRegionOp(Operation *op) {
  return isa<scf::IfOp, scf::ForOp, scf::WhileOp, scf::IndexSwitchOp>(op);
}

// How many divergent regions enclose `op`: the count of ancestor control-flow
// ops, one per region the op sits in. An op directly in a function body has
// depth 0.
unsigned divergentNestingDepth(Operation *op) {
  unsigned depth = 0;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (isDivergentRegionOp(parent))
      ++depth;
  return depth;
}

struct VentusBoundedDivergence
    : public PassWrapper<VentusBoundedDivergence, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VentusBoundedDivergence)

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // A region op's own body is one level deeper than the op, so the check is
    // on `depth + 1`. Every level is entered by one of these ops, so walking
    // them is sufficient to bound the depth of any nested operation.
    WalkResult result =
        mod.walk([&](Operation *op) -> WalkResult {
          if (!isDivergentRegionOp(op))
            return WalkResult::advance();
          unsigned depth = divergentNestingDepth(op) + 1;
          if (depth > kMaxDivergentNesting) {
            op->emitError()
                << "Ventus lowering (V1): divergent control-flow nesting depth "
                << depth << " exceeds the validated producer bound of "
                << kMaxDivergentNesting
                << " nested regions (VentusLLVMInputContract.md); flatten the "
                   "nesting or use data-plane predicates";
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted())
      signalPassFailure();
  }

  StringRef getName() const override { return "ventus-bounded-divergence"; }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createVentusBoundedDivergencePass() {
  return std::make_unique<VentusBoundedDivergence>();
}

} // namespace mlir::triton::ventus
