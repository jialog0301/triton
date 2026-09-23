// P3 hardware-divergence arm: rewrite a data-plane `llvm.select` into a
// control-flow `cond_br` diamond.
//
// `tl.where` lowers to `arith.select` and then, in the core conversion, to
// `LLVM::SelectOp` -- a pure data-plane choice with no branch. Ventus hardware
// control flow, in contrast, is driven by divergent branches (`vbeq`/`setrpc`/
// `join`), which the Ventus LLVM generates from `cond_br`. Both shapes compute
// the same value: by the time this pass runs, the condition and both arms are
// already-evaluated SSA values, so branching on the condition and yielding the
// chosen arm is semantics-preserving. The only thing that changes is which
// codegen path the divergence analysis sees (README 4.6).
//
// This runs on the MLIR LLVM dialect, after `TritonGPUToLLVM` has produced the
// `llvm.select` ops and before `llvm.to_module` translates the module. It is
// only added to the pipeline when the `predication` option selects the hardware
// arm, so the software arm's IR is byte-for-byte what it was.
#include "Target/TritonVentusToLLVM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::triton::ventus {
namespace {

struct VentusSelectToBranch
    : public PassWrapper<VentusSelectToBranch, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VentusSelectToBranch)

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<LLVM::SelectOp> selects;
    mod.walk([&](LLVM::SelectOp op) { selects.push_back(op); });

    IRRewriter rewriter(&getContext());
    for (LLVM::SelectOp op : selects) {
      // Only scalar `i1` conditions are divergence: a vector condition is a
      // per-element mask, which the hardware handles as data, not control.
      if (!op.getCondition().getType().isInteger(1))
        continue;
      rewriter.setInsertionPoint(op);
      rewrite(rewriter, op);
    }
  }

  static void rewrite(IRRewriter &rewriter, LLVM::SelectOp op) {
    Location loc = op.getLoc();
    Block *currentBlock = op->getBlock();
    Region *region = currentBlock->getParent();
    Block *joinBlock = currentBlock->splitBlock(rewriter.getInsertionPoint());
    // The result is produced by whichever arm runs, so it arrives at the join
    // point as a block argument (same idiom as the predicated load/store
    // lowering in VentusLoadStoreOpToLLVM.cpp).
    BlockArgument result = joinBlock->addArgument(op.getType(), loc);
    Block *thenBlock =
        rewriter.createBlock(region, Region::iterator(joinBlock));
    Block *elseBlock =
        rewriter.createBlock(region, Region::iterator(joinBlock));

    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, op.getCondition(), thenBlock,
                           ValueRange{}, elseBlock, ValueRange{});

    rewriter.setInsertionPointToEnd(thenBlock);
    LLVM::BrOp::create(rewriter, loc, op.getTrueValue(), joinBlock);

    rewriter.setInsertionPointToEnd(elseBlock);
    LLVM::BrOp::create(rewriter, loc, op.getFalseValue(), joinBlock);

    rewriter.setInsertionPointToStart(joinBlock);
    rewriter.replaceOp(op, result);
  }

  StringRef getName() const override { return "ventus-select-to-branch"; }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createVentusSelectToBranchPass() {
  return std::make_unique<VentusSelectToBranch>();
}

} // namespace mlir::triton::ventus
