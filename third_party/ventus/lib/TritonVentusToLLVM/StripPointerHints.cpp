// Keeps the layout passes from spending per-thread ownership on this target.
//
// Triton's coalescer turns a pointer argument's `tt.divisibility` into
// `sizePerThread`, i.e. into several *consecutive* elements per thread. On
// Ventus that is a pessimization: with four consecutive elements per thread the
// 32 lanes of a warp stride through memory by the vector width instead of
// covering consecutive addresses. Measured on the same kernel and grid, with an
// essentially identical instruction stream (351 vs 349 instructions, same
// `vlw12.v` x8 / `vsw12.v` x4), the two layouts differ by 2.7x in cycle time
// (85845 ns warp-contiguous vs 229185 ns per-thread contiguous; README 4.4).
//
// The divisibility of a pointer argument is a *true* fact -- the reference
// launcher's device buffers come from page-granular allocations -- so this pass
// does not correct a lie. It drops a hint whose only consumer here uses it in
// the wrong direction. Nothing becomes incorrect: the divisibility is only ever
// a promise to the optimizer, and the layout stays warp-contiguous, which is
// what the hardware's 32-lane vector unit wants.
//
// This is a target preference with no upstream hook: `tritongpu-coalesce` takes
// no target parameters, so a backend that wants per-thread width 1 has to say
// so before the pass runs. If upstream ever adds such a hook, this pass should
// be replaced by it.
#include "Target/TritonVentusToLLVM.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::ventus {
namespace {

// Hints that steer per-thread ownership or per-thread contiguity.
// `tt.contiguity` and `tt.constancy` are included because the coalescer
// consults them the same way when it picks `sizePerThread`.
constexpr StringLiteral kPointerHints[] = {"tt.divisibility", "tt.contiguity",
                                           "tt.constancy", "tt.max_contiguous",
                                           "tt.multiple_of"};

struct StripPointerHints
    : public PassWrapper<StripPointerHints, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StripPointerHints)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](triton::FuncOp func) {
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        if (!isa<triton::PointerType>(func.getArgument(i).getType()))
          continue;
        for (StringLiteral hint : kPointerHints)
          func.removeArgAttr(i, hint);
      }
    });
  }

  StringRef getName() const override { return "ventus-strip-pointer-hints"; }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createStripPointerHintsPass() {
  return std::make_unique<StripPointerHints>();
}

} // namespace mlir::triton::ventus
