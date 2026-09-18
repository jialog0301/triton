#ifndef TRITON_THIRD_PARTY_VENTUS_TARGET_TRITONVENTUSTOLLVM_H
#define TRITON_THIRD_PARTY_VENTUS_TARGET_TRITONVENTUSTOLLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "triton/Analysis/AxisInfo.h"

#include "llvm/IR/Module.h"

#include <memory>

namespace mlir::triton::ventus {

// Emit a convergent call to an OpenCL work-item builtin (e.g. the mangled
// `_Z12get_local_idj`), creating the declaration in the module if needed.
Value emitWorkItemBuiltinCall(OpBuilder &rewriter, Location loc,
                              ModuleOp moduleOp, StringRef symbol,
                              Value index);

// Drop the pointer-argument hints the coalescer would spend on per-thread
// ownership (see StripPointerHints.cpp): one element per thread keeps a warp's
// accesses contiguous, which is what this target's 32-lane vector unit wants.
std::unique_ptr<OperationPass<ModuleOp>> createStripPointerHintsPass();

// TritonGPU -> LLVM dialect conversion driven by the Ventus TargetInfo.
// V1 baseline: elementwise / masked load-store / SPMD control flow on
// riscv32 with warp size 32, global addrspace 1, local (shared) addrspace 3.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToVentusLLVMPass();

// V1 global tt.load/tt.store lowering (per-element; masked accesses go
// through llvm.masked.load/store on <1 x T> vectors).
void populateVentusLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit);

// Post-translation attachment of the Ventus ABI envelope on the emitted
// llvm::Module: target triple, data layout, the `ventus_kernel` calling
// convention on the entry kernel, target features and OpenCL-style kernel
// argument metadata. Nothing here rewrites IR structure; it only attaches
// module/function-level properties the external Ventus LLVM 16 toolchain
// consumes.
void finalizeLLVMModule(llvm::Module &module);

} // namespace mlir::triton::ventus

#endif // TRITON_THIRD_PARTY_VENTUS_TARGET_TRITONVENTUSTOLLVM_H
