// TritonGPU -> LLVM dialect conversion for the Ventus backend (V1).
//
// Mirrors the third-party driver pattern used by the NVIDIA and AMD backends:
// the driver lives outside Triton core and injects target differences through
// a TargetInfoBase implementation. V1 scope is the non-MMA Spike baseline:
// elementwise and masked load/store on riscv32, warp size 32, global address
// space 1, local (shared) address space 3. Anything outside that scope fails
// compilation loudly instead of silently producing wrong code.
#include "Target/TritonVentusToLLVM.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::ventus {
namespace {

// Work-item builtins the pinned Ventus libclc object exports (verified with
// llvm-nm against toolchain/version.json). The mangled names follow the
// C++ mangling clang uses for the OpenCL builtins in the ABI goldens.
constexpr StringRef kWorkGroupIdBuiltin = "_Z12get_group_idj";
constexpr StringRef kWorkItemIdBuiltin = "_Z12get_local_idj";

[[noreturn]] void unimplemented(const llvm::Twine &what) {
  llvm::report_fatal_error("Ventus lowering (V1): not implemented: " + what);
}

} // namespace

// Defined outside the anonymous namespace so the load/store lowering TU can
// share it (see emitWorkItemBuiltinCall in the header).
Value emitWorkItemBuiltinCall(OpBuilder &rewriter, Location loc,
                              ModuleOp moduleOp, StringRef symbol,
                              Value index) {
  auto i32Ty = rewriter.getI32Type();
  // The declaration goes at module level, but the call must be emitted at
  // the caller's insertion point (inside the function body).
  auto savedPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
  auto fnOp =
      LLVM::lookupOrCreateFn(rewriter, moduleOp, symbol, {i32Ty}, i32Ty);
  rewriter.restoreInsertionPoint(savedPoint);
  if (failed(fnOp))
    return Value();
  (*fnOp)->setAttr("convergent", UnitAttr::get(rewriter.getContext()));
  auto call = LLVM::CallOp::create(rewriter, loc, *fnOp, index);
  return call.getResult();
}

namespace {

class VentusTargetInfo : public TargetInfoBase {
  // [[noreturn]] lets every unsupported-out-of-scope override omit a return.
  [[noreturn]] static void unsupported(StringRef what) {
    llvm::report_fatal_error("Ventus lowering (V1): unsupported: " + what);
  }

public:
  bool supportMaximumMinimum() const override { return false; }

  // One CTA per work-group (toolchain identity: cta_to_work_group
  // one_to_one), so the cluster CTA id is always 0.
  Value getClusterCTAId(RewriterBase &rewriter, Location loc) const override {
    TritonLLVMOpBuilder b(loc, rewriter);
    return b.i32_val(0);
  }
  Value ballot(RewriterBase &, Location, Type, Value) const override {
    unsupported("ballot");
  }

  Value getGlobalTimer(RewriterBase &, Location) const override {
    unsupported("global timer");
  }

  // Triton program instances map one-to-one onto Ventus work-groups; the
  // group id comes from the libclc work-item ABI.
  Value programId(RewriterBase &rewriter, Location loc, ModuleOp moduleOp,
                  ProgramIDDim axis) const override {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto axisValue = b.i32_val(static_cast<int>(axis));
    return emitWorkItemBuiltinCall(rewriter, loc, moduleOp,
                                   kWorkGroupIdBuiltin, axisValue);
  }
  StringRef getAtomicSyncScope(MemSyncScope scope) const override {
    switch (scope) {
    case MemSyncScope::GPU:
      return "device";
    case MemSyncScope::SYSTEM:
      return {};
    case MemSyncScope::CTA:
      // Work-group scope has no dedicated sync scope in V1.
      unsupported("CTA-scoped atomics");
    }
    llvm_unreachable("unknown memory synchronization scope");
  }

  void barrier(Location, RewriterBase &, triton::gpu::AddrSpace) const override {
    // Milestone 2 lowers this to llvm.riscv.ventus.barrier.
    unsupported("barrier");
  }

  void clusterBarrier(Location, RewriterBase &, Operation *) const override {
    unsupported("cluster barrier");
  }

  void warpSync(Location, RewriterBase &) const override {
    unsupported("warp sync");
  }

  void storeDShared(RewriterBase &, Location, Value, Value, Value,
                    Value) const override {
    unsupported("shared memory store");
  }

  Value loadDShared(RewriterBase &, Location, Value, Value, Type, Value,
                    Operation *) const override {
    unsupported("shared memory load");
  }

  Value shuffleXor(RewriterBase &, Location, Value, int) const override {
    unsupported("shuffle xor");
  }
  Value shuffleUp(RewriterBase &, Location, Value, int) const override {
    unsupported("shuffle up");
  }
  Value shuffleIdx(RewriterBase &, Location, Value, int) const override {
    unsupported("shuffle idx");
  }
  Value shuffleIdx(RewriterBase &, Location, Value, Value) const override {
    unsupported("shuffle idx");
  }

  Value permute(RewriterBase &, Location, Value, Value, Value) const override {
    unsupported("shuffle permute");
  }

  bool warpReduce(RewriterBase &, Location, SmallVector<Value> &,
                  triton::ReduceOp, unsigned) const override {
    // Returning false falls back to the generic reduction tree, which then
    // fails loudly on the shuffle primitives above.
    return false;
  }

  std::string getMulhiFuncName(Type) const override {
    unsupported("mulhi helper");
  }

  void printf(RewriterBase &, Value, int, ValueRange,
              ArrayRef<bool>) const override {
    unsupported("device printf");
  }

  void printf(RewriterBase &, StringRef, ValueRange,
              ArrayRef<bool>) const override {
    unsupported("device printf");
  }

  void assertFail(RewriterBase &rewriter, Location loc, StringRef, StringRef,
                  StringRef, int) const override {
    // llvm.trap() is always available in the LLVM backend; no runtime
    // formatting dependency is required for the V1 baseline.
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.trap", TypeRange{},
                                    {});
    LLVM::UnreachableOp::create(rewriter, loc);
  }
  int getSharedAddressSpace() const override { return 3; }

  int getAddressSpace(Attribute addressSpace) const override {
    if (isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace))
      return 3;
    unsupported("address space mapping");
  }

  bool supportVectorizedAtomics() const override { return false; }
  bool supportBitwidth16Elementwise() const override { return false; }
  bool supportBitwidth32Elementwise() const override { return true; }
  bool isCuda() const override { return false; }
};

//===----------------------------------------------------------------------===//
// Ventus work-item mapping patterns
//===----------------------------------------------------------------------===//

struct GpuThreadIdConversion
    : public ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp> {
  using ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::ThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getDimension() != mlir::gpu::Dimension::x)
      return op.emitError(
          "Ventus lowering (V1): only the x work-item dimension is modeled");
    auto moduleOp = op->getParentOfType<ModuleOp>();
    TritonLLVMOpBuilder b(op.getLoc(), rewriter);
    auto localId = emitWorkItemBuiltinCall(rewriter, op.getLoc(), moduleOp,
                                           kWorkItemIdBuiltin, b.i32_val(0));
    if (!localId)
      return failure();
    // Core helpers mask the thread id with (num_warps * warp_size - 1);
    // local_size equals that bound, so no additional masking is required.
    rewriter.replaceOp(op, localId);
    return success();
  }
};

struct GpuWarpIdConversion
    : public ConvertOpToLLVMPattern<triton::gpu::WarpIdOp> {
  using ConvertOpToLLVMPattern<triton::gpu::WarpIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::WarpIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
    TritonLLVMOpBuilder b(op.getLoc(), rewriter);
    auto localId = emitWorkItemBuiltinCall(rewriter, op.getLoc(), moduleOp,
                                           kWorkItemIdBuiltin, b.i32_val(0));
    if (!localId)
      return failure();
    rewriter.replaceOpWithNewOp<arith::DivUIOp>(
        op, localId, b.i32_val(threadsPerWarp));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Function boundary conversion
//===----------------------------------------------------------------------===//

// Core amendFuncOp appends global-scratch and profile-scratch pointer
// arguments to every function. The Ventus kernel ABI (see the ABI goldens)
// takes user arguments only, so the boundary conversion is done here without
// amending the signature.
struct VentusFuncOpConversion : public ConvertOpToLLVMPattern<triton::FuncOp> {
  using ConvertOpToLLVMPattern<triton::FuncOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<LLVM::LLVMFuncOp> maybeNewFuncOp =
        mlir::convertFuncOpToLLVMFuncOp(funcOp, rewriter,
                                        *getTypeConverter());
    if (failed(maybeNewFuncOp))
      return failure();
    LLVM::LLVMFuncOp newFuncOp = *maybeNewFuncOp;
    if (triton::isKernel(funcOp)) {
      newFuncOp.setLinkage(LLVM::Linkage::External);
    } else {
      newFuncOp.setLinkage(LLVM::Linkage::Internal);
      newFuncOp->setAttr("noinline", UnitAttr::get(funcOp.getContext()));
    }
    rewriter.eraseOp(funcOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conversion driver
//===----------------------------------------------------------------------===//

class VentusLLVMConversionTarget : public ConversionTarget {
public:
  explicit VentusLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
    // Control flow is lowered in a second application after the
    // axis-info-dependent patterns have finished (NVIDIA/AMD driver order).
    addLegalDialect<cf::ControlFlowDialect>();
    addIllegalDialect<triton::TritonDialect, triton::gpu::TritonGPUDialect,
                      mlir::gpu::GPUDialect>();
  }
};

struct ConvertTritonGPUToVentusLLVM
    : public PassWrapper<ConvertTritonGPUToVentusLLVM,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTritonGPUToVentusLLVM)

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    VentusTargetInfo targetInfo;

    mlir::LowerToLLVMOptions option(ctx);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(ctx, option, targetInfo);

    // Phase 0: function boundary. The kernel signature must be settled before
    // bodies are converted.
    {
      // Only the function boundary is illegal here; body ops stay legal
      // (unknown) so phase 0 rewrites signatures alone, mirroring the
      // NVIDIA/AMD driver split.
      ConversionTarget target(*ctx);
      target.addLegalDialect<LLVM::LLVMDialect>();
      target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
      target.addIllegalOp<triton::FuncOp>();
      RewritePatternSet patterns(ctx);
      patterns.add<VentusFuncOpConversion>(typeConverter,
                                           patternBenefitDefault);
      if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Phase 1: Triton/TritonGPU ops. Control flow stays legal until phase 2.
    {
      VentusLLVMConversionTarget target(*ctx);
      RewritePatternSet patterns(ctx);
      ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
      const int benefit = patternBenefitPrioritizeOverLLVMConversions;
      mlir::triton::populateElementwiseOpToLLVMPatterns(
          typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
      mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                                   patterns, benefit);
      // Global tt.load/tt.store: backend-owned in Triton, see the V1
      // per-element lowering in VentusLoadStoreOpToLLVM.cpp.
      populateVentusLoadStoreOpToLLVMPatterns(typeConverter, patterns,
                                              axisInfoAnalysis, benefit);
      mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);
      mlir::triton::populateControlFlowOpToLLVMPattern(
          typeConverter, patterns, targetInfo, benefit);
      mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                  targetInfo, benefit);
      patterns.add<GpuThreadIdConversion, GpuWarpIdConversion>(typeConverter,
                                                               benefit);
      mlir::arith::populateArithToLLVMConversionPatterns(typeConverter,
                                                         patterns);
      mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
      mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
      mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);
      if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Phase 2: control flow.
    {
      ConversionTarget target(*ctx);
      target.addLegalDialect<LLVM::LLVMDialect>();
      target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
      target.addIllegalDialect<cf::ControlFlowDialect>();
      RewritePatternSet patterns(ctx);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            patterns);
      if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }

  StringRef getName() const override {
    return "convert-triton-gpu-to-ventus-llvm";
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToVentusLLVMPass() {
  return std::make_unique<ConvertTritonGPUToVentusLLVM>();
}

//===----------------------------------------------------------------------===//
// LLVM module envelope finalization
//===----------------------------------------------------------------------===//

static constexpr StringRef kTargetTriple = "riscv32";
static constexpr StringRef kTargetDataLayout =
    "e-m:e-p:32:32-i64:64-n32-S128-A5-G1";
static constexpr StringRef kTargetCpu = "ventus-gpgpu";
static constexpr StringRef kTargetFeatures =
    "+32bit,+a,+m,+relax,+zdinx,+zfinx,+zhinx,+zve32f,+zve32x,+zvl32b,"
    "-64bit,-save-restore";

void finalizeLLVMModule(llvm::Module &module) {
  module.setTargetTriple(llvm::Triple(kTargetTriple));
  module.setDataLayout(kTargetDataLayout);

  auto *i32Ty =
      llvm::IntegerType::getInt32Ty(module.getContext());
  // Module flags observed in the verified ABI goldens.
  module.addModuleFlag(llvm::Module::Error, "wchar_size", 4);
  module.addModuleFlag(llvm::Module::Error, "target-abi",
                       llvm::MDString::get(module.getContext(), "ilp32"));
  module.addModuleFlag(llvm::Module::Error, "SmallDataLimit", 8);

  llvm::Function *kernel = nullptr;
  for (llvm::Function &fn : module.functions()) {
    if (fn.isDeclaration())
      continue;
    if (!kernel && fn.hasExternalLinkage())
      kernel = &fn;
    fn.addFnAttr("target-cpu", kTargetCpu);
    fn.addFnAttr("target-features", kTargetFeatures);
    fn.addFnAttr(llvm::Attribute::getWithVScaleRangeArgs(
        module.getContext(), 1, 2048));
  }

  if (!kernel)
    llvm::report_fatal_error(
        "Ventus finalize: no externally linked kernel function found");

  // The `ventus_kernel` calling convention is NOT set here: the consumer
  // LLVM numbers that convention differently (its CC 104 is amdgpu_cs_chain
  // and prints as that keyword). The keyword is injected on the textual IR
  // by the backend's llir stage instead.

  // OpenCL kernel argument metadata, mirroring the ABI goldens: every pointer
  // argument lives in the global address space, everything else is a plain
  // scalar.
  llvm::LLVMContext &ctx = module.getContext();
  llvm::SmallVector<llvm::Metadata *> addrSpaces, accessQuals, types,
      typeQuals;
  for (const llvm::Argument &arg : kernel->args()) {
    uint32_t space = 0;
    if (auto *ptrTy =
            llvm::dyn_cast<llvm::PointerType>(arg.getType()))
      space = ptrTy->getAddressSpace();
    addrSpaces.push_back(llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(i32Ty, space)));
    accessQuals.push_back(
        llvm::MDString::get(ctx, "none"));
    std::string typeName;
    llvm::raw_string_ostream os(typeName);
    arg.getType()->print(os);
    types.push_back(llvm::MDString::get(ctx, os.str()));
    typeQuals.push_back(llvm::MDString::get(ctx, ""));
  }
  auto attach = [&](StringRef name, llvm::ArrayRef<llvm::Metadata *> values) {
    kernel->setMetadata(name, llvm::MDNode::get(ctx, values));
  };
  attach("kernel_arg_addr_space", addrSpaces);
  attach("kernel_arg_access_qual", accessQuals);
  attach("kernel_arg_type", types);
  attach("kernel_arg_base_type", types);
  attach("kernel_arg_type_qual", typeQuals);
}

} // namespace mlir::triton::ventus
