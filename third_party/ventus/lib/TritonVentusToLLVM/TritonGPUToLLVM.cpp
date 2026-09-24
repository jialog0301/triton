// TritonGPU -> LLVM dialect conversion for the Ventus backend (V1).
//
// Mirrors the third-party driver pattern used by the NVIDIA and AMD backends:
// the driver lives outside Triton core and injects target differences through
// a TargetInfoBase implementation. The implemented scope is the non-MMA
// baseline: elementwise and masked load/store on riscv32, warp size 32, global
// address space 1, the CTA's shared (LDS) window in address space 3, and the
// work-group barrier. Anything outside that scope fails compilation loudly
// instead of silently producing wrong code.
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
#include "triton/Analysis/Utility.h"
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

// Shared with the work-item mapping patterns in this file and declared in the
// header so other translation units can reach it too.
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

//===----------------------------------------------------------------------===//
// Shared memory (LDS) and work-group barrier emission
//===----------------------------------------------------------------------===//

// The shared address space is 3 both in LLVM and in the toolchain's ABI
// goldens; the name the core `getStackPointer` helper looks up for the base of
// the CTA's shared window.
constexpr unsigned kSharedAddressSpace = 3;
constexpr StringLiteral kSharedMemoryGlobal = "global_smem";
// Set by the core `allocate-shared-memory` pass: the total shared bytes the
// kernel needs, and the per-op offset into them.
constexpr StringLiteral kSharedMemorySizeAttr = "ttg.shared";

// P3 warp-exchange scratch. This target has no shuffle primitive
// (Ventus-GPGPU-doc: no lane-to-lane data movement), so the warp-local
// reduction tree and the warp-local gather move their values through LDS
// instead. The core allocator knows nothing about that traffic (a reduction's
// intra-warp part produces no `tt.reduce` scratch), so the backend reserves its
// own region after the core allocation and records its start in a module attr
// that the shuffle hooks read back.
//
// Each warp gets `kPerWarpScratchBytes`; 32 lanes x 8 bytes covers the widest
// element (`f64`/pointer) the V1 element set can reach. One value at a time
// occupies the slot: each hook does store -> barrier -> load -> barrier, so a
// later store (the next bit, or the next operand) cannot race an earlier load.
constexpr unsigned kPerWarpScratchBytes = 32 * 8;
constexpr StringLiteral kWarpScratchOffsetAttr = "ventus.warp_scratch_offset";
constexpr StringLiteral kHistogramOffsetAttr = "ventus.histogram_offset";

// The Ventus barrier is `barrier x0, x0, <imm>`, a custom-0 R-type
// instruction whose 5-bit immediate is a fence bitmask
// (`VentusInstrInfo.td`: `BarrierBase`, func7 0b0000010). The mapping is read
// off the pinned clang: OpenCL's `barrier(CLK_LOCAL_MEM_FENCE)` lowers to
// `llvm.riscv.ventus.barrier(i32 1)`, `CLK_GLOBAL_MEM_FENCE` to `2`, and the
// two combined to `3`. Triton's AddrSpace mask is wider than those two bits,
// so local requests fold onto bit 0 and anything global onto bit 1 (the
// tensor-memory spaces have no Ventus counterpart yet).
unsigned barrierImmediate(triton::gpu::AddrSpace targets) {
  unsigned bits = static_cast<unsigned>(targets);
  unsigned imm = 0;
  if (bits & static_cast<unsigned>(triton::gpu::AddrSpace::Local))
    imm |= 1;
  if (bits & (static_cast<unsigned>(triton::gpu::AddrSpace::GlobalRead) |
              static_cast<unsigned>(triton::gpu::AddrSpace::GlobalWrite)))
    imm |= 2;
  return imm;
}

// Emit the work-group rendezvous.
//
// The barrier is INLINE ASM rather than a call to
// `llvm.riscv.ventus.barrier`: that intrinsic exists only in the pinned
// Ventus LLVM 16, while the consumer-side LLVM validates every declaration
// against its own builtin table and would reject it (toolchain README,
// "Textual IR Boundary Facts"). The instruction itself is unambiguous, so a
// side-effecting, operand-less asm block emits exactly the same machine
// instruction and is opaque to both compilers. `opt -passes=verify` and `llc`
// in the elf stage check that the toolchain still accepts it.
//
// Semantics: every warp of the work-group waits until all warps that have not
// finished executing have reached the barrier
// (`gpgpu/docs/Ventus-GPGPU-doc.md`, custom instruction 2;
// `pipeline/warp_schedule.scala` tracks the per-work-group arrival mask).
// That is the CTA-wide rendezvous Triton asks for.
void emitBarrier(RewriterBase &rewriter, Location loc,
                 triton::gpu::AddrSpace targets) {
  MLIRContext *ctx = rewriter.getContext();
  std::string asmString =
      ("barrier x0, x0, " + llvm::Twine(barrierImmediate(targets))).str();
  LLVM::InlineAsmOp::create(
      rewriter, loc, /*resultTypes=*/TypeRange{},
      /*operands=*/ValueRange{}, asmString,
      /*constraints=*/"", /*has_side_effects=*/true,
      /*is_align_stack=*/false, LLVM::TailCallKind::None,
      LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
      /*operand_attrs=*/ArrayAttr::get(ctx, {}));
}

// Shared (LDS) elements are addressed by a single pointer into the CTA's
// window; a packed vector operand means "these elements are contiguous", not
// "use a wide instruction". The emitter therefore splits vectors into
// elements: the caller's promise is alignment to the width of `val` (see
// `TargetInfoBase`), which is only reproducible element by element here, and
// the LDS window is a small on-chip SRAM where a misaligned wide access is
// simply not worth the risk. This is the V1 trade-off; nothing downstream
// depends on the access width.
SmallVector<Value> splitElements(Location loc, Value val,
                                 RewriterBase &rewriter) {
  if (!isa<VectorType>(val.getType()))
    return {val};
  return unpackLLVector(loc, val, rewriter);
}

// Rebuild the value the caller asked for. `expectVector` matters for `<1 x T>`:
// `unpackLLVector` on a scalar returns the scalar itself, so a one-element
// vector would otherwise come back as a bare `T` and fail the caller's type
// expectations.
Value packElements(Location loc, ArrayRef<Value> vals, bool expectVector,
                   RewriterBase &rewriter) {
  if (!expectVector && vals.size() == 1)
    return vals.front();
  return packLLVector(loc, vals, rewriter);
}

Value zeroOfType(Location loc, Type ty, RewriterBase &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (auto intTy = dyn_cast<IntegerType>(ty))
    return LLVM::ConstantOp::create(rewriter, loc, ty,
                                    rewriter.getIntegerAttr(intTy, 0));
  if (auto floatTy = dyn_cast<FloatType>(ty))
    return LLVM::ConstantOp::create(
        rewriter, loc, ty,
        rewriter.getFloatAttr(floatTy,
                              APFloat(floatTy.getFloatSemantics(), 0)));
  return b.undef(ty);
}

// The width of one element of an LDS access, in bytes. `elemTy` comes from a
// shared-memory descriptor and can be a pointer, which has no integer width.
unsigned elemBytesOf(Type elemTy) {
  if (isa<LLVM::LLVMPointerType>(elemTy))
    return 4; // riscv32
  return std::max(1u, elemTy.getIntOrFloatBitWidth() / 8);
}

void emitElementStore(RewriterBase &rewriter, Location loc, Value ptrElem,
                      Value val, Value pred) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned elemBytes = elemBytesOf(val.getType());
  if (!pred) {
    auto stored = LLVM::StoreOp::create(rewriter, loc, val, ptrElem);
    stored.setAlignment(elemBytes);
    return;
  }
  // `pred ? store : nothing` as an llvm.cond_br diamond, the same idiom the
  // global load/store lowering uses (masked intrinsics cannot cross the two
  // LLVM revisions).
  Block *currentBlock = rewriter.getInsertionBlock();
  Region *region = currentBlock->getParent();
  Block *joinBlock = currentBlock->splitBlock(rewriter.getInsertionPoint());
  Block *storeBlock = rewriter.createBlock(region, Region::iterator(joinBlock));

  rewriter.setInsertionPointToEnd(currentBlock);
  LLVM::CondBrOp::create(rewriter, loc, pred, storeBlock, ValueRange{},
                         joinBlock, ValueRange{});
  rewriter.setInsertionPointToEnd(storeBlock);
  auto stored = LLVM::StoreOp::create(rewriter, loc, val, ptrElem);
  stored.setAlignment(elemBytes);
  LLVM::BrOp::create(rewriter, loc, joinBlock);
  rewriter.setInsertionPointToStart(joinBlock);
}

// One LDS store: `val` is written to `ptr` when `pred` holds. Packed vectors
// are split (see `splitElements`).
void emitSharedStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
                     Value pred) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> vals = splitElements(loc, val, rewriter);
  if (vals.size() == 1) {
    emitElementStore(rewriter, loc, ptr, vals.front(), pred);
    return;
  }
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
  Type elemTy = vals.front().getType();
  for (auto [i, v] : llvm::enumerate(vals)) {
    Value ptrElem = b.gep(ptrTy, elemTy, ptr, b.i32_val(i));
    emitElementStore(rewriter, loc, ptrElem, v, pred);
  }
}

// One LDS load of `loadTy` from `ptr`. A false predicate yields zero: the
// callers only ask for a guarded value and discard it in that case (matching
// how the AMD driver handles the same hook).
Value emitSharedLoad(RewriterBase &rewriter, Location loc, Value ptr,
                     Type loadTy, Value pred) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type elemTy = loadTy;
  SmallVector<Value> results;
  if (auto vecTy = dyn_cast<VectorType>(loadTy)) {
    elemTy = vecTy.getElementType();
    results.reserve(vecTy.getNumElements());
  }
  unsigned elemBytes = elemBytesOf(elemTy);
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());

  unsigned numElems =
      isa<VectorType>(loadTy) ? cast<VectorType>(loadTy).getNumElements() : 1u;
  for (unsigned i = 0; i < numElems; ++i) {
    Value ptrElem = i == 0 ? ptr : b.gep(ptrTy, elemTy, ptr, b.i32_val(i));
    auto loaded = LLVM::LoadOp::create(rewriter, loc, elemTy, ptrElem);
    loaded.setAlignment(elemBytes);
    Value v = loaded.getRes();
    if (pred)
      v = b.select(pred, v, zeroOfType(loc, elemTy, rewriter));
    results.push_back(v);
  }
  return packElements(loc, results, /*expectVector=*/isa<VectorType>(loadTy),
                      rewriter);
}

// The base of the CTA's shared window, as this toolchain wants to see it.
//
// Core lowering reaches shared memory through `LLVM::getStackPointer`, which
// resolves the module symbol `global_smem` and offsets into it. NVIDIA and AMD
// declare that symbol with external linkage and size 0, because their runtime
// binds a dynamic base. Ventus has no such runtime hook: the toolchain places
// an `addrspace(3)` object inside the work-group's on-chip window and
// addresses it off the base register `crt0.o` sets up (entry does
// `addi s0, s0, <size>`, accesses go to `s0 - <size>` + offset), so the size
// written here is the size of that object. Sizing it from `ttg.shared` keeps
// it equal to the allocation the launcher must pass as `ldsSize`, by
// construction; the assembler reports the same number (plus its 4-byte
// alignment pad) in the VRES record's `lds` field, which is what the launcher
// actually reads.
//
// A kernel with no shared memory gets no object at all: nothing can reference
// it, and a missing symbol makes the core's own lookup fail loudly instead of
// quietly addressing a zero-byte window.
void createSharedGlobal(ModuleOp mod, LLVMTypeConverter &typeConverter,
                        uint64_t size) {
  OpBuilder b(mod.getBodyRegion());
  Type elemTy = typeConverter.convertType(b.getIntegerType(8));
  // 16-byte alignment: 4 x i32 is the widest access shared lowering produces,
  // the same bound NVIDIA and AMD request.
  auto arrayTy = LLVM::LLVMArrayType::get(elemTy, size);
  LLVM::GlobalOp::create(b, mod.getLoc(), arrayTy, /*isConstant=*/false,
                         LLVM::Linkage::Internal, kSharedMemoryGlobal,
                         /*value=*/LLVM::ZeroAttr::get(mod.getContext()),
                         /*alignment=*/16, kSharedAddressSpace);
}

void createSharedMemoryGlobal(ModuleOp mod, LLVMTypeConverter &typeConverter) {
  auto sizeAttr = mod->getAttrOfType<IntegerAttr>(kSharedMemorySizeAttr);
  if (!sizeAttr)
    llvm::report_fatal_error(
        llvm::Twine("Ventus lowering (V1): module has no ") +
        StringRef(kSharedMemorySizeAttr) +
        " attribute (run allocate-shared-memory first)");
  uint64_t coreSize = sizeAttr.getValue().getZExtValue();

  // P3: reserve the warp-exchange region. The core allocator only sizes the
  // cross-warp part of a reduction (and cross-warp gathers); the intra-warp
  // tree moves values lane to lane, and this target has no shuffle, so those
  // bytes have to be added here. The region starts right after the core
  // allocation, and its start is recorded for the shuffle hooks to find.
  // A reduction needs the exchange only when its axis is spread across lanes
  // (`threadsPerWarp[axis] > 1`); a gather needs it only on the warp-local
  // path, which is exactly what `isWarpLocal` decides.
  bool needsWarpScratch = false;
  mod.walk([&](triton::GatherOp gatherOp) {
    if (GatherLoweringHelper(gatherOp).isWarpLocal())
      needsWarpScratch = true;
  });
  mod.walk([&](triton::ReduceOp reduceOp) {
    auto srcTy = cast<RankedTensorType>(reduceOp.getOperand(0).getType());
    auto threadsPerWarp =
        triton::gpu::getThreadsPerWarp(srcTy.getEncoding(), srcTy.getShape());
    if (threadsPerWarp[reduceOp.getAxis()] > 1)
      needsWarpScratch = true;
  });
  uint64_t numWarps = triton::gpu::lookupNumWarps(mod);
  uint64_t size = coreSize;
  if (needsWarpScratch) {
    mod->setAttr(
        kWarpScratchOffsetAttr,
        IntegerAttr::get(IntegerType::get(mod.getContext(), 32), size));
    size += uint64_t(kPerWarpScratchBytes) * numWarps;
  }

  std::optional<int64_t> histogramBins;
  mod.walk([&](triton::HistogramOp histogram) {
    if (histogramBins)
      llvm::report_fatal_error("Ventus V1 supports one histogram operation per kernel");
    auto resultType = histogram.getType();
    if (!resultType || !resultType.getElementType().isInteger(32))
      llvm::report_fatal_error("Ventus V1 histogram requires int32 output bins");
    histogramBins = resultType.getDimSize(0);
  });
  if (histogramBins) {
    mod->setAttr(kHistogramOffsetAttr,
                 IntegerAttr::get(IntegerType::get(mod.getContext(), 32), size));
    size += uint64_t(*histogramBins) * 4 * numWarps *
            triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  }
  if (size == 0)
    return;
  mod->setAttr(kSharedMemorySizeAttr,
               IntegerAttr::get(IntegerType::get(mod.getContext(), 32), size));
  createSharedGlobal(mod, typeConverter, size);
}

//===----------------------------------------------------------------------===//
// P3 warp-local exchange through LDS
//===----------------------------------------------------------------------===//

// The address of `lane`'s slot in the reserved warp-exchange region. The region
// lives after the core allocation (see createSharedMemoryGlobal), with one
// `kPerWarpScratchBytes` slot per warp so that concurrently running warps never
// alias.
Value warpScratchSlot(RewriterBase &rewriter, Location loc, Type elemTy,
                      Value lane) {
  Operation *parent = rewriter.getInsertionBlock()->getParentOp();
  ModuleOp mod = parent->getParentOfType<ModuleOp>();
  FunctionOpInterface func = dyn_cast<FunctionOpInterface>(parent);
  if (!func)
    func = parent->getParentOfType<FunctionOpInterface>();
  auto offsetAttr = mod->getAttrOfType<IntegerAttr>(kWarpScratchOffsetAttr);
  if (!offsetAttr)
    llvm::report_fatal_error(
        "Ventus lowering (V1): lane exchange requested but "
        "no warp scratch was reserved");
  uint64_t offset = offsetAttr.getValue().getZExtValue();

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ptrTy =
      LLVM::LLVMPointerType::get(rewriter.getContext(), kSharedAddressSpace);
  Value warp = mlir::getLaneAndWarpId(rewriter, loc).second;
  Value base = LLVM::getStackPointer(rewriter, func);
  Value warpOff =
      b.add(b.i32_val(offset), b.mul(warp, b.i32_val(kPerWarpScratchBytes)));
  Value warpBase = b.gep(ptrTy, i8_ty, base, warpOff);
  Value laneByte = b.mul(lane, b.i32_val(elemBytesOf(elemTy)));
  return b.gep(ptrTy, i8_ty, warpBase, laneByte);
}

// Publish `val` for the current lane, rendezvous across the CTA, read the value
// `srcLane` published, rendezvous again so the slot is free for the next
// exchange. Two barriers per round-trip is the price of doing lane traffic
// through memory; the reduce tree and the warp-local gather are block-uniform,
// so a CTA-wide barrier is the correct (and only available) rendezvous.
Value exchangeThroughLds(RewriterBase &rewriter, Location loc, Value val,
                         Value srcLane) {
  Value lane = mlir::getLaneAndWarpId(rewriter, loc).first;
  unsigned elemBytes = elemBytesOf(val.getType());

  Value selfPtr = warpScratchSlot(rewriter, loc, val.getType(), lane);
  auto stored = LLVM::StoreOp::create(rewriter, loc, val, selfPtr);
  stored.setAlignment(elemBytes);
  emitBarrier(rewriter, loc, triton::gpu::AddrSpace::Local);

  Value srcPtr = warpScratchSlot(rewriter, loc, val.getType(), srcLane);
  auto loaded = LLVM::LoadOp::create(rewriter, loc, val.getType(), srcPtr);
  loaded.setAlignment(elemBytes);
  emitBarrier(rewriter, loc, triton::gpu::AddrSpace::Local);
  return loaded.getRes();
}

} // namespace

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
    return emitWorkItemBuiltinCall(rewriter, loc, moduleOp, kWorkGroupIdBuiltin,
                                   axisValue);
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

  void barrier(Location loc, RewriterBase &rewriter,
               triton::gpu::AddrSpace targets) const override {
    emitBarrier(rewriter, loc, targets);
  }

  void clusterBarrier(Location, RewriterBase &, Operation *) const override {
    unsupported("cluster barrier");
  }

  void warpSync(Location, RewriterBase &) const override {
    unsupported("warp sync");
  }

  void storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    Value ctaId, Value val, Value pred) const override {
    // Cross-CTA shared traffic needs a cluster, which one CTA per work-group
    // rules out (toolchain identity: cta_to_work_group one_to_one).
    if (ctaId)
      unsupported("cross-CTA shared memory transfer");
    emitSharedStore(rewriter, loc, ptr, val, pred);
  }

  Value loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    Value ctaId, Type elemTy, Value pred,
                    Operation *localLoadOp) const override {
    (void)localLoadOp;
    if (ctaId)
      unsupported("cross-CTA shared memory transfer");
    return emitSharedLoad(rewriter, loc, ptr, elemTy, pred);
  }

  // P3: this target has no lane-shuffle instruction (Ventus-GPGPU-doc), so the
  // four shuffle primitives are served from the reserved warp-exchange region
  // (see createSharedMemoryGlobal). The lane index maths mirrors the semantics
  // the core shuffles have: xor/up/idx with a warp-size clamp.
  Value shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value lane = mlir::getLaneAndWarpId(rewriter, loc).first;
    return exchangeThroughLds(rewriter, loc, val, b.xor_(lane, b.i32_val(i)));
  }
  Value shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                  int i) const override {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value lane = mlir::getLaneAndWarpId(rewriter, loc).first;
    // PTX `shfl.up` with clamp 0 keeps the caller's value when it would fall
    // below lane 0; reading the caller's own slot is exactly that value.
    Value shifted = b.sub(lane, b.i32_val(i));
    Value src = b.select(b.icmp_ult(lane, b.i32_val(i)), lane, shifted);
    return exchangeThroughLds(rewriter, loc, val, src);
  }
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    return exchangeThroughLds(rewriter, loc, val, b.i32_val(i & 0x1f));
  }
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   Value i) const override {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    return exchangeThroughLds(rewriter, loc, val, b.and_(i, b.i32_val(0x1f)));
  }

  Value permute(RewriterBase &, Location, Value, Value, Value) const override {
    unsupported("shuffle permute");
  }

  bool warpReduce(RewriterBase &, Location, SmallVector<Value> &,
                  triton::ReduceOp, unsigned) const override {
    // The generic reduction tree is used: it drives the intra-warp part through
    // `shuffleXor`, which is served from LDS above. No redux hardware here.
    return false;
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
    rewriter.replaceOpWithNewOp<arith::DivUIOp>(op, localId,
                                                b.i32_val(threadsPerWarp));
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
        mlir::convertFuncOpToLLVMFuncOp(funcOp, rewriter, *getTypeConverter());
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
// Work-group barrier lowering
//===----------------------------------------------------------------------===//

// `ttg.barrier` carries the fence mask of the synchronization it needs. The
// core lowering would route it through `gpu.barrier`, which drops the mask, so
// this backend lowers the op directly and hands the mask to the target. The
// pattern is registered above the core one (see the driver): the hardware
// instruction encodes the mask, so preserving it costs nothing.
struct VentusBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    emitBarrier(rewriter, op.getLoc(), op.getAddrSpace());
    rewriter.eraseOp(op);
    return success();
  }
};

// A GPU-dialect barrier can still reach this pass (an assert lowering or a
// gluon source may create one) and the GPU dialect is illegal here, so it has
// to be lowered too. There is no mask on it: `gpu.barrier` means a full
// work-group rendezvous with the memory ordering the model implies, which is
// the local + global fence.
struct VentusGpuBarrierOpConversion
    : public ConvertOpToLLVMPattern<mlir::gpu::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    emitBarrier(rewriter, op.getLoc(), triton::gpu::AddrSpace::All);
    rewriter.eraseOp(op);
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

    // The shared window has to exist before any pattern asks for its base:
    // `getStackPointer` resolves it by name.
    createSharedMemoryGlobal(mod, typeConverter);

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
          typeConverter, patterns, axisInfoAnalysis, benefit);
      mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                                   patterns, benefit);
      // Global tt.load/tt.store: backend-owned in Triton, see the V1
      // per-element lowering in VentusLoadStoreOpToLLVM.cpp.
      populateVentusLoadStoreOpToLLVMPatterns(typeConverter, patterns,
                                              axisInfoAnalysis, benefit);
      // Index generation via LinearLayout: shared with the NVIDIA/AMD
      // drivers (ttg::toLinearLayout + emitIndices), replacing the former
      // backend-private 1-D blocked formula.
      mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                     patterns, benefit);
      // View ops (expand_dims / broadcast / reshape / splat / ...): also
      // shared with NVIDIA and AMD. These make the rank-1 index result of
      // make_range usable as a 2-D tile index, and they bring the core splat
      // lowerings, so the backend carries none of its own.
      mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                                 benefit);
      // A reduction that spans warps stages each round through LDS, which is
      // expressed as a `ttg.convert_layout`. Without this registration the core
      // reduce lowering cannot materialize that staging and the op stays
      // illegal (the intra-warp part alone does not need it).
      mlir::triton::populateConvertLayoutOpToLLVMPatterns(
          typeConverter, targetInfo, patterns, benefit);
      // Reductions that span warps stage through LDS and rendezvous; the
      // intra-warp part drives the shuffle hooks, which this target serves from
      // its reserved warp-exchange region (see createSharedMemoryGlobal).
      mlir::triton::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                                   targetInfo, benefit);
      // A gather whose source spans warps moves its elements through the LDS
      // window: write, work-group barrier, indexed read. That is the shape
      // this milestone implements (the warp-local variants need register
      // shuffles, which fail loudly in the target's overrides).
      mlir::triton::populateGatherOpToLLVMPatterns(typeConverter, patterns,
                                                    targetInfo, benefit);
      mlir::triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
                                                      targetInfo, benefit);
      // Above the core `ttg.barrier` -> `gpu.barrier` pattern, so the fence
      // mask survives into the instruction's immediate.
      patterns.add<VentusBarrierOpConversion, VentusGpuBarrierOpConversion>(
          typeConverter, PatternBenefit(benefit + 1));
      mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);
      mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                       targetInfo, benefit);
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

  auto *i32Ty = llvm::IntegerType::getInt32Ty(module.getContext());
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
    fn.addFnAttr(
        llvm::Attribute::getWithVScaleRangeArgs(module.getContext(), 1, 2048));
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
  llvm::SmallVector<llvm::Metadata *> addrSpaces, accessQuals, types, typeQuals;
  for (const llvm::Argument &arg : kernel->args()) {
    uint32_t space = 0;
    if (auto *ptrTy = llvm::dyn_cast<llvm::PointerType>(arg.getType()))
      space = ptrTy->getAddressSpace();
    addrSpaces.push_back(
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32Ty, space)));
    accessQuals.push_back(llvm::MDString::get(ctx, "none"));
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
