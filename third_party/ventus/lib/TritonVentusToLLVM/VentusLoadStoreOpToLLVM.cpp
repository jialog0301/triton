// V1 element lowering for the Ventus backend: global tt.load/tt.store and the
// floating-point elementwise set.
//
// Two facts about upstream Triton drive this file's shape:
//  * Global tt.load/tt.store lowering is backend-owned (the core
//    populateMemoryOpToLLVMPatterns only covers local/scratch memory), so
//    NVIDIA and AMD each carry their own LoadStoreOpToLLVM.cpp.
//  * The core elementwise registration covers the integer set only; every
//    backend re-registers the floating-point ops.
//
// Index generation is NOT here: tt.make_range goes through the core
// populateMakeRangeOpToLLVMPattern, i.e. ttg::toLinearLayout + emitIndices.
// That is the same path NVIDIA and AMD use, so index generation is layout-
// and rank-generic (any order, any rank, multiple warps) instead of a
// backend-private formula.
//
// Load/store themselves are layout-agnostic: they walk the unpacked
// per-thread element lists, whatever encoding produced them. Predicated
// accesses become an `llvm.cond_br` diamond rather than llvm.masked.* :
// the masked intrinsics changed arity between LLVM releases (the alignment
// operand was removed upstream) while the pinned Ventus LLVM 16 still
// requires it, so no single declaration satisfies both toolchains. Plain
// loads plus control flow mean the same thing in both. Vectorized masked
// accesses are a later optimization, not a correctness dependency.
#include "Target/TritonVentusToLLVM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::ventus {
namespace {

Value zeroOf(OpBuilder &rewriter, Location loc, Type llvmElemTy) {
  if (auto intTy = dyn_cast<IntegerType>(llvmElemTy))
    return LLVM::ConstantOp::create(rewriter, loc, llvmElemTy,
                                    rewriter.getIntegerAttr(intTy, 0));
  if (auto floatTy = dyn_cast<FloatType>(llvmElemTy))
    return LLVM::ConstantOp::create(
        rewriter, loc, llvmElemTy,
        rewriter.getFloatAttr(floatTy,
                              APFloat(floatTy.getFloatSemantics(), 0)));
  llvm::report_fatal_error("Ventus lowering (V1): bad element type");
}

// `pred ? load(ptr) : other`, as an llvm.cond_br diamond. The result arrives
// in the join block as a block argument.
Value emitPredicatedLoad(ConversionPatternRewriter &rewriter, Location loc,
                         Value ptrElem, Type llvmElemTy, Value pred,
                         Value otherVal, unsigned elemBytes) {
  Block *currentBlock = rewriter.getInsertionBlock();
  Region *region = currentBlock->getParent();
  Block *joinBlock = currentBlock->splitBlock(rewriter.getInsertionPoint());
  Block *loadBlock = rewriter.createBlock(region, Region::iterator(joinBlock));
  BlockArgument result = joinBlock->addArgument(llvmElemTy, loc);

  rewriter.setInsertionPointToEnd(currentBlock);
  LLVM::CondBrOp::create(rewriter, loc, pred, loadBlock, ValueRange{},
                         joinBlock, ValueRange{otherVal});

  rewriter.setInsertionPointToEnd(loadBlock);
  auto loaded = LLVM::LoadOp::create(rewriter, loc, llvmElemTy, ptrElem);
  loaded.setAlignment(elemBytes);
  LLVM::BrOp::create(rewriter, loc, loaded.getRes(), joinBlock);

  rewriter.setInsertionPointToStart(joinBlock);
  return result;
}

// `if (pred) store(ptr, val)`; no result.
void emitPredicatedStore(ConversionPatternRewriter &rewriter, Location loc,
                         Value ptrElem, Value val, Value pred,
                         unsigned elemBytes) {
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

// Local copy of the core's (private) trivial elementwise mapping: each tensor
// op becomes per-element scalar ops over the unpacked operands.
template <typename SourceOp, typename DestOp>
struct VentusElementwiseOpConversion
    : public gpu::ElementwiseOpConversionBase<
          SourceOp, VentusElementwiseOpConversion<SourceOp, DestOp>> {
  using Base = gpu::ElementwiseOpConversionBase<
      SourceOp, VentusElementwiseOpConversion<SourceOp, DestOp>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  SmallVector<DestOp> createDestOps(SourceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    Type elemTy,
                                    gpu::MultipleOperandsRange operands,
                                    Location loc) const {
    return {DestOp::create(rewriter, loc, elemTy, operands[0],
                           adaptor.getAttributes().getValue())};
  }
};

// Math ops the Ventus LLVM 16 cannot expand itself.
//
// The core converts `math::ExpOp` (and friends) to the same `math.*` op, which
// the dialect export turns into an `llvm.exp.f32`-style intrinsic. The pinned
// Ventus llc cannot lower that intrinsic on this target: `llc` crashes in
// SelectionDAG's `ExpandFPLibCall` -> `RISCVTargetLowering::LowerCall` while
// trying to materialise a libm call. Clang's own OpenCL output never hits this
// because it emits a direct call to the libclc-provided, Itanium-mangled symbol
// (`_Z3expf` for `exp(float)`, verified against the reference softmax's
// object0.riscv). This pattern reproduces that: it emits a plain `LLVM::CallOp`
// to the mangled symbol, which `opt`/`llc`/`ld.lld` all accept.
//
// It is registered with a higher benefit than the core pattern so it shadows
// `math::ExpOp -> math::ExpOp`; only the ops listed at registration need it.
// `symbol` is the libclc export for the fp32 variant.
template <typename SourceOp>
struct VentusLibcallOpConversion
    : public gpu::ElementwiseOpConversionBase<
          SourceOp, VentusLibcallOpConversion<SourceOp>> {
  using Base = gpu::ElementwiseOpConversionBase<
      SourceOp, VentusLibcallOpConversion<SourceOp>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  VentusLibcallOpConversion(StringRef symbol, LLVMTypeConverter &converter,
                            ModuleAxisInfoAnalysis &axisInfoAnalysis,
                            PatternBenefit benefit)
      : Base(converter, axisInfoAnalysis, benefit), symbol(symbol) {}

  SmallVector<Value> createDestOps(SourceOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy,
                                   gpu::MultipleOperandsRange operands,
                                   Location loc) const {
    auto funcTy = LLVM::LLVMFunctionType::get(elemTy, {elemTy});
    gpu::appendOrGetExternFuncOp(rewriter, op, symbol, funcTy);
    Operation *call =
        LLVM::CallOp::create(rewriter, loc, funcTy,
                             SymbolRefAttr::get(rewriter.getContext(), symbol),
                             operands[0]);
    return {call->getResult(0)};
  }

private:
  StringRef symbol;
};

struct VentusLoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp> {
  using ConvertOpToLLVMPattern<triton::LoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *typeConverter = getTypeConverter();

    // Judge on the original TTIR type: adapted tensor operands are packed
    // into LLVM structs.
    bool isTensor = isa<RankedTensorType>(op.getType());
    Type elemTy = isTensor
                      ? cast<RankedTensorType>(op.getType()).getElementType()
                      : op.getType();
    auto llvmElemTy = typeConverter->convertType(elemTy);
    unsigned elemBytes = std::max(1u, llvmElemTy.getIntOrFloatBitWidth() / 8);

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    SmallVector<Value> ptrs = isTensor ? unpackLLElements(loc, llPtr, rewriter)
                                       : SmallVector<Value>{llPtr};
    SmallVector<Value> masks;
    if (llMask)
      masks = isTensor ? unpackLLElements(loc, llMask, rewriter)
                       : SmallVector<Value>{llMask};
    SmallVector<Value> others;
    if (llOther)
      others = isTensor ? unpackLLElements(loc, llOther, rewriter)
                        : SmallVector<Value>{llOther};

    SmallVector<Value> results;
    for (auto [i, ptrElem] : llvm::enumerate(ptrs)) {
      Value pred = masks.empty() ? Value{} : masks[i];
      if (!pred) {
        auto loaded = LLVM::LoadOp::create(rewriter, loc, llvmElemTy, ptrElem);
        loaded.setAlignment(elemBytes);
        results.push_back(loaded.getRes());
        continue;
      }
      Value other =
          others.empty() ? zeroOf(rewriter, loc, llvmElemTy) : others[i];
      results.push_back(emitPredicatedLoad(rewriter, loc, ptrElem, llvmElemTy,
                                           pred, other, elemBytes));
    }

    Value out =
        packLLElements(loc, typeConverter, results, rewriter, op.getType());
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct VentusStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::StoreOp> {
  using ConvertOpToLLVMPattern<triton::StoreOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *typeConverter = getTypeConverter();

    bool isTensor = isa<RankedTensorType>(op.getValue().getType());
    Type elemTy =
        isTensor
            ? cast<RankedTensorType>(op.getValue().getType()).getElementType()
            : op.getValue().getType();
    auto llvmElemTy = typeConverter->convertType(elemTy);
    unsigned elemBytes = std::max(1u, llvmElemTy.getIntOrFloatBitWidth() / 8);

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getValue();
    Value llMask = adaptor.getMask();

    SmallVector<Value> ptrs = isTensor ? unpackLLElements(loc, llPtr, rewriter)
                                       : SmallVector<Value>{llPtr};
    SmallVector<Value> vals = isTensor ? unpackLLElements(loc, llVal, rewriter)
                                       : SmallVector<Value>{llVal};
    SmallVector<Value> masks;
    if (llMask)
      masks = isTensor ? unpackLLElements(loc, llMask, rewriter)
                       : SmallVector<Value>{llMask};

    for (auto [i, ptrElem] : llvm::enumerate(ptrs)) {
      Value pred = masks.empty() ? Value{} : masks[i];
      if (!pred) {
        auto stored = LLVM::StoreOp::create(rewriter, loc, vals[i], ptrElem);
        stored.setAlignment(elemBytes);
        continue;
      }
      emitPredicatedStore(rewriter, loc, ptrElem, vals[i], pred, elemBytes);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateVentusLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit) {
  patterns.add<VentusLoadOpConversion, VentusStoreOpConversion>(typeConverter,
                                                                benefit);

  // Floating-point ops are re-registered per backend; the core populate only
  // covers the integer set. Mirrors the NVIDIA additions.
  patterns.add<VentusElementwiseOpConversion<arith::AddFOp, LLVM::FAddOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<VentusElementwiseOpConversion<arith::SubFOp, LLVM::FSubOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<VentusElementwiseOpConversion<arith::MulFOp, LLVM::FMulOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<VentusElementwiseOpConversion<arith::DivFOp, LLVM::FDivOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<VentusElementwiseOpConversion<arith::RemFOp, LLVM::FRemOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<VentusElementwiseOpConversion<arith::ExtFOp, LLVM::FPExtOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<VentusElementwiseOpConversion<arith::TruncFOp, LLVM::FPTruncOp>>(
      typeConverter, axisInfoAnalysis, benefit);

  // Math ops the Ventus llc cannot expand (see VentusLibcallOpConversion).
  // Registered one benefit above the core `math.* -> math.*` patterns so they
  // shadow them; the symbol is the libclc Itanium-mangled fp32 export.
  PatternBenefit libcallBenefit(benefit.getBenefit() + 1);
  patterns.add<VentusLibcallOpConversion<math::ExpOp>>(
      "_Z3expf", typeConverter, axisInfoAnalysis, libcallBenefit);
  patterns.add<VentusLibcallOpConversion<math::Exp2Op>>(
      "_Z4exp2f", typeConverter, axisInfoAnalysis, libcallBenefit);
  patterns.add<VentusLibcallOpConversion<math::LogOp>>(
      "_Z3logf", typeConverter, axisInfoAnalysis, libcallBenefit);
  patterns.add<VentusLibcallOpConversion<math::Log2Op>>(
      "_Z4log2f", typeConverter, axisInfoAnalysis, libcallBenefit);
  patterns.add<VentusLibcallOpConversion<math::SinOp>>(
      "_Z3sinf", typeConverter, axisInfoAnalysis, libcallBenefit);
  patterns.add<VentusLibcallOpConversion<math::CosOp>>(
      "_Z3cosf", typeConverter, axisInfoAnalysis, libcallBenefit);
  patterns.add<VentusLibcallOpConversion<math::SqrtOp>>(
      "_Z4sqrtf", typeConverter, axisInfoAnalysis, libcallBenefit);
  patterns.add<VentusLibcallOpConversion<math::RsqrtOp>>(
      "_Z5rsqrtf", typeConverter, axisInfoAnalysis, libcallBenefit);
  patterns.add<VentusLibcallOpConversion<math::ErfOp>>(
      "_Z3erff", typeConverter, axisInfoAnalysis, libcallBenefit);
}

} // namespace mlir::triton::ventus
