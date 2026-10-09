# Ventus P3 byte-permute design

## Scope

Implement the `TargetInfoBase::permute(a, b, selector)` hook used by the core within-warp `ttg.convert_layout` lowering. This is an **in-register byte permutation**, not a lane exchange or the tensor `tl.permute` API. Preserve the existing LDS implementation of `shuffleIdx` for lane transfers.

## Semantics and alternatives

The generic lowering emits NVIDIA `prmt` selectors: each of the four low selector nibbles picks one byte from the concatenation of two 32-bit operands. Use shifts, masks, and OR on i32 values to construct the four output bytes; apply a 3-bit index mask, and verify whether the high nibble bit has sign-replication meaning before settling on the final implementation. A target intrinsic would need toolchain changes; LDS round-trips would misinterpret an in-register operation and unnecessarily synchronize. Pure integer operations are portable across the existing Ventus LLVM boundary.

## Test and limits

First create a regression exercising the packed 8/16-bit `convert_layout` path and a focused selector semantic test (including source selection, byte ordering, and sign behavior if applicable); confirm the old hook fails. Rebuild after native changes and compare Spike output with the host reference. Keep changes scoped to the Ventus target hook; reject or defer selector forms that cannot be implemented faithfully. Existing reduction/gather and full Ventus tests must remain passing except for the known toolchain-identity snapshot mismatch.
