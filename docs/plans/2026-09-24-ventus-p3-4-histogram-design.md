# Ventus P3.4 Single-CTA Histogram Design

## Goal

Support `tl.histogram` on Ventus V1 for single-CTA kernels with `int32`
inputs and `int32` output bins, without relying on shared-memory atomics.

## Scope

The first supported shape is a normal Triton `tl.histogram(input, num_bins)`
where the input is one-dimensional `int32`, the result is `int32`, and the
kernel executes one CTA at a time. Valid input values in `[0, num_bins)` are
counted; out-of-range values and masked-off values are ignored. Cross-CTA
combination is deferred and must not silently be treated as supported.

## Design

The lowering uses a private LDS histogram slice for each lane. Each lane
initializes and updates only its own slice using ordinary LDS load/add/store
operations. A CTA barrier makes all slices visible. Each lane then reads the
per-lane slice for the output bins and sums the slices into the result tensor.
This avoids the currently unreliable `atomicrmw`/`amoadd.w` path in LDS while
preserving the histogram semantics for all lanes.

The backend reserves the additional LDS bytes after the core shared-memory
allocation and records the offset in a module attribute. The lowering rejects
cross-CTA histogram lowering until a separate implementation exists.

## Verification

Add a Ventus unit test that compiles and executes a single-CTA histogram on
Spike. The test covers repeated values, out-of-range values, and a mask, and
asserts that the emitted LLIR contains no `atomicrmw` for the histogram and
that execution has zero mismatches. Existing Ventus tests remain the
regression suite for LDS, barriers, reductions, gathers, and divergence.
