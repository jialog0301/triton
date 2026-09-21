# Triton-for-Ventus V1 Documentation Synchronization Design

> **Status:** Completed historical synchronization record. Verification was
> performed for the documentation-sync change; this file is not active V1 or
> architecture authority. See the
> [final V1 scope](2026-08-28-triton-for-ventus-v1-scope.md) and the completed
> [sync implementation record](2026-08-28-triton-for-ventus-v1-doc-sync.md).

## Purpose

Make the Triton-for-Ventus planning documents consistently describe the first
version as containing both a general FP32 correctness path and a fixed-profile
Ventus Tensor Core path.

## Authoritative V1 Scope

Add one authoritative V1 specification covering:

- A selectable Ventus Triton backend that produces Ventus ELF artifacts.
- RV32, `i1`, `i32`, and `f32` support.
- One-dimensional SPMD builtins and 32-lane warp execution.
- Basic blocked layouts, elementwise operations, pointer arithmetic, masked
  global loads and stores.
- Static AS3 shared memory and full work-group barriers.
- A correctness-first shared-memory reduction subset.
- Ordinary blocked FP32 FMA with mandatory canonical `32x32x32` case as an
  accepted-set reference/fallback, not a general matrix or `tl.dot` path;
  additional static blocked patterns require explicit lowering and correctness
  tests before entering the accepted set.
- One fixed FP32 MMA profile matching the default 32-lane RTL:
  logical `A[M,K]=[4,8] x B[K,N]=[8,4] + C[M,N]=[4,4]`, with physical
  `B^T[N,K]=[4,8]`, emitted as `vftta.vv`.
- Kernel ABI, ELF resource metadata, artifact identity, reference launching,
  intermediate artifacts, capability checks, and deterministic diagnostics.

MMA support means one constrained vertical slice, not general `tl.dot` support.
It requires a contract-freeze milestone that aligns mandatory Spike and RTL
results, plus CycleSim when that backend capability is available, with the RTL
shape and lane-fragment semantics.

## Documents To Synchronize

- `2026-08-23-triton-for-ventus-learning-roadmap.md`: update first-stage
  constraints, milestones, matmul sequence, Tensor Facility section, capability
  matrix, and completion criteria.
- `2026-08-20-triton-ventus-iree-shared-mlir-design.md`: permit a narrowly scoped
  Ventus MMA encoding/operation and retain standard dialects for all ordinary
  arithmetic and memory semantics.
- `2026-08-20-triton-ventus-iree-implementation.md`: add MMA contract, LLVM
  codegen, Triton layout/lowering, metadata, and end-to-end tests to baseline
  implementation tasks.
- `2026-08-25-ventus-env-facility-snapshot.md`: preserve implementation facts,
  but change the project-planning recommendation and Initial Triton Profile to
  include the fixed RTL MMA vertical slice.

The older complete-model architecture document remains focused on IREE and only
receives a cross-reference if necessary; its model deployment scope is not
redefined by this Triton operator milestone.

## Diagram Changes

Update both Draw.io diagrams so that fixed FP32 MMA is shown as a V1 core path,
not a future exclusion. Keep low precision MMA, arbitrary MMA shapes, masked
MMA, async copy, shuffle/ballot, atomics, clusters, warp specialization, and the
physical hardware driver in the future/unsupported area.

## Validation

- Search all synchronized documents for stale claims that V1 excludes the fixed
  MMA slice or requires only ordinary FMA; preserve the distinction between the
  fixed profile, the `32x32x32` reference/fallback, and general `tl.dot`.
- Confirm every remaining statement distinguishes fixed-profile V1 MMA from
  general or low-precision MMA.
- Run Draw.io structural validation on both diagrams.
- Re-export PNG and editable SVG outputs.
- Review the final diff to ensure facility facts were not rewritten as already
  completed compiler support.
