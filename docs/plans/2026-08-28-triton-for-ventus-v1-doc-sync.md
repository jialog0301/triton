# Triton-for-Ventus V1 Documentation Synchronization Implementation Plan

> **Status:** Completed historical implementation record. The synchronization
> was reviewed and verified with consistency searches, diagram validation/export,
> and `git diff --check`. This file is not active authority. See the completed
> [sync design](2026-08-28-triton-for-ventus-v1-doc-sync-design.md) and the
> [final V1 scope](2026-08-28-triton-for-ventus-v1-scope.md).

**Goal:** Synchronize the Triton-for-Ventus planning documents and diagrams around a V1 that includes a fixed-profile FP32 MMA path.

**Architecture:** The synchronization added one authoritative V1 scope document,
then updated the learning roadmap, shared MLIR design, implementation plan, and
facility snapshot. The sync design/plan remain historical records rather than
members of the durable architecture authority chain.

**Tech Stack:** Markdown, Draw.io XML, draw.io CLI, validation scripts

---

### Task 1: Add The Authoritative V1 Specification

**Files:**
- Create: `docs/plans/2026-08-28-triton-for-ventus-v1-scope.md`

Document the complete V1 scope, fixed RTL Profile, supported operations,
MMA contract, fallback behavior, ABI/artifact requirements, milestones,
validation matrix, and explicit exclusions.

### Task 2: Synchronize The Planning Documents

**Files:**
- Modify: `docs/plans/2026-08-23-triton-for-ventus-learning-roadmap.md`
- Modify: `docs/plans/2026-08-20-triton-ventus-iree-shared-mlir-design.md`
- Modify: `docs/plans/2026-08-20-triton-ventus-iree-implementation.md`
- Modify: `docs/plans/2026-08-25-ventus-env-facility-snapshot.md`

Replace stale statements that V1 excludes the fixed MMA slice or requires only
ordinary FMA. Keep the facts that RTL Tensor Core support is experimental and
that the compiler/simulator contract still needs to be frozen. State that V1
includes only fixed FP32 `vftta.vv` support matching the default 32-lane RTL
profile, while ordinary FMA remains an accepted-set reference/fallback.

### Task 3: Update The Draw.io Diagrams

**Files:**
- Modify: `docs/plans/triton-for-ventus-architecture.drawio`
- Modify: `docs/plans/triton-for-ventus-lowering-architecture.drawio`
- Regenerate: matching PNG and editable SVG exports

Move fixed-profile FP32 MMA into the V1/core path and retain general MMA,
low-precision MMA, masked MMA, and other unimplemented facilities as future
capabilities. Preserve valid routing and the central lowering emphasis.

### Task 4: Verify Consistency

Run searches for stale MMA exclusions and verify that each remaining exclusion
refers to general/low-precision/unsupported MMA rather than the fixed V1 slice.
Run Draw.io validation on both source files and re-export both PNG and SVG
outputs. Inspect `git diff --stat` and `git diff --check`; do not commit.

**Verification status:** Completed for the synchronization revision. Later
documentation corrections are governed by the final V1 scope and current review
diff, not by reopening this historical task list.
