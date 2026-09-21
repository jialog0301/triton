# Ventus Environment Facility Synchronization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Synchronize Triton-for-Ventus planning documents with the committed facilities and separately identified local experiments in the inspected `ventus-env` checkout.

**Architecture:** Create one evidence-based facility snapshot as the volatile source of truth, then make narrow corrections in the existing architecture, implementation, and learning documents. Keep committed component revisions separate from dirty Driver/CycleSim experiments and preserve the distinction between current facilities and future Triton/IREE work.

**Tech Stack:** Markdown, Git, Triton planning documents, Ventus LLVM/POCL/Driver/GPGPU/CycleSim source references.

---

### Task 1: Add The Ventus Facility Snapshot

**Files:**
- Create: `docs/plans/2026-08-25-ventus-env-facility-snapshot.md`

**Step 1: Record repository identities**

Record the inspected `ventus-env` commit, component commits, and dirty state.

**Step 2: Record compiler and ABI facilities**

Document target identity, data layout, calling convention, argument ABI,
address spaces, work-item builtins, barriers, divergence ownership, and the raw
resource section.

**Step 3: Record runtime and hardware facilities**

Document POCL metadata layers, driver backends, default RTL profile, CycleSim
differences, experimental tensor support, and unsupported facilities.

**Step 4: Record validation facilities**

Document ABI goldens, regression tests, OpenCL CTS, and existing AI operator
testcases.

### Task 2: Correct The Architecture Documents

**Files:**
- Modify: `docs/plans/2026-08-18-ventus-ml-inference-architecture-design.md`
- Modify: `docs/plans/2026-08-20-triton-ventus-iree-shared-mlir-design.md`

**Step 1: Mark superseded decisions**

Clarify that the August 18 document is historical where it conflicts with the
August 20 shared-target design.

**Step 2: Correct current facility assumptions**

Add the LLVM version boundary, simulator-only runtime status, GVM, existing
barrier intrinsics, libclc ID representation, artifact metadata layers, and
experimental tensor support.

### Task 3: Correct The Implementation Plan

**Files:**
- Modify: `docs/plans/2026-08-20-triton-ventus-iree-implementation.md`

**Step 1: Expand reproducibility identity**

Replace the single simulator identity with per-component identities and dirty
state handling.

**Step 2: Update Triton integration mechanics**

Use the current external plugin and `GPUTarget` model instead of requiring
initial core registration changes.

**Step 3: Add prerequisite hardening**

Make LLVM-version resolution, resource consumption, centralized launch ABI,
capability queries, and explicit ELF loading prerequisites.

### Task 4: Correct The Learning Roadmap

**Files:**
- Modify: `docs/plans/2026-08-23-triton-for-ventus-learning-roadmap.md`

**Step 1: Qualify the baseline profile**

Describe warp size 32, maximum eight warps, and maximum 256 work-items as the
pinned default RTL profile rather than universal properties.

**Step 2: Correct hardware and simulator capabilities**

Describe `VFTTA_VV` as experimental existing hardware, atomics and subgroup
collectives as unsupported, and CycleSim metrics as limited.

**Step 3: Add current validation assets**

Reference `regression-test.py`, OpenCL CTS, and existing AI operator kernels.

### Task 5: Verify Documentation Consistency

**Files:**
- Verify: `docs/plans/*.md`

**Step 1: Search stale terminology**

Search for claims that tensor hardware does not exist, physical hardware is a
current backend, resources are versioned or consumed, and stable ID intrinsics
already exist.

**Step 2: Review the diff**

Run `git diff -- docs/plans` and confirm all changed statements are supported by
the facility snapshot.

**Step 3: Verify scope**

Run `git status --short` and confirm no source or generated files were changed.
