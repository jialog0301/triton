# Ventus P3.4 Histogram Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement correct single-CTA Ventus V1 `tl.histogram` lowering for `int32` input/output without LDS atomics.

**Architecture:** Reserve a backend-owned LDS region containing one `num_bins`-element `int32` slice per lane. Each lane performs ordinary load/add/store updates to its private slice, synchronizes with a CTA barrier, and sums all lane slices for the result. Cross-CTA histogram lowering is rejected explicitly.

**Tech Stack:** Triton MLIR lowering, MLIR LLVM dialect, Ventus LDS (`addrspace(3)`), Spike launcher, pytest.

---

### Task 1: Add the failing Ventus histogram regression test

**Files:**
- Create: `python/test/unit/ventus/test_histogram.py`
- Modify: `third_party/ventus/backend/launcher.py`

**Step 1: Add a kernel and host launch plan**

Define a `tl.histogram` kernel with an `int32` input, `int32` output, a mask,
and a fixed single-CTA launch. The host plan should provide repeated valid
bins, at least one out-of-range value, and masked-off values, then compare all
output bins against a Python reference.

**Step 2: Add compile and execution assertions**

Compile for `GPUTarget("ventus", "ventus-gpgpu", 32)`, assert the generated
LLIR has no `atomicrmw`, and execute through Spike. Assert
`result["num_mismatches"] == 0`.

**Step 3: Run the test to verify it fails**

Run:

```bash
source .venv/bin/activate
export TRITON_HOME="$PWD/.triton-home"
export PYTHONPATH="$PWD/python"
python -m pytest python/test/unit/ventus/test_histogram.py -q -s --tb=short
```

Expected: the test fails because the current lowering emits LDS `atomicrmw`
and the Spike result contains zero counts.

### Task 2: Add backend LDS reservation for histogram slices

**Files:**
- Modify: `third_party/ventus/lib/TritonVentusToLLVM/TritonGPUToLLVM.cpp`

**Step 1: Detect the supported histogram shape**

Walk the module for `triton::HistogramOp`. Require one histogram operation,
single-CTA operation, and an `int32` result. For unsupported cross-CTA or
incompatible shapes, emit a diagnostic/fatal backend error rather than
producing incorrect code.

**Step 2: Reserve and record LDS space**

Extend the existing `createSharedMemoryGlobal` sizing logic with a histogram
region after the core allocation and warp scratch region. Reserve
`lanes_per_workgroup * num_bins * sizeof(i32)` bytes and record its byte offset
and bin count in module attributes. Preserve all existing LDS reservations.

**Step 3: Run the compile-only test**

Run the focused histogram test and inspect the LLIR. Expected: the shared
global size includes the histogram region and the test still fails at runtime,
but the failure is now in the new lowering path rather than the old atomic
path.

### Task 3: Replace atomic updates with lane-private LDS updates

**Files:**
- Modify: `lib/Conversion/TritonGPUToLLVM/HistogramOpToLLVM.cpp`

**Step 1: Add lane-slice address calculation**

Use the current lane id and the recorded histogram region offset to form the
pointer for `lane_id * num_bins + bin`. Use the existing shared load/store
target hooks so accesses remain ordinary `addrspace(3)` operations.

**Step 2: Initialize and update slices**

Initialize each lane's slice to zero. For every source element, retain the
existing bounds, thread predicate, and mask conditions; then load the selected
bin, add one, and store it back to the current lane's private slice.

**Step 3: Synchronize and sum slices**

Insert the existing local barrier after updates. For each output bin, load the
same bin from every lane slice, sum the values, and pack the result using the
existing layout/index machinery.

**Step 4: Run the focused test**

Run the test from Task 1. Expected: it passes with zero mismatches and the LLIR
contains no `atomicrmw` or `amoadd.w` for histogram updates.

### Task 4: Document the supported P3.4 boundary

**Files:**
- Modify: `third_party/ventus/README.md`

**Step 1: Document the implementation and limitation**

Add a P3.4 subsection describing lane-private LDS slices, the barrier, the
single-CTA/int32 boundary, and the deferred cross-CTA case.

**Step 2: Run the relevant Ventus suite**

Run:

```bash
source .venv/bin/activate
export TRITON_HOME="$PWD/.triton-home"
export PYTHONPATH="$PWD/python"
python -m pytest python/test/unit/ventus/ -q -s --tb=short
```

Expected: the histogram regression and all existing Ventus tests pass, apart
from any pre-existing toolchain identity mismatch documented in the worktree.

**Step 3: Inspect the final diff**

Run `git status --short`, `git diff --check`, and review the changed files
before committing the implementation.
