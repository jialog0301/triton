# Ventus Byte Permute Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Ventus's register-local byte-permute target hook for packed within-warp layout conversions.

**Architecture:** Lower the NVIDIA-compatible selector nibble semantics to ordinary integer shifts, masks, and selects in the Ventus target hook. Leave core conversion and LDS exchange unchanged.

**Tech Stack:** Triton MLIR LLVM dialect, Gluon `convert_layout`, pytest and Spike.

---

### Task 1: Verify selector behavior and write failing regression

**Files:** `python/test/unit/ventus/test_byte_permute.py`

1. Confirm the core conversion's nibble values and the target's NVIDIA `prmt` contract.
2. Add a packed 8/16-bit layout-conversion kernel whose source and target differ across lane/register bits, and compare its output to input through the existing Spike test launcher. Assert the resulting TTGIR retains the conversion and that its LLIR contains the expected bit-level reconstruction.
3. Run `python -m pytest python/test/unit/ventus/test_byte_permute.py -q -s --tb=short`; confirm failure at the currently unsupported hook, not due to an unrelated frontend or environment error.

### Task 2: Implement minimal integer byte-permute hook

**Files:** `third_party/ventus/lib/TritonVentusToLLVM/TritonGPUToLLVM.cpp`

1. For each of four output bytes, extract the selected byte from `a` or `b`, respecting selector bit 3 if relevant, shift it into output position, and OR it into i32. Do not introduce lane-varying scalar values outside block tensor semantics.
2. Run `make` in the Triton worktree before native tests.
3. Re-run the focused test and verify the Spike result and LLVM IR. Add coverage for differing source bytes and the special sign selector if applicable.

### Task 3: Document and validate

**Files:** `third_party/ventus/README.md`

1. Document the register-local nature of the hook and the tested selector/element boundaries.
2. Run the Ventus suite with `python -m pytest python/test/unit/ventus/ -q -s --tb=short` and inspect `git diff --check` / `git status --short`; report the pre-existing toolchain identity mismatch separately.
