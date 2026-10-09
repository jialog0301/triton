# Ventus P4 Implementation Plan: textual-boundary cleanup and first-class builtins

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stop the ad-hoc growth of Triton-side/ Ventus-side textual LLVM IR boundaries: replace the one-off string hacks with a single tested normalization layer, add an end-to-end boundary sentinel that catches new boundaries before they hit real kernels, and lay out the (gated) LLVM/MLIR alignment path that makes the barrier and the work-item builtins first-class.

**Architecture:** All consumer-side-only textual features are funneled through a table-driven `normalize()` in `third_party/ventus/backend/textual.py`, applied once in `make_llir`. A pytest sentinel compiles a corpus of representative kernels end-to-end to ELF so any new consumer-emitted keyword/literal fails a test instead of a user's kernel. The barrier and shuffle emission points get explicit, documented extension seams so a future first-class builtin (or hardware shuffle) replaces exactly one function.

**Tech Stack:** Triton backend Python (`triton.backends.ventus.compiler`), MLIR LLVM dialect translation, pinned Ventus LLVM 16 (`opt`/`llc`/`ld.lld`), pytest. The consumer LLVM is pinned at `b010a18d2b648cab83c83967ff26b8fde11acdc6`; the Ventus toolchain is LLVM 16 `d4f2063fe81cbbefda34da60d4cd5c46bce3d231` (`cmake/llvm-info.json`, `toolchain/version.json`).

---

## 0. Prerequisite (do before Task 1)

Land the in-flight P3 work so the baseline is clean:

```bash
cd /home/weijiale/Code/cuda2rvv/triton/.worktrees/triton-ventus-v1
source .venv/bin/activate
export TRITON_HOME="$PWD/.triton-home" PYTHONPATH="$PWD/python"
cd "$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')" && ninja triton && cd -
python -m pytest python/test/unit/ventus/ -q --tb=short
```

Then commit the uncommitted byte-permute (`third_party/ventus/backend/compiler.py`, `launcher.py`, `lib/TritonVentusToLLVM/TritonGPUToLLVM.cpp`, `python/test/unit/ventus/test_byte_permute.py`, the two `docs/plans/2026-09-27-*` files) as its own commit. Do not start P4 on top of a dirty tree.

Also fix the stale README §8/§9 lines that still say warp-inner shuffle / histogram are undone (they landed in P3; §4.7 and the histogram commit are the record).

---

## Task 1: Boundary sentinel corpus + test

**Files:**
- Create: `python/test/unit/ventus/test_textual_boundary.py`

**Intent:** one test that compiles a corpus of representative kernels to ELF (through `opt -passes=verify` + `llc` + `ld.lld`). Today it must fail on the kernels that hit an unhandled consumer-side feature (softmax: `-inf`, `nocreateundeforpoison`). After Tasks 2-4 it must pass. This is the regression net for every future boundary.

**Step 1: Write the failing test**

```python
"""End-to-end sentinel for the Triton/Ventus textual LLVM IR boundary.

The Triton-side LLVM and the pinned Ventus LLVM 16 are different revisions, so
the emitted text can contain keywords/literals the Ventus parser predates. Each
such feature is handled in `triton.backends.ventus.textual`; this test compiles a
corpus that exercises the known features so a new one fails here, not in a user's
kernel. Compile-only: no simulator needed.
"""

import importlib
from pathlib import Path

import pytest

import triton
import triton.backends as triton_backends
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import compile as triton_compile

REPO_ROOT = Path(__file__).resolve().parents[4]
VENTUS_ROOT = REPO_ROOT / "third_party/ventus"
TARGET = GPUTarget("ventus", "ventus-gpgpu", 32)


@pytest.fixture
def ventus_backend(monkeypatch, tmp_path):
    isolated = tmp_path / "triton" / "backends"
    isolated.mkdir(parents=True)
    (isolated / "ventus").symlink_to(VENTUS_ROOT / "backend", target_is_directory=True)
    monkeypatch.setattr(triton_backends, "__path__", [str(isolated), *triton_backends.__path__])
    importlib.invalidate_caches()
    try:
        monkeypatch.setitem(triton_backends.backends, "ventus",
                            triton_backends._discover_backends()["ventus"])
        yield
    finally:
        for name in ("triton.backends.ventus.driver", "triton.backends.ventus.compiler",
                     "triton.backends.ventus.textual", "triton.backends.ventus"):
            import sys
            sys.modules.pop(name, None)
        importlib.invalidate_caches()


@triton.jit
def _vector_add(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(z_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)


@triton.jit
def _reduce(x_ptr, z_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    tl.store(z_ptr + pid, tl.sum(x, axis=0))


@triton.jit
def _softmax(x_ptr, z_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    tl.store(z_ptr + offs, e / tl.sum(e, axis=0), mask=offs < n)


CORPUS = {
    "vector_add": (_vector_add, {"x_ptr": "*fp32", "y_ptr": "*fp32", "z_ptr": "*fp32", "n": "i32"},
                   {"BLOCK": 32}, 1),
    "reduce": (_reduce, {"x_ptr": "*fp32", "z_ptr": "*fp32", "n": "i32"}, {"BLOCK": 32}, 1),
    "softmax": (_softmax, {"x_ptr": "*fp32", "z_ptr": "*fp32", "n": "i32"}, {"BLOCK": 32}, 1),
}


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_kernel_reaches_elf(ventus_backend, monkeypatch, tmp_path, name):
    fn, signature, constexprs, num_warps = CORPUS[name]
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / name))
    triton_compile(ASTSource(fn=fn, signature=signature, constexprs=constexprs),
                   target=TARGET, options={"num_warps": num_warps})
    assert list((tmp_path / name).rglob("*.elf")), f"{name}: no ELF produced"
```

**Step 2: Run it to verify it fails**

Run:
```bash
source .venv/bin/activate
export TRITON_HOME="$PWD/.triton-home" PYTHONPATH="$PWD/python"
python -m pytest python/test/unit/ventus/test_textual_boundary.py -q -s --tb=short
```
Expected: `test_kernel_reaches_elf[softmax]` FAILs inside `make_elf` (`opt` rejects the text). `vector_add` and `reduce` PASS.

**Step 3: Commit the red test**

```bash
git add python/test/unit/ventus/test_textual_boundary.py
git commit -m "test(ventus): add an end-to-end sentinel for the textual IR boundary"
```

---

## Task 2: Table-driven textual normalizer, migrate `or disjoint`

**Files:**
- Create: `third_party/ventus/backend/textual.py`
- Modify: `third_party/ventus/backend/compiler.py` (remove `_OR_DISJOINT`/`_strip_or_disjoint` at `:119-130`, import and call `textual.normalize` at `:257`)
- Test: `python/test/unit/ventus/test_textual_normalizer.py`

**Step 1: Write the failing unit test**

```python
"""Unit tests for the consumer-side textual IR normalizer (pure Python)."""

from triton.backends.ventus import textual


def test_or_disjoint_is_stripped():
    assert textual.normalize("%x = or disjoint i32 %a, %b\n") == "%x = or i32 %a, %b\n"


def test_plain_or_is_untouched():
    assert textual.normalize("%x = or i32 %a, %b\n") == "%x = or i32 %a, %b\n"


def test_rules_are_documented():
    assert textual.RULES, "the normalizer must have at least one rule"
    for rule in textual.RULES:
        assert rule.name and rule.rationale
```

**Step 2: Run it to verify it fails**

Run: `python -m pytest python/test/unit/ventus/test_textual_normalizer.py -q --tb=short`
Expected: FAIL (`ModuleNotFoundError: triton.backends.ventus.textual`).

**Step 3: Write the module**

```python
"""Normalize consumer-side LLVM IR text for the pinned Ventus LLVM 16 parser.

The Triton-side LLVM and the Ventus toolchain are different revisions, so the
text the consumer emits can contain features the Ventus parser predates. Every
such feature gets one `Rule` here, with the reason it is safe, and `normalize()`
applies them in order. This replaces the ad-hoc string edits in `compiler.py`
and is covered by `test_textual_normalizer.py` plus the end-to-end sentinel
`test_textual_boundary.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    replacement: str
    rationale: str


RULES = (
    Rule(
        name="or_disjoint",
        pattern=re.compile(r"\bor\s+disjoint\s+"),
        replacement="or ",
        rationale="`disjoint` is an LLVM 17 keyword and a promise about the "
        "operands, not part of the result; the LLVM 16 parser rejects it.",
    ),
)


def normalize(text: str) -> str:
    """Apply every rule in order; return the Ventus-LLVM-16-parseable text."""
    for rule in RULES:
        text = rule.pattern.sub(rule.replacement, text)
    return text
```

**Step 4: Run the unit test to verify it passes**

Run: `python -m pytest python/test/unit/ventus/test_textual_normalizer.py -q --tb=short`
Expected: PASS.

**Step 5: Rewire `compiler.py`**

Delete `_OR_DISJOINT` and `_strip_or_disjoint` (`compiler.py:119-130`), add `from triton.backends.ventus import textual` near the imports, and change the `make_llir` return (`compiler.py:257`) to:

```python
        return textual.normalize(_apply_ventus_kernel_convention(ret, entry_name))
```

**Step 6: Run the Ventus suite to verify no regression**

Run: `python -m pytest python/test/unit/ventus/ -q --tb=short`
Expected: the same pass set as before, plus the new normalizer test.

**Step 7: Commit**

```bash
git add third_party/ventus/backend/textual.py third_party/ventus/backend/compiler.py \
        python/test/unit/ventus/test_textual_normalizer.py
git commit -m "refactor(ventus): funnel textual IR edits through a documented normalizer"
```

---

## Task 3: Special floating-point literals (`-inf` / `+inf` / `nan`)

**Context (observed):** compiling the softmax corpus member fails at `opt -passes=verify` with

```
error: expected value token
  %22 = phi float [ %20, %19 ], [ -inf, %3 ]
```

The consumer emits `-inf`; the LLVM 16 lexer wants a bit pattern.

**Files:**
- Modify: `third_party/ventus/backend/textual.py`
- Test: `python/test/unit/ventus/test_textual_normalizer.py`

**Step 1: Write the failing tests** (append)

```python
def test_float_neg_inf_becomes_hex():
    assert textual.normalize("phi float [ %a, %b ], [ -inf, %c ]\n") == \
        "phi float [ %a, %b ], [ 0xFF800000, %c ]\n"


def test_float_pos_inf_and_nan_become_hex():
    assert "0x7F800000" in textual.normalize("fadd float %a, +inf\n")
    assert "0x7FC00000" in textual.normalize("fadd float %a, nan\n")


def test_double_neg_inf_becomes_hex():
    assert "0xFFF0000000000000" in textual.normalize("phi double [ -inf, %c ]\n")
```

**Step 2: Run to verify failure**

Run: `python -m pytest python/test/unit/ventus/test_textual_normalizer.py -q --tb=short`
Expected: the three new tests FAIL.

**Step 3: Add the rules**

```python
    Rule(
        name="float_neg_inf",
        pattern=re.compile(r"\bfloat\s+-inf\b"),
        replacement="float 0xFF800000",
        rationale="LLVM 16 spells -inf as a bit pattern; the consumer prints -inf.",
    ),
    Rule(
        name="float_pos_inf",
        pattern=re.compile(r"\bfloat\s+\+?inf\b"),
        replacement="float 0x7F800000",
        rationale="Same as float_neg_inf, for +inf.",
    ),
    Rule(
        name="float_nan",
        pattern=re.compile(r"\bfloat\s+nan\b"),
        replacement="float 0x7FC00000",
        rationale="Same as float_neg_inf, for nan.",
    ),
    Rule(
        name="double_neg_inf",
        pattern=re.compile(r"\bdouble\s+-inf\b"),
        replacement="double 0xFFF0000000000000",
        rationale="Same as float_neg_inf, for double.",
    ),
```

**Important:** the observed softmax failure is a **bare** `-inf` inside a `phi float`, i.e. the type is not adjacent to the literal. Before writing the regex, dump the actual text and place the rule accordingly:

```bash
python -m pytest python/test/unit/ventus/test_textual_boundary.py::test_kernel_reaches_elf[softmax] -q -s --tb=long 2>&1 | grep -n "inf"
```
If the type is not adjacent, add a bare-literal rule scoped to the `float` case (the only one the corpus reaches) and a `double` sibling, and note the limitation in the `rationale`. If a bare form is genuinely ambiguous, escalate to Task 6 (alignment) instead of guessing.

**Step 4: Run to verify pass**

Run: `python -m pytest python/test/unit/ventus/test_textual_normalizer.py -q --tb=short`
Expected: PASS.

**Step 5: Commit**

```bash
git add third_party/ventus/backend/textual.py python/test/unit/ventus/test_textual_normalizer.py
git commit -m "fix(ventus): normalize special float literals across the textual boundary"
```

---

## Task 4: Newer attribute keywords (`nocreateundeforpoison`, …)

**Context (observed):** the same softmax compile then fails with

```
error: unterminated attribute group
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
```

`nocreateundeforpoison` is a newer function-attribute keyword the LLVM 16 lexer does not know.

**Files:**
- Modify: `third_party/ventus/backend/textual.py`
- Test: `python/test/unit/ventus/test_textual_normalizer.py`

**Step 1: Write the failing test**

```python
def test_newer_function_attribute_keywords_are_dropped():
    src = "attributes #2 = { nocallback nocreateundeforpoison nofree nosync }\n"
    out = textual.normalize(src)
    assert "nocreateundeforpoison" not in out
    assert "nocallback nofree nosync" in out
```

**Step 2: Run to verify failure**

Run: `python -m pytest python/test/unit/ventus/test_textual_normalizer.py -q --tb=short`
Expected: FAIL.

**Step 3: Add the rule** (drop the keyword plus one adjacent space so the list stays well-formed)

```python
    Rule(
        name="attr_nocreateundeforpoison",
        pattern=re.compile(r"\bnocreateundeforpoison\s+"),
        replacement="",
        rationale="A newer function attribute keyword the LLVM 16 lexer "
        "rejects; dropping it only forgoes an optimization hint.",
    ),
```

**Step 4: Re-run the sentinel and collect the next boundary, if any**

Run:
```bash
python -m pytest python/test/unit/ventus/test_textual_boundary.py -q -s --tb=short
```
Expected: `softmax` either PASSES or fails with a *new* keyword/literal. If new, add a rule the same way (test → run-fail → rule → run-pass) and note it in the commit message. Repeat until `softmax` passes.

**Step 5: Commit**

```bash
git add third_party/ventus/backend/textual.py python/test/unit/ventus/test_textual_normalizer.py
git commit -m "fix(ventus): drop consumer-only attribute keywords at the textual boundary"
```

---

## Task 5: Make the sentinel green and widen the corpus

**Files:**
- Modify: `python/test/unit/ventus/test_textual_boundary.py`

**Step 1: Add the remaining representative kernels**

Add corpus entries (reuse the definitions from the P3 tests verbatim so the sentinel and the functional tests cannot drift): 2-D tile, `row_swap` (LDS + barrier), warp `gather`, `histogram`, and a `tl.where` kernel compiled in **both** `predication` arms.

**Step 2: Run**

Run: `python -m pytest python/test/unit/ventus/test_textual_boundary.py -q -s --tb=short`
Expected: all corpus members PASS. Any failure is a newly surfaced boundary — handle it with the Task 3/4 loop.

**Step 3: Commit**

```bash
git add python/test/unit/ventus/test_textual_boundary.py
git commit -m "test(ventus): widen the textual-boundary sentinel corpus"
```

---

## Task 6: Barrier and shuffle extension seams

**Intent:** make the two "we will replace this when the toolchain/hardware catches up" points explicit and single-sited, so a future first-class builtin (or hardware shuffle) is a one-function change.

**Files:**
- Modify: `third_party/ventus/lib/TritonVentusToLLVM/TritonGPUToLLVM.cpp` (`emitBarrier` ~`:143`, `exchangeThroughLds` ~`:415`)
- Modify: `third_party/ventus/README.md` (§8 陷阱清单)

**Step 1: Annotate the seams**

Above `emitBarrier`, add:

```cpp
// EXTENSION SEAM (P4): when the consumer LLVM learns the Ventus builtin table
// (or the toolchain revision is aligned), replace this inline-asm emission with
// a real `llvm.riscv.ventus.barrier` declaration. Only this function changes.
```

Above `exchangeThroughLds`, add:

```cpp
// EXTENSION SEAM (P4): a hardware lane-shuffle instruction, if the target ever
// grows one, replaces this LDS round-trip here. `shuffleXor/Up/Idx` and
// `warpReduce` stay as-is.
```

**Step 2: Document in README §8**

Add one bullet stating the two seams and that they are the single change points for first-class builtins / hardware shuffle.

**Step 3: Rebuild and run the suite**

```bash
cd "$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')" && ninja triton && cd -
python -m pytest python/test/unit/ventus/ -q --tb=short
```
Expected: unchanged pass set.

**Step 4: Commit**

```bash
git add third_party/ventus/lib/TritonVentusToLLVM/TritonGPUToLLVM.cpp third_party/ventus/README.md
git commit -m "docs(ventus): mark the barrier and shuffle extension seams"
```

---

## Task 7 (gated): LLVM/MLIR revision-alignment record

**Do not start until the user confirms the alignment effort.** The toolchain README states the shared-library Tasks 4-7 "may be reconsidered only after an explicit LLVM/MLIR revision-alignment record." This task produces that record and a spike, not a full migration.

**Files:**
- Create: `docs/plans/<date>-ventus-llvm-alignment.md`
- Modify: `third_party/ventus/toolchain/README.md` (link the record)

**Step 1: Record the gap**

Write down: consumer LLVM `b010a18d…` vs Ventus LLVM 16 `d4f2063…`; the exact features that forced the normalizer (each `Rule` in `textual.py` plus the calling-convention and barrier builtins); and the two candidate strategies — (a) build the Triton consumer against the Ventus LLVM fork, or (b) add the Ventus target/builtins to the consumer LLVM.

**Step 2: Spike the smaller one**

Attempt (b) for a single builtin (`llvm.riscv.ventus.barrier`) in a throwaway branch: declare it in the consumer's builtin table and confirm a compiled kernel passes `opt -passes=verify` with no inline asm. Record the result either way.

**Step 3: Commit the record**

```bash
git add docs/plans/<date>-ventus-llvm-alignment.md third_party/ventus/toolchain/README.md
git commit -m "docs(ventus): record the LLVM/MLIR alignment path to first-class builtins"
```

---

## 判据 / 命令汇总

```bash
source .venv/bin/activate
export TRITON_HOME="$PWD/.triton-home" PYTHONPATH="$PWD/python"

# pure-Python (no GPU, no toolchain needed)
python -m pytest python/test/unit/ventus/test_textual_normalizer.py -q --tb=short

# end-to-end boundary sentinel (needs the Ventus toolchain)
python -m pytest python/test/unit/ventus/test_textual_boundary.py -q -s --tb=short

# full backend suite
python -m pytest python/test/unit/ventus/ -q --tb=short

# C++ changes only
cd "$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')" && ninja triton
```

**P4 (this phase) 完成的定义：**
1. `test_textual_boundary.py` 的整个 corpus 编到 ELF 通过（含 softmax）。
2. 所有文本改写集中在 `textual.py`，每条规则有单元测试与 rationale；`compiler.py` 不再持有一次性 regex。
3. barrier / shuffle 的替换点各只有一处，且 README §8 记录。
4. LLVM/MLIR 对齐有书面记录 + 一个 first-class builtin 的 spike 结论（Task 7，gated）。

## 范围外（不在本阶段）

- 真正的 LLVM 版本迁移（Task 7 只出记录与 spike）。
- 硬件 shuffle（决定 skip，留 seam；见 Task 6）。
- P5 的 M4K8N4 MMA（依赖 M3 契约）。
- `barriersub`、`tt.scan`、跨 CTA histogram。

## 风险

1. **float 字面量的类型歧义**：`phi float ... [ -inf, ... ]` 的类型不紧贴字面量；若 bare `-inf` 无法安全区分 float/double，不要猜，升级到 Task 7 的 alignment，或改在 lowering 侧不发 `-inf`（softmax 的掩码用有限大负数）。
2. **sentinel 需要完整 Ventus 工具链**：与 P3 测试同环境，不能进 CI 的共享 runner（见 README §2.2）。
3. **`nocreateundeforpoison` 只是第一个**：同类新关键字可能还有（`nofpclass`、`icmp samesign` 等）。sentinel 的价值就是让下一个在这里现形，而不是靠人枚举。
