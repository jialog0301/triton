# Ventus LLVM/MLIR Alignment Record

> **Status:** investigation only. This document satisfies the toolchain README's
> prerequisite ("Tasks 4-7 may be reconsidered only after an explicit LLVM/MLIR
> revision-alignment record"). It does **not** migrate anything.

**Goal:** record the revision gap between the Triton-side LLVM/MLIR and the pinned
Ventus toolchain, inventory every consumer-side-only feature the backend works
around today, state the two candidate alignment strategies, and record a spike
result for the smaller one (teach the consumer a first-class Ventus builtin).

**Why now:** P4 replaced the ad-hoc text edits with a documented normalizer
(`backend/textual.py`) and two `EXTENSION SEAM (P4)` points. That work made the
boundary *explicit*; this record decides when and how to remove it.

---

## 1. The revision gap

| Side | Component | Version | Pin (in-repo) |
|---|---|---|---|
| Triton consumer | LLVM/MLIR cache | **24.0.0git** (`LLVM_VERSION_STRING`) | `cmake/llvm-info.json` → `llvm-b010a18d-ubuntu-x64-1`, build 1 |
| Ventus toolchain | clang/opt/llc/ld.lld | **16.0.0git** | `toolchain/version.json` → `llvm_commit` (submodule `origin/add-new-owner-12-gd4f2063fe81c`, `d4f2063`) |

The consumer cache string `b010a18d` is the build hash, not the git commit; the
`*-info.json` files carry anonymized `0000…` hashes in this worktree, so the
authoritative build identity remains `toolchain/version.json` plus the cache
directory name. **Gap: 8 major LLVM releases (16 → 24).**

The two sides never exchange bitcode or object code in-process: the boundary is
**checked textual LLVM IR** (`compiler.py::make_llir` emits text) followed by
absolute-path `opt`/`llc`/`ld.lld` subprocesses (`compiler.py::make_elf`). That
design is why a version gap is survivable at all — but every feature that changed
in the gap has to be handled on the Triton side.

## 2. Boundary inventory (what the gap costs today)

Every item below is a consumer-emitted feature the Ventus LLVM 16 does not accept,
and the producer-local fix that exists now.

| # | Feature (consumer side) | Failure on Ventus side | Fix | Location |
|---|---|---|---|---|
| 1 | calling convention 104 printed as `amdgpu_cs_chain` | Ventus parser rejects the keyword | attach `define ventus_kernel` on the text | `compiler.py::_apply_ventus_kernel_convention` |
| 2 | `or disjoint` | LLVM 17 keyword, `expected type` | normalizer rule `or_disjoint` | `textual.py` |
| 3 | `nocreateundeforpoison` attribute | LLVM 18+ keyword, `unterminated attribute group` | normalizer rule `attr_nocreateundeforpoison` | `textual.py` |
| 4 | bare `-inf`/`+inf`/`nan` in a typed instruction | `floating point constant invalid for type` | normalizer rules `float_*`/`double_*`/`bare_special_float` (64-bit IEEE-754 bit patterns) | `textual.py` |
| 5 | `llvm.masked.*` arity changed | no single declaration fits both | lower masked access to an `llvm.cond_br` diamond | `VentusLoadStoreOpToLLVM.cpp` |
| 6 | `llvm.riscv.ventus.barrier` is a Ventus-16 builtin | consumer translator validates against its own intrinsic table | emit inline asm `barrier x0, x0, N` | `TritonGPUToLLVM.cpp::emitBarrier` (EXTENSION SEAM) |
| 7 | `llvm.exp.f32` and friends | Ventus llc crashes in `ExpandFPLibCall` → `LowerCall` | lower to a direct call to the libclc Itanium-mangled symbol (`_Z3expf`) | `VentusLoadStoreOpToLLVM.cpp::VentusLibcallOpConversion` |
| 8 | work-item builtins (`get_local_id`, …) | — (works: same libclc ABI) | none | — |

Items 1–4 are textual and now live in `textual.py` (rules + rationale + unit
tests); items 5–7 are lowering-level. The end-to-end sentinel
`python/test/unit/ventus/test_textual_boundary.py` compiles a corpus through the
whole chain so a **new** item fails a test rather than a user's kernel.

## 3. Two candidate strategies

### (a) Build the Triton consumer against the Ventus LLVM 16 fork
Align the versions so the in-process translator *is* the Ventus LLVM, and the
textual boundary disappears (could pass a module/bitcode directly).

- **Pros:** removes the entire boundary class at once (items 1–7).
- **Cons:** Triton's consumer LLVM is chosen by a large upstream build
  (`cmake/llvm-info.json`, patched `triton-lang/llvm-project`); swapping it for a
  RISC-V/GPGPU fork with a `ventus-gpgpu` target means re-porting Triton's local
  LLVM patches onto LLVM 16 and maintaining that forever. Effectively a fork of
  Triton's LLVM dependency. High recurring cost.

### (b) Teach the consumer LLVM the Ventus builtins
Keep the consumer LLVM, add `llvm.riscv.ventus.*` to its intrinsic definitions so
item 6 (and any future first-class builtin) can be declared and exported directly.

- **Pros:** narrow; only the intrinsic table changes; the textual barrier hack and
  the two extension seams are removed.
- **Cons:** still a custom consumer LLVM build (item 6's symbol must be added to
  `llvm/IR/IntrinsicsRISCV.td` / the RISCV target), and the Ventus semantics must
  match the consumer's expectations across the 16↔24 gap.

## 4. Spike result: can the consumer declare a Ventus intrinsic? — **No, not today**

Checked the actual consumer LLVM cache (`llvm-b010a18d-ubuntu-x64-1`, 24.0.0git):

```
$ grep -rn "ventus" <cache>/include/llvm/IR/IntrinsicsRISCV.td        # (no matches)
$ grep -rl  "ventus" <cache>/include/llvm/                      # (no matches)
```

The consumer LLVM 24 has **no `ventus` intrinsics at all** — not `ventus.barrier`,
not `vftta`, nothing. It does know other RISC-V intrinsics (e.g. `riscv_sf_vc_*`
via MLIR's `VCIXOps.td`), so the table is extensible in principle, but **strategy
(b) requires editing and rebuilding the consumer LLVM**; it cannot be done from
the backend Python/C++ alone. This matches the toolchain README's rule that a
first-class builtin "may be reconsidered only after an explicit alignment record"
— the record is this document; the alignment itself is a separate decision.

**Consequence for P3/P5:** the `EXTENSION SEAM (P4)` at `emitBarrier` and the
`VentusLibcallOpConversion` table stay. They are correct and tested; they are not
a migration target until either strategy (a) or (b) is funded.

## 5. Recommendation

1. **Do not align now.** The textual normalizer + sentinel + lowering-level
   bitwise fixes (items 1–7) keep the backend correct end-to-end on the current
   pins, at a cost that is bounded and now tested.
2. **When to revisit:** the trigger is a boundary that cannot be handled
   producer-locally without changing values/semantics — e.g. a consumer feature
   whose Ventus equivalent is not a bit-pattern/kind rewrite, or a first-class
   builtin the MMA path (P5) genuinely needs for correctness (not just speed).
3. **If revisiting:** prefer **(b)** (teach the consumer the specific builtins)
   over **(a)** (adopt LLVM 16 wholesale), because it is additive and does not
   fork Triton's whole LLVM dependency. Either way the first step is a dedicated
   LLVM-patch branch plus a `cmake/llvm-info.json` re-pin and a full rebuild.

## 6. Verification of this record

- Boundary inventory §2 matches the code: `grep -n 'name="' third_party/ventus/backend/textual.py`
  lists rules 2–4; `grep -rn "EXTENSION SEAM" third_party/ventus/lib/` finds seam 6.
- Spike §4 is reproducible: the two `grep` commands above against the resolved
  consumer cache.
- The sentinel `python/test/unit/ventus/test_textual_boundary.py` is the living
  check that §2 stays complete: add a kernel, and an unhandled boundary fails it.
