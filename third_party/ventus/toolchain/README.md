# Ventus Toolchain Identity

`version.json` is the machine-readable identity for this worktree. It currently
records a `pre_hardening` snapshot and deliberately sets `release_valid` to
`false`. The observed tools can support development, ABI goldens, and the
M1/M2 non-MMA Spike baseline, but they are not a coordinated release identity.

## Triton Environment

Use the worktree's `.venv` and set:

```bash
export TRITON_HOME=/home/weijiale/Code/cuda2rvv/triton/.worktrees/triton-ventus-v1/.triton-home
```

The resolved consumer cache must be
`llvm-b010a18d-ubuntu-x64-1`, matching `cmake/llvm-info.json`. Do not use
`cmake/llvm-build-info.json` as the consumer pin. Build current-checkout native
code with the editable worktree environment; Python source and
`python/triton/_C/libtriton.so` must both resolve inside this worktree.

Never set Triton's `LLVM_SYSPATH` to the Ventus LLVM 16 prefix. Never source
`ventus-env/env.sh` in the normal Triton build shell or globally prepend the
Ventus install `bin` or `lib` directories. Runtime-only loader changes belong in
a dedicated subshell.

## Ventus Tools

All Ventus tools are invoked by absolute path under:

```text
/home/weijiale/Code/cuda2rvv/ventus-env/install
```

Ventus Clang 16 is an ABI oracle, not Triton's build dependency. The supported
boundary is checked textual LLVM IR followed by absolute Ventus `opt`, `llc`,
and `ld.lld` subprocesses. LLVM bitcode and LLVM object code never cross the
Triton/Ventus version boundary in-process.

The final link prunes with `--gc-sections` and roots the kernel with `-u`: the
installed `riscv32clc.o` is a 22 MB single object (all OpenCL builtins, not an
archive), so an unpruned link produces a ~14 MB image, while `--gc-sections`
alone would delete the kernel itself, which nothing in the image references and
the runtime finds by symbol name.

The installed linker script names `_start`, supplied by `crt0.o`, as the ELF
entry. Task 2 ABI goldens confirm that `crt0.o` supplies `_start`,
`libworkitem.a` supplies the observed work-item builtins, and `riscv32clc.o`
references that work-item ABI. The final link command remains owned and tested by
the producer-local backend stage.

## Textual IR Boundary Facts

Two producer-side rules follow from the revision gap above and are worth stating
explicitly, because both were established by experiment against the installed
toolchain.

**The `ventus_kernel` calling convention is attached on the text.** In the
pinned Ventus LLVM 16 that convention is calling-convention number 104. The
Triton-side LLVM numbers 104 as `amdgpu_cs_chain` and prints that foreign
keyword, which the Ventus parser rejects. The backend therefore leaves the
calling convention unset in the translated module and rewrites the kernel's
`define` line to `define ventus_kernel ...` on the emitted text; the elf stage's
`opt -passes=verify` run then fails loudly if the Ventus toolchain does not
accept the result.

**Predicated accesses use control flow, not `llvm.masked.*`.** The masked
load/store intrinsics changed arity between the two revisions: Ventus LLVM 16
requires the trailing `i32 immarg` alignment operand, while the Triton-side LLVM
removed it. No single declaration satisfies both, and the Triton-side translator
validates declarations against its own intrinsic table, so the intrinsics cannot
cross this boundary. V1 lowers a masked access to an `llvm.cond_br` diamond with
the loaded value and the `other` operand meeting in the join block. Vectorized
masked accesses remain a later optimization.

## Local Core Advance

The Triton source tree in this worktree has been advanced past its base commit
`310241f824` by cherry-picking selected upstream commits, rather than rebasing
onto `origin/main` (which is 263 commits ahead). Each cherry-pick is
patch-equivalent to upstream, so a future rebase drops them automatically:

| Commit | Subject |
|---|---|
| `7a7f10458e` | `[LAYOUTS] Dispatch dot-operand lowering through MmaEncodingTrait` (#11025) |
| `a70115aa99` | `[LAYOUTS] Dispatch shared-layout lowering through SharedEncodingTrait` (#11764) |
| `12fa7984ca` | `[OptimizeThreadLocality] Handle multi-use thread locality results` (#11682) |

This advance does **not** change the LLVM/MLIR revision consumed by Triton (still
`b010a18d2b648cab83c83967ff26b8fde11acdc6`, build 1) and therefore does not affect
the Tasks 4-7 gate below. `version.json` records the resulting `triton_commit`,
worktree content hash, and `libtriton` hash.

Commits that depend on intervening upstream refactors are deliberately **not**
cherry-picked (`92ff4362da`, `51593ac6b6`); see `third_party/ventus/README.md`
section 2.1.

## Conditional Shared-Library Tasks

Implementation-plan Tasks 4-7 are deferred. Triton consumes LLVM/MLIR revision
`b010a18d2b648cab83c83967ff26b8fde11acdc6` while the installed Ventus toolchain
uses LLVM 16 revision `d4f2063fe81cbbefda34da60d4cd5c46bce3d231`; these
revisions have not been deliberately aligned.

V1 therefore does not build a shared compiled Ventus dialect, GPU-to-Ventus
conversion library, ABI conversion library, or LLVM-based compiler utility
library. Their V1 responsibilities remain producer-local: target and ABI
validation in the Triton adapter and manifest, SPMD/barrier lowering in the
Triton backend, checked textual `kernel.ventus.ll`, absolute-path Ventus LLVM 16
subprocesses, and producer-local ELF inspection/link orchestration. Tasks 4-7
may be reconsidered only after an explicit LLVM/MLIR revision-alignment record.

## Runtime Status

The current runtime build and install events were not captured by one
coordinated provenance manifest. `version.json` therefore records observed
binary hashes, build IDs, and RUNPATHs without claiming an exact source-to-binary
derivation.

The installed Spike is the M1/M2 non-MMA baseline. It does not contain the
current dirty `vftta_vv.h` experiment. That experiment is intentional pre-M3
work, but it must be revised and rebuilt against the newer Chisel M4K8N4 design
after M3 establishes a verified source/generated-RTL/testbench record.

The runtime-side resource units are recorded. Task 8 replaced the legacy four-
`uint16` payload with the tested 24-byte `VRES` v1 record containing four
`uint32` resource fields. The transport remains non-release because it has no
runtime consumer and POCL still uses hard-coded resource values. Subsequent
runtime hardening must parse the versioned record, remove or isolate hard-coded
consumption, and validate limits before `release_valid` can become true.

## Updating Identity

An update must record commits, normalized dirty content hashes, full binary
hashes, build IDs, RUNPATHs, build/install events, and the resolved Triton LLVM
cache. Unknown provenance remains explicitly unknown. A coordinated rebuild may
change build and install content hashes because CMake relocates RUNPATHs; matching
build IDs plus documented RUNPATH changes are not evidence of stripping.

Any ABI change requires a new artifact ABI version or updated golden references.
This README is explanatory; `version.json` remains the final machine-readable
identity source.
