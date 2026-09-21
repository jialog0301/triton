# Ventus Environment Facility Snapshot

## 1. Purpose And Status

This document records the facilities inspected in
`/home/weijiale/Code/cuda2rvv/ventus-env`. It is the implementation-fact source
of truth for the Triton-for-Ventus plans. Architecture documents describe the
desired design; this snapshot describes what the inspected checkout implements.
The authoritative project scope is
[Triton-for-Ventus V1 Scope](2026-08-28-triton-for-ventus-v1-scope.md). Any V1
target stated below is a plan layered on these facility facts, not a claim that
Triton compiler integration already exists.

Triton/Python provenance, consumer LLVM cache resolution, installed Ventus
binary hashes/build-ids, and release blockers are maintained in
[Triton/Ventus Toolchain Version Audit](2026-08-31-triton-ventus-toolchain-version-audit.md).

The committed revisions identify the source baselines. The superproject and
several submodules are dirty, so commits alone are not a reproducible identity;
the uncommitted changes and generated/build artifacts must be recorded by
source diff/content hash and must not be treated as stable ABI or runtime
facilities.

## 2. Inspected Revisions

| Component | Commit | State |
| --- | --- | --- |
| `ventus-env` | `7e9790708d58ebf697d74fa8dadbaafa1232ca1d` | dirty superproject |
| LLVM | `d4f2063fe81cbbefda34da60d4cd5c46bce3d231` | clean |
| POCL | `0c9a0bf6c51a922cd483e11281ddad70667b681b` | one source change present |
| Driver | `86a4860f6112e2f016d0f86c2653a75337f4f1f2` | log-level changes present |
| GPGPU/RTL | `f5853809f114192b99657e7021d1e50dfe961fe1` | untracked documents present |
| CycleSim | `335ba24d2c7763c87e8c9bfe889c9074763a32f7` | multiple source/test changes present |
| Spike | `abe4323aa551fc5e2f80c46322cc3feb6be49a6d` | `vftta_vv.h` modified |
| OpenCL CTS | `a08f46ba93e23283df625599eb1cd4507fb21ec1` | clean |
| Testcases | `e4630621280554f691580b03b266e0c95b7db47d` | generated outputs modified |
| OCL ICD | `2faf7063c91fdb0f2471f98bf5b2c49c8f907423` | build artifacts present |
| SystemC | `5fc1469339775b54b1fcc2020ac744a58be5b50e` | generated build files modified |
| Rodinia | `b9890b87cb83b108f5cabf3b84568e627299d1dc` | submodule pin |

Release artifacts must use a clean checkout or record reviewed patch/content
hashes in addition to the commits above.

## 3. Build And Execution Stack

The committed stack builds SystemC, LLVM/libclc, OCL ICD, Spike, RTL simulation,
CycleSim, GVM, the Ventus driver, POCL, Rodinia, OpenCL CTS, and testcases.

The normal execution path is:

```text
OpenCL host program
  -> ocl-icd
  -> POCL Ventus device
  -> Ventus Clang/LLVM
  -> objectN.riscv ELF
  -> packed arguments and launch metadata
  -> libventus_driver.so
  -> Spike / RTL simulation / CycleSim / GVM
```

`VENTUS_BACKEND` selects `spike`, `rtlsim`, `cyclesim`, or `gvm`. No physical
hardware backend is selectable in this checkout. The current Ventus C driver is
a simulator-facing API, not yet a stable production HAL boundary.

Primary sources are `build-ventus.sh`, `env.sh`,
`pocl/lib/CL/devices/ventus/pocl_ventus.cc`, and
`driver/driver/auto_select/ventus.cpp`.

Normal Triton build shells must use explicit absolute `VENTUS_*` tool variables;
they must not source `env.sh` or globally prepend the Ventus install `bin`/`lib`
directories. Runtime tests that need Ventus loader/PATH settings must confine
them to a separate subshell.

## 4. Ventus LLVM Target Contract

### 4.1 Target Identity

```text
target:      riscv32 or riscv32-unknown-unknown
mcpu:        ventus-gpgpu
pointer:     32 bits
data layout: e-m:e-p:32:32-i64:64-n32-S128-A5-G1
default ISA: rv32ima_zhinx_zfinx_zdinx_zve32f
```

The target currently inherits LLVM's `RocketModel`; it has no stable
Ventus-specific scheduling model.

The Ventus compiler is LLVM 16.0.0git. The current Triton checkout uses much
newer LLVM/MLIR revisions. This version gap is a boundary condition handled by
producer-specific adapters plus textual LLVM IR/tool invocation. It is not a
requirement to align LLVM versions before M1; a shared compiled C++ MLIR library
is deferred until revisions are deliberately aligned.
For the audited Triton checkout, the consumer pin is
`cmake/llvm-info.json` `b010a18d...`, build 1. `cmake/llvm-build-info.json`
`941a04e...` is the workflow producer pin, not the local consumer pin. The
boundary forbids in-process LLVM object-code linking and cross-version bitcode;
Ventus `opt` must verify LLVM-16-compatible textual IR before `llc`.
The current stable cache symlink points at a different revision identity, but
normal `build_helpers` resolution uses the revisioned package directory,
validates `version.txt`/downloads as needed, passes that directory as
`LLVM_SYSPATH`, and only then rewrites the stable symlink. A private
`TRITON_HOME` is therefore an isolation/concurrency/reproducibility requirement,
not evidence that normal resolution necessarily consumes the symlink's previous
target. No semantic incompatibility between those Triton LLVM packages is
claimed without tests.

The audited Python environment has no `direct_url.json` for pip Triton `3.7.1`;
its package provenance is unknown even though the venv directory is under a
separate checkout. `PYTHONPATH` was unset normally. The observed import mismatch
was reproduced only after explicitly pointing `PYTHONPATH` at the current
checkout's Python source, which then lacked a matching current-checkout native
module resolution.

### 4.2 Kernel Calling Convention And Arguments

OpenCL kernels use LLVM calling convention `ventus_kernel` (ID 104). LLVM IR
retains direct formal arguments, while the backend loads values from a packed
argument buffer whose base is supplied in scalar register `a0`/`x10`.

Top-level arguments follow LLVM data-layout alignment. Values smaller than four
bytes use four-byte-aligned top-level slots. Pointer values are RV32 addresses.

Evidence includes `llvm/include/llvm/IR/CallingConv.h`,
`clang/lib/CodeGen/TargetInfo.cpp`,
`llvm/lib/Target/RISCV/RISCVISelLowering.cpp`, and the VentusGPGPU
`kernel_args.ll` and `parameter-vector-struct-types.ll` tests.

### 4.3 Address Spaces

| Address space | Meaning |
| ---: | --- |
| 0 | flat/generic |
| 1 | global |
| 3 | local/work-group shared |
| 4 | constant |
| 5 | private/per-work-item |

LLVM address spaces are not the physical RTL decoder. Current RTL routes LDS
through the address range rooted at `0x70000000`.

The backend currently treats all address-space casts as no-ops. Triton and IREE
must not rely on arbitrary cross-space casts until this behavior is reviewed
against physically distinct memories.

### 4.4 Work-Item Builtins

This revision has no stable LLVM ID intrinsics. libclc calls
`__builtin_riscv_*` routines implemented by
`libclc/riscv32/lib/workitem/workitem.S`. The implementation reads work-group
IDs from CSRs, combines `CSR_TID` with `vid.v` for work-item IDs, and reads
sizes/offsets from launch metadata. Stable ID intrinsics remain future LLVM
hardening work.

### 4.5 Barriers

Barrier support already exists:

```text
llvm.riscv.ventus.barrier
llvm.riscv.ventus.barrier.with.scope
llvm.riscv.ventus.subgroup.barrier
llvm.riscv.ventus.subgroup.barrier.with.scope
```

The compiler and RTL support a full work-group barrier baseline. Distinct
subgroup semantics are not sufficiently verified and remain unsupported for the
initial Triton target. Barriers must be control-flow uniform across all
participating work-group warps.

### 4.6 Divergence And Register Domains

Ventus LLVM owns uniform/varying analysis, SGPR/VGPR selection, vector branch
selection, `SETRPC`/`JOIN` insertion, mixed-PHI repair, load legalization,
spilling, and `REGEXT` insertion.

MLIR producers should emit per-work-item scalar semantics and ordinary CFG.
They must not emit explicit `<32 x T>` SIMT values, vector branches, `SETRPC`,
or `JOIN`.

The backend restricts several normal machine CFG passes and relies on repair
passes. Producer equivalence beyond Clang-shaped IR needs differential tests.
The inspected source has a finite per-warp SIMT stack, but this review did not
establish an architectural nesting limit, overflow detection, or overflow
recovery contract. Therefore bounded divergent control flow is a compiler-plan
requirement, not an existing general hardware guarantee.

### 4.7 Resource Section

Each entry kernel receives `.ventus.resource.<kernel>`. Its payload is exactly
four target-endian `uint16` values:

```text
VGPR, SGPR, LDS bytes, PDS bytes
```

It has no magic, version, record size, target identity, argument information,
or extensibility fields. Internal 32-bit counters are truncated to 16 bits. The
active POCL path does not parse this section and instead supplies hard-coded
resource values. Versioning, range checking, parsing, and runtime consumption
are prerequisites for a reliable Triton/IREE artifact.
The inspected names do not by themselves fix aggregation units: LDS is emitted
as bytes, while SGPR/VGPR must still be reconciled as per-wavefront versus
work-group/resident totals, and PDS units/aggregation differ across parameter,
resource, and driver structures. A canonical resource-unit contract was not
found.

## 5. Runtime ABI And Metadata Layers

Current execution uses five distinct structures:

1. **Packed kernel argument buffer.** POCL owns the current packing logic.
2. **Device kernel metadata block.** A reserved 64-byte block containing entry,
   argument base, work dimension, global/local sizes, global offsets, and
   reserved printf fields.
3. **Host driver launch metadata.** Grid, warp/work-group geometry, LDS/PDS,
   SGPR/VGPR, PDS base, and kernel name.
4. **ELF resource section.** The raw four-value record above.
5. **Proposed artifact manifest.** A future versioned Triton/IREE package that
   references and validates the other structures.

The host launch structure is duplicated in POCL and driver backends. GVM's copy
already differs by omitting the trailing kernel-name field. A shared, versioned
C ABI is required before IREE relies on this boundary.

POCL currently depends on writable current-directory files, external `nm`,
shell pipelines, and filename-based ELF loading. Production integration should
use LLVM object APIs and define explicit in-memory or secure temporary-file ELF
loading.

`driver/common/loadelf.cpp` currently enumerates and loads `PT_LOAD` segments,
but does not validate artifact ABI, target identity, entry symbol, content hash,
or a manifest. The current capability API is also insufficient as a stable
artifact contract: the inspected RTL/GVM backends expose only a small parameter
subset, while the inspected current CycleSim worktree returns unimplemented for
`vt_dev_caps`.

RTL simulation virtual memory is explicitly incomplete in
`driver/driver/rtlsim_device/ventus.cpp`; allocated “virtual” addresses are
currently physical-address-style simulator addresses. Completion is likewise
not a deterministic production contract: the inspected RTL/GVM paths append a
fixed 5000-step drain because L2 flush completion is not observable, and the
inspected current CycleSim worktree ignores the timeout while stepping until
idle. These are source observations, not committed stable runtime promises.

## 6. Execution Profiles

### 6.1 Default RTL Profile

```text
2 SMs
32 lanes per warp
8 warp slots per SM
8 work-group slots per SM
at most 8 warps per work-group
at most 256 work-items per work-group
128 KiB LDS per SM
32-bit addresses and SEW=32
```

These are profile parameters, not universal source invariants. Triton's initial
artifact profile should require and record 32 lanes. Runtime overrides must not
silently change a compiled artifact's ABI.

The RTL uses extended SGPR/VGPR indices through `REGEXT/REGEXTI`; it is not a
simple fixed 32-SGPR/32-VGPR architecture.

### 6.2 CycleSim Differences

CycleSim is a capability-gated functional and cycle-level experiment, but differs
from the default RTL and is not an independent proof of RTL architectural behavior
or performance:

- it accepts wavefront sizes up to 32;
- it models 256 SGPR and 256 VGPR slots per warp;
- CTA residency does not consume SGPR/VGPR metadata;
- LDS capacity is 256 MiB rather than 128 KiB;
- it does not model the RTL L1 cache or LDS bank conflicts;
- its public API exposes simulated time but not stable barrier, coalescing,
  bank-conflict, cache-miss, spill, or occupancy counters.

CycleSim results must be combined with compiler metadata and mandatory RTL
measurements; they cannot replace RTL validation.

The audited CycleSim and `libcyclesim_driver` build/install pairs have matching
build IDs but different content hashes because CMake install relocated RUNPATHs
from build/absolute paths to `$ORIGIN`; this is not evidence of stripping or a
source-code mismatch. Their Aug 17 and Aug 21 install timestamps prove separate
install events and the absence of one coordinated generation manifest, not an
incompatible source generation. Record both full hashes, build IDs, and
RUNPATHs, then perform one coordinated rebuild/install with a unified manifest
before release or M3. The complete values are in the toolchain audit and must
ultimately be serialized by Task 1 `version.json`.

### 6.3 Shared Memory

Static AS3 objects and full work-group barriers are suitable baseline compiler
targets. Dynamic OpenCL `__local` arguments and host-managed LDS uploads are not
generally supported across all runtime backends.

RTL LDS is banked. Same-bank accesses replay, and identical reads are not
broadcast automatically. Shared-memory layouts need RTL validation or static
bank analysis; CycleSim does not model these costs.

### 6.4 Tensor Facility

The Chisel source contains an experimental FP32 `VFTTA_VV` tensor facility.
`gpgpu/ventus/src/top/parameters.scala` sets default `num_thread=32` and
`tc_dim=Seq(4,8,4)`. `execution.scala` passes these as
`TensorCoreFP32(vl, DimM, DimN, DimK, ...)`. The explicit mapping is logical
`M=constructor DimM`, logical `K=constructor DimN`, and logical `N=constructor
DimK`. Thus the labeled logical order is `(M,K,N)` while constructor positional
order is `(M,N,K)`; no unlabeled tuple may be used as a contract value.

The inspected checked-in `gpgpu/driver/rtl/GPGPU_top.v` appears older/smaller
than that current Chisel default: review found 4 dot units with 4-element dot
reduction rather than the source-implied 16 outputs with 8-element reductions.
No verified generation record currently pins/hashes the Chisel source and binds
it to the effective parameters, generated Verilog, and testbench. Generated RTL
cannot yet be called the source of truth for profile A.

At source level, `TensorCoreFP32` maps A/B through physical lanes `0..31`, maps
C and meaningful D results through lanes `0..15`, initializes all 32 output
lanes to zero, and `vTCexe` currently sets writeback mask true for all 32 lanes.
Thus the inspected wrapper source would write upper lanes `16..31` as zero; this
requires generated-RTL/testbench verification before it is frozen, and is not a
universal Ventus invariant. The source dot product is an 8-element multiply
tree plus final C addition for the default profile, so exact tree, rounding,
NaN, exception, and latency/backpressure behavior must be recorded rather than
summarized as mathematical FMA.

The actual assembler spelling is the tied-destination form with three explicit
operands, `vftta.vv vd, vs2, vs1`. The `vm` field seen in
`spike/riscv/insns/vftta_vv.h` is the encoded mask-control bit/comment notation,
not a fourth assembly operand. Mandatory architectural binding is old
`vd=C[M,N]` accumulator, `vs1=A[M,K]`, `vs2=physical B^T[N,K]`, and new
`vd=D[M,N]`. Spike currently implements dual behavior: a `vl==4`, e32
2x2x2-like case and another `vl`-dependent path. The
inspected CycleSim implementation/fixtures represent a different smaller
`2x2x4`-style operation and legacy four-register fixtures. Neither is evidence for
profile A until corrected and aligned. No stable LLVM intrinsic, lane-fragment
ABI, masking/`vm`, architectural-`vl` versus fixed-width, register
range/alignment/overlap, latency/issue/backpressure, `fflags`
production/aggregation/writeback, or resource-unit contract was found. If the
architectural `fflags` writeback is not observable in a pinned backend, that field
remains explicitly unresolved and M3 cannot silently treat it as zero or absent.
Full-active, unmasked, e32, fixed width 32 are proposed compiler
requirements and are not all independently hardware-enforced today.

No Triton compiler integration for this instruction was found in the inspected
facility. The V1 project plans a narrow contract freeze and later integration,
but those requirements and milestone definitions belong to the authoritative
[V1 scope](2026-08-28-triton-for-ventus-v1-scope.md), not this fact snapshot.

### 6.5 Unsupported Or Unproven Facilities

- shuffle and ballot;
- subgroup collectives and subgroup barrier semantics;
- atomics, including basic atomic operations;
- async global-to-LDS copy;
- CTA clusters and warp specialization;
- dynamic local kernel arguments;
- arbitrary gather/scatter and native RVV reductions;
- general FP16, BF16, FP8, and low-bit execution;
- general/low-precision MMA, arbitrary MMA shapes, and masked MMA;
- physical hardware through the current driver;
- broad OpenCL 2.0 conformance despite POCL advertising OpenCL 2.0.
- independent AS4 constant-memory allocation/upload/coherence guarantees;
- general user-visible AS5/PDS memory support;
- unbounded divergent control flow or defined SIMT-stack overflow behavior;
- native FP max semantics matching the planned NaN/signed-zero policy;
- deterministic cross-backend timeout, completion, and cache-flush observability.

## 7. Validation Assets

Use three tiers:

1. Minimal OpenCL and LLVM ABI goldens.
2. `ventus-env/regression-test.py` for stack regression.
3. Selected `ventus-env/OpenCL-CTS` topics.

Existing semantic references under `testcases/_get_case/AIops`, MNIST, and
MNIST_conv_small include GEMM, Softmax, Conv, fused operators, BatchNorm,
Pool2D, ReLU, and inference examples. They are useful runtime references, but
small goldens should remain the ABI specification.

## 8. Local Experimental Changes

Uncommitted Spike, Driver, POCL, and CycleSim experiments include the current
`vftta_vv.h` patch, a POCL source change, log-level and progress controls,
changed wait/timeout behavior, `REGEXTI` fixes and tests, stale operand-response
handling, initialization fixes, and new CTest coverage. They must be committed
and pinned before CI, ABI identity, or Triton documentation treats them as
stable.
GPGPU has untracked documents; SystemC, testcases, OCL ICD, and the superproject
also contain generated/build artifacts. Commit identity alone is insufficient
for dirty components; use source diff/content hashes as specified by the
toolchain audit.

## 9. Derived Project Target Summary (Not Existing Integration)

The project intends to target the inspected facilities through a producer-local
Triton adapter, textual LLVM IR/tool boundary, versioned artifacts, mandatory
Spike validation, and a narrow fixed FP32 MMA contract.
This is a concise derived summary, not a second normative profile or an assertion
that the inspected `ventus-env` already contains a Triton backend. The complete
profile, accepted kernel set, milestones, and exclusions are defined only by
[Triton-for-Ventus V1 Scope](2026-08-28-triton-for-ventus-v1-scope.md).

Use Triton's external plugin mechanism first. The selected Triton revision uses
`GPUTarget(backend, arch, warp_size)`; target triple, pointer width, ABI version,
and tool paths belong in backend options and artifact identity.

The LLVM version gap is handled textually for M1. Centralized launch ABI,
resource-section consumption, capability queries, and explicit ELF loading
remain project integration work; milestone definitions are intentionally not
duplicated here.
