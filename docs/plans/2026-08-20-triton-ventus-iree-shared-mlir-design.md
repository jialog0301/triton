# Triton Ventus Backend and IREE Shared MLIR Design

## 1. Purpose

Current implementation facts and pinned component identities are recorded in
`2026-08-25-ventus-env-facility-snapshot.md`. This document defines the desired
architecture and does not imply that the versioned artifact, IREE HAL
integration, or physical hardware backend already exists.
The authoritative Triton-for-Ventus V1 scope, profile, milestones, and
exclusions are defined in
[Triton-for-Ventus V1 Scope](2026-08-28-triton-for-ventus-v1-scope.md); where
this broader architecture discusses later model/runtime work, it is not a V1
commitment.

This document defines a Ventus inference architecture with one model deployment
mainline and one operator-development side path. The design has five goals:

1. Use PyTorch for model development and NVIDIA training, then export models to
   IREE for Ventus inference deployment.
2. Use Triton as an operator laboratory for kernel development, tuning, and
   validation of the MLIR and Ventus LLVM paths.
3. Preserve Triton's GPU tiling, layout, thread, warp, and work-group semantics
   inside independently developed operator kernels.
4. Reuse the same low-level Ventus target support from Triton and IREE.
5. Package tuned Triton kernels as versioned Ventus artifacts that IREE can
   select, embed, load, and dispatch.

The first implementation milestone targets an inference-oriented subset:

- Elementwise operations and broadcasting.
- Masked global loads and stores.
- Basic reductions and softmax building blocks.
- A fixed static blocked FP32 `32x32x32` matrix pattern for correctness and
  reference/fallback.
- A fixed RTL-profile FP32 MMA vertical slice after M3 contract freeze, not
  general `tl.dot`.
- General-path local size `[32,1,1]` or `[64,1,1]`; MMA exactly `[32,1,1]`.

General TorchInductor-to-Ventus model execution is not a baseline requirement.
General/low-precision MMA, arbitrary matrix shapes, shuffle/ballot, atomics,
asynchronous copies, clusters, warp specialization, FP16/BF16/FP8, and a
physical driver are excluded from V1. The one target-specific exception is the
M3/M4-gated fixed FP32 `VentusMmaProfileA` vertical slice.
The completed RTL compatibility review leaves the basic backend actionable, but
adds M1/M2 contract work for bounded divergence, restricted AS4/AS5 use,
resource units, ELF/capability identity, simulator physical addresses, and
deterministic completion. It does not freeze MMA.

## 2. Core Decisions

The design adopts the following decisions:

1. Use `torch.export` and IREE as the complete-model inference mainline.
2. Use a native TritonGPU backend for operator development instead of lowering
   Triton back to Linalg.
3. Keep TTIR and TTGIR as the Triton-side optimization and distribution layers.
4. Lower both Triton kernels and IREE dispatch kernels against the same Ventus
   target contract; until LLVM revisions align, use producer-specific adapters
   and a producer-owned LLVM 16 compatibility stage plus textual LLVM IR/tool
   boundary rather than assuming one compiled MLIR library.
5. Treat the Ventus LLVM fork as a modifiable project component, not a fixed
   external black box.
6. Use Ventus LLVM for kernel ABI lowering, uniform/varying handling, divergent
   control flow, reconvergence, instruction selection, register allocation,
   resource collection, and ELF generation.
7. Use IREE as the model compiler, executable packaging layer, and production
   runtime.
8. Keep POCL as an ABI oracle and correctness reference, not as a required
   production runtime dependency.
9. Build the shared target layer primarily from standard MLIR dialects, with a
   small amount of custom Ventus IR only where standard dialects cannot express
   a stable target contract.
10. Permit small, focused changes to Triton common code when target hooks or
   layout support cannot remain completely out of tree.

## 3. Why Triton-to-Linalg Is Not the Main Path

The rejected integration path is:

```text
Triton operator kernel
  -> Triton-to-Linalg
  -> IREE tiling and distribution
  -> Ventus
```

This path turns a GPU-tiled SPMD program back into structured tensor
operations and then asks IREE to distribute it again. That creates several
problems:

- TTGIR layout and scheduling decisions can be lost.
- Tiling and distribution are performed twice.
- Triton pointer arithmetic, masks, shared memory, barriers, and atomics are
  difficult to reconstruct as Linalg operations.
- Failures span too many compiler layers.
- Independently tuned Triton kernels cannot be preserved reliably.

Triton-to-Linalg remains useful for semantic migration, compatibility
experiments, and compiler research, but it is not the production backend
architecture for Ventus.

## 4. Target Architecture

### 4.1 Model Inference Mainline

```text
PyTorch -> torch.export -> Torch MLIR / StableHLO -> IREE Flow dispatch
                                                     |
                         +---------------------------+-------------------------+
                         |                                                     |
                         v                                                     v
             Dispatch device codegen                                Stream/HAL lowering
       Linalg/SCF/Vector/GPU or selected                          buffers/dependencies/
               Triton artifact                                   bindings/commands
                         |
          Shared Ventus MLIR target layer
                         |
                 MLIR LLVM Dialect
                         |
                  Native LLVM IR
                         |
       Modified Ventus LLVM (-mcpu=ventus-gpgpu)
                         |
             Ventus ELF + kernel metadata
                         |
                         +---------------------------+-------------------------+
                                                     |
                                                     v
                                          IREE Ventus executable
                                                     |
                                          IREE HAL Ventus driver
                                                     |
                                           Ventus C driver API
                                                      |
                              Spike / CycleSim / RTL simulation / GVM
```

Physical hardware requires a future concrete driver backend and is not part of
the inspected execution baseline.

### 4.2 Triton Operator Development Side Path

```text
Triton Python DSL
  -> TTIR
  -> TTGIR
  -> TritonGPUToVentus target adapter
  -> Shared Ventus MLIR target layer
  -> MLIR LLVM Dialect
  -> Modified Ventus LLVM
  -> Ventus ELF + metadata
  -> Spike / CycleSim / RTL tuning and validation
  -> versioned Ventus kernel library
  -> optional selection by IREE dispatch codegen
```

Triton is not the baseline full-model graph owner. It provides an operator DSL,
autotuning input space, MLIR validation source, LLVM backend stress suite, and
producer of optimized kernels. IREE remains responsible for complete-model
dispatch formation, buffer planning, executable packaging, and deployment.

The two paths share the physical target contract and low-level lowering, but
not TTGIR, Linalg tiling, or other high-level scheduling IR.

The inspected Ventus compiler is an LLVM 16 fork while Triton uses a newer
LLVM/MLIR revision. Therefore the diagrams in this section are conceptual: the
V1 concrete baseline is producer-local Triton and IREE adapters lowering through
`VentusLLVM16Compatibility` to `kernel.ventus.ll`, followed by pinned Ventus LLVM
tool invocation. The compiled
`ventus-mlir` tree described below remains the target architecture after
deliberate revision alignment; it is not a V1 prerequisite or an already
implemented shared library.

For V1, “shared low-level lowering” means a shared tested semantic/ABI contract,
not a required compiled utility. Triton Task 10 implements SPMD lowering
producer-locally. The compiled shared utilities in the later component layout
are a conditional post-alignment replacement only and are not a V1 dependency.

### 4.3 Ventus Execution Mapping

The initial mapping is:

| Triton concept | Ventus concept |
| --- | --- |
| Triton program / CTA | Ventus block / OpenCL work-group |
| Triton warp | Ventus warp |
| Triton thread | Ventus work-item / lane |
| Warp size | 32 |
| `program_id` | Work-group ID |
| Local thread ID | Work-item ID |
| Global memory | LLVM address space 1 |
| Shared memory | LLVM address space 3 |
| Constant memory | LLVM address space 4 |
| Private memory | LLVM address space 5 |
| CTA barrier | Ventus work-group barrier |

The initial pinned target profile requires 32 lanes per warp and uses one
Triton CTA per Ventus work-group. The default RTL profile supports at most eight
warps and 256 work-items per work-group. The source and runtime metadata are
parameterized, so the required warp size must be recorded and validated rather
than treated as a universal Ventus invariant. The first version does not enable
multi-CTA cluster semantics. A kernel may still launch any number of
work-groups.

For V1, the AS4 row means a validated read-only ABI convention, not an
independent constant-memory allocation/upload guarantee. AS5/PDS is only a
spill/private-resource convention, not general user-visible memory support.
Likewise, Ventus LLVM may own reconvergence lowering only for bounded and
validated divergent CFG; the inspected SIMT stack overflow behavior is not yet
an architectural guarantee.

V1 fixes `riscv32`/`ventus-gpgpu`, RV32 pointers, and scalar `i1/i32/f32`.
Only one-dimensional grids are supported. General kernels use local size
exactly `[32,1,1]` or `[64,1,1]`; the MMA vertical slice uses exactly
`[32,1,1]`. The RTL eight-warp/256-work-item limits are validation ceilings and
do not enlarge this accepted geometry.

Ventus must not be modeled as a normal CPU RVV target or as an explicitly
vectorized `<32 x T>` function. TTGIR lowering produces per-logical-thread
scalar semantics. Ventus LLVM determines which values belong in scalar or
vector registers and lowers divergent branches to Ventus SIMT control flow.

## 5. Shared Ventus MLIR Target Layer

### 5.1 Design Principle

The shared layer is not a complete custom IR. It is a target support component
composed primarily of standard MLIR dialects:

```text
arith + math + scf + cf + memref + gpu + vector + LLVM
                              +
      thin Ventus target, ABI, metadata, and special operations
```

The rule for adding custom IR is:

```text
If a standard dialect expresses the semantics accurately:
  use the standard dialect.

If Ventus LLVM can infer and lower the semantics reliably:
  keep generic control flow and arithmetic.

If Triton and IREE require a shared, verified target contract that standard
dialects cannot express:
  add a typed Ventus attribute, interface, or operation.

If Ventus-specific semantics must survive and be transformed by several MLIR
passes:
  add a Ventus operation.
```

### 5.2 Standard Dialect Responsibilities

The shared pipeline should continue to use:

- `arith` and `math` for scalar arithmetic, comparisons, and math operations.
- `scf` for structured loops and branches.
- `cf` for lowered control flow.
- `memref` for shaped buffer views before LLVM conversion.
- `gpu` for generic thread, block, grid, and barrier semantics where useful.
- `vector` for explicit short-vector operations that are independent of the
  Ventus SIMT warp representation.
- `LLVM` for pointers, GEPs, loads, stores, calls, function ABI, and intrinsics.

The first version must not introduce Ventus-specific add, multiply, compare,
load, store, loop, or generic branch operations.

### 5.3 Thin Ventus Contract

The first shared target layer should define typed target and kernel contracts,
preferably as attributes attached to `module`, `func.func`, or `llvm.func`:

```text
ventus.target
ventus.kernel
ventus.kernel_abi
ventus.workgroup_size
ventus.required_warp_size
ventus.shared_memory_size
ventus.target_features
```

Illustrative form:

```mlir
module attributes {
  ventus.target = #ventus.target<
    arch = "ventus-gpgpu",
    pointer_width = 32,
    warp_size = 32
  >
} {
  func.func @kernel(...)
      attributes {
        ventus.kernel,
        ventus.workgroup_size = [32, 1, 1],
        ventus.kernel_abi = #ventus.kernel_abi<packed_args_v1>
      }
}
```

The pinned LLVM fork already provides work-group and subgroup Ventus barrier
intrinsics, including scoped forms. The only likely custom operation required
in the first version is a typed work-group barrier if `gpu.barrier` cannot
preserve the memory scope and semantics required by those intrinsics:

```text
ventus.workgroup_barrier
```

Thread ID, work-group ID, block dimension, and grid dimension should remain
generic GPU operations. The inspected toolchain currently realizes them through
libclc calls and linked `__builtin_riscv_*` assembly routines, not stable LLVM
ID intrinsics. Stable ID intrinsics are a backend-hardening goal rather than an
existing baseline contract.

The first version must not define generic target-specific duplicates such as:

```text
ventus.split
ventus.join
ventus.vbranch
ventus.load
ventus.store
ventus.reduce
ventus.async_copy
```

The narrow exception is a proposed `VentusMmaEncodingAttr` plus
`VentusDotOperandEncodingAttr` for the fixed profile. They remain proposed until
M3 freezes the mandatory RTL/Spike contract, with CycleSim capability-gated,
and these attributes must not become a generic
`ventus.mma` operation or arbitrary-shape encoding.

In particular, SCF/CF/LLVM branches remain generic so that Ventus LLVM passes
such as divergent-branch conversion and mixed-PHI repair remain the single
owner of SIMT reconvergence lowering.

### 5.4 Shared Component Layout After Revision Alignment

After LLVM/MLIR revisions are deliberately aligned, the target architecture is
a compiled shared component independent of both Triton and IREE:

```text
ventus-mlir/
  include/ventus/
    Dialect/Ventus/
      VentusAttrs.td
      VentusOps.td
      VentusInterfaces.td
    Conversion/GPUToVentusLLVM/
      GPUToVentusLLVM.h
    Conversion/VentusToLLVM/
      VentusToLLVM.h
      Passes.td
    Target/
      VentusABI.h
      VentusMetadata.h
      VentusTarget.h
  lib/
    Dialect/Ventus/
    Conversion/GPUToVentusLLVM/
    Conversion/VentusToLLVM/
    Target/
```

The dependency direction is:

```text
Triton adapter --------\
                        -> ventus-mlir -> Ventus LLVM
IREE adapter ----------/

ventus-mlir must not depend on Triton.
ventus-mlir must not depend on IREE.
```

## 6. Triton Backend Design

### 6.1 Backend Structure

The Triton integration should be mostly out of tree:

```text
third_party/ventus/
  backend/
    compiler.py
    driver.py
  include/
    TritonGPUToVentus/
      TargetInfo.h
      LLVMCompatibility.h
      Passes.h
      Passes.td
  lib/
    TritonGPUToVentus/
      TargetInfo.cpp
      LLVMCompatibility.cpp
      SPMDOpToLLVM.cpp
      MemoryOpToLLVM.cpp
      FuncOpToLLVM.cpp
      TritonGPUToLLVM.cpp
  triton_ventus.cc
```

Small common Triton changes are allowed for target hooks, layout queries, or
generic conversion support. Target-specific code should not be added to common
passes when an interface or extension point can express it.

### 6.2 Reuse of TritonGPUToLLVM

The backend should implement a Ventus `TargetInfo` compatible with Triton's
common lowering infrastructure and reuse common patterns where possible:

- Elementwise operations.
- Control flow.
- Tensor view and pointer operations.
- Masked loads and stores.
- Reductions supported by the common target interface.
- Shared-memory allocation analysis.
- Layout decomposition.

Ventus-specific patterns should be limited to:

- SPMD builtins.
- Kernel function ABI and attributes.
- Address-space conversion.
- Work-group barrier lowering.
- Unsupported shuffle, ballot, reduction, and atomic cases.
- Target data layout and module flags.
- Artifact and metadata emission.
- Producer-side LLVM 16 compatibility checking and deterministic
  `kernel.ventus.ll` emission.

The initial implementation should not copy the complete NVIDIA, AMD, or PPU
conversion backend.

### 6.3 Ventus LLVM 16 Compatibility Boundary

`VentusLLVM16Compatibility` is a first-class Triton Ventus Backend component,
not part of the Ventus LLVM provider implementation. Its checker API may be
named `LLVM16CompatibilityChecker`. The formal input is producer-local MLIR LLVM
Dialect/native IR before serialization; the formal artifact is
`kernel.ventus.ll`, a deterministic textual subset satisfying the tested Ventus
LLVM 16 contract.

The component constrains producer IR rather than translating arbitrary LLVM
versions. It must not use string substitution. Post-serialization normalization
is permitted only through structured parsing and an explicit allowlist, and it
must never silently remove semantic attributes, metadata, or module flags.
Unknown features warn and fail closed.

The initial allowlist covers the required `riscv32` triple and RV32 data layout,
32-bit device pointers, `ventus_kernel`, `i1/i32/f32`, legal AS1/AS3, restricted
read-only AS4 and backend-only AS5, tested builtin/barrier signatures, and
ordinary scalar CFG/phi/load/store/GEP/allowlisted calls. The denylist covers
untested or LLVM 16-unsupported attributes, intrinsics, metadata/module flags,
scalable vectors, i64 device pointers/addresses, atomics, EH/`invoke`,
unsupported calls, all address-space casts, AS4 writes, user AS5, and
unsupported memory order/scope. More precisely, V1 denies every
`addrspacecast`, including otherwise well-typed casts, and denies the complete
atomic family represented by `atomicrmw`, `cmpxchg`, atomic load/store, and
`fence` with any ordering/scope. This contract is based on tested compatibility;
it does not assert that every newer textual feature is inherently semantically
incompatible.

M1 has three distinct compile gates: internal checker, absolute Ventus LLVM 16
`opt` parse/verify, and absolute Ventus LLVM 16 `llc` codegen to object. Syntax
acceptance and target/codegen semantics are distinct evidence; `opt -verify`
alone is insufficient. Task 12 ELF/Spike execution is a fourth gate. The
manifest records `kernel.ventus.ll` SHA-256, deterministic diagnostics, absolute
argv/stdout/stderr/status for `opt` and `llc`, and the object hash.

After Gate 3, `third_party/ventus/backend/compiler.py` owns the producer-local V1
object-to-ELF stage. It invokes absolute `VENTUS_LLD` with a Task 1-pinned linker
script, object input, crt0, libclc/work-item runtime inputs, and kernel entry/init
contract, then emits the ELF. Concrete installed filenames must come from Task 1
and ABI goldens. The manifest records all input identities/hashes, absolute
argv/stdout/stderr/status, ELF hash, and validation result before Spike runs.

This pipeline is invariant for Tasks 11, 18, 19, 20, 20B, and future Triton
compiler paths: each module emits/hashes `kernel.ventus.ll`, runs Gates 1-3,
links through the defined ELF stage, and records evidence. Direct bypass is not
an accepted V1 implementation.
Gate 4 evidence adds the ELF hash, launcher input/manifest hash, pinned Spike
binary identity, execution result/status, and test-result hash. Missing pinned
Spike/runtime is a milestone/release failure, not a compatible artifact result.

### 6.4 TTGIR Layout Scope

The first version should prioritize existing generic encodings:

```text
BlockedEncodingAttr
SliceEncodingAttr
SharedEncodingAttr
DotOperandEncodingAttr where generic lowering is possible
proposed VentusMmaEncodingAttr / VentusDotOperandEncodingAttr for the fixed profile
```

Initial constraints:

- Warp size is fixed at 32.
- `BlockedEncodingAttr` is the primary distributed layout.
- Work-group size is a compile-time specialization.
- No CTA clusters.
- No warp specialization.
- No NVIDIA TMA or `cp.async` semantics.
- No general target-specific MMA encoding or arbitrary `tl.dot`.

Ordinary FMA support has mandatory canonical accepted case static blocked FP32
`M=N=K=32`, used only as a correctness reference/fallback rather than a general
matrix path. Additional static blocked patterns enter the accepted set only with
explicit lowering and correctness tests; none of this implies general `tl.dot`.
The current Chisel source contains an experimental FP32
`VFTTA_VV` tensor facility, but the source default, checked-in generated Verilog,
Spike, and CycleSim do not yet identify one profile and no stable LLVM
intrinsic, lane layout, masking, or resource contract exists. V1 nevertheless
targets one fixed vertical slice after M3 contract freeze:

```text
VentusMmaProfileA
logical D[M,N] = A[M,K] x B[K,N] + C[M,N]
(M,K,N) = (4,8,4)
logical M=constructor DimM, K=constructor DimN, N=constructor DimK
labeled logical order=(M,K,N), constructor positional order=(M,N,K)
physical B fragment = B^T[N,K] = [4,8]
instruction = vftta.vv
old vd=C[M,N], vs1=A[M,K], vs2=physical B^T[N,K]; new vd=D[M,N]
dtype = FP32, local size = [32,1,1], full active warp, no masked MMA
```

M3 starts by regenerating RTL from pinned Chisel and creating one verified
generation record that pins/hashes the source and binds it to effective parameters,
generated Verilog, and testbench. Dimensions are always labeled with logical
`M=DimM`, `K=DimN`, `N=DimK`; logical order is `(M,K,N)` and constructor positional
order is `(M,N,K)`, never an unlabeled tuple. The default source implies 16 meaningful
outputs with 8-element reductions, while the inspected checked-in
`gpgpu/driver/rtl/GPGPU_top.v` appears to contain only 4 dot units with
4-element dots. Until that mismatch is resolved, neither generated Verilog nor
simulator behavior can freeze profile A.

The complete M3 contract also freezes A/B lanes `0..31`, C/D meaningful lanes
`0..15`, all-32-lane writeback and unused-lane values, tied-destination
three-explicit-operand syntax `vftta.vv vd, vs2, vs1`, mandatory old
`vd=C`/`vs1=A`/`vs2=B^T` to new `vd=D` binding, versus quarantined legacy CycleSim
four-register fixtures, exact FP32 tree/rounding/NaN/exceptions, `fflags`
production/aggregation/architectural writeback, encoded `vm` mask-control bit,
unmasked/e32/full-active/fixed width versus
architectural `vl`, register range/alignment/overlap, latency/issue/backpressure,
and resource units. M4 is disabled unless the manifest records every field.

M4 lowers only a Triton `tl.dot` proven to match this contract. Any other shape,
dtype, mask, layout, or active-lane condition is rejected, except that a pattern
already in the explicit blocked accepted set may use ordinary FMA fallback.

## 7. IREE Integration

### 7.1 IREE Is the Model Deployment Mainline

IREE owns complete-model deployment:

- StableHLO input conversion.
- Graph optimization and dispatch formation in Flow.
- Buffer lifetime, allocation, transfer, and dependency planning in Stream.
- Device executable codegen for generic kernels.
- Selection and import of precompiled optimized Triton artifacts.
- Executable packaging and HAL dispatch generation.
- Runtime device, buffer, command, queue, and synchronization management.

Triton does not replace Flow, Stream, or HAL. A Triton kernel may replace the
device-codegen result for one selected dispatch, but it does not replace the
surrounding resource and runtime lowering.

### 7.2 Device Codegen and Stream/HAL Lowering

IREE device codegen and Stream/HAL lowering are cooperating branches after a
Flow dispatch is formed, not a simple pipeline in which codegen starts after
HAL lowering:

```text
                         IREE Flow dispatch
                                |
              +-----------------+-----------------+
              |                                   |
              v                                   v
      Dispatch device codegen              Stream/HAL lowering
      Linalg/SCF/Vector/GPU                 buffer lifetime
      or selected Triton ELF                allocation/transfer
              |                             dependencies/commands
              +-----------------+-----------------+
                                |
                                v
                     IREE HAL executable
                  entry point + bindings + workload
                                |
                                v
                          HAL dispatch
```

The HAL driver must not contain Linalg tiling, kernel fusion, or target MLIR
codegen. It consumes a completed executable artifact, argument bindings, and
launch geometry.

### 7.3 Runtime Boundary

IREE is the production runtime owner:

```text
IREE Runtime
  -> IREE HAL Ventus driver
  -> Ventus C driver API
  -> Spike / CycleSim / RTL simulation / GVM
```

Physical-device execution remains capability-gated future work.

POCL is not part of this final chain. It remains useful for:

- Producing ABI reference LLVM IR and ELF files.
- Verifying calling conventions and address spaces.
- Cross-checking argument packing and launch metadata.
- Running differential tests against the IREE HAL path.
- Detecting ABI changes after Ventus toolchain upgrades.

### 7.4 First Triton-to-IREE Integration Mode

The first IREE integration should use precompiled external kernel artifacts:

```text
Triton kernel
  -> Ventus ELF + metadata
  -> imported into an IREE executable
  -> IREE HAL dispatch
```

This mode keeps Triton and IREE compiler versions loosely coupled and allows
each side to be tested independently.

Long-term integration may invoke Triton code generation from the IREE compiler
or select from a tuned kernel database:

```text
IREE dispatch
  -> kernel selection
       -> precompiled Triton artifact
       or Triton codegen invocation
  -> IREE executable
```

That is not a first-stage dependency.

### 7.5 Kernel Selection and Fallback

The intended selection policy is:

```text
IREE dispatch or @ventus.* custom call
  -> matching tuned Triton artifact exists
       -> import ELF and metadata
  -> no compatible artifact exists
       -> use IREE generic Ventus codegen
```

Selection must validate shape, dtype, ABI version, target features, toolchain
identity, work-group constraints, and resource limits. A missing optimized
kernel is not a model compilation failure when generic IREE codegen supports
the operation.

### 7.6 Why IREE Is Needed When PyTorch Can Run Inference

PyTorch can evaluate `model.eval()`, but that does not provide a Ventus device
backend automatically. Direct PyTorch execution on Ventus would require a
substantial runtime and framework integration:

- A PyTorch device, allocator, stream, event, and copy implementation.
- ATen operator registration or a complete compiler fallback path.
- Kernel compilation, caching, launch wrappers, and error handling.
- Model weight loading, temporary-buffer management, and device synchronization.
- Deployment with Python or LibTorch and a large dynamic dispatcher.

IREE solves a different problem from PyTorch. PyTorch remains the model
development, training, and export environment; IREE cross-compiles the exported
model into a target executable and provides a compact deployment runtime:

```text
PyTorch:
  model semantics, training, checkpoints, export

IREE compiler:
  graph optimization, dispatch formation, buffer planning, target codegen,
  executable packaging

IREE runtime/HAL:
  device, allocator, buffers, executable loading, commands, queues, dispatch,
  synchronization
```

This division avoids implementing the complete PyTorch backend and ATen device
ecosystem before Ventus can run an AOT inference model. It also keeps the target
deployment independent of Python and permits compile-time memory planning.

IREE is not mandatory in principle. A future direct PyTorch Ventus backend may
be added for interactive development, but it is not the baseline deployment
architecture.

### 7.7 Model and Weight Packaging

Model parameters should not all become large MLIR literals. The export and
packaging path should preserve parameter names and bindings, then convert
`state_dict` or safetensors data into a separate aligned constant archive:

```text
PyTorch checkpoint / safetensors
  -> name and signature mapping
  -> dtype conversion where required
  -> target layout and alignment packing
  -> constant archive

IREE executable
  + constant archive
  + selected Ventus kernel artifacts
  + dispatch and ABI metadata
  + model configuration
  -> versioned Ventus model package
```

The package must version model ABI, kernel artifact ABI, target architecture,
toolchain identity, and weight layout independently enough to reject
incompatible combinations during loading.

## 8. Unified Kernel Artifact and ABI

### 8.1 Artifact Contract

Triton and IREE should exchange a stable target artifact rather than TTIR,
TTGIR, or IREE internal IR:

```text
VentusKernelArtifact
  - ELF bytes
  - entry-point symbol
  - target architecture and features
  - argument layout
  - work-group size
  - grid calculation contract
  - shared-memory size
  - private-memory size
  - SGPR and VGPR usage
  - ABI version
```

The artifact schema must be versioned. It may initially use a simple typed C++
structure plus JSON for debugging, followed by a stable binary schema when
IREE executable packaging is implemented.

### 8.2 Kernel ABI

The first ABI should be compatible with the selected Ventus toolchain revision
and POCL-generated reference kernels. The contract must specify:

- `ventus_kernel` calling convention.
- RV32 device pointers.
- Packed argument block layout.
- Scalar size and alignment.
- Buffer binding to argument offset mapping.
- Global address space 1 pointers.
- Shared address space 3 objects.
- Constant address space 4 objects.
- Private address space 5 objects.
- Work dimension, global size, local size, and global offset metadata.
- Static shared-memory accounting. Dynamic local arguments remain a separate,
  currently incomplete runtime capability.
- ELF resource section naming and contents.
- AS4 as a restricted read-only convention and AS5/PDS as spill/resource only.
- Canonical LDS-byte, PDS, SGPR, and VGPR units/aggregation domains.
- ELF content/target/entry identity, an independent versioned capability record,
  simulator physical-address convention, and deterministic timeout/completion/
  cache-flush observability.

Argument packing and ELF resource parsing should live in shared target code,
not be implemented independently in Triton and IREE.

The current runtime has four distinct metadata layers: the packed argument
buffer, a 64-byte device kernel metadata block, a host driver launch structure,
and the raw ELF resource section. The versioned artifact manifest must identify
and validate these layers rather than assuming one unified metadata object
already exists. Current POCL does not consume the resource section and supplies
hard-coded LDS/PDS/SGPR/VGPR values.
The existing common ELF loader's `PT_LOAD` handling and current `vt_dev_caps`
surface are implementation inputs, not sufficient validation/capability APIs.

### 8.3 Toolchain Versioning

Development must pin one `ventus-env` revision and its LLVM, POCL, driver,
Spike, CycleSim, GPGPU/RTL, SystemC, CTS, and testcase revisions. Dirty
components require reviewed patch or content hashes. The target artifact should record a toolchain
identity so that incompatible ELF files fail during loading instead of
producing silent execution errors.

The inspected Ventus compiler is an LLVM 16 fork, while the selected Triton
revision uses substantially newer LLVM/MLIR revisions. "Shared MLIR support"
therefore initially means a shared semantic specification, textual test corpus,
ABI schema, and producer-specific adapters. A single shared C++ MLIR library is
valid only after both producers use a deliberately aligned LLVM revision.

## 9. Responsibility Boundaries

### 9.1 Triton Owns

- Operator-level Triton Python compilation.
- TTIR and TTGIR optimization.
- Tensor-to-thread/warp/work-group layout.
- Triton pointer and mask semantics.
- Shared-memory tiling chosen by Triton.
- Conversion of Triton-specific operations to the shared target level.
- Operator autotuning and performance experiments.
- Stress testing of shared MLIR and Ventus LLVM lowering.
- Production of versioned optimized kernel artifacts.
- Producer-side Ventus LLVM 16 compatibility checking, `kernel.ventus.ll`
  emission, and recording all three compile-gate results.

### 9.2 Shared Ventus MLIR Support Owns

- Ventus target attributes and verification.
- Kernel ABI attributes and conversion.
- Generic GPU builtin lowering to Ventus LLVM builtins.
- Work-group barrier lowering.
- Ventus address-space contract.
- LLVM target data layout and function attributes.
- Shared metadata and artifact structures.
- Ventus LLVM tool invocation utilities where practical.

For the unaligned V1 baseline, shared support defines semantic/ABI facts while
each producer owns its compatibility implementation. Task 8's LLVM input
contract is provider-side; Triton's compatibility component is producer-side.

### 9.3 Ventus LLVM Owns

- Concrete implementation of the Ventus kernel calling convention and kernel
  argument loads.
- Uniform/varying instruction classification.
- Scalar/vector register-domain selection.
- Divergent branch conversion.
- SIMT reconvergence and join insertion.
- Mixed PHI repair.
- Address-space and load/store legalization.
- VV-to-VX/VF target optimization.
- RVV/Ventus instruction selection.
- Register allocation.
- Spill handling and `REGEXT` insertion.
- Machine scheduling and target-specific peephole optimization.
- Final SGPR/VGPR/LDS/PDS resource information.
- Assembly, object, and ELF generation.

The Ventus LLVM fork is expected to change as Triton and IREE expose missing or
fragile backend behavior. Priority improvements include stable builtins and
intrinsics, divergence propagation, shared-memory support, versioned resource
metadata, a Ventus-specific scheduling model, and new instruction patterns.

### 9.4 IREE Owns

- Model-level dispatch formation.
- Buffer lifetime and memory planning.
- Generic Ventus dispatch codegen.
- Optimized Triton kernel selection and artifact import.
- Executable packaging.
- Buffer binding and push constants.
- Command buffers, synchronization, and queues.
- ELF loading and entry-point lookup.
- Launch geometry and native Ventus driver integration.

## 10. Implementation Phases

### Phase 0: Freeze Contracts and Golden References

- Pin the Ventus toolchain revision.
- Generate OpenCL/POCL LLVM IR and ELF golden files.
- Document calling convention, data layout, address spaces, argument packing,
  builtin representation, and resource sections.
- Define versioned kernel artifact metadata.

Exit criteria:

- A minimal hand-written LLVM kernel can be compiled, loaded, and launched.
- The same argument packer can launch a POCL reference ELF.

### Phase 1: Shared Ventus MLIR Target Support (Post-Alignment Alternative)

This compiled-library phase is conditional on deliberate LLVM/MLIR revision
alignment. Without alignment, V1 implements the same semantic contract in each
producer-local adapter and crosses a textual LLVM IR/tool boundary.

- Add target and kernel ABI attributes.
- Implement verifiers.
- Lower generic GPU IDs and barrier semantics to Ventus LLVM-compatible IR.
- Emit the Ventus kernel calling convention, data layout, and attributes.
- Add LLVM IR and ELF metadata tests.

Exit criteria:

- A standard-dialect MLIR vector-add kernel lowers to ABI-correct Ventus LLVM
  IR and ELF.

### Phase 2: Ventus LLVM Contract and Backend Hardening

- Define a verified LLVM input contract shared by IREE and Triton.
- Add stable Ventus builtins/intrinsics for IDs, barriers, and other baseline
  target semantics.
- Add tests for kernel calling convention, argument loading, divergence,
  mixed PHIs, address spaces, and multi-return control flow.
- Version the `.ventus.resource.<kernel>` payload or its replacement.
- Expose SGPR/VGPR/LDS/PDS resource information through a stable parser.
- Establish a Ventus-specific scheduling-model roadmap instead of treating the
  inherited Rocket model as a permanent performance model.

Exit criteria:

- OpenCL, standard MLIR, and hand-written LLVM inputs use the same documented
  Ventus LLVM contract.
- Ventus LLVM emits verified ELF and versioned resource metadata for all ABI
  reference kernels.

### Phase 3: Minimal Triton Ventus Operator Backend

- Register the Triton backend and target.
- Add TTIR and TTGIR stages.
- Implement Ventus `TargetInfo` and reuse common TritonGPU-to-LLVM patterns.
- Support blocked layouts, SPMD builtins, elementwise operations, and masked
  global memory operations.
- Compile Triton kernels to the unified artifact.
- Enforce `riscv32`/`ventus-gpgpu`, RV32, warp 32, one CTA per work-group,
  one-dimensional grid, and general local sizes `[32,1,1]`/`[64,1,1]`.
- Include versioned manifest, canonical argument packing, ELF/resource parsing,
  range validation, and launch-time ABI/resource rejection in the M1 baseline.
- Emit `kernel.ventus.ll` through the Triton-owned compatibility checker and
  require internal checking, absolute Ventus `opt` parse/verify, and absolute
  Ventus `llc` object codegen before Spike execution.

Exit criteria:

- Triton vector add and masked copy run correctly on Ventus Spike.
- Fill, copy, multiply, fused elementwise, broadcast, reshape/view, select, and
  ReLU also run through the same Task 12 Spike pipeline.
- LLVM IR is structurally comparable with POCL golden references.

### Phase 4: V1 M2 Triton Operator Coverage

- Support static shared memory in address space 3.
- Lower work-group barriers.
- Validate multi-warp work-groups.
- Implement correctness-first fixed-extent `sum`/`max` only after the shared-memory
  and barrier path is available.
- Implement blocked-layout FMA matmul without target-specific MMA.
- Limit blocked-layout FMA to the accepted static FP32 `32x32x32` pattern.

Exit criteria:

- The accepted blocked FP32 `32x32x32` reference/fallback is numerically correct
  on Spike; CycleSim is an additional capability-gated validation backend.
- Static AS3/barrier tests pass before dependent multi-warp reduction tests.

### Phase 5: V1 M3/M4 Fixed MMA Contract And Integration

- M3 freezes `VentusMmaProfileA` only after one verified generation record binds
  pinned Chisel source/hash, effective parameters, generated Verilog, and testbench,
  and the regenerated structure has the expected 16 dot units and 8-element reductions.
- Freeze explicit logical `M=DimM`, `K=DimN`, `N=DimK`, logical order `(M,K,N)`,
  constructor positional order `(M,N,K)`, and no unlabeled tuple.
- Align mandatory Spike/RTL shape, lane fragments, 16 meaningful outputs versus
  32 writeback lanes, old `vd=C`/`vs1=A`/`vs2=B^T` to new `vd=D` binding,
  full-active/unmasked/e32/fixed-width behavior, exact FP semantics, `fflags`
  production/aggregation/writeback, register constraints, latency/backpressure,
  and resource contracts. CycleSim is capability-gated and is not independent
  RTL architectural/performance proof.
- M4 introduces the narrow proposed `VentusMmaEncodingAttr` and
  `VentusDotOperandEncodingAttr` and lowers only matching FP32 M4K8N4 Triton
  `tl.dot` to `vftta.vv`; it remains disabled unless all M3 manifest fields are frozen.
- Negative tests reject swapped `vs1`/`vs2` and a non-tied accumulator.

### Phase 6: Post-V1 IREE Ventus Model Runtime Baseline

- Implement the IREE Ventus executable format.
- Embed or reference Ventus ELF artifacts.
- Implement argument packing and buffer binding.
- Implement ELF loading, symbol lookup, launch, and synchronization.
- Connect Spike and capability-gated CycleSim backends.

Exit criteria:

- IREE HAL loads and dispatches a precompiled Triton vector-add artifact.
- Results match the POCL reference path.

### Phase 7: Post-V1 IREE Kernel Library Selection And Tuning

- Import versioned Triton artifacts into IREE executables.
- Match kernels by operation, shape, dtype, target features, ABI, and resource
  constraints.
- Fall back to IREE generic codegen when no compatible tuned kernel exists.
- Package model executable, weights, selected kernels, and metadata together.
- Measure and tune memory coalescing, work-group geometry, register pressure,
  and shared-memory usage.
- Store tuned ELF artifacts, compatibility metadata, and performance records in
  a versioned kernel library.

Exit criteria:

- An exported PyTorch inference model runs through the IREE mainline.
- At least one dispatch uses a tuned Triton artifact and another can use the
  generic IREE Ventus codegen fallback.

- General/low-precision MMA, arbitrary shapes, masked MMA, shuffle/ballot,
  atomics, async copy, clusters, warp specialization, and FP16/BF16/FP8 remain
  excluded from V1.

After V1, consider:

- Warp-level shuffle or reduction support.
- Specialized shared-memory layouts.
- Asynchronous copies and software pipelining.
- Atomics and more complex scans.
- Direct IREE-to-Triton codegen invocation.

## 11. Testing Strategy

### 11.1 MLIR Unit Tests

- Target attribute parsing and verification.
- Kernel ABI validation.
- Address-space conversion.
- GPU builtin lowering.
- Barrier lowering.
- LLVM function attributes and calling convention.
- Failure tests for unsupported features.

### 11.2 Triton Compiler Tests

- TTIR and TTGIR snapshots for supported kernels.
- Blocked layout decomposition.
- Masked load/store lowering.
- Reduction lowering.
- Shared-memory allocation and barriers.
- Artifact metadata generation.

### 11.3 Differential ABI Tests

For each basic kernel, compare:

```text
OpenCL C -> Ventus Clang -> LLVM IR / ELF
Triton   -> Ventus MLIR  -> LLVM IR / ELF
IREE     -> Ventus MLIR  -> LLVM IR / ELF
```

Check:

- Calling convention.
- Address spaces.
- Kernel attributes.
- Argument offsets and alignment.
- Builtin declarations.
- Resource sections.
- Launch geometry.

Comparison is over normalized semantic and ABI facts, not byte-identical LLVM
IR. The baseline includes one OpenCL/Clang golden and one Triton-shaped positive
vector add. Negative coverage includes unknown attributes, unsupported
intrinsics, scalable vectors, i64 device pointers/addresses, `atomicrmw`,
`cmpxchg`, atomic load/store, `fence`/ordering/scope, `invoke`/EH, an otherwise
well-typed `addrspacecast`, AS4 writes, user AS5, and unsupported metadata/module
flags. All `addrspacecast` operations are rejected in V1.

### 11.4 Runtime Tests

Run each kernel through:

```text
POCL reference path
IREE HAL Ventus path
```

Use Spike for correctness testing. RTL is mandatory for M3/M4 architectural
validation. CycleSim is a capability-gated functional/cycle experiment and is
not an independent proof of RTL architecture or performance.
For M1/M2 milestone, release, and required CI, the pinned Spike/runtime is
mandatory: missing tools or a skipped required suite is a hard gate failure.
Only explicitly non-milestone local convenience runs may omit it with a
non-green, non-acceptance status. CycleSim alone remains capability-gated.

### 11.5 Initial Kernel Matrix

The first regression suite should include:

- Fill and copy.
- Vector add and fused elementwise chains.
- Boundary masks.
- Broadcasting.
- ReLU and select.
- Row reduction and softmax components.
- Static shared-memory copy with a barrier.
- Basic blocked matrix multiplication.
- Fixed `VentusMmaProfileA` differential golden after M3 freeze.

## 12. Risks and Controls

### 12.1 Duplicating Ventus LLVM Responsibilities

Risk: MLIR explicitly models vector registers or split/join control flow and
conflicts with Ventus LLVM.

Control: keep arithmetic and branches generic; let Ventus LLVM own
uniform/varying and reconvergence lowering.

### 12.2 Overdesigning the Custom Dialect

Risk: a large VentusGPU dialect duplicates Arith, SCF, GPU, MemRef, and LLVM.

Control: add custom operations only when a verified cross-pass target semantic
cannot be represented by standard dialects.

### 12.3 Triton and IREE Backend Drift

Risk: the two frontends emit different ABI attributes, builtins, or metadata.

Control: share typed attributes, lowering libraries, artifact definitions, and
differential tests.

### 12.4 Toolchain ABI Instability

Risk: Ventus LLVM, POCL, driver, and simulator revisions become incompatible.

Control: pin `ventus-env`, record toolchain identity in artifacts, and retain
POCL-generated golden references.

### 12.5 Cross-Version Textual IR Drift

Risk: newer producer IR parses in some tools but uses an untested attribute,
intrinsic, metadata flag, pointer form, or target semantic that Ventus LLVM 16
cannot codegen correctly. Ad hoc string replacement can also produce parseable
but semantically altered IR.

Control: make `VentusLLVM16Compatibility` an explicit M1 deliverable, constrain
producer translation, allow only structured allowlisted normalization, fail
closed on unknown features, and require internal, absolute `opt`, and absolute
`llc` gates. The existing textual boundary remains correct; it is now an
owned, testable artifact rather than an implicit handoff.

### 12.6 Confusing Operator and Model Responsibilities

Risk: Triton grows into a second model runtime, or IREE HAL accumulates kernel
tiling and codegen responsibilities.

Control: IREE exclusively owns complete-model graph, buffer, executable, and
runtime planning. Triton owns independent operator development and tuning.
Optimized kernels cross the boundary only as versioned artifacts.

### 12.7 Premature Performance Work

Risk: matrix layouts, asynchronous copies, or atomics delay the basic runtime
and ABI bring-up.

Control: complete elementwise, masks, reductions, ELF loading, and IREE HAL
before adding target-specific performance features.

## 13. Final Architecture Summary

The final design is:

```text
Model inference mainline:
  PyTorch -> torch.export -> StableHLO -> IREE Flow
  -> IREE generic codegen or tuned-kernel selection
  -> IREE Stream/HAL -> model package

Triton operator side path:
  Triton DSL -> TTIR -> TTGIR -> TritonGPUToVentus
  -> VentusLLVM16Compatibility -> kernel.ventus.ll
  -> absolute Ventus opt/llc -> tuning/validation -> versioned kernel library

Shared target layer:
  Standard MLIR dialects
  + thin Ventus target/ABI/metadata contract
  + shared Ventus-to-LLVM lowering

Machine-code layer:
  Modified Ventus LLVM -> ELF + resource metadata

Runtime layer:
  IREE HAL Ventus -> Ventus C driver API
  -> Spike / CycleSim / RTL simulation / GVM

Physical hardware backend:
  future capability requiring a concrete driver implementation

Reference path:
  OpenCL/POCL -> ABI and correctness oracle
```

The most important boundary is that Triton and IREE do not share TTGIR or
IREE's high-level codegen IR. They share the Ventus hardware contract, LLVM
lowering, kernel ABI, artifact metadata, and runtime interface.

This architecture lets IREE remain the complete-model compiler and runtime,
while Triton acts as an operator DSL, tuner, MLIR validation source, Ventus LLVM
stress suite, and producer of optimized kernels. It avoids Triton-to-Linalg and
prevents IREE and Triton from maintaining independent Ventus LLVM stacks.
