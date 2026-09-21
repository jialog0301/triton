# Triton Ventus and IREE Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an IREE-based Ventus model inference mainline, a native Triton operator development and tuning side path, and shared MLIR/LLVM infrastructure for versioned Ventus ELF kernel artifacts.

**Architecture:** IREE owns complete-model import, dispatch formation, buffer planning, executable packaging, and runtime execution. Triton retains TTIR and TTGIR as an independent operator laboratory for kernel development, tuning, shared-contract validation, and Ventus LLVM stress testing; tuned kernels enter IREE only as versioned artifacts. In V1, Triton implements producer-local lowering, LLVM 16 compatibility, and object-to-ELF orchestration. A standard-dialect-based shared Ventus MLIR target library is only a conditional post-alignment alternative. The modifiable Ventus LLVM fork owns concrete ABI lowering, divergence, machine code, resource metadata, and object generation; `third_party/ventus/backend/compiler.py` owns the V1 external tool pipeline and final ELF link.

The Triton backend follows the explicit stage contract `TTIR -> TTGIR -> LLIR ->
binary`. TTIR owns Triton-level semantics and common high-level optimization;
TTGIR owns layout, thread distribution, and target configuration; LLIR owns
address spaces, kernel ABI, and producer-side LLVM lowering; binary generation
only invokes the pinned external tools, captures evidence, and returns bytes. The
binary stage must not perform additional complex IR transformations.

The backend also follows a complete-closure design even when implementation is
sequenced across tasks:

```text
target selection -> TTIR -> TTGIR -> LLIR -> external toolchain -> ELF
    -> artifact validation -> runtime loading -> launch ABI -> execution
```

Each boundary is tested independently. Successful IR or ELF generation is not
considered proof that a kernel can be loaded, launched, or executed correctly.

**Tech Stack:** Triton TTIR/TTGIR, MLIR TableGen and standard dialects, LLVM IR, Ventus LLVM, CMake/Ninja, Python/pytest, LLVM lit/FileCheck, IREE compiler and HAL, Ventus driver, Spike, CycleSim, POCL/OpenCL.

**Authoritative V1 scope:** [Triton-for-Ventus V1 Scope](2026-08-28-triton-for-ventus-v1-scope.md)
defines the binding V1 profile, milestones, and exclusions. In this broader plan,
IREE model integration and general inference operators are post-V1/non-gating;
V1 is the basic FP32 Triton backend plus the fixed M3/M4-gated FP32 MMA vertical
slice, not general `tl.dot`.

---

**Facility baseline:** Current implementation facts, component revisions, dirty
state, and capability limits are recorded in
`docs/plans/2026-08-25-ventus-env-facility-snapshot.md`.
Exact Triton/Python/LLVM-cache provenance, installed Ventus binary identity, and
release blockers are recorded in
`docs/plans/2026-08-31-triton-ventus-toolchain-version-audit.md`.

## Preconditions

- Work in an isolated worktree created from the selected Triton revision.
- Every new terminal/OpenCode window must enter and verify the environment using
  section 8, "New window enters the verified Worktree environment", in
  `docs/plans/2026-08-31-triton-ventus-toolchain-version-audit.md`. That section
  is the single source of truth for `cd`, venv activation, `TRITON_HOME`,
  explicit `VENTUS_*` paths, identity checks, rebuild recovery, and the
  runtime-test subshell; do not maintain a second divergent command block here.
- Use a worktree-local `TRITON_HOME`; verify that the resolved cache matches the
  selected checkout's `cmake/llvm-info.json` consumer pin by checking the
  revisioned directory, `version.txt`, and CMake `LLVM_SYSPATH`. The private
  cache is required for isolation, concurrency, and reproducibility; do not use
  the stable symlink's current target as proof of the consumed revision. Do not
  interpret `cmake/llvm-build-info.json` as the local consumer pin.
- Pin one `ventus-env` revision and all of its submodule revisions before writing ABI-sensitive code.
- Resolve the inspected Ventus LLVM 16 versus current Triton LLVM/MLIR version
  gap before implementing a shared C++ MLIR library. Use producer-specific
  adapters and a textual LLVM IR/tool invocation boundary until revisions are
  deliberately aligned.
- Treat `ventus-env/llvm` as a source dependency that this project may modify;
  maintain backend changes as reviewed commits against the pinned THU-DSP-LAB
  LLVM fork.
- Record the absolute paths of the Ventus `clang`, `opt`, `llc`, `ld.lld`, Spike, CycleSim, POCL, and driver installations.
- Do not set `LLVM_SYSPATH` to the Ventus LLVM 16 prefix when building Triton.
  Keep the version boundary at verified textual LLVM IR and absolute-path
  Ventus LLVM 16 subprocesses; never exchange LLVM bitcode across it.
- Treat `VentusLLVM16Compatibility` as a first-class M1 Triton Ventus Backend
  stage. Its formal input is producer-local MLIR LLVM Dialect/native IR before
  serialization and its formal output is `kernel.ventus.ll`, a checked textual
  subset accepted by the pinned Ventus LLVM 16 contract. It is not a general
  LLVM version translator and must not use textual string replacement.
- Do not prepend the Ventus install `bin` or `lib` directories to the normal
  Triton build shell's `PATH`/`LD_LIBRARY_PATH`, and never source
  `ventus-env/env.sh` there. Use explicit `VENTUS_*` paths; confine temporary
  runtime loader/PATH changes to a dedicated test subshell.
- Treat the current driver as a simulator-facing API supporting Spike, RTL
  simulation, CycleSim, and GVM. Physical hardware support is not a baseline.
- Centralize the packed argument ABI, 64-byte device metadata block, host launch
  structure, and ELF resource parsing before IREE directly consumes artifacts.
- Treat AS4 constant as a validated ABI convention, not an independent V1
  memory facility. Treat AS5/PDS only as a spill/private-resource convention,
  not general user-visible memory support.
- Bound and validate divergent control flow. The inspected SIMT stack depth and
  overflow behavior are not yet a V1 hardware guarantee.
- Keep the design document as architecture context, but treat
  `docs/plans/2026-08-28-triton-for-ventus-v1-scope.md` as the authoritative V1
  source of truth.
- Fix the V1 profile to `riscv32`/`ventus-gpgpu`, RV32 pointers, warp 32,
  `1 CTA = 1 work-group`, one-dimensional grid, general local size `[32,1,1]`
  or `[64,1,1]`, and MMA local size exactly `[32,1,1]`.
- Limit V1 scalar types to `i1/i32/f32` and layouts to Blocked/Slice/Shared plus
  proposed `VentusMmaEncodingAttr`/`VentusDotOperandEncodingAttr` pending M3 freeze.
- Do not implement general or low-precision MMA, arbitrary `tl.dot` shapes,
  shuffle/ballot, atomics, asynchronous copies, CTA clusters, warp
  specialization, FP16/BF16/FP8, or a physical driver in V1.
- Do not make TorchInductor-to-Ventus full-model execution a baseline
  dependency. Triton is initially an operator tool and artifact producer.

### Backend configuration and capability source of truth

All stages must consume one resolved Ventus target configuration rather than
independently interpreting `self.target.arch`, `options.arch`, or environment
variables. The configuration is represented by the current `VentusOptions` at
the Python boundary and by an equivalent `VentusTargetProfile` fact set at the
native/artifact boundaries. It must cover at least:

- `riscv32`, `ventus-gpgpu`, 32-bit device pointers, and ABI revision.
- Warp size, supported local sizes, one-dimensional grid, and `1 CTA = 1
  work-group`.
- Supported address spaces, scalar types, barriers, atomics, clusters, warp
  specialization, and asynchronous-copy capabilities.
- Toolchain identities, linker/runtime inputs, resource units, and resource
  limits.

The profile is a single source of target facts for option validation,
`TargetInfo`, artifact validation, compiler command construction, and runtime
launch checks. Capability decisions must not be duplicated as scattered
architecture conditionals in individual passes.

For the V1 general profile, `num_warps` is restricted to `1` or `2`, implying
local sizes `[32,1,1]` or `[64,1,1]` with warp size 32. The Python option parser
and TTGIR/native preflight must both enforce this rule. The MMA profile remains
the separate exact `[32,1,1]` profile defined by the scope document.

V1 milestone mapping across the tasks is fixed as follows: M1 is the basic
backend plus ABI/manifest/resource validation, canonical packing, and mandatory
Spike execution; M2 is static AS3/full-work-group barrier, fixed-extent
`sum`/`max`, and the static blocked FP32 `32x32x32` reference/fallback; M3 is
mandatory RTL/Spike contract alignment with capability-gated CycleSim; M4 is the exact-profile Triton `tl.dot`
to `vftta.vv` lowering.

## Binding Execution Order And Dependency Gates

The instruction to execute this plan task-by-task means the following order,
not the physical order of every later section:

| Order | Tasks | Gate and result |
| ---: | --- | --- |
| 1 | Tasks 1-3 | Pin facilities, capture ABI goldens, and define every mandatory V1 manifest field. |
| 2 | Tasks 4-7 | At each task, execute the producer-local/textual-boundary V1 baseline stated in its gate. Execute the compiled shared-library body only after deliberate LLVM/MLIR revision alignment; otherwise record it as deferred and continue. |
| 3 | Tasks 8-10, then Task 10A, Task 11, and Task 12 | Task 8 defines/hardens the provider-side Ventus LLVM input contract. Task 10A owns the producer-side compatibility stage and all three M1 compile gates; Task 11 depends on it, and Task 12 is the fourth ELF/Spike execution gate. |
| 4 | Task 19, then Task 18 | Implement static AS3 plus full work-group barrier before reduction; reduction may depend on that shared-memory path. |
| 5 | Task 20 | Compile and execute the mandatory canonical blocked FP32 `32x32x32` FMA reference/fallback. Additional static blocked patterns require their own explicit lowering and correctness tests. |
| 6 | Task 20A (M3) | Freeze the contract only. Mandatory Spike and RTL differential results must exist; CycleSim is capability-gated. No Triton `tl.dot` compile/execute claim yet. |
| 7 | Task 20B (M4) | Implement exact-profile Triton encodings/lowering, code generation, metadata/diagnostics, mandatory Spike/RTL differential execution, and capability-gated CycleSim. |
| 8 | Task 22 V1 suites | Run host/compiler, mandatory Spike, M2 capability-gated CycleSim, and mandatory M3/M4 RTL gates. |
| 9 | Tasks 13-17 and Task 21 | Post-V1, non-gating IREE/model work. These tasks must not begin before V1 completion and cannot satisfy a V1 gate. |

Tasks 13-17 appear earlier than some V1 tasks only because this document also
preserves the broader future architecture. Their numbering is not execution
priority. Shared memory always precedes reduction. M3 always precedes M4.

**No-bypass invariant:** Every Ventus Triton compilation in Tasks 11, 18, 19,
20, and 20B, and every future Triton compiler path, must emit and hash its own
`kernel.ventus.ll`, run Gates 1-3 for that module, link the resulting object
through the producer-local V1 ELF stage defined in Task 10A, and record all gate
and link evidence in the manifest. No task may invoke `llc`, `ld.lld`, or Spike
through a direct path that bypasses this pipeline.

### Task 1: Freeze the Ventus toolchain identity

**Audit gate:** Start from
`docs/plans/2026-08-31-triton-ventus-toolchain-version-audit.md`. Before creating
the release identity, verify the selected Triton commit, Python/site-package and
native `libtriton` provenance, the `cmake/llvm-info.json` consumer hash/build,
the revisioned cache path/`version.txt` and CMake `LLVM_SYSPATH` under an isolated
`TRITON_HOME`, and all referenced Ventus build/install binary hashes, build-ids,
and RUNPATHs. A shared symlink name or `cmake/llvm-build-info.json` is not proof
of the LLVM consumed by this checkout. If pip provenance is unavailable, encode
it as unknown rather than deriving it from the venv's parent checkout.

**Files:**
- Create: `third_party/ventus/toolchain/README.md`
- Create: `third_party/ventus/toolchain/version.json`
- Create: `python/test/unit/ventus/test_toolchain_identity.py`

**Step 1: Write the failing identity test**

Add a test that loads `third_party/ventus/toolchain/version.json` and requires:

```python
REQUIRED_KEYS = {
    "triton_commit",
    "triton_consumer_llvm_hash",
    "triton_consumer_llvm_build",
    "triton_home",
    "triton_llvm_cache_path",
    "python_executable",
    "python_version",
    "python_triton_provenance",
    "libtriton_content_hash",
    "ventus_env_commit",
    "llvm_commit",
    "pocl_commit",
    "driver_commit",
    "spike_commit",
    "cyclesim_commit",
    "gpgpu_commit",
    "systemc_commit",
    "opencl_cts_commit",
    "testcases_commit",
    "dirty_components",
    "dirty_component_content_hashes",
    "ventus_install_prefix",
    "tool_binary_hashes",
    "runtime_build_manifest",
    "runtime_install_manifest",
    "runtime_binary_build_ids",
    "runtime_binary_runpaths",
    "target_triple",
    "mcpu",
    "pointer_width",
    "warp_size",
    "capability_record_version",
    "simulator_address_convention",
    "resource_unit_contract",
    "completion_contract_version",
    "ventus_linker_script",
    "ventus_crt0_input",
    "ventus_libclc_input",
    "ventus_workitem_input",
    "ventus_kernel_entry_or_init",
}
```

Assert the initial target contract:

```python
assert identity["target_triple"] == "riscv32"
assert identity["mcpu"] == "ventus-gpgpu"
assert identity["pointer_width"] == 32
assert identity["warp_size"] == 32
```

Also record and validate the supported local-size set
`[[32, 1, 1], [64, 1, 1]]`, the MMA local size `[32, 1, 1]`, one CTA per
  work-group, one-dimensional grid, and scalar types `i1/i32/f32`.
Define the capability record independently of the current `vt_dev_caps` API;
the latter is backend-incomplete and cannot be serialized as the V1 contract.
The resource-unit contract must label LDS in bytes, PDS allocation/limit units,
and whether SGPR/VGPR values are per-wavefront, per-CTA, or total resident use.
The simulator address convention must state that RTL simulation currently uses
physical-address-style device values while virtual memory remains incomplete.
Discover and record the actual installed absolute paths and hashes for the V1
linker script, crt0, libclc, work-item runtime support, and kernel entry/init
contract. Task 2 ABI goldens must confirm their roles. Do not invent or assume
concrete library filenames.
The identity test must also reject a Triton LLVM cache whose resolved hash/build
does not match `cmake/llvm-info.json`, Python/native imports from another
checkout, absent required full binary hashes, unreviewed dirty patches, or a
missing coordinated simulator/driver build-install manifest. Build/install
content hashes may differ when CMake relocates RUNPATH; matching build-id plus
documented RUNPATH change must not be mislabeled as stripping or source mismatch. Do not use
`cmake/llvm-build-info.json` as the consumer expectation.

**Step 2: Run the test and verify failure**

Run:

```bash
pytest -q python/test/unit/ventus/test_toolchain_identity.py
```

Expected: FAIL because `version.json` does not exist.

**Step 3: Add the pinned identity**

Populate `version.json` from the selected clean `ventus-env` checkout. Do not
use placeholders such as `main`, `latest`, or an empty commit. Reject a release
identity with nonempty `dirty_components` unless every patch is represented by
a reviewed content hash. Rebuild/install Spike and the complete CycleSim/driver
set from coordinated pins when required by the audit; record both build and
install full hashes, build-ids, RUNPATHs, and install events rather than
relabeling old installed binaries with current dirty source state.

Document in `README.md`:

- How each commit was obtained.
- The expected compiler commands.
- The dedicated Python environment, isolated `TRITON_HOME`, consumer LLVM cache
  resolution, and current-checkout native build procedure.
- The supported simulator/runtime combination.
- Absolute Ventus tool paths and the rule that Ventus Clang is an ABI oracle,
  not Triton's build dependency; `LLVM_SYSPATH` must not point Triton at Ventus
  LLVM 16.
- The rule that the normal Triton shell never sources `ventus-env/env.sh` or
  globally prepends Ventus `bin`/`lib`; runtime environment changes belong in a
  separate test subshell.
- The rule that this README/audit is explanatory and `version.json` is the final
  machine-readable release identity.
- The rule that ABI changes require an artifact ABI version change or updated golden references.

**Step 4: Run the test and verify success**

Run:

```bash
pytest -q python/test/unit/ventus/test_toolchain_identity.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add third_party/ventus/toolchain python/test/unit/ventus/test_toolchain_identity.py
git commit -m "test: pin Ventus toolchain identity"
```

### Task 2: Capture POCL ABI golden references

**Files:**
- Create: `third_party/ventus/test/abi/vector_add.cl`
- Create: `third_party/ventus/test/abi/masked_copy.cl`
- Create: `third_party/ventus/test/abi/barrier_local.cl`
- Create: `third_party/ventus/test/abi/golden/vector_add.ll`
- Create: `third_party/ventus/test/abi/golden/masked_copy.ll`
- Create: `third_party/ventus/test/abi/golden/barrier_local.ll`
- Create: `third_party/ventus/test/abi/check_golden.py`
- Create: `python/test/unit/ventus/test_abi_goldens.py`

**Step 1: Write the failing golden-reference test**

The test must verify that every OpenCL source has a matching LLVM IR golden and that each golden contains:

```text
ventus_kernel
target datalayout
addrspace(1)
```

The barrier golden must also contain the Ventus barrier intrinsic expected by the pinned toolchain.

**Step 2: Run the test and verify failure**

Run:

```bash
pytest -q python/test/unit/ventus/test_abi_goldens.py
```

Expected: FAIL because the fixtures are missing.

**Step 3: Add minimal OpenCL kernels**

Use one-dimensional kernels with explicit scalar arguments and no optional OpenCL extensions:

```c
__kernel void vector_add(__global const float *a,
                         __global const float *b,
                         __global float *c,
                         int n) {
  int i = get_global_id(0);
  if (i < n)
    c[i] = a[i] + b[i];
}
```

Add a masked copy and a static `__local` memory kernel containing `barrier(CLK_LOCAL_MEM_FENCE)`.

**Step 4: Generate and normalize LLVM goldens**

Use the pinned Ventus Clang command:

```bash
<ventus-clang> -target riscv32 -mcpu=ventus-gpgpu -cl-std=CL2.0 \
  -S -emit-llvm third_party/ventus/test/abi/vector_add.cl \
  -o third_party/ventus/test/abi/golden/vector_add.ll
```

Repeat for the other kernels. Normalize only nondeterministic module paths and timestamps; do not remove ABI-relevant attributes or metadata.

**Step 5: Run the test and inspect ABI facts**

Run:

```bash
pytest -q python/test/unit/ventus/test_abi_goldens.py
```

Expected: PASS.

Record the observed calling convention, data layout, builtin declarations, function attributes, address spaces, and barrier intrinsic in comments in `check_golden.py` or the toolchain README.

**Step 6: Commit**

```bash
git add third_party/ventus/test/abi python/test/unit/ventus/test_abi_goldens.py
git commit -m "test: add Ventus ABI golden kernels"
```

### Task 3: Define the versioned Ventus kernel artifact schema

**Files:**
- Create: `third_party/ventus/include/Target/VentusArtifact.h`
- Create: `third_party/ventus/lib/Target/VentusArtifact.cpp`
- Create: `third_party/ventus/test/Target/artifact-roundtrip.cpp`
- Create: `third_party/ventus/include/Target/CMakeLists.txt`
- Create: `third_party/ventus/lib/Target/CMakeLists.txt`

**Step 1: Write a failing C++ round-trip test**

Construct metadata containing:

```cpp
VentusKernelMetadata metadata;
metadata.abiVersion = 1;
metadata.entryPoint = "vector_add";
metadata.targetArch = "ventus-gpgpu";
metadata.pointerWidth = 32;
metadata.warpSize = 32;
metadata.workgroupSize = {32, 1, 1};
metadata.sharedMemoryBytes = 0;
metadata.privateMemoryBytes = 0;
metadata.arguments = {
    VentusKernelArgument::buffer(0, 0, 4),
    VentusKernelArgument::buffer(1, 4, 4),
    VentusKernelArgument::buffer(2, 8, 4),
    VentusKernelArgument::scalarI32(12),
};
```

The round-trip fixture must cover all mandatory V1 fields from the scope, not
only the abbreviated C++ example:

- `artifact_abi_version`, `target_triple`, `mcpu`, pointer width, warp size, and
  pinned RTL profile revision/content hash.
- Toolchain identity plus Ventus LLVM, driver, Spike, CycleSim, and RTL
  revision/content hashes, with simulator identity distinct from RTL identity.
- ELF content hash, entry-point symbol, ELF class, endianness, and machine
  validation result.
- `ventus_kernel` calling convention and every packed argument's kind, binding,
  byte offset, size, and alignment.
- One-dimensional grid calculation, global/local/offset metadata, one CTA per
  work-group, and the exact accepted local size.
- Required features, operation, dtype, layout, shape, active-warp, and profile
  constraints, including optional frozen `VentusMmaProfileA`.
- Static shared bytes, private bytes, VGPR, SGPR, LDS/PDS/resource limits, and
  range-validation status.
- Raw `.ventus.resource.<kernel>` four-target-endian-`uint16` values, parser
  version/source, and whether hard-coded resource consumption was rejected.
- Canonical units and aggregation domains for LDS bytes, PDS, SGPR, and VGPR,
  including per-wavefront/per-CTA definitions and validated device limits.
- Capability-record identity independent of `vt_dev_caps`, simulator physical
  address convention, and deterministic completion/timeout/cache-flush contract.
- References/content hashes for TTIR, TTGIR, LLVM IR, assembly, ELF, launcher
  input, and test results, including fallback/skip/tolerance records.
- The `kernel.ventus.ll` reference/content hash, internal compatibility result,
  absolute Ventus `opt` and `llc` argv/stdout/stderr/status/tool identity, and
  generated object content hash.

Serialize it, deserialize it, and compare all fields.

**Step 2: Build and verify failure**

Run the narrow CMake/Ninja target selected for Ventus unit tests.

Expected: compilation failure because the artifact types do not exist.

**Step 3: Implement the minimal schema**

Implement typed structures for every field listed in Step 1. Reject missing
mandatory fields, inconsistent grid/local-size metadata, unknown ABI versions,
invalid ELF identity, resource overflow, and a simulator/profile identity that
does not match a required frozen MMA contract.

Use a deterministic JSON debug representation first. Keep the in-memory structure independent of IREE and Triton.

**Step 4: Run the unit test**

Expected: PASS with a deterministic serialized representation.

**Step 5: Commit**

```bash
git add third_party/ventus/include/Target third_party/ventus/lib/Target third_party/ventus/test/Target
git commit -m "feat: define Ventus kernel artifact metadata"
```

### Task 4: Add the thin Ventus MLIR target dialect (Conditional Post-Alignment Alternative)

**Dependency gate:** V1 does not require this compiled dialect. Without deliberate
LLVM/MLIR revision alignment, record this task as deferred and keep target,
kernel ABI, geometry, and feature validation in the producer-local Triton
adapter and versioned manifest. The files and steps below apply only after
alignment.

**Files:**
- Create: `third_party/ventus/include/Dialect/Ventus/IR/Dialect.h`
- Create: `third_party/ventus/include/Dialect/Ventus/IR/VentusDialect.td`
- Create: `third_party/ventus/include/Dialect/Ventus/IR/VentusAttrs.td`
- Create: `third_party/ventus/include/Dialect/Ventus/IR/VentusOps.td`
- Create: `third_party/ventus/lib/Dialect/Ventus/IR/Dialect.cpp`
- Create: `third_party/ventus/include/Dialect/Ventus/IR/CMakeLists.txt`
- Create: `third_party/ventus/lib/Dialect/Ventus/IR/CMakeLists.txt`
- Create: `third_party/ventus/test/Dialect/Ventus/invalid-target.mlir`
- Create: `third_party/ventus/test/Dialect/Ventus/parse-print.mlir`

**Step 1: Write failing parse/verification tests**

Add a valid module containing a target attribute and kernel attributes. Add invalid cases for:

- Pointer width other than 32.
- Warp size other than 32.
- Zero work-group dimension.
- Work-group size other than `[32,1,1]` or `[64,1,1]` on the general path, and
  anything other than `[32,1,1]` for an MMA kernel.
- Unknown ABI version.

**Step 2: Run `triton-opt` and verify failure**

Run:

```bash
triton-opt third_party/ventus/test/Dialect/Ventus/parse-print.mlir
```

Expected: FAIL because the dialect is not registered.

**Step 3: Implement only the thin contract**

Add typed attributes equivalent to:

```text
#ventus.target<arch = "ventus-gpgpu", pointer_width = 32, warp_size = 32>
#ventus.kernel_abi<packed_args_v1>
ventus.kernel
ventus.workgroup_size
ventus.shared_memory_size
ventus.target_features
```

Do not add target-specific arithmetic, memory, branch, reduction, or generic
MMA operations. The fixed-profile MMA attributes are added only after M3 freeze.

Leave `VentusOps.td` empty except for dialect boilerplate unless the barrier requirement is already proven by the golden IR analysis.

**Step 4: Register the dialect in the Ventus plugin**

Wire generated headers and dialect registration into the Ventus plugin CMake targets without registering it globally when the plugin is not built.

**Step 5: Run parse/print and invalid tests**

Expected: valid input round-trips; each invalid case fails with a specific verifier message.

**Step 6: Commit**

```bash
git add third_party/ventus/include/Dialect third_party/ventus/lib/Dialect third_party/ventus/test/Dialect
git commit -m "feat: add thin Ventus MLIR target dialect"
```

### Task 5: Lower generic GPU builtins and barriers in a shared library (Conditional Post-Alignment Alternative)

**Dependency gate:** Without revision alignment, do not build this shared pass.
The V1 baseline lowers Triton SPMD builtins producer-locally in Task 10, matches
the pinned POCL/LLVM textual contract, and invokes Ventus LLVM as an external
tool. Barrier lowering is implemented producer-locally after Task 19 establishes
static AS3 and the full-work-group barrier contract.

**Files:**
- Create: `third_party/ventus/include/Conversion/GPUToVentusLLVM/GPUToVentusLLVM.h`
- Create: `third_party/ventus/include/Conversion/GPUToVentusLLVM/Passes.td`
- Create: `third_party/ventus/lib/Conversion/GPUToVentusLLVM/GPUToVentusLLVM.cpp`
- Create: `third_party/ventus/test/Conversion/gpu-to-ventus-llvm.mlir`
- Create: `third_party/ventus/test/Conversion/barrier-to-ventus-llvm.mlir`

**Step 1: Write failing FileCheck tests**

Test lowering for:

```text
gpu.thread_id x/y/z
```

Check for the exact builtin or intrinsic representation observed in the pinned POCL LLVM goldens.

**Step 2: Run the conversion test and verify failure**

Run:

```bash
triton-opt --convert-gpu-to-ventus-llvm \
  third_party/ventus/test/Conversion/gpu-to-ventus-llvm.mlir | FileCheck \
  third_party/ventus/test/Conversion/gpu-to-ventus-llvm.mlir
```

Expected: FAIL because the pass is unavailable.

**Step 3: Implement minimal conversion patterns**

Generate LLVM Dialect calls or operations matching Ventus LLVM's supported builtin contract. Do not emit explicit `<32 x T>` values and do not emit split/join instructions.

If `gpu.barrier` lacks required scope information, introduce only:

```text
ventus.workgroup_barrier
```

and lower it to `llvm.riscv.ventus.barrier` or the exact pinned intrinsic.

**Step 4: Run the conversion tests**

Expected: PASS and no illegal GPU builtin operations remain.

**Step 5: Compare generated native LLVM IR to goldens**

Translate the MLIR LLVM module and compare calling patterns, types, and attributes against the OpenCL references. Differences must be documented before acceptance.

**Step 6: Commit**

```bash
git add third_party/ventus/include/Conversion third_party/ventus/lib/Conversion third_party/ventus/test/Conversion
git commit -m "feat: lower GPU builtins to Ventus LLVM"
```

### Task 6: Emit the Ventus kernel ABI from a shared library (Conditional Post-Alignment Alternative)

**Dependency gate:** Without revision alignment, do not compile this shared
conversion. The V1 baseline emits calling convention, data layout, target
attributes, argument layout, and manifest fields from the Triton adapter and
validates the resulting textual LLVM IR with pinned Ventus `opt`/`llc` tools.

**Files:**
- Create: `third_party/ventus/include/Conversion/VentusToLLVM/VentusToLLVM.h`
- Create: `third_party/ventus/include/Conversion/VentusToLLVM/Passes.td`
- Create: `third_party/ventus/lib/Conversion/VentusToLLVM/VentusToLLVM.cpp`
- Create: `third_party/ventus/test/Conversion/kernel-abi-to-llvm.mlir`
- Create: `third_party/ventus/test/Conversion/kernel-argument-layout.mlir`

**Step 1: Write failing ABI FileCheck tests**

Require the lowered kernel to contain:

- The Ventus kernel calling convention.
- The pinned RV32 data layout.
- Target CPU/features required by Ventus LLVM.
- ABI version metadata.
- Work-group size metadata and V1 geometry validation.
- Address-space-correct pointer arguments or packed argument loads.

**Step 2: Run and verify failure**

Expected: FAIL because the conversion pass does not exist.

**Step 3: Implement function and module conversion**

Convert typed Ventus attributes into LLVM function/module properties. Reject kernels whose argument types cannot be represented by `packed_args_v1`.

Keep argument-offset computation in the shared `VentusArtifact` library so IREE and Triton cannot diverge.

**Step 4: Run ABI tests and native LLVM verification**

Run `mlir-translate`, then absolute Ventus `opt` parse/verify, then absolute
Ventus `llc -mcpu=ventus-gpgpu -filetype=obj` on the generated LLVM IR.

Expected: all tests PASS, Ventus LLVM parses/verifies the module, and the Ventus
target emits an object. `opt -verify` alone is insufficient.

**Step 5: Commit**

```bash
git add third_party/ventus/include/Conversion/VentusToLLVM third_party/ventus/lib/Conversion/VentusToLLVM third_party/ventus/test/Conversion
git commit -m "feat: emit Ventus kernel ABI from MLIR"
```

### Task 7: Add shared compiled Ventus tool utilities (Conditional Post-Alignment Alternative)

**Dependency gate:** A shared C++ utility library is post-alignment. The V1
baseline is producer-local safe subprocess invocation from
`third_party/ventus/backend/compiler.py`, plus producer-local ELF inspection and
the common serialized schema from Task 3. It must use argument arrays, explicit
tool paths, captured diagnostics, secure temporary files, and textual LLVM IR;
it must not require linking Triton's LLVM against Ventus LLVM 16.

**Files:**
- Create: `third_party/ventus/include/Target/VentusCompiler.h`
- Create: `third_party/ventus/lib/Target/VentusCompiler.cpp`
- Create: `third_party/ventus/include/Target/VentusELF.h`
- Create: `third_party/ventus/lib/Target/VentusELF.cpp`
- Create: `third_party/ventus/test/Target/compiler-smoke.cpp`
- Create: `third_party/ventus/test/Target/elf-resource-parser.cpp`

**Step 1: Write failing compiler and ELF parser tests**

The compiler test supplies a minimal verified LLVM module and expects nonempty ELF bytes. The parser test reads a POCL golden ELF and expects:

- Kernel symbol found.
- `.ventus.resource.<kernel>` found.
- Resource values parsed or explicitly reported absent according to the pinned toolchain.

**Step 2: Run tests and verify failure**

Expected: compilation failure because utility APIs do not exist.

**Step 3: Implement safe tool invocation**

Use argument arrays, temporary directories, captured stdout/stderr, and explicit tool paths. Do not use `shell=True`. Include the exact reproduction command in errors.

**Step 4: Implement ELF resource parsing**

Use LLVM Object APIs rather than ad hoc byte offsets. Validate ELF class,
machine, endianness, entry symbol, content hash, artifact ABI/manifest identity,
and toolchain identity before accepting an artifact. Loading `PT_LOAD` segments
alone is not artifact validation.

**Step 5: Run tests**

Expected: PASS against both generated and POCL-reference ELF files.

**Step 6: Commit**

```bash
git add third_party/ventus/include/Target third_party/ventus/lib/Target third_party/ventus/test/Target
git commit -m "feat: compile and inspect Ventus ELF artifacts"
```

### Task 8: Harden the modifiable Ventus LLVM backend contract

**Ownership:** This is the provider-side contract owned by the Ventus LLVM
repository. It defines what the pinned backend accepts and hardens backend
behavior. It does not implement the producer-side compatibility checker or emit
`kernel.ventus.ll`; Task 10A owns those Triton responsibilities.

**Files in `ventus-env/llvm`:**
- Create: `llvm/docs/VentusLLVMInputContract.md`
- Create: `llvm/test/CodeGen/RISCV/VentusGPGPU/kernel-abi.ll`
- Create: `llvm/test/CodeGen/RISCV/VentusGPGPU/divergent-control-flow.ll`
- Create: `llvm/test/CodeGen/RISCV/VentusGPGPU/mixed-phi.ll`
- Create: `llvm/test/CodeGen/RISCV/VentusGPGPU/address-spaces.ll`
- Create: `llvm/test/CodeGen/RISCV/VentusGPGPU/resource-section.ll`
- Modify as required: `llvm/include/llvm/IR/IntrinsicsRISCV.td`
- Modify as required: `llvm/lib/Target/RISCV/RISCVISelLowering.cpp`
- Modify as required: `llvm/lib/Target/RISCV/RISCVTargetMachine.cpp`
- Modify as required: `llvm/lib/Target/RISCV/RISCVAsmPrinter.cpp`
- Modify as required: `llvm/lib/Target/RISCV/VentusProgramInfo.h`
- Modify as required: Ventus MachineFunction passes under `llvm/lib/Target/RISCV/`

**Step 1: Document the LLVM input contract before changing behavior**

Specify:

- Target triple, CPU, data layout, and pointer width.
- `ventus_kernel` calling convention and packed kernel argument rules.
- Address spaces 1, 3, 4, and 5.
- Stable ID and barrier intrinsics accepted from MLIR.
- Supported scalar types, control-flow forms, atomics, and calls.
- Ownership of uniform/varying propagation, mixed PHIs, divergent branches,
  reconvergence, spills, and resource collection.
- Unsupported IR that must be rejected before instruction selection.
- AS4 restrictions as a read-only validated convention rather than an
  independent memory guarantee, and AS5/PDS as spill/private-resource only.
- The maximum validated divergent nesting/loop form and deterministic rejection
  beyond it; do not assume SIMT stack overflow is detected by hardware.

**Step 2: Write failing LLVM lit tests from three producer shapes**

Create equivalent kernels shaped like output from:

```text
OpenCL/Clang
IREE standard-dialect lowering
Triton TTGIR lowering
```

Require all three to lower to the same ABI and target semantics without
requiring accidental Clang-specific IR patterns.

**Step 3: Run the narrow Ventus LLVM suite and verify failures**

Run the repository's configured `llvm-lit` command for:

```text
llvm/test/CodeGen/RISCV/VentusGPGPU
```

Expected: new tests expose missing intrinsics, fragile divergence propagation,
unversioned resource metadata, or inconsistent ABI handling.

**Step 4: Add stable builtins and correctness fixes**

Implement only behavior required by failing tests. Prefer explicit LLVM
intrinsics for local/group IDs and barriers over pattern matching expanded
libclc implementations. Keep `VentusFixMixedPHI`,
`VentusInsertJoinToVBranch`, `VentusLegalizeLoad`, and
`VentusRegextInsertion` as backend-owned responsibilities.

**Step 5: Version resource metadata**

Replace or wrap the raw four-`uint16` resource payload with a documented,
self-identifying versioned record containing at least:

```text
magic, version, record size, VGPR, SGPR, LDS, PDS
```

Update the shared ELF parser to reject unknown versions safely.

**Step 6: Establish backend performance baselines**

Record generated instruction count, VGPR, SGPR, LDS, PDS, spills, and
CycleSim cycles for the ABI kernels. Document that the inherited `RocketModel`
is provisional and identify the first instructions requiring a Ventus-specific
scheduling model.

**Step 7: Run LLVM, MLIR, and ABI differential tests**

Expected:

- Ventus LLVM lit tests PASS.
- OpenCL, IREE-shaped, and Triton-shaped LLVM kernels compile.
- Resource records parse with the expected version.
- No MLIR layer emits explicit Ventus split/join machine semantics.

**Step 8: Commit in the Ventus LLVM repository**

```bash
git add llvm/docs/VentusLLVMInputContract.md llvm/test/CodeGen/RISCV/VentusGPGPU llvm/include/llvm/IR/IntrinsicsRISCV.td llvm/lib/Target/RISCV
git commit -m "feat: define stable Ventus LLVM kernel contract"
```

### Task 9: Register the minimal Triton Ventus backend

**Files:**
- Create: `third_party/ventus/backend/__init__.py`
- Create: `third_party/ventus/backend/compiler.py`
- Create: `third_party/ventus/backend/driver.py`
- Create: `third_party/ventus/triton_ventus.cc`
- Create: `third_party/ventus/CMakeLists.txt`
- Test: `python/test/unit/ventus/test_backend_registration.py`

Implement the first version as an external plugin selected through
`TRITON_PLUGIN_DIRS`. Modify Triton core setup or CMake only after a concrete
plugin or target-interface gap is demonstrated.

**Step 1: Write the failing registration test**

Use the actual discovery API in this Triton revision. Construct a target using
the current `GPUTarget(backend, arch, warp_size)` shape and verify that exactly
one registered backend supports it. Conceptually:

```python
target = GPUTarget("ventus", "ventus-gpgpu", 32)
backend = get_single_compatible_backend(target)
assert backend.binary_ext == "elf"
assert target.warp_size == 32
assert target.arch == "ventus-gpgpu"
```

Store `riscv32`, pointer width, ABI revision, and tool paths in `VentusOptions`
and the toolchain/artifact identity; they are not fields of `GPUTarget`.

**Step 2: Run and verify failure**

Expected: FAIL because no Ventus backend is registered.

**Step 3: Implement backend scaffolding**

Add:

- A `GPUTarget` representation that does not impersonate CUDA.
- Option parsing for `num_warps`, `num_stages`, target features, and toolchain paths.
- Dialect/pass registration through `triton_ventus.cc`.
- Compilation stage placeholders for TTIR, TTGIR, LLIR, and ELF.

The stage placeholders must preserve the stage contract above: `ttir` and
`ttgir` may run common Triton optimization and target configuration, `llir` is
the insertion point for producer-local Ventus lowering, and `elf` is reserved
for external-tool orchestration. The binary stage must not silently add target
semantics that were absent from LLIR. The backend hash/options identity must
eventually include tool versions and the contents of every code-generation
input, not only their filesystem paths.

Do not implement a runtime launcher in this task.

**Step 4: Build and run registration tests**

Expected: the backend is discoverable and reports stable target identity.

**Step 5: Commit**

```bash
git add third_party/ventus python/test/unit/ventus/test_backend_registration.py
git commit -m "feat: register Triton Ventus backend"
```

### Task 10: Implement Ventus TargetInfo and SPMD lowering

**Files:**
- Create: `third_party/ventus/include/TritonGPUToVentus/TargetInfo.h`
- Create: `third_party/ventus/lib/TritonGPUToVentus/TargetInfo.cpp`
- Create: `third_party/ventus/lib/TritonGPUToVentus/SPMDOpToLLVM.cpp`
- Create: `third_party/ventus/test/TritonGPUToVentus/spmd.mlir`
- Modify: common Triton target interfaces only where required by compilation errors

**Step 1: Write failing TTGIR lowering tests**

Cover:

- `tt.get_program_id` for x/y/z.
- Thread and work-group dimensions required by blocked layouts.
- One CTA per work-group.
- Rejection of CTA clusters.
- Warp size 32 verification.
- Rejection of non-1D grids and local sizes outside `[32,1,1]`/`[64,1,1]`.

**Step 2: Run and verify failure**

Expected: Ventus conversion pass cannot lower SPMD operations.

**Step 3: Implement the smallest TargetInfo**

Implement the V1 SPMD builtin lowering producer-locally in
`SPMDOpToLLVM.cpp`, matching the exact builtin/libclc signatures established by
Task 2 goldens and Task 8's provider contract. Explicitly report unsupported
target operations rather than borrowing NVVM operations. Shared Ventus MLIR
lowering utilities may replace this code only as the conditional post-alignment
alternative in Tasks 4-7; Task 10 must not depend on them in V1.

For shuffle, ballot, and atomics, return unsupported until a test and Ventus LLVM contract are defined.

This pass is the SPMD component of the larger LLIR lowering stage, not a
complete TTGIR-to-LLVM conversion. It must keep the Triton-specific boundary
(`tt.*` and `ttg.*` profile/layout facts) separate from producer-independent
GPU semantics where practical. The implementation should preserve the option
to factor standard `gpu.block_id`, `gpu.thread_id`, `gpu.grid_dim`, and
`gpu.block_dim` lowering into a separately compiled Ventus component if a
future IREE or aligned-MLIR integration needs the same semantics. It must not
generate Ventus machine control-flow operations such as `JOIN` or `SETRPC`, nor
explicit SIMT vectors; those remain provider-side responsibilities.

Validate the profile before applying conversion patterns. In particular, reject
missing or non-32 `ttg.threads-per-warp`, `ttg.num-ctas` other than 1, local
sizes other than `[32,1,1]` and `[64,1,1]`, multi-CTA layouts, and warp
specialization with ordinary MLIR diagnostics. A y/z SPMD query is not itself
a multidimensional launch: under the fixed one-dimensional contract it lowers
to zero for IDs and one for sizes/counts. Actual runtime grid dimensions remain
the responsibility of artifact/launch validation.

The supported target facts should be centralized in `VentusTargetProfile` (or
an equivalent profile object), rather than repeated as independent constants
in each lowering pattern. The pass may reuse common TritonGPU-to-LLVM patterns
only after confirming that their target operations and address-space semantics
are accepted by the Ventus profile.

**Step 4: Run TTGIR lowering tests**

Expected: SPMD tests PASS; unsupported cluster, warp, local-size, atomics,
shuffle/ballot, and warp-specialization inputs fail with deterministic
diagnostics. The test must prove that no NVVM/ROCDL operation or explicit
SIMT-vector machine semantics are emitted.

**Step 5: Commit**

```bash
git add third_party/ventus/include/TritonGPUToVentus third_party/ventus/lib/TritonGPUToVentus third_party/ventus/test/TritonGPUToVentus
git commit -m "feat: lower Triton SPMD ops for Ventus"
```

### Task 10A: Add the producer-side Ventus LLVM 16 compatibility stage

**Dependencies and ownership:** Tasks 1, 2, 3, 8, 9, and 10 must be complete.
Task 8 is provider-side; this independent task closes the producer-side gap and
is owned by the Triton Ventus Backend. Task 11 may not begin until this task's
positive and negative contract tests and all three compile gates pass.

**Files:**
- Create: `third_party/ventus/include/TritonGPUToVentus/LLVMCompatibility.h`
- Create: `third_party/ventus/lib/TritonGPUToVentus/LLVMCompatibility.cpp`
- Create: `third_party/ventus/test/TritonGPUToVentus/llvm16-compatible.mlir`
- Create: `third_party/ventus/test/TritonGPUToVentus/llvm16-incompatible.mlir`
- Create: `python/test/unit/ventus/test_llvm16_contract.py`
- Create: `python/test/unit/ventus/test_elf_link_contract.py`
- Modify: `third_party/ventus/backend/compiler.py`
- Modify: the Task 3 manifest/artifact implementation as required to record the
  compatibility artifact and gate results

**Formal stage contract:** `VentusLLVM16Compatibility` (with checker API
`LLVM16CompatibilityChecker`) receives producer-local MLIR LLVM Dialect/native
LLVM IR before serialization. It emits `kernel.ventus.ll`, a deterministic
Ventus LLVM 16-compatible textual LLVM IR subset, plus structured diagnostics.
This is not an arbitrary newer-to-older LLVM translator. Prefer constraining the
MLIR LLVM Dialect and native translation output. Any post-serialization
normalization must parse and rewrite a structured, explicitly allowlisted
construct; string replacement is forbidden.

**Step 1: Write failing positive contract tests**

Add an MLIR fixture and Python test covering:

- Required `riscv32` triple, pinned RV32 data layout, 32-bit device pointers,
  and `ventus_kernel` calling convention.
- Supported `i1/i32/f32`, legal AS1/AS3, restricted read-only AS4, and AS5 only
  for backend spill/private-resource conventions.
- Frozen builtin/barrier signatures and ordinary scalar CFG, `phi`, load/store,
  GEP, and allowlisted calls.
- At least one OpenCL/Clang LLVM 16 golden comparison using normalized semantic
  and ABI facts, not byte-identical IR.
- One Triton-shaped positive vector-add module that reaches object codegen.

**Step 2: Write the failing negative matrix**

In `llvm16-incompatible.mlir` and the Python test, require deterministic failure
for every category below:

```text
unknown attribute
unsupported intrinsic
scalable vector
i64 device pointer/address
atomicrmw
cmpxchg
atomic load
atomic store
fence or unsupported atomic ordering/scope
invoke/EH
otherwise well-typed addrspacecast
AS4 write
user-visible AS5
unsupported metadata/module flag
```

All `addrspacecast` operations are denied in V1, including otherwise well-typed
casts between pointer address spaces. Also reject unsupported calls and every
atomic form or ordering/scope, including `atomicrmw`, `cmpxchg`, atomic load,
atomic store, and `fence`, plus any LLVM 16-unsupported or untested attribute,
intrinsic, metadata, or module flag. Unknown features must emit a warning and
then fail closed. Never silently delete semantic attributes or flags.

**Step 3: Run the tests and verify failure**

After rebuilding Triton for the new native/compiler files, run:

```bash
pytest -s --tb=short python/test/unit/ventus/test_llvm16_contract.py
```

Expected: FAIL because the checker, emission stage, diagnostics, and manifest
records do not exist.

**Step 4: Implement the internal compatibility checker**

Implement structural inspection over MLIR LLVM Dialect/native IR. Encode a
small explicit allowlist derived from Task 2 OpenCL/Clang goldens and Task 8's
tested provider contract. Distinguish syntax acceptance from target/codegen
semantics. Diagnostics must be deterministic and contain category, operation or
location, observed feature, required contract, and action. Do not claim that
every newer LLVM textual feature is semantically incompatible; reject features
outside the tested contract because they are unproven.

**Step 5: Emit `kernel.ventus.ll` and run three independent M1 gates**

Write the checked textual IR deterministically, then execute in order:

1. Internal `LLVM16CompatibilityChecker` over producer-local IR before
   serialization.
2. Absolute Ventus LLVM 16 `opt` parse/verify, for example
   `"${VENTUS_OPT}" -verify -disable-output kernel.ventus.ll` after verifying
   `VENTUS_OPT` is an absolute pinned path.
3. Absolute Ventus LLVM 16 `llc` target codegen to an object, for example
   `"${VENTUS_LLC}" -mtriple=riscv32 -mcpu=ventus-gpgpu -filetype=obj kernel.ventus.ll -o kernel.ventus.o`.

Gate 2 proves LLVM 16 textual syntax/IR verification. Gate 3 proves the tested
target can select and emit object code for the module. `opt -verify` alone is
insufficient. Use argv subprocess invocation without `shell=True`; capture the
exact absolute argv, stdout, stderr, exit status, and tool identity.

**Step 6: Save compatibility evidence in the manifest**

Record the `kernel.ventus.ll` path/reference and SHA-256 content hash, internal
checker result/diagnostics, both absolute tool invocations and outputs, and the
object content hash. A failed gate must not leave an artifact marked valid.
The recorded tool evidence must include the exact argv, return code, stdout,
stderr, input/output hashes, target profile, and tool identities. Cache identity
must include the contents or immutable identities of the Ventus tools, linker
script, crt0, libclc/work-item inputs, and other code-generation inputs; paths
alone are insufficient.

**Step 6A: Link the Gate 3 object to a Ventus ELF**

Implement the producer-local V1 object-to-ELF owner in
`third_party/ventus/backend/compiler.py`. It runs only after Gate 3 and is a
prerequisite for Gate 4 Spike execution, although it is not numbered as a
separate compatibility gate. `test_elf_link_contract.py` must require an argv
equivalent to the following, with every path absolute and pinned by Task 1:

```text
${VENTUS_LLD}
  -T ${VENTUS_LINKER_SCRIPT}
  -e ${VENTUS_KERNEL_ENTRY_OR_INIT}
  ${KERNEL_OBJECT}
  ${VENTUS_CRT0_OBJECT}
  ${VENTUS_LIBCLC_OBJECT_OR_ARCHIVE}
  ${VENTUS_WORKITEM_OBJECT_OR_ARCHIVE}
  -o ${KERNEL_ELF}
```

Use an argument array and no shell. The test must verify object input, ELF
output, linker script, kernel entry/init handling, crt0, libclc, and work-item
runtime inputs, plus captured argv/stdout/stderr/exit status and tool identity.
The concrete installed filenames and whether libclc/work-item support is an
object or archive must be discovered and pinned by Task 1 and Task 2 ABI
goldens; the plan must not guess library names. Record input hashes, ELF content
hash, entry symbol, linker-script/runtime-input identities, command evidence,
and validation status in the manifest. A link failure or missing pinned input is
a hard compile/environment failure and cannot produce a valid artifact.

**Step 7: Run positive, golden, and negative tests**

Expected: the OpenCL/Clang normalized facts and Triton-shaped vector add pass all
three gates; each negative fails at the internal checker with its stable
category and does not proceed as a valid artifact. Add a focused test proving a
syntactically parseable module can still fail `llc` target semantics.
The link-contract test must also pass and produce an ELF whose hash and entry
identity are recorded.

**Step 8: Commit example**

```bash
git add third_party/ventus/include/TritonGPUToVentus/LLVMCompatibility.h \
  third_party/ventus/lib/TritonGPUToVentus/LLVMCompatibility.cpp \
  third_party/ventus/test/TritonGPUToVentus/llvm16-compatible.mlir \
  third_party/ventus/test/TritonGPUToVentus/llvm16-incompatible.mlir \
  python/test/unit/ventus/test_llvm16_contract.py \
  python/test/unit/ventus/test_elf_link_contract.py \
  third_party/ventus/backend/compiler.py
git commit -m "feat: enforce Ventus LLVM 16 IR compatibility"
```

### Task 11: Lower blocked-layout elementwise and masked memory kernels

**Dependency:** Task 10A must pass first. Memory/elementwise lowering must
produce IR accepted by the compatibility allowlist and may not bypass
`kernel.ventus.ll` emission, any of the three compile gates, or the V1 ELF link
stage.

**Files:**
- Create: `third_party/ventus/lib/TritonGPUToVentus/TritonGPUToLLVM.cpp`
- Create: `third_party/ventus/lib/TritonGPUToVentus/MemoryOpToLLVM.cpp`
- Create: `third_party/ventus/test/TritonGPUToVentus/elementwise.mlir`
- Create: `third_party/ventus/test/TritonGPUToVentus/masked-load-store.mlir`
- Create: `python/test/unit/ventus/test_compile_elementwise.py`

**Step 1: Write failing MLIR and Python compilation tests**

Use a blocked layout with:

```text
threadsPerWarp = [32]
warpsPerCTA = [1] or [2]
sizePerThread = [1]
```

Test add, multiply, compare/select, boundary masks, and global AS1 load/store.
Add negative tests for user-visible AS5 pointers/allocations, unsupported AS4
writes or kernel arguments, and every `addrspacecast`, including an otherwise
well-typed cast.

**Step 2: Run and verify failure**

Expected: conversion cannot legalize TritonGPU load/store and elementwise operations for Ventus.

**Step 3: Reuse common TritonGPU-to-LLVM patterns**

Register common elementwise, view, control-flow, and pointer patterns using
`VentusTargetInfo`. Add target-specific memory patterns only where the common
lowering emits unsupported target operations. Keep the responsibilities
explicit: arithmetic and ordinary CFG should reuse standard/common lowering;
layout materialization, pointer/address-space handling, masked memory, and
target-specific ABI details are Ventus adaptations. Do not add a separate
Ventus arithmetic pass merely for ordinary `arith` operations. Reject
unsupported operations before instruction selection instead of allowing a
generic fallback to guess Ventus semantics.

When the native implementation grows, keep the libraries separable so that
artifact/schema support, target transforms, GPU-to-LLVM lowering, runtime
bindings, and the thin Python plugin binding do not become one monolithic
target. Any shared implementation intended for IREE must be shared as source
compiled against IREE's MLIR revision, or through the textual LLVM/artifact
boundary; do not exchange compiled MLIR/LLVM objects across revisions.

**Step 4: Compile to LLVM IR and ELF**

Expected:

- No Triton or TritonGPU operations remain.
- Global pointers use AS1.
- The internal checker accepts the producer-local IR, `kernel.ventus.ll` is
  saved, absolute Ventus `opt` parses/verifies it, and absolute Ventus `llc`
  compiles it to an object.
- ELF contains the expected kernel symbol and resource section.
- The Gate 3 object is linked only by `compiler.py` using the pinned absolute
  `VENTUS_LLD` contract, and the manifest records the link and ELF evidence.

Compilation evidence must include the exact argv, return code, stdout/stderr,
input IR reference/hash, target profile, tool identities, and output hashes.
The cache key must cover the Triton/backend source identity, target/profile,
options, external tool contents/versions, linker script, crt0, libclc/work-item
inputs, and any other input that can affect code generation. A filesystem path
alone is not sufficient to invalidate stale binaries.

**Step 5: Run compilation tests**

Expected: PASS without requiring a Ventus device.

**Step 6: Commit**

```bash
git add third_party/ventus/lib/TritonGPUToVentus third_party/ventus/test/TritonGPUToVentus python/test/unit/ventus/test_compile_elementwise.py
git commit -m "feat: compile Triton elementwise kernels for Ventus"
```

### Task 12: Execute the M1 basic kernel matrix on Ventus Spike

**Dependency and fourth gate:** Tasks 10A and 11 must have passed the internal
checker, absolute Ventus `opt` parse/verify, and absolute Ventus `llc` object
codegen gates. This task's ELF/Spike execution is the fourth M1 gate; it cannot
be replaced by `opt -verify` or object emission.

**Files:**
- Create: `third_party/ventus/runtime/argument_packer.py`
- Create: `third_party/ventus/runtime/reference_launcher.py`
- Create: `python/test/unit/ventus/test_vector_add_spike.py`
- Create: `python/test/unit/ventus/test_masked_copy_spike.py`
- Create: `python/test/unit/ventus/test_basic_kernels_spike.py`

**Step 1: Write failing end-to-end tests**

Compile and execute the complete M1 matrix: fill, copy, vector add, multiply,
fused elementwise, masked AS1 copy/load/store, broadcast, reshape/view, select,
and ReLU. Keep the current dedicated vector-add and masked-copy tests; use one
parameterized `test_basic_kernels_spike.py` for the remaining cases rather than
creating one file per operation. Allocate through the existing Ventus
test/runtime facilities, launch on Spike, and compare with NumPy or PyTorch CPU
results.

Mark tests with a Ventus-specific pytest marker. A convenience local developer
run may report a clearly non-green skip when the explicitly selected local
runtime suite is not requested or provisioned. In every M1/M2 milestone,
release, or required CI workflow, missing or mismatched pinned Spike/runtime is
a hard environment/gate failure, never an accepted green skip. Only CycleSim
may use capability-gated skip semantics.
Add launch failures for ELF hash/target/entry mismatch, capability-record
mismatch, resource-unit mismatch, timeout, and unobservable cache-flush
completion. Add bounded divergent `if`/loop cases plus an over-limit case that
must fail before launch with a deterministic diagnostic.

Treat each boundary as a separate assertion rather than treating ELF creation
as execution proof. The M1 evidence must independently establish accepted LLVM
input, object emission, ELF section/symbol identity, artifact loading, argument
packing, launch geometry, resource validation, completion/timeout observation,
and numerical result correctness.

**Step 2: Run and verify failure**

Expected: FAIL because the artifact cannot be packed/launched.

**Step 3: Implement reference argument packing**

Use the shared artifact schema for offsets and alignment. Do not infer layout independently in Python.

**Step 4: Implement a test-only reference launcher**

This launcher may use the existing Ventus test harness or POCL-compatible metadata path, but it must consume the same ELF and metadata that IREE will later consume.
It must not treat the current `vt_dev_caps` response as a complete capability
contract. Validate the independent capability record and ELF identity, translate
only the documented simulator physical-address convention, and expose
deterministic completion, timeout, cache-flush status, and diagnostic fields.

**Step 5: Run Spike tests**

Expected: PASS for every M1 operation above, including aligned, tail-masked, and
non-multiple-of-workgroup inputs. Save the ELF hash, launcher input/manifest
hash, pinned Spike binary identity, execution status/result, and test-result
hash as Gate 4 evidence.
Handle zero logical workload by skipping allocation and launch in the reference
launcher; the current driver does not define zero-byte device allocation.

**Step 6: Commit**

```bash
git add third_party/ventus/runtime python/test/unit/ventus/test_vector_add_spike.py python/test/unit/ventus/test_masked_copy_spike.py python/test/unit/ventus/test_basic_kernels_spike.py
git commit -m "test: run Triton Ventus kernels on Spike"
```

### Task 13: Establish the PyTorch export and StableHLO model contract

**Status and gate:** Post-V1, non-gating. Do not begin Tasks 13-17 until Tasks
1-10, 10A, 11-12, 19, 18, 20, 20A, 20B, and the V1 portion of Task 22 have completed.

**Files in the model frontend/IREE integration repository:**
- Create: `tests/ventus/models/export_smoke.py`
- Create: `tests/ventus/models/export_prefill.py`
- Create: `tests/ventus/models/export_decode.py`
- Create: `tests/ventus/models/check_stablehlo.py`
- Create: `docs/ventus/StableHLOContract.md`
- Create: model weight-binding and constant-archive tests

**Step 1: Write failing export tests for a fixed inference module**

Use a small inference-only `nn.Module` with parameter weights, elementwise
operations, a reduction, and matmul. Create separate prefill/decode wrappers if
the selected model has autoregressive state.

Check that `torch.export.export()` preserves:

- Functional ATen graph semantics.
- Shape constraints.
- Graph signature.
- Parameter and buffer names.
- Dynamic scalar inputs required at runtime.

**Step 2: Run and verify failure**

Expected: missing decomposition, unsupported operation, or absent StableHLO
conversion exposes the first frontend gap.

**Step 3: Add the minimum Torch MLIR and Ventus frontend passes**

Reuse generic torch-mlir decomposition and value-semantics passes. Add Ventus
legal-op policy and custom rewrites only for demonstrated gaps. Preserve coarse
target operations as `stablehlo.custom_call @ventus.*` only when a generic
StableHLO decomposition is undesirable.

**Step 4: Define and verify the StableHLO contract**

Document supported primitives, custom calls, dynamic dimensions, function
signatures, parameter bindings, and explicit compile errors. Do not include
Ventus work-group, warp, or machine instructions in StableHLO.

**Step 5: Add weight packaging tests**

Verify that checkpoint parameters are mapped by stable names into an aligned
constant archive with recorded dtype, shape, layout, offset, and size. Do not
freeze all weights as large MLIR literals.

**Step 6: Run frontend tests**

Expected: the selected model exports reproducibly to the documented StableHLO
contract and produces deterministic weight-binding metadata.

**Step 7: Commit in the frontend/integration repository**

Use that repository's commit style, for example:

```bash
git commit -m "feat: define Ventus StableHLO export contract"
```

### Task 14: Add IREE generic Ventus dispatch device codegen

**Status and gate:** Post-V1, non-gating. This task cannot satisfy or precede a
V1 compiler, execution, MMA, or release gate.

**Files in the IREE repository:**
- Create: `compiler/src/iree/compiler/Codegen/Ventus/VentusTarget.cpp`
- Create: `compiler/src/iree/compiler/Codegen/Ventus/VentusPassPipeline.cpp`
- Create: `compiler/src/iree/compiler/Codegen/Ventus/CMakeLists.txt`
- Create: `compiler/src/iree/compiler/Codegen/Ventus/test/elementwise.mlir`
- Create: `compiler/src/iree/compiler/Codegen/Ventus/test/reduction.mlir`
- Create: `compiler/src/iree/compiler/Codegen/Ventus/test/dispatch-workgroup.mlir`
- Depend on: shared Ventus MLIR target support

**Step 1: Write failing device-codegen tests**

Start with IREE dispatch bodies containing elementwise, masked boundary, and
simple reduction workloads. Check that tiling/distribution produces standard:

```text
arith / scf / memref / gpu / vector
```

plus the thin Ventus target/ABI attributes, not a duplicate full hardware IR.

**Step 2: Run and verify failure**

Run the narrow IREE Codegen lit target.

Expected: FAIL because no Ventus executable target or pass pipeline exists.

**Step 3: Implement the minimal generic codegen pipeline**

Implement:

```text
IREE dispatch
  -> Linalg tiling/fusion
  -> work-group distribution
  -> bufferization
  -> SCF/Vector/GPU/MemRef
  -> shared Ventus target lowering
  -> LLVM Dialect
```

Use one-dimensional work-groups first. Keep Flow/Stream/HAL operations out of
the device module.

**Step 4: Compile through the modified Ventus LLVM backend**

Expected: generic IREE vector add lowers to a verified Ventus ELF using the
same LLVM input contract and resource metadata as Triton and OpenCL.

**Step 5: Run generic codegen tests**

Expected: elementwise and baseline reduction tests PASS; unsupported operations
fail before LLVM instruction selection with explicit diagnostics.

**Step 6: Commit in the IREE repository**

```bash
git add compiler/src/iree/compiler/Codegen/Ventus
git commit -m "feat: add generic IREE Ventus device codegen"
```

### Task 15: Add the IREE Ventus executable artifact format

**Status and gate:** Post-V1, non-gating; depends on V1 completion and the
versioned artifact proven by the Triton baseline.

**Files:**
- Modify in the IREE repository: `compiler/src/iree/compiler/Codegen/Ventus/VentusTarget.cpp`
- Create in the IREE repository: `compiler/src/iree/compiler/Codegen/Ventus/VentusExecutable.cpp`
- Modify in the IREE repository: `compiler/src/iree/compiler/Codegen/Ventus/CMakeLists.txt`
- Create in the IREE repository: `compiler/src/iree/compiler/Codegen/Ventus/test/import_triton_artifact.mlir`
- Import or depend on: shared `VentusArtifact` definitions

**Step 1: Write a failing IREE compiler test**

Create an IREE executable containing one external Ventus kernel artifact with:

- ELF bytes.
- Entry point.
- Work-group size.
- Four arguments matching vector add.

Check that serialization preserves all metadata.

**Step 2: Run and verify failure**

Run the narrow IREE lit test target.

Expected: FAIL because the Ventus executable target is unavailable.

**Step 3: Implement artifact import and verification**

Validate ABI version, toolchain identity, pointer width, entry-point symbol, and argument bindings during IREE compilation.

**Step 4: Run compiler tests**

Expected: valid artifacts serialize; incompatible ABI/toolchain artifacts fail with explicit diagnostics.

**Step 5: Commit in the IREE repository**

```bash
git add compiler/src/iree/compiler/Codegen/Ventus
git commit -m "feat: add IREE Ventus executable artifacts"
```

### Task 16: Implement the IREE HAL Ventus driver baseline

**Status and gate:** Post-V1, non-gating; must not precede V1 completion.

**Files:**
- Create in the IREE repository: `runtime/src/iree/hal/drivers/ventus/ventus_device.c`
- Create in the IREE repository: `runtime/src/iree/hal/drivers/ventus/ventus_executable.c`
- Create in the IREE repository: `runtime/src/iree/hal/drivers/ventus/ventus_allocator.c`
- Create in the IREE repository: `runtime/src/iree/hal/drivers/ventus/ventus_command_buffer.c`
- Create in the IREE repository: `runtime/src/iree/hal/drivers/ventus/CMakeLists.txt`
- Create in the IREE repository: `runtime/src/iree/hal/drivers/ventus/test/ventus_driver_test.cc`

**Step 1: Write failing HAL tests**

Test:

- Device creation.
- Buffer allocation and mapping rules.
- Executable loading and entry-point lookup.
- Argument block packing.
- One direct dispatch.
- Toolchain/ABI mismatch rejection.

**Step 2: Run and verify failure**

Expected: FAIL because the driver is not registered.

**Step 3: Implement the minimal synchronous driver**

Initially support one queue and synchronous submission. Wrap the existing
Ventus C driver API rather than POCL, and keep Spike, CycleSim, RTL simulation,
and GVM selection behind a driver option. Do not claim physical-device support
until a concrete backend exists.

**Step 4: Implement executable loading**

Parse the shared artifact, locate the entry point through an ELF object API, and
consume validated resource metadata. The current driver backends do not provide
a consistent complete-ELF byte-loading contract, so the baseline may use a
secure temporary ELF file; add an explicit in-memory ELF module API before
production deployment.

**Step 5: Run HAL unit tests**

Expected: PASS without asynchronous command-buffer features.

**Step 6: Commit in the IREE repository**

```bash
git add runtime/src/iree/hal/drivers/ventus
git commit -m "feat: add baseline IREE HAL Ventus driver"
```

### Task 17: Dispatch a precompiled Triton kernel through IREE HAL

**Status and gate:** Post-V1, non-gating; must not precede V1 completion and all
of Tasks 13-16.

**Files:**
- Create in the IREE repository: `tests/e2e/ventus/triton_vector_add.mlir`
- Create in the IREE repository: `tests/e2e/ventus/run_triton_vector_add.py`
- Add fixture artifact generated by Task 11 under the IREE test data policy

**Step 1: Write the failing end-to-end test**

Package the Triton vector-add ELF into an IREE executable, bind three buffers plus `n`, dispatch it through the Ventus HAL driver, and compare output with a CPU reference.

**Step 2: Run and verify failure**

Expected: FAIL at missing dispatch integration or launch metadata.

**Step 3: Complete binding and launch geometry integration**

Map IREE bindings to the shared artifact argument descriptors. Use compile-time work-group size and runtime workload to compute the number of work-groups.

**Step 4: Run on Spike**

Expected: PASS for full and tail work-groups.

**Step 5: Run the POCL differential test**

Run the equivalent OpenCL kernel through POCL and compare device output and launch geometry.

Expected: matching numerical output.

**Step 6: Commit in the IREE repository**

```bash
git add tests/e2e/ventus
git commit -m "test: dispatch Triton artifacts through IREE Ventus HAL"
```

### Task 18: Add Triton reductions needed by inference

**Dependency:** Task 19 static shared memory and full work-group barrier must be
complete first. Although Task 18 appears first numerically, execute Task 19
before Task 18. The no-bypass invariant applies to every reduction module.

**Files:**
- Create: `third_party/ventus/test/TritonGPUToVentus/reduce.mlir`
- Create: `python/test/unit/ventus/test_reduce.py`
- Modify: `third_party/ventus/lib/TritonGPUToVentus/TargetInfo.cpp`
- Modify or create target reduction patterns only as required

**Step 1: Write failing compile and execution tests**

Start with non-empty, fixed/static-extent reductions limited to one or two
warps and not requiring unsupported cross-warp shuffle semantics:

- One-warp sum and max.
- Two-warp sum and max through shared memory and a work-group barrier.
- `max` cases containing NaNs, `-0.0`/`+0.0`, and infinities. A native RTL
  FP compare/max instruction is insufficient evidence for the normative policy.
- Deterministic reduction order for both `sum` and `max`.
- Negative cases rejecting empty extent, dynamic extent, and any extent/warp
  count outside the documented 1-2 warp contract.

**Step 2: Run and verify failure**

Expected: reduction lowering reports unsupported target operations.

**Step 3: Implement the minimal correct strategy**

Prefer common Triton reduction lowering when it can use supported Ventus operations. If shuffle is unavailable, use shared memory and barriers rather than inventing unsupported shuffle intrinsics.
Use explicit classify/compare/select/canonicalization for `max` when needed to
implement propagate-NaN and canonical `+0.0` ties exactly.
Preserve the fixed deterministic order and the scope's infinity behavior.

**Step 4: Run compile and Spike tests**

Expected: PASS within documented floating-point tolerances.

**Step 5: Commit**

```bash
git add third_party/ventus/test/TritonGPUToVentus python/test/unit/ventus/test_reduce.py third_party/ventus/lib/TritonGPUToVentus
git commit -m "feat: support Triton reductions on Ventus"
```

### Task 19: Add static shared memory and barriers

**Pipeline invariant:** Every shared-memory/barrier module follows the no-bypass
compatibility and ELF pipeline before mandatory Spike execution.

**Files:**
- Create: `third_party/ventus/test/TritonGPUToVentus/shared-memory.mlir`
- Create: `python/test/unit/ventus/test_shared_memory.py`
- Modify: shared-memory allocation and memory lowering integration as required
- Modify: artifact metadata emission for shared-memory bytes

**Step 1: Write failing tests**

Compile and run a kernel that:

- Allocates static shared memory.
- Stores one value per thread.
- Executes a work-group barrier.
- Reads another thread's stored value after the barrier.

**Step 2: Run and verify failure**

Expected: shared allocation or barrier lowering is unsupported.

**Step 3: Implement AS3 allocation and resource accounting**

Map Triton shared memory to Ventus AS3 and emit exact shared-memory byte usage into the artifact. Reject dynamic shared memory until its ABI is specified.

**Step 4: Run multi-warp Spike tests**

Expected: PASS for one- and multi-warp work-groups.

**Step 5: Run capability-gated CycleSim smoke test**

Expected: when exact backend capability is available, the kernel completes
without barrier deadlock and produces correct output. Otherwise record an
explicit capability-gated skip reason. Spike results from Step 4 remain
mandatory and cannot be replaced by this skip.

**Step 6: Commit**

```bash
git add third_party/ventus/test/TritonGPUToVentus python/test/unit/ventus/test_shared_memory.py third_party/ventus
git commit -m "feat: support Ventus shared memory and barriers"
```

### Task 20: Add blocked-layout FMA matrix multiplication

**Pipeline invariant:** Every blocked-FMA module follows the no-bypass
compatibility and ELF pipeline before mandatory Spike execution.

**Files:**
- Create: `python/test/unit/ventus/test_matmul.py`
- Create: `third_party/ventus/test/TritonGPUToVentus/dot.mlir`
- Modify: `third_party/ventus/backend/compiler.py`
- Modify: common/generic dot lowering registration as required

**Step 1: Write failing correctness tests**

Use exactly the static 2D FP32 blocked `M=N=K=32` canonical pattern, full
active lanes, and general-path local size `[32,1,1]`. Require ordinary
multiply-add lowering and do not accept FP16, arbitrary shape/layout, dynamic
extent, or MMA fragments.

**Step 2: Run and verify failure**

Expected: dot layout or dot lowering is unsupported.

**Step 3: Implement generic blocked dot lowering**

Implement only the explicitly accepted blocked FP32 `32x32x32` pattern. Reject
all other `tl.dot` shapes/layouts deterministically. Fallback must occur before
MMA selection or through an explicitly tested rematerialization to blocked
layout; do not silently convert an existing MMA fragment back to blocked form.

**Step 4: Run mandatory Spike and capability-gated CycleSim tests**

Expected: Spike numerical correctness against PyTorch CPU with documented
tolerance is mandatory. CycleSim runs only when capability is available and
otherwise records an explicit gated skip reason.

**Step 5: Record baseline performance and resource usage**

Store compile-time metadata and simulator cycles as non-gating diagnostics. Do not optimize by adding target-specific MMA in this task.

**Step 6: Commit**

```bash
git add python/test/unit/ventus/test_matmul.py third_party/ventus/test/TritonGPUToVentus/dot.mlir third_party/ventus/backend/compiler.py
git commit -m "feat: add baseline Triton matmul for Ventus"
```

### Task 20A: Freeze the fixed RTL MMA contract (M3, Contract Only)

**Dependencies:** Tasks 1-10, Task 10A, Tasks 11-12, Task 19, Task 18, and Task
20 must pass. M3 does
not add Triton `tl.dot` lowering and must not claim Triton compile/execute.

**Files in `ventus-env` and Triton:**
- Create in GPGPU/RTL: `docs/VentusMmaProfileA.md`
- Create in GPGPU/RTL: `test/ventus_mma_profile_a/` using the repository's RTL
  testbench layout, including deterministic input/output/trace goldens
- Create in Spike: the repository-local `vftta.vv` semantic test and golden
- Modify in CycleSim only if capability is present: its `vftta.vv` implementation
  and exact-profile regression fixture
- Create in Ventus LLVM: `llvm/test/CodeGen/RISCV/VentusGPGPU/vftta-profile-a.ll`
- Modify in Ventus LLVM, choosing one boundary:
  `llvm/include/llvm/IR/IntrinsicsRISCV.td` and RISCV instruction lowering files
  for an intrinsic, or the controlled inline-asm acceptance/codegen path
- Create: `third_party/ventus/toolchain/mma_profile_a.json`
- Create: `third_party/ventus/test/mma/profile_a_contract.py`
- Create: `third_party/ventus/test/mma/golden/profile_a_inputs.json`
- Create: `third_party/ventus/test/mma/golden/profile_a_spike.json`
- Create: `third_party/ventus/test/mma/golden/profile_a_cyclesim.json` when capable
- Create: `third_party/ventus/test/mma/golden/profile_a_rtl.json`
- Create: `python/test/unit/ventus/test_mma_contract.py`

**Step 0: Block on Chisel/source-generated RTL consistency**

Pin and regenerate RTL from
`gpgpu/ventus/src/top/parameters.scala`,
`gpgpu/ventus/src/pipeline/execution.scala`, and
`gpgpu/dependencies/fpuv2/src/main/scala/Tensor.scala`. Create one verified
generation record that pins/hashes the Chisel sources and binds them to effective
parameters (`num_thread=32`; logical `M=constructor DimM`, logical `K=constructor
DimN`, logical `N=constructor DimK`; labeled logical order `(M,K,N)=(4,8,4)`;
constructor positional order `(M,N,K)`), generation command, generated Verilog,
and RTL testbench. Inspect the regenerated design for 16
meaningful dot units and 8-element reduction trees. Fail M3 if it still matches
the inspected older/smaller `gpgpu/driver/rtl/GPGPU_top.v` profile with 4 dot
units/4-element dots, or if the verified generation record is missing or
inconsistent. Never persist an unlabeled tuple.

**Step 1: Write failing contract and golden tests**

Require one canonical contract:

```text
VentusMmaProfileA
logical D[M,N] = A[M,K] x B[K,N] + C[M,N]
(M,K,N) = (4,8,4)
A[M,K] = [4,8], B[K,N] = [8,4], C/D[M,N] = [4,4]
physical vftta B fragment = B^T[N,K] = [4,8]
instruction = vftta.vv
old vd=C[M,N], vs1=A[M,K], vs2=physical B^T[N,K]; new vd=D[M,N]
FP32, one full active 32-lane warp, local_size = [32,1,1], no masked MMA
```

The test must fail unless the manifest fixes lane-to-fragment elements and mandatory
architectural binding old `vd=C[M,N]`, `vs1=A[M,K]`, `vs2=physical B^T[N,K]`,
new `vd=D[M,N]`; A/B physical lanes `0..31`, C/D meaningful
lanes `0..15`, all-32-lane writeback and upper-lane behavior, exact FP32
multiplication/accumulation tree and per-stage rounding, NaN/signed-zero/
exception behavior, `fflags` production, cross-lane/pipeline aggregation and
architectural writeback, active-lane requirements, encoded `vm` mask-control bit,
unmasked, `e32`, fixed
hardware width versus architectural `vl`, register range/alignment/overlap,
latency/issue/ready-valid backpressure, dot/add pipeline resource units, target
feature spelling, intrinsic or controlled-asm contract version, RTL revision,
and simulator revision/capability records. Add negative fixtures for transposed
or `[8,4]` physical B notation, partial active warp, swapped `vs1`/`vs2`,
non-tied accumulator, stale
profile hash, missing RTL result, unlabeled dimensions, non-e32, masked `vm`,
variable-width `vl`, illegal register overlap, and unspecified unused lanes.
Require the architectural spelling/semantics to have three explicit operands with
tied destination, `vftta.vv vd, vs2, vs1`; `vm` is an encoded mask-control bit/comment
notation, not a fourth assembly operand. Remove or quarantine legacy four-register
CycleSim fixtures so they cannot define profile A. If architectural `fflags` is not
observable in a backend, record it as unresolved and fail the M3 complete-manifest gate.

**Step 2: Run the tests and verify failure**

Run:

```bash
pytest -s --tb=short python/test/unit/ventus/test_mma_contract.py
```

Run the narrow Spike semantic test, Ventus LLVM lit test, and RTL testbench
command documented by each pinned repository.

Expected: FAIL because the profile manifest/goldens and at least one aligned
semantic or codegen implementation do not exist. A missing CycleSim capability
is an expected explicit gated skip, not a failure. A missing Spike result or RTL
result is a failure and cannot be skipped.

**Step 3: Establish the RTL source-of-truth golden**

Use deterministic finite FP32 operands covering positive/negative values,
zeros, cancellation, and nontrivial accumulation. Record raw lane fragments,
instruction inputs, output fragments, architectural state needed to identify
active lanes, and RTL revision/content hash. Run RTL simulation and store the
actual result/trace, including observable `fflags` production/aggregation/writeback.
Do not synthesize an expected RTL record from Spike or a
Python reference. M3 cannot complete without this RTL result.
This step uses only the Step 0 regenerated-and-hashed RTL/testbench pair. Verify
16 meaningful C/D outputs and the wrapper's 32-lane writeback, including whether
lanes `16..31` are zero. Freeze the observed behavior rather than assuming it.

**Step 4: Align Spike and capability-gated CycleSim**

Implement or correct Spike semantics until its lane-level output matches the RTL
golden and the ordered FP32 reference, including the frozen architectural
`fflags`. If CycleSim advertises exact
`VentusMmaProfileA` capability, align and run it against the same inputs; if it
does not, emit a machine-readable skip containing observed operation shape,
revision, and `reason=profile_a_unavailable`. Never reinterpret the existing
smaller CycleSim operation as profile A.
The current Spike file has dual behavior keyed by architectural `vl`, and the
inspected CycleSim fixtures include a smaller `2x2x4`-style operation; classify
both as non-profile-A until corrected and covered by the same manifest.

**Step 5: Define and test the LLVM codegen boundary**

Prefer a typed intrinsic such as `llvm.riscv.ventus.vftta.vv` when the pinned
Ventus LLVM can validate operands and resources. Otherwise use only a controlled
inline-asm emitter with fixed `vftta.vv` spelling, constraints, clobbers, and
operand order; arbitrary user Triton inline asm must not reach this path. The lit
test must check LLVM IR acceptance, exact assembly, object/ELF emission, and
absence of a scalar-FMA replacement.

Run the pinned Ventus LLVM suite command for
`llvm/test/CodeGen/RISCV/VentusGPGPU/vftta-profile-a.ll`.

Expected: PASS and assembly contains exactly the contract-approved
`vftta.vv` form.

**Step 6: Freeze the versioned profile manifest**

Write `mma_profile_a.json` with the complete contract, intrinsic/asm choice,
toolchain and RTL identities, mandatory Spike/RTL capabilities, capability-gated
CycleSim status, resource rules,
and content hashes of every golden. Extend Task 3 artifact validation so an MMA
artifact can require this exact frozen profile and rejects version/hash mismatch.
The manifest must include every Step 0/Step 1 field; absent fields make M4
unavailable rather than defaulted.

**Step 7: Run M3 differential verification**

Run the Python contract test, Spike semantic test, Ventus LLVM codegen test, and
RTL test. Run CycleSim only when exact-profile capability is reported.

Expected: Spike and RTL match the same result and `fflags` reference and each other; RTL result is
present and non-skipped; CycleSim either matches or records the approved gated
skip. No Triton `tl.dot` compile/execute result is required or claimed by M3.

**Step 8: Commit examples**

Commit separately in the affected repositories when executing this plan:

```bash
git commit -m "test: freeze Ventus MMA profile A contract"
git commit -m "feat: add Ventus vftta profile A codegen"
```

### Task 20B: Lower the fixed Triton `tl.dot` profile to `vftta.vv` (M4)

**Dependency:** Task 20A has a frozen manifest with mandatory Spike and RTL
results and every verified-generation identity, dimension mapping, lane/writeback,
operand binding, syntax, FP/`fflags`,
mask/`vl`, register, latency/backpressure, and resource field above. Otherwise
M4 is disabled. M4 is the compile-and-execute milestone.
The no-bypass invariant applies to every M4 module, including diagnostics and
fallback paths that produce an artifact.

**Files:**
- Create: `third_party/ventus/include/Dialect/VentusGPU/IR/VentusGPUAttrDefs.td`
- Create: `third_party/ventus/lib/Dialect/VentusGPU/IR/VentusGPUAttrs.cpp`
- Create: `third_party/ventus/lib/TritonGPUToVentus/VentusMmaLowering.cpp`
- Modify: `third_party/ventus/lib/TritonGPUToVentus/TritonGPUToLLVM.cpp`
- Modify: `third_party/ventus/backend/compiler.py`
- Modify: `third_party/ventus/triton_ventus.cc`
- Create: `third_party/ventus/test/TritonGPUToVentus/mma-profile-a.mlir`
- Create: `third_party/ventus/test/TritonGPUToVentus/mma-profile-a-invalid.mlir`
- Create: `third_party/ventus/test/TritonGPUToVentus/mma-profile-a-codegen.mlir`
- Create: `python/test/unit/ventus/test_mma_profile_a_compile.py`
- Create: `python/test/unit/ventus/test_mma_profile_a_diagnostics.py`
- Create: `python/test/unit/ventus/test_mma_profile_a_spike.py`
- Create: `python/test/unit/ventus/test_mma_profile_a_differential.py`
- Create: `third_party/ventus/test/mma/golden/triton_profile_a_manifest.json`

**Step 1: Write failing encoding and legality tests**

Add only narrow `VentusMmaEncodingAttr` and
`VentusDotOperandEncodingAttr` representations for the frozen profile. Tests
must require exact FP32 M4K8N4 logical semantics, physical `B^T[N,K]=[4,8]`,
full active 32-lane warp, local size `[32,1,1]`, fixed fragment mapping, no
instruction-level mask, `e32`, fixed hardware width, legal register
range/alignment/overlap, sufficient resources, and exact frozen contract hash.

Negative tests must cover wrong M/N/K, dtype, physical B shape/orientation,
swapped `vs1`/`vs2`, non-tied accumulator, mask, active lanes, warp size, local size, layout, dynamic shape,
resource overflow, absent feature, and stale ABI/profile version. Diagnostics
must be deterministic and include operation, observed shape/dtype/layout,
`required_shape=M4K8N4`, required profile/hash, observed capability, and action.
Only a pattern already in the explicit blocked accepted set may report FMA
fallback; all others report `unsupported=no_general_tl_dot`.

**Step 2: Run tests and verify failure**

Run:

```bash
pytest -s --tb=short python/test/unit/ventus/test_mma_profile_a_compile.py
pytest -s --tb=short python/test/unit/ventus/test_mma_profile_a_diagnostics.py
```

Run the relevant `triton-opt` lit files from the Triton build directory after
rebuilding native/compiler changes as required by this repository.

Expected: FAIL because the encodings, legality verifier, and lowering do not
exist.

**Step 3: Implement the exact Triton encodings and selection gate**

Represent only the frozen lane fragments and dot operands. In the TTIR/TTGIR
pipeline, select MMA only after proving every legality condition and profile
identity. Keep ordinary blocked FMA selection separate: `32x32x32` is the
mandatory canonical reference/fallback, and an additional static blocked shape
is eligible only after its own explicit lowering and correctness test has added
it to the accepted set. Never advertise or infer general `tl.dot` support.

**Step 4: Lower to the M3 LLVM boundary**

Lower legal fragments to the exact intrinsic or controlled inline-asm contract
frozen in M3. Preserve old `vd=C`, `vs1=A`, `vs2=physical B^T`, new `vd=D`,
constraints, clobbers, FP32 accumulator semantics, and `fflags`. Reject rather
than silently scalarize an otherwise MMA-encoded illegal
fragment. A fallback is allowed only before MMA fragment formation or through an
explicitly tested conversion to an accepted blocked layout.

**Step 5: Verify LLVM IR, assembly, ELF, and artifact metadata**

The codegen test must check the selected intrinsic/asm in LLVM IR, exact
`vftta.vv` assembly, entry-point symbol, ELF identity/resource section, and a
manifest containing `VentusMmaProfileA`, operation/shape/layout constraints,
full-active-warp requirement, static resources, toolchain identity, frozen RTL
profile hash, mandatory operand binding, `fflags` behavior, and references to
TTIR/TTGIR/LLVM/assembly/ELF/test artifacts.

Expected: legal profile emits one contract-approved `vftta.vv`; illegal profiles
emit the specified diagnostic or accepted-set FMA fallback and never emit
`vftta.vv`.

**Step 6: Execute mandatory Spike correctness tests**

Compile a Triton kernel using exact profile A, launch through the reference
launcher, and compare output fragments, logical D, and architectural `fflags`
against the M3 ordered FP32 reference. Cover at least the canonical M3 operands and a second deterministic
input set. Spike is mandatory; tool absence is an environment failure for M4,
not an accepted skip.

Run:

```bash
pytest -s --tb=short python/test/unit/ventus/test_mma_profile_a_spike.py
```

Expected: PASS with the frozen profile/ABI hash in the result record.

**Step 7: Execute mandatory Spike/RTL and capability-gated CycleSim differential goldens**

Feed the same Triton-produced ELF, argument packing, launch metadata, and input
fragments to each backend. RTL execution and result/trace are mandatory and
cannot skip. CycleSim runs only when it advertises exact profile A; otherwise the
test records the same machine-readable capability-gated skip format frozen in
M3. Compare output bits or the explicitly frozen FP32 tolerance/NaN policy,
compare `fflags` production/aggregation/writeback, and verify resource metadata
before launch.

Run:

```bash
pytest -s --tb=short python/test/unit/ventus/test_mma_profile_a_differential.py
```

Expected: Spike and RTL match the M3 goldens and logical reference; CycleSim
matches or has the approved capability skip. Any missing RTL result, stale
profile hash, metadata mismatch, or unexpected fallback is FAIL.

**Step 8: Run the full M4 regression set**

Run the encoding/lowering lit tests, compile and diagnostics pytest tests,
mandatory Spike execution, differential test, artifact round-trip/resource
tests, and the Task 20 ordinary-FMA regression to prove the fallback remains
independent of MMA.

Expected: all required tests PASS; no test or documentation claims general
`tl.dot`.

**Step 9: Commit example**

```bash
git add third_party/ventus python/test/unit/ventus
git commit -m "feat: lower Ventus MMA profile A from Triton"
```

### Task 21: Build the tuned Triton kernel library and IREE selection

**Status and gate:** Post-V1, non-gating IREE/kernel-library expansion. It may
begin only after V1 completion and does not participate in the V1 critical path.

**Files:**
- Create: `third_party/ventus/kernel_library/schema.json`
- Create: `third_party/ventus/kernel_library/README.md`
- Create: `third_party/ventus/tools/package_kernel.py`
- Create: `python/test/unit/ventus/test_kernel_library.py`
- Create in the IREE repository: `compiler/src/iree/compiler/Codegen/Ventus/VentusKernelSelection.cpp`
- Create in the IREE repository: `compiler/src/iree/compiler/Codegen/Ventus/test/kernel_selection.mlir`
- Create in the IREE repository: a model-level Ventus e2e test using `torch.export`/StableHLO input

**Step 1: Write failing kernel-library tests**

Require each tuned record to contain:

```text
operation
shapes and dynamic constraints
dtypes
layout requirements
target architecture and features
toolchain identity
artifact ABI version
ELF identity and entry point
argument layout
work-group size
VGPR/SGPR/LDS/PDS
measured backend and performance metric
```

Reject records with missing compatibility fields or unverifiable ELF symbols.

**Step 2: Run and verify failure**

Expected: FAIL because no schema, package tool, or selector exists.

**Step 3: Package independently tuned Triton operators**

Start with the kernels already validated by prior tasks. Store ELF artifacts
outside source files according to repository artifact policy, with deterministic
content hashes and human-readable metadata for debugging.

Triton remains an independent operator tool. Do not add a PyTorch device
backend or make TorchInductor a model graph owner in this task.

**Step 4: Write failing IREE selection and fallback tests**

Cover:

- Exact shape/dtype/target match selects a Triton artifact.
- ABI, toolchain, feature, or resource mismatch rejects the artifact.
- No match selects generic IREE Ventus codegen instead of failing compilation.
- A `stablehlo.custom_call @ventus.*` can select a compatible tuned artifact.

**Step 5: Implement minimal IREE kernel selection**

Perform selection during dispatch device codegen, before HAL executable
packaging. Keep Stream/HAL responsible only for buffers, bindings, workload,
commands, and runtime dependencies.

**Step 6: Run a complete-model IREE inference test**

Compile an exported PyTorch inference graph through:

```text
torch.export -> StableHLO -> IREE Flow -> dispatch codegen
```

Require at least one dispatch to use a tuned Triton artifact and at least one
supported dispatch to exercise generic IREE Ventus codegen fallback. Execute
through IREE HAL on Spike and compare with the PyTorch CPU reference.

**Step 7: Record tuning provenance**

For each selected kernel, retain CycleSim/hardware measurements and final LLVM
resource metadata. Never select kernels using TTGIR parameters alone.

**Step 8: Commit in the relevant repositories**

```bash
git add third_party/ventus/kernel_library third_party/ventus/tools/package_kernel.py python/test/unit/ventus/test_kernel_library.py
git commit -m "feat: package tuned Ventus Triton kernels"
```

Commit IREE selector and model e2e tests separately using the IREE repository's
message style.

### Task 22: Add continuous verification and compatibility gates

**Files:**
- Create: `.github/workflows/ventus-compile.yml`
- Create: `.github/workflows/ventus-spike.yml`
- Create: `python/test/unit/ventus/conftest.py`
- Create: `third_party/ventus/test/README.md`

**Step 1: Add failing CI configuration tests or local validation scripts**

Require separate suites for:

- Host-only MLIR and artifact tests.
- Ventus toolchain compilation tests.
- Internal LLVM 16 compatibility, absolute `opt` parse/verify, and absolute
  `llc` target-object gates, with `kernel.ventus.ll` hashes and diagnostics.
- Spike execution tests.
- Optional CycleSim tests.
- AS4/AS5 restriction and bounded-divergence diagnostics.
- Deterministic runtime timeout/completion/cache-flush and ELF identity failures.
- M3 source/parameter/generated-Verilog/testbench hash consistency.

**Step 2: Add toolchain caching and identity checks**

CI must fail if the installed toolchain identity differs from `version.json`.
The milestone/release workflow must provision and run the pinned Spike/runtime;
missing Spike, missing runtime inputs, or a skipped required Spike suite fails
the workflow. A convenience developer workflow may omit the suite only when it
is explicitly non-milestone and reports a non-green, non-acceptance status.

**Step 3: Add ABI differential checks**

Regenerate temporary POCL outputs and compare ABI-relevant facts rather than requiring byte-identical LLVM IR.

**Step 4: Run the complete local verification matrix**

Run:

```bash
pytest -q python/test/unit/ventus
ninja check-triton-ventus
```

Then run the IREE Ventus unit and end-to-end test targets in the IREE repository.

Expected: all required suites, including pinned Spike, PASS. CycleSim may remain
optional but must report its capability-gated skip reason.

**Step 5: Commit**

```bash
git add .github/workflows python/test/unit/ventus third_party/ventus/test/README.md
git commit -m "ci: verify Triton Ventus backend"
```

## Completion Criteria

The V1 Triton project is complete when all of the following are true. IREE
model integration tasks above remain a broader post-V1 plan and are not V1
release gates:

1. A pinned Ventus toolchain and ABI reference set exists.
2. Producer-specific adapters lower through a documented textual LLVM IR/tool
   boundary. The Triton-owned `VentusLLVM16Compatibility` stage emits
   `kernel.ventus.ll`; internal checking, absolute Ventus LLVM 16 `opt`
   parse/verify, and absolute Ventus LLVM 16 `llc` object codegen all pass before
   ELF/Spike execution. A shared compiled C++ MLIR library is required only if
   LLVM revisions are aligned.
3. Ventus LLVM has a documented and tested input contract for OpenCL-, IREE-,
   and Triton-shaped LLVM IR, plus versioned resource metadata.
4. M1 basic kernels, ABI/manifest/resource validation, canonical packing, and
   launch-time mismatch rejection run on Spike for the fixed profile. The M1
   contract includes canonical resource units, independent capabilities, ELF
   identity validation, AS4/AS5 diagnostics, bounded divergence, simulator
   physical addresses, and deterministic completion/timeout/cache-flush status.
   Fill, copy, vector add, multiply, fused elementwise, masked AS1 memory,
   broadcast, reshape/view, select, and ReLU all execute on the pinned Spike.
   Missing Spike/runtime is a hard milestone/release failure.
5. M2 static AS3/full-work-group barriers, fixed-extent `sum`/`max`, and the
   blocked FP32 `32x32x32` reference/fallback are correct.
6. M3 freezes the contract and differentially validates mandatory RTL and Spike
   `VentusMmaProfileA` results; CycleSim is capability-gated and cannot replace
   either mandatory result. M3 begins with matching hashes for Chisel source,
   effective parameters, generated Verilog, and testbench in one verified
   generation record, and records every dimension mapping, lane/writeback,
   operand binding, syntax, exact-FP/`fflags`, mask/`vl`, register, latency/backpressure,
   and resource field required by Task 20A.
7. M4 lowers only exact FP32 M4K8N4 `tl.dot` with physical `B^T[4,8]` and full
   active warp to `vftta.vv`.
8. Unsupported general/low-precision MMA, arbitrary `tl.dot`, shuffle/ballot,
   atomics, async copy, cluster, warp specialization, FP16/BF16/FP8, and
   physical-driver requests fail early with deterministic diagnostics.
9. Intermediate artifacts, resource/ABI decisions, fallback reasons, and
   simulator capability/skip records are reproducible.
   Every Triton compiler path obeys the no-bypass invariant and records
   `kernel.ventus.ll`, Gates 1-3, V1 ELF link, and Gate 4 evidence including ELF
   hash, launcher input/manifest hash, Spike identity, execution status/result,
   and test-result hash.
10. Later IREE integration consumes the same versioned ABI, artifact, and
   target contract without being required for V1 completion.
