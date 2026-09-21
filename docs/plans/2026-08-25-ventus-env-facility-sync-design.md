# Ventus Environment Facility Synchronization Design

## Goal

Synchronize the Triton-for-Ventus planning documents with the facilities
actually present in `/home/weijiale/Code/cuda2rvv/ventus-env`.

The committed revisions form the reproducible baseline. Uncommitted Driver and
CycleSim worktree changes are recorded separately and are not treated as stable
ABI or runtime contracts.

## Source Of Truth

Add one facility snapshot that records:

- The `ventus-env` superproject and submodule identities.
- Dirty-worktree state.
- The implemented LLVM target and kernel ABI.
- POCL argument packing and launch metadata.
- Driver backends and current runtime limitations.
- RTL and CycleSim execution profiles.
- Regression, CTS, and operator-reference facilities.
- A capability matrix separating stable, partial, experimental, and absent
  facilities.

Existing architecture and implementation plans should link to this snapshot
instead of duplicating volatile implementation details.

## Synchronization Rules

1. Describe committed code as the baseline.
2. Describe local Driver and CycleSim changes only in an explicit experimental
   section.
3. Distinguish implemented behavior from proposed Triton/IREE architecture.
4. Distinguish LLVM address spaces from physical RTL address decoding.
5. Distinguish ELF resource metadata, packed kernel arguments, the device
   kernel metadata block, and host driver launch metadata.
6. Treat FP32 and ordinary FMA as the first supported Triton profile.
7. Treat tensor instructions as present but experimental until their compiler
   and simulator contracts agree.
8. Treat physical hardware execution, atomics, shuffle/ballot, dynamic local
   arguments, and broad low-precision support as unavailable or unproven.

## Documents To Update

- Mark the 2026-08-18 architecture as superseded where it conflicts with the
  later shared-target design.
- Update the shared MLIR design with the LLVM version boundary, current runtime
  backends, current barrier and builtin contracts, and artifact distinctions.
- Update the implementation plan with current Triton plugin APIs, complete
  toolchain identity fields, and prerequisite ABI/runtime hardening.
- Update the learning roadmap with the pinned target profile, experimental
  tensor facility, realistic simulator metrics, and current validation assets.

## Verification

- Search all Ventus plans for stale claims about future tensor hardware,
  physical hardware availability, stable ID intrinsics, versioned resource
  records, and universal fixed warp size.
- Review the final diff for internal consistency and source references.
- Verify that only documentation files are modified.
