# Ventus launch profile design

## Goal

Make the reference launcher select execution parameters from a named, minimal
Ventus launch profile so the ELF generation model and Spike driver's launch
metadata cannot silently diverge. The first supported profile is the existing
legacy `numw=2,numt=8` ELF.

## Scope

This change is limited to `third_party/ventus/reference_launcher`. It does not
modify the installed Spike driver or connect profiles to Triton's TTGIR,
layout, or lowering passes.

## Profile contract

The launcher owns this first-stage contract:

```cpp
struct VentusLaunchProfile {
  uint32_t lanes_per_warp;
  uint32_t warps_per_workgroup;
  uint32_t local_size_x;
  uint32_t local_size_y;
  uint32_t local_size_z;
  uint32_t vector_length;
  uint32_t lds_size;
  uint32_t pds_size;
  uint32_t sgpr_usage;
  uint32_t vgpr_usage;
};
```

Built-in profiles are `legacy-8x2`, `v1-32`, and `v1-64`. The legacy profile
has eight lanes per warp, two warps per workgroup, and local size `[16,1,1]`.
The V1 profiles retain the current 32- and 64-lane launch shapes.

`--profile` selects a built-in profile and defaults to `legacy-8x2`. Explicit
resource options remain accepted as overrides; local size must match the
profile's declared shape, while omitted resource options use profile defaults.

## Data flow

Argument parsing resolves the profile first, applies explicit overrides, and
validates the resulting launch configuration. The resolved values are then
used consistently for driver metadata (`wf_size`, `wg_size`), kernel metadata,
allocation sizes, logs, and the child-process execution path.

The driver-facing `wf_size` is `lanes_per_warp`, and `wg_size` is
`warps_per_workgroup`. The launcher does not attempt to override the driver's
fixed program load base, start PC, PDS size, or work-group run count.

## Errors and compatibility

Unknown profiles, malformed profile/resource values, and mismatched explicit
local sizes fail during argument validation with exit code 2. Existing command
lines without `--profile` resolve to the legacy profile. The launcher remains
an initial vector-add smoke launcher and continues to require two input
buffers.

## Tests

Extend the existing Python test file rather than adding a new test file. Test
help output, legacy profile defaults, V1 profile selection, unknown profiles,
local-size mismatch, and the existing invalid-local failure. Build the native
launcher and run the focused pytest suite; no Triton `make` is required because
only launcher code and tests change.
