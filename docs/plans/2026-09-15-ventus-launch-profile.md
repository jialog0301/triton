# Ventus Launch Profile Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the Ventus reference launcher profile-driven, with a default legacy-8x2 profile that can launch the existing 16-lane legacy ELF.

**Architecture:** Keep profiles as built-in C++ data in the reference launcher. Parse and resolve one profile before execution, use its values for local-shape validation, driver metadata, allocations, and manifests, and preserve explicit CLI resource overrides where they are compatible. Do not change the installed Spike driver or Triton lowering.

**Tech Stack:** C++17 launcher, installed Ventus `libspike_driver.so`, Make, pytest.

---

### Task 1: Add failing CLI coverage for profile selection

**Files:**
- Modify: `third_party/ventus/reference_launcher/test_launcher.py`
- Modify: `third_party/ventus/reference_launcher/ventus_spike_smoke.cpp`

**Step 1: Write the failing tests**

Extend the existing launcher subprocess tests to require `--profile` in help,
accept `--profile legacy-8x2` with `--local 16,1,1`, reject an unknown profile,
and reject an explicit local size that disagrees with the selected profile.
Use a nonexistent ELF and an args-log path for the positive parse test so the
test verifies resolved profile values without requiring Spike execution.

**Step 2: Run tests to verify they fail**

Run:

```bash
pytest -s --tb=short third_party/ventus/reference_launcher/test_launcher.py
```

Expected: the new profile-related assertions fail because the CLI has no
profile option and still only accepts local sizes 32 and 64.

### Task 2: Implement the minimal built-in profile model

**Files:**
- Modify: `third_party/ventus/reference_launcher/ventus_spike_smoke.cpp`

**Step 1: Add profile data and resolution**

Add `VentusLaunchProfile` and built-in `legacy-8x2`, `v1-32`, and `v1-64`
records. Add a profile name to `Config`, make `legacy-8x2` the default, parse
`--profile`, and resolve profile defaults before validating the command line.
Track whether each override was explicitly supplied so omitted `--lds-size`,
`--pds-size`, `--sgpr`, and `--vgpr` inherit profile values.

**Step 2: Replace hard-coded launch derivation**

Use profile `lanes_per_warp` and `warps_per_workgroup` for `MetaData.wf_size`
and `MetaData.wg_size`. Use profile vector length and resource values in the
kernel metadata/allocation path. Validate local dimensions against the resolved
profile shape, and keep one-dimensional grid validation.

**Step 3: Run the focused tests**

Run:

```bash
make -C third_party/ventus/reference_launcher
pytest -s --tb=short third_party/ventus/reference_launcher/test_launcher.py
```

Expected: all launcher tests pass. The positive parse test may report the
expected child driver failure while asserting the args-log contents; no Spike
positive run is required in this environment.

### Task 3: Record the resolved profile in manifests and documentation

**Files:**
- Modify: `third_party/ventus/reference_launcher/ventus_spike_smoke.cpp`
- Modify: `third_party/ventus/reference_launcher/README.md`
- Modify: `third_party/ventus/reference_launcher/test_launcher.py`

**Step 1: Extend logs**

Record the profile name and all resolved profile fields in the execution log
and args-log. Keep the existing ELF hash and launch fields intact.

**Step 2: Update usage documentation**

Document the default `legacy-8x2` profile, the three built-in profile names,
their lane/local shapes, and how explicit resource arguments override defaults.
Remove the statement that local size is restricted to only 32 or 64 lanes.

**Step 3: Add log assertions and run tests**

Add deterministic assertions for profile and local values in the args-log test,
then run the focused pytest command again. Expected: all tests pass.

### Task 4: Verify and commit the implementation

**Files:**
- Review: `third_party/ventus/reference_launcher/Makefile`
- Review: `third_party/ventus/reference_launcher/ventus_spike_smoke.cpp`
- Review: `third_party/ventus/reference_launcher/README.md`
- Review: `third_party/ventus/reference_launcher/test_launcher.py`

**Step 1: Run static/build verification**

Run:

```bash
make -C third_party/ventus/reference_launcher
third_party/ventus/reference_launcher/ventus_spike_smoke --help
pytest -s --tb=short third_party/ventus/reference_launcher/test_launcher.py
```

Expected: native compilation succeeds, help exits 0, and the focused suite is
green. Do not run the repository-wide `make`, because this change is isolated
to launcher C++/Python code and does not alter Triton compiler code.

**Step 2: Review the diff**

Run `git diff --check` and inspect that no Spike driver sources or unrelated
user changes are included.

**Step 3: Commit**

```bash
git add third_party/ventus/reference_launcher
git commit -m "feat: drive Ventus launcher from execution profiles"
```
