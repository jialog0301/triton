"""End-to-end checks for the Ventus V1 pipeline: ttir -> ttgir -> llir -> elf.

The kernel mirrors the ABI golden third_party/ventus/test/abi/vector_add.cl:
a masked fp32 elementwise add over the global range. The compile-level tests
stop at the linked ELF (Gate 3); `test_*_on_spike` runs that ELF on the
instruction-level Spike simulator through the reference launcher, which is
V1's mandatory M1 execution gate (Gate 4).
"""

import importlib
import subprocess
import sys
from importlib.metadata import EntryPoint, EntryPoints
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
TOOL_ROOT = Path("/home/weijiale/Code/cuda2rvv/ventus-env/install/bin")

TARGET = GPUTarget("ventus", "ventus-gpgpu", 32)
TARGET_DATA_LAYOUT = "e-m:e-p:32:32-i64:64-n32-S128-A5-G1"


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, x + y, mask=mask)


@triton.jit
def vector_add_2d_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK_M: tl.constexpr,
                         BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = tl.arange(0, BLOCK_N)[None, :]
    offs = offs_m * BLOCK_N + offs_n
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, x + y, mask=mask)


@pytest.fixture
def ventus_backend(monkeypatch, tmp_path):
    isolated = tmp_path / "triton" / "backends"
    isolated.mkdir(parents=True)
    (isolated / "ventus").symlink_to(VENTUS_ROOT / "backend", target_is_directory=True)
    monkeypatch.setattr(triton_backends, "__path__",
                        [str(isolated), *triton_backends.__path__])
    entry_points = EntryPoints(
        (EntryPoint(name="ventus", value="triton.backends.ventus",
                    group="triton.backends"), ))
    monkeypatch.setattr(triton_backends, "entry_points", lambda: entry_points)
    importlib.invalidate_caches()
    try:
        discovered = triton_backends._discover_backends()
        registration = discovered["ventus"]
        monkeypatch.setitem(triton_backends.backends, "ventus", registration)
        yield registration
    finally:
        for name in ("triton.backends.ventus.driver",
                     "triton.backends.ventus.compiler", "triton.backends.ventus"):
            sys.modules.pop(name, None)
        importlib.invalidate_caches()


def _compile_vector_add(monkeypatch, tmp_path):
    return _compile(vector_add_kernel, {"BLOCK": 32}, 1, monkeypatch, tmp_path)


def _compile(fn, constexprs, num_warps, monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    src = ASTSource(
        fn=fn,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "z_ptr": "*fp32",
                   "n": "i32"},
        constexprs=constexprs,
    )
    return triton_compile(src, target=TARGET, options={"num_warps": num_warps})


def _stage_file(cache_dir: Path, suffix: str) -> Path:
    files = list(cache_dir.rglob(f"*{suffix}"))
    assert len(files) == 1, f"expected exactly one {suffix} artifact, got {files}"
    return files[0]


def test_vector_add_pipeline_stages(ventus_backend, monkeypatch, tmp_path):
    _compile_vector_add(monkeypatch, tmp_path)

    cache = tmp_path / "cache"
    ttgir = _stage_file(cache, ".ttgir").read_text()
    llir = _stage_file(cache, ".llir").read_text()
    elf = _stage_file(cache, ".elf").read_bytes()

    # TTGIR: blocked layout, no residual layout conversions.
    assert "#ttg.blocked" in ttgir
    assert "convert_layout" not in ttgir

    # LLIR: the Ventus ABI envelope, verified against the ABI goldens.
    assert 'target triple = "riscv32"' in llir
    assert f'target datalayout = "{TARGET_DATA_LAYOUT}"' in llir
    # `ventus_kernel` is attached on the textual IR: the consumer LLVM numbers
    # that convention differently and would print a foreign keyword.
    assert "define ventus_kernel void @vector_add_kernel(" in llir
    assert "ptr addrspace(1)" in llir
    assert "@_Z12get_local_idj" in llir  # work-item thread id
    assert "@_Z12get_group_idj" in llir  # program id
    assert "kernel_arg_addr_space" in llir

    # ELF: loadable image with the standard Ventus entry point.
    assert elf[:4] == b"\x7fELF"
    assert elf[4] == 1  # ELFCLASS32


def test_llir_round_trips_through_ventus_opt(ventus_backend, monkeypatch,
                                             tmp_path):
    _compile_vector_add(monkeypatch, tmp_path)
    llir_path = _stage_file(tmp_path / "cache", ".llir")

    # The Ventus toolchain must accept the emitted text as-is; `opt` also
    # round-trips the `ventus_kernel` keyword.
    out = tmp_path / "roundtrip.ll"
    subprocess.run([str(TOOL_ROOT / "opt"), "-passes=verify", "-S",
                    str(llir_path), "-o", str(out)],
                   capture_output=True, text=True, check=True)
    assert "define ventus_kernel void @vector_add_kernel(" in out.read_text()


def test_vector_add_on_spike(ventus_backend, monkeypatch, tmp_path):
    """Gate 4: the M1 mandatory Spike execution baseline for vector_add.

    Compiles the vector_add kernel through the full pipeline, stages the ELF
    at a short path (the driver's char[128] path buffer overflows on a Triton
    cache path), launches it on the Ventus Spike device via libspike_driver,
    and checks the device output against the CPU reference. The canonical tail
    N=100 is the boundary case: 4 work-groups of 32 lanes, of which only 4
    lanes of the last group are active.
    """
    _compile_vector_add(monkeypatch, tmp_path)
    elf_path = _stage_file(tmp_path / "cache", ".elf")
    launcher = importlib.import_module("triton.backends.ventus.launcher")
    result = launcher.run_vector_add(
        launcher.LaunchSpec(elf=elf_path, n_elements=100,
                            local_size=32, keep_log=True))
    assert len(result["spike_log"]) > 0, "spike produced no execution log"
    # The instruction log must actually reach the kernel body and execute the
    # elementwise add and the masked store.
    entry = int(result["entry"], 16)
    body = [line for line in result["spike_log"].splitlines()
            if line.startswith(f"core   0: 0x{entry:08x}")]
    assert body, "kernel entry never executed on Spike"
    assert any("vfadd" in line for line in result["spike_log"].splitlines())
    assert any("vsw12.v" in line for line in result["spike_log"].splitlines())
    assert result["grid"] == 4
    assert result["num_mismatches"] == 0
    # The resolved profile must reach the driver, not just the summary: these
    # are the values handed to `vt_start`, and spike asserts
    # `thread_number == vlen/elen` internally, so a wrong `wf_size` would
    # abort the run rather than silently pass.
    assert result["lanes_per_warp"] == 32
    assert result["warps_per_workgroup"] == 1
    assert result["driver_wf_size"] == 32
    assert result["driver_wg_size"] == 1
    assert result["profile"] == "v1-32"


def test_vector_add_manifest_gate4(ventus_backend, monkeypatch, tmp_path):
    """The M1 versioned artifact manifest validates after Spike execution.

    Builds the compile-side manifest over the real cache artifacts, runs the
    kernel on Spike through the reference launcher, records the Gate 4
    evidence, and requires `ventus.validate()` to accept the result. This is
    the machine-checkable form of "Gate 4 happened": the C++ validator rejects
    any manifest whose `launcher_input`/`test_result` references or result
    records are missing, so a compile-only artifact can never masquerade as
    one that executed.
    """
    _compile_vector_add(monkeypatch, tmp_path)
    cache = tmp_path / "cache"
    launcher = importlib.import_module("triton.backends.ventus.launcher")
    manifest = importlib.import_module("triton.backends.ventus.manifest")

    facts = manifest.CompileFacts(
        cache_dir=cache, kernel_name="vector_add_kernel",
        elf_name="vector_add_kernel", local_size=32, grid_size=4,
        op_name="vector_add")
    out = tmp_path / "vector_add_kernel.manifest.json"
    md = manifest.build_compile_manifest(facts, out)
    assert not manifest.validate(md), "compile-side manifest must not validate"

    result = launcher.run_vector_add(
        launcher.LaunchSpec(elf=facts.stage("elf"), n_elements=100,
                            local_size=32, keep_log=True))
    result["exit_status"] = 0
    manifest.record_execution(
        md, out,
        {"elf_sha256": result["elf_sha256"], "grid": result["grid"],
         "local_size": result["local_size"],
         "capability_identity":
             manifest.capability_identity(manifest.load_identity()),
         "spike": manifest.tool_identity(manifest.load_identity(), "spike")},
        result, spike_log=Path(result["staged_elf"] + ".log"))
    assert {a.kind for a in md.artifacts} >= {
        "ttir", "ttgir", "llvm_ir", "assembly", "elf", "launcher_input",
        "test_result"}
    assert manifest.validate(md), "gate-4 manifest must validate"

    reloaded = manifest.load_manifest(out)
    assert reloaded is not None
    assert {a.kind for a in reloaded.artifacts} >= {
        "ttir", "ttgir", "llvm_ir", "assembly", "elf", "launcher_input",
        "test_result"}


def test_launch_profile_consistency(ventus_backend, monkeypatch, tmp_path):
    """A profile must agree with the kernel it launches.

    `v1-32` declares one 32-lane warp, so a `num_warps=2` kernel (a 64-lane
    Triton block) and an unknown profile name are both refused before any
    driver call. Without this, the extra lanes would be launched but never
    accounted for by the kernel, aliasing results instead of failing.
    """
    _compile_vector_add(monkeypatch, tmp_path)
    elf_path = _stage_file(tmp_path / "cache", ".elf")
    launcher = importlib.import_module("triton.backends.ventus.launcher")

    with pytest.raises(ValueError, match="unknown launch profile"):
        launcher.LaunchSpec(elf=elf_path, n_elements=32, profile="v9-999")

    # A known but non-V1 shape is rejected by name, not silently launched.
    with pytest.raises(ValueError, match="not launchable by this"):
        launcher.LaunchSpec(elf=elf_path, n_elements=32, profile="legacy-8x2")

    # v1-32 declares one warp, so a two-warp kernel is refused before launch.
    with pytest.raises(ValueError, match="num_warps"):
        launcher.run_vector_add(
            launcher.LaunchSpec(elf=elf_path, n_elements=64, num_warps=2,
                                local_size=32, profile="v1-32"))

    # v1-32 declares local_size_x=32, so a 64-lane launch is refused.
    with pytest.raises(ValueError, match="local_size"):
        launcher.run_vector_add(
            launcher.LaunchSpec(elf=elf_path, n_elements=64, local_size=64,
                                profile="v1-32"))

    # The two-warp shape is legal under the matching profile.
    assert launcher.BUILTIN_PROFILES["v1-64"].warps_per_workgroup == 2
    assert launcher.BUILTIN_PROFILES["v1-64"].local_size_x == 64


def test_launch_profiles_are_self_consistent():
    """The shared vocabulary and the launchable subset stay coherent."""
    launcher = importlib.import_module("triton.backends.ventus.launcher")
    for name, profile in launcher.LAUNCH_PROFILE_VOCABULARY.items():
        assert profile.lanes_per_warp * profile.warps_per_workgroup == \
            profile.local_size_x, name
        assert profile.local_size_y == 1 and profile.local_size_z == 1, name
        assert profile.vector_length == profile.lanes_per_warp * 32, name
    assert launcher.LAUNCH_PROFILE_VOCABULARY["legacy-8x2"].local_size_x == 16
    assert launcher.LAUNCH_PROFILE_VOCABULARY["v1-32"].local_size_x == 32
    assert launcher.LAUNCH_PROFILE_VOCABULARY["v1-64"].local_size_x == 64
    # This launcher only realizes the V1 32-lane warp; the legacy shape stays
    # with the C++ smoke tool rather than being launched with the wrong lanes.
    assert set(launcher.BUILTIN_PROFILES) == {"v1-32", "v1-64"}
    assert all(p.lanes_per_warp == launcher.V1_LANES_PER_WARP
               for p in launcher.BUILTIN_PROFILES.values())


def test_vector_add_spike_exact_multiple(ventus_backend, monkeypatch,
                                         tmp_path):
    """Non-boundary control: N=32 (single work-group, full active warp)."""
    _compile_vector_add(monkeypatch, tmp_path)
    elf_path = _stage_file(tmp_path / "cache", ".elf")
    launcher = importlib.import_module("triton.backends.ventus.launcher")
    result = launcher.run_vector_add(
        launcher.LaunchSpec(elf=elf_path, n_elements=32, local_size=32))
    assert result["grid"] == 1
    assert result["num_mismatches"] == 0


def test_vector_add_on_cyclesim(ventus_backend, monkeypatch, tmp_path):
    """The cycle-level driver runs the same ELF and reports model time.

    spike is functional-only, so the installed `libcyclesim_driver.so` is the
    timing source of truth; the measurement loop has to close there. Two things
    the functional driver hides are exercised by this launch: cyclesim sizes
    private memory over the whole grid at `pdsBaseAddr` (spike gives each
    work-group a fixed segment and ignores the address), and it reads a trailing
    `kernel_name` field that spike's metadata struct does not have. Both were
    wrong before, and both are invisible with grid=1.

    One cyclesim launch per process: SystemC refuses a second simulation, so the
    launcher raises rather than letting the model abort. That makes this the only
    cyclesim launch in the suite.
    """
    _compile_vector_add(monkeypatch, tmp_path)
    elf_path = _stage_file(tmp_path / "cache", ".elf")
    launcher = importlib.import_module("triton.backends.ventus.launcher")
    result = launcher.run_vector_add(
        launcher.LaunchSpec(elf=elf_path, n_elements=100, local_size=32,
                            driver="cyclesim"))
    assert result["driver"] == "cyclesim"
    assert result["grid"] == 4
    assert result["num_mismatches"] == 0
    # Model time, not host time: it comes from the simulator's own clock, so it
    # cannot be zero for a kernel that ran.
    assert result["simulated_time_ns"] > 0


def test_vector_add_on_rtl(ventus_backend, monkeypatch, tmp_path):
    """The Verilator RTL model runs the same ELF and reports model time.

    This is the hardware description itself, so it is the ground truth the
    SystemC cycle model is calibrated against. The installed model is the
    `nocache` variant (L1 D-cache and L2 removed), which is why its cycle count
    is not directly a cache-accurate number.

    The grid is limited to the RTL's SM count: `gpgpu/ventus/src/top/parameters.scala`
    declares `num_sm = 2`, and a grid beyond one wave aborts inside the model
    (`Assertion failed: UNDEFINED INSTRUCTION @ SM 0 warp 1 PC 0x90004000`) --
    the third work-group starts at the metadata buffer instead of the kernel
    entry. n=64 is therefore the largest shape that runs today: grid=2, every
    lane active.

    Like cyclesim, the rtlsim driver can only be opened once per process, so this
    is the only RTL launch in the suite.
    """
    _compile_vector_add(monkeypatch, tmp_path)
    elf_path = _stage_file(tmp_path / "cache", ".elf")
    launcher = importlib.import_module("triton.backends.ventus.launcher")
    result = launcher.run_vector_add(
        launcher.LaunchSpec(elf=elf_path, n_elements=64, local_size=32,
                            driver="rtlsim"))
    assert result["driver"] == "rtlsim"
    assert result["grid"] == 2
    assert result["num_mismatches"] == 0
    assert result["simulated_time_ns"] > 0


def test_vector_add_tiled_on_spike(ventus_backend, monkeypatch, tmp_path):
    """A tile larger than the warp: the grid follows the tile, not the lanes.

    With `BLOCK=256` and 32 lanes each program covers 8 elements, so `n=1024`
    needs 4 programs rather than the 32 a one-element-per-lane launch would use.
    This is the shape that measured fastest on the cycle model (see README 6),
    and it is the case where deriving the grid from `local_size` would launch 32
    programs and leave 28 of them fully masked off -- correct, but eight times
    the per-program overhead.
    """
    _compile(vector_add_kernel, {"BLOCK": 256}, 1, monkeypatch, tmp_path)
    elf_path = _stage_file(tmp_path / "cache", ".elf")
    launcher = importlib.import_module("triton.backends.ventus.launcher")
    result = launcher.run_vector_add(
        launcher.LaunchSpec(elf=elf_path, n_elements=1024, local_size=32,
                            elements_per_program=256))
    assert result["grid"] == 4
    assert result["num_mismatches"] == 0


def test_2d_tile_vector_add_on_spike(ventus_backend, monkeypatch, tmp_path):
    """A rank-2 tile compiles and executes on Spike.

    Both `tl.arange` results are rank-1 tensors carrying a slice encoding of a
    rank-2 blocked parent, so their per-thread offsets come from the parent
    layout's dim-0 and dim-1 bases (order=[1, 0]) rather than from any 1-D
    formula. A wrong mapping does not fail compilation, it writes wrong
    values, so the assertion is the device result.
    """
    block_m, block_n, local_size = 4, 8, 32
    # The launcher derives the grid from `n_elements` and `local_size`, so a
    # tile of local_size elements per program is what keeps every offset
    # covered exactly once.
    assert block_m * block_n == local_size

    _compile(vector_add_2d_kernel, {"BLOCK_M": block_m, "BLOCK_N": block_n}, 1,
             monkeypatch, tmp_path)
    elf_path = _stage_file(tmp_path / "cache", ".elf")
    launcher = importlib.import_module("triton.backends.ventus.launcher")
    result = launcher.run_vector_add(
        launcher.LaunchSpec(elf=elf_path, n_elements=100,
                            local_size=local_size,
                            kernel_name="vector_add_2d_kernel"))
    assert result["grid"] == 4
    assert result["num_mismatches"] == 0


def test_vector_add_two_warps_on_spike(ventus_backend, monkeypatch, tmp_path):
    """num_warps=2 (the `v1-64` profile): the layout's warp basis.

    With 64 elements per program and two warps, the element offset of a lane
    depends on its warp id, not only on its lane id. A wrong warp basis makes
    the second warp write the first warp's offsets; with every lane active
    (N=128) that aliasing shows up as mismatched values. This is also the
    first end-to-end use of the `v1-64` profile, which was declared but never
    launched.
    """
    _compile(vector_add_kernel, {"BLOCK": 64}, 2, monkeypatch, tmp_path)
    elf_path = _stage_file(tmp_path / "cache", ".elf")
    launcher = importlib.import_module("triton.backends.ventus.launcher")
    result = launcher.run_vector_add(
        launcher.LaunchSpec(elf=elf_path, n_elements=128, local_size=64,
                            num_warps=2, profile="v1-64"))
    assert result["grid"] == 2
    assert result["driver_wf_size"] == 32
    assert result["driver_wg_size"] == 2
    assert result["num_mismatches"] == 0
