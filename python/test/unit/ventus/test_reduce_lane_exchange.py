"""P3 warp-local data exchange: reduction and gather through LDS.

This target has no shuffle instruction (Ventus-GPGPU-doc: no lane-to-lane data
movement), so `shuffleXor`/`shuffleUp`/`shuffleIdx` are served from a
per-warp scratch region. The backend reserves that region after the core
allocation (`createSharedMemoryGlobal`) and the hooks exchange
store -> barrier -> load -> barrier. These tests cover the two consumers:

* `tt.reduce`: the intra-warp reduction tree runs on `shuffleXor`. A 32-lane
  `tl.sum` is purely intra-warp, which the core allocator gives no scratch for,
  so this is the case that pins the reservation (README 4.7).
* `tt.gather`: the warp-local path runs on `shuffleIdx` (`isWarpLocal`), the
  other user of the region.

Both are executed on Spike against a host reference. A cyclesim number is
recorded in README 4.7 from the same shape, but a second timing launch in this
suite is impossible: cyclesim can only be opened once per process and
`test_pipeline_vector_add.py` already owns that one launch.
"""

import importlib
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import compile as triton_compile

TARGET = GPUTarget("ventus", "ventus-gpgpu", 32)
REDUCE_SIGNATURE = {"x_ptr": "*fp32", "z_ptr": "*fp32", "n": "i32"}
GATHER_SIGNATURE = {"x_ptr": "*fp32", "idx_ptr": "*i32", "z_ptr": "*fp32"}


@triton.jit
def reduce_kernel(x_ptr, z_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    s = tl.sum(x, axis=0)
    tl.store(z_ptr + pid, s)


@triton.jit
def warp_gather_kernel(x_ptr, idx_ptr, z_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    idx = tl.load(idx_ptr + offs)
    z = tl.gather(x, idx, axis=0)
    tl.store(z_ptr + offs, z)


def _compile(monkeypatch, tmp_path, fn, signature, block, num_warps=1):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    src = ASTSource(fn=fn, signature=dict(signature), constexprs={"BLOCK": block})
    return triton_compile(src, target=TARGET, options={"num_warps": num_warps})


def _stage_file(cache_dir: Path, suffix: str) -> Path:
    files = list(cache_dir.rglob(f"*{suffix}"))
    assert len(files) == 1, f"expected exactly one {suffix} artifact, got {files}"
    return files[0]


def test_reduce_reserves_warp_scratch(monkeypatch, tmp_path):
    """An intra-warp reduction needs its own LDS region.

    `ttg.shared` is 0 from the core allocator for this shape (no cross-warp
    traffic), so a non-zero `ttg.shared` and a `global_smem` sized to match can
    only come from the backend's reservation.
    """
    kernel = _compile(monkeypatch, tmp_path, reduce_kernel, REDUCE_SIGNATURE, 32)
    assert kernel.metadata.shared == 32 * 8
    llir = _stage_file(tmp_path / "cache", ".llir").read_text()
    assert f"@global_smem = internal addrspace(3) global [{32 * 8} x i8]" in llir


def test_reduce_on_spike(monkeypatch, tmp_path):
    """A 32-lane `tl.sum` reduces across lanes through LDS, not shuffle.

    BLOCK=32 equals the warp, so the whole reduction is intra-warp: the generic
    tree calls `shuffleXor`, which this target serves from the scratch region.
    n=100 with a 32-element tile makes the last program's lanes partially
    masked, so the masked-load semantics are checked too.
    """
    _compile(monkeypatch, tmp_path, reduce_kernel, REDUCE_SIGNATURE, 32)
    elf_path = _stage_file(tmp_path / "cache", ".elf")
    launcher = importlib.import_module("triton.backends.ventus.launcher")
    result = launcher.run_reduce(
        launcher.LaunchSpec(elf=elf_path, n_elements=100, local_size=32, kernel_name="reduce_kernel",
                            elements_per_program=32))
    assert result["grid"] == 4
    assert result["num_mismatches"] == 0


def test_reduce_two_warps_on_spike(monkeypatch, tmp_path):
    """A reduction spread across lanes *and* warps uses both paths.

    BLOCK=64 with two warps reduces the lane bases through `shuffleXor` (LDS)
    and the warp bases through the core's cross-warp shared staging. The two
    scratch regions are distinct: the core allocation for the cross-warp part,
    the reserved region at `warp_scratch_offset` for the lane part.
    """
    _compile(monkeypatch, tmp_path, reduce_kernel, REDUCE_SIGNATURE, 64, num_warps=2)
    elf_path = _stage_file(tmp_path / "cache", ".elf")
    launcher = importlib.import_module("triton.backends.ventus.launcher")
    result = launcher.run_reduce(
        launcher.LaunchSpec(elf=elf_path, n_elements=128, local_size=64, num_warps=2, profile="v1-64",
                            kernel_name="reduce_kernel", elements_per_program=64))
    assert result["grid"] == 2
    assert result["num_mismatches"] == 0


def test_warp_gather_on_spike(monkeypatch, tmp_path):
    """A warp-local gather indexes lanes through LDS (`shuffleIdx`).

    The indices reverse the block, so the gather is local to the warp and takes
    the `emitWarpLocalGather` path. Every lane asks for a different lane's
    value, which only works if the scratch write/read is ordered by the
    barrier pair.
    """
    _compile(monkeypatch, tmp_path, warp_gather_kernel, GATHER_SIGNATURE, 32)
    elf_path = _stage_file(tmp_path / "cache", ".elf")
    launcher = importlib.import_module("triton.backends.ventus.launcher")
    result = launcher.run_warp_gather(
        launcher.LaunchSpec(elf=elf_path, n_elements=32, local_size=32, kernel_name="warp_gather_kernel",
                            elements_per_program=32))
    assert result["num_mismatches"] == 0
