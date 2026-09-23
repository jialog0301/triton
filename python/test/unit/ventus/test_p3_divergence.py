"""P3 divergence A/B: the software-predicate and hardware-divergence arms.

`tl.where` lowers to a data-plane `arith.select` -> `LLVM::SelectOp` by default
(the "software" arm). The "hardware" arm runs the `select-to-branch` pass,
which rewrites each scalar `llvm.select` into a `cond_br` diamond so the Ventus
LLVM sees control-flow divergence and emits `vbeq`/`setrpc`/`join`. Both compute
the same values; the difference is only the code shape (README 4.6).

The mask is lane parity, not a threshold, so every warp diverges: a condition
like `offs < n` only splits the one boundary warp. Both arms are compiled once
per arm and inspected at the LLIR (shape) and on Spike (values).
"""

import importlib
from pathlib import Path

import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import compile as triton_compile

TARGET = GPUTarget("ventus", "ventus-gpgpu", 32)
WHERE_SIGNATURE = {"x_ptr": "*fp32", "z_ptr": "*fp32"}


@triton.jit
def where_kernel(x_ptr, z_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    z = tl.where(offs % 2 == 0, x * 2.0, 0.0)
    tl.store(z_ptr + offs, z)


@triton.jit
def two_nested_if_kernel(x_ptr, z_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    if n > 0:
        if n > 1:
            tl.store(z_ptr + offs, x)


@triton.jit
def three_nested_if_kernel(x_ptr, z_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    if n > 0:
        if n > 1:
            if n > 2:
                tl.store(z_ptr + offs, x)


def _compile(monkeypatch, tmp_path, predication):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    src = ASTSource(fn=where_kernel, signature=dict(WHERE_SIGNATURE), constexprs={"BLOCK": 32})
    return triton_compile(src, target=TARGET, options={"num_warps": 1, "predication": predication})


def _stage_file(cache_dir: Path, suffix: str) -> Path:
    files = list(cache_dir.rglob(f"*{suffix}"))
    assert len(files) == 1, f"expected exactly one {suffix} artifact, got {files}"
    return files[0]


def test_predication_option_is_accepted(monkeypatch, tmp_path):
    assert _compile(monkeypatch, tmp_path, "software")
    assert _compile(monkeypatch, tmp_path, "hardware")


def test_predication_rejects_unknown_value(monkeypatch, tmp_path):
    with pytest.raises((TypeError, ValueError)):
        _compile(monkeypatch, tmp_path, "sideways")


def _llir(monkeypatch, tmp_path, predication):
    _compile(monkeypatch, tmp_path, predication)
    return _stage_file(tmp_path / "cache", ".llir").read_text()


def test_select_arm_is_data_plane(monkeypatch, tmp_path):
    ll = _llir(monkeypatch, tmp_path, "software")
    assert "select i1" in ll
    assert "br i1" not in ll


def test_branch_arm_is_control_plane(monkeypatch, tmp_path):
    ll = _llir(monkeypatch, tmp_path, "hardware")
    assert "br i1" in ll
    assert "select i1" not in ll


@pytest.mark.parametrize("predication", ["software", "hardware"])
def test_where_arm_on_spike(monkeypatch, tmp_path, predication):
    """Both arms compute the parity-mask result on Spike.

    The launch is `grid=4, 32 lanes`: every warp has a half-taken mask, so the
    hardware arm's `vbeq`/`join` really executes and a wrong merge shows up as
    a mismatch. Spike can be opened repeatedly in one process, so both arms run
    in the same test binary.
    """
    _compile(monkeypatch, tmp_path, predication)
    elf_path = _stage_file(tmp_path / "cache", ".elf")
    launcher = importlib.import_module("triton.backends.ventus.launcher")
    result = launcher.run_where(
        launcher.LaunchSpec(elf=elf_path, n_elements=128, local_size=32, kernel_name="where_kernel",
                            elements_per_program=32))
    assert result["num_mismatches"] == 0


def _compile_if(monkeypatch, tmp_path, fn):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    src = ASTSource(fn=fn, signature={"x_ptr": "*fp32", "z_ptr": "*fp32", "n": "i32"}, constexprs={"BLOCK": 32})
    return triton_compile(src, target=TARGET, options={"num_warps": 1})


def test_two_nested_divergent_regions_are_accepted(monkeypatch, tmp_path):
    """The validated bound is two nested regions, so two must still compile."""
    assert _compile_if(monkeypatch, tmp_path, two_nested_if_kernel)


def test_three_nested_divergent_regions_are_rejected(monkeypatch, tmp_path, capfd):
    """One region past the bound is refused, with the bound named.

    `scf-to-cf` runs after the check, so the diagnostic comes from the
    bounded-divergence pass on the explicit `scf.if` nesting. The Python
    exception is the pass manager's generic message; the specific bound is in
    the MLIR diagnostic on stderr (captured at the fd level, since LLVM writes
    it around Python's streams).
    """
    with pytest.raises(Exception, match="PassManager::run failed"):
        _compile_if(monkeypatch, tmp_path, three_nested_if_kernel)
    err = capfd.readouterr().err
    assert "divergent control-flow nesting depth 3" in err
    assert "ventus-bounded-divergence" in err
