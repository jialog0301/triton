"""Exercise Ventus's packed, register-local byte-permute layout conversion."""

import importlib
from pathlib import Path

import pytest
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import compile as triton_compile
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from test_pipeline_vector_add import ventus_backend  # noqa: F401


@g.jit
def packed_layout_kernel(x_ptr, z_ptr, SRC: gl.constexpr, DST: gl.constexpr):
    src_m = gl.arange(0, 16, layout=gl.SliceLayout(1, SRC))[:, None]
    src_n = gl.arange(0, 32, layout=gl.SliceLayout(0, SRC))[None, :]
    x = gl.load(x_ptr + src_m * 32 + src_n).to(gl.uint8)
    y = gl.convert_layout(x, DST).to(gl.int32)
    dst_m = gl.arange(0, 16, layout=gl.SliceLayout(1, DST))[:, None]
    dst_n = gl.arange(0, 32, layout=gl.SliceLayout(0, DST))[None, :]
    gl.store(z_ptr + dst_m * 32 + dst_n, y)


def _stage(cache_dir: Path, suffix: str):
    matches = list(cache_dir.rglob(f"*{suffix}"))
    assert len(matches) == 1, matches
    return matches[0]


@pytest.mark.parametrize("source,dest", [
    (gl.BlockedLayout([2, 2], [4, 8], [1, 1], [1, 0]),
     gl.BlockedLayout([1, 4], [8, 4], [1, 1], [1, 0])),
])
def test_packed_layout_conversion_on_spike(ventus_backend, monkeypatch, tmp_path, source, dest):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    src = g.GluonASTSource(packed_layout_kernel, {"x_ptr": "*i32", "z_ptr": "*i32"},
                           {"SRC": source, "DST": dest})
    triton_compile(src, target=GPUTarget("ventus", "ventus-gpgpu", 32),
                   options={"num_warps": 1})
    assert "convert_layout" in _stage(tmp_path / "cache", ".ttgir").read_text()
    launcher = importlib.import_module("triton.backends.ventus.launcher")
    values = [(i * 37 + 129) % 256 for i in range(512)]
    result = launcher.run_layout_permute(
        launcher.LaunchSpec(elf=_stage(tmp_path / "cache", ".elf"), n_elements=512,
                            elements_per_program=512, kernel_name="packed_layout_kernel",
                            result_dtype="i32"), values)
    assert result["num_mismatches"] == 0
