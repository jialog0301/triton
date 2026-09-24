"""P3.4 single-CTA histogram execution on Ventus Spike."""

import importlib
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import compile as triton_compile

from test_pipeline_vector_add import ventus_backend


TARGET = GPUTarget("ventus", "ventus-gpgpu", 32)
SIGNATURE = {"x_ptr": "*i32", "z_ptr": "*i32"}


@triton.jit
def histogram_kernel(x_ptr, z_ptr, BLOCK: tl.constexpr, BINS: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    mask = (offs & 1) == 0
    hist = tl.histogram(x, BINS, mask=mask)
    tl.store(z_ptr + tl.arange(0, BINS), hist)


def _stage_file(cache_dir: Path, suffix: str) -> Path:
    files = list(cache_dir.rglob(f"*{suffix}"))
    assert len(files) == 1, f"expected exactly one {suffix} artifact, got {files}"
    return files[0]


def _compile(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    src = ASTSource(fn=histogram_kernel, signature=dict(SIGNATURE),
                    constexprs={"BLOCK": 32, "BINS": 4})
    return triton_compile(src, target=TARGET, options={"num_warps": 1})


def test_histogram_single_cta_on_spike(ventus_backend, monkeypatch, tmp_path):
    _compile(monkeypatch, tmp_path)
    cache = tmp_path / "cache"
    llir = _stage_file(cache, ".llir").read_text()
    assert "atomicrmw" not in llir
    elf_path = _stage_file(cache, ".elf")
    launcher = importlib.import_module("triton.backends.ventus.launcher")
    result = launcher.run_histogram(
        launcher.LaunchSpec(elf=elf_path, n_elements=32, local_size=32,
                            kernel_name="histogram_kernel", elements_per_program=32,
                            result_dtype="i32"),
        values=[0, 1, 1, 2, 3, 3, 3, 4] * 4,
        num_bins=4,
        masked_even_lanes=True,
    )
    assert result["num_mismatches"] == 0
