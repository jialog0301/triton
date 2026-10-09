"""End-to-end sentinel for the Triton/Ventus textual LLVM IR boundary.

The Triton-side LLVM and the pinned Ventus LLVM 16 are different revisions, so
the emitted text can contain keywords/literals the Ventus parser predates. Each
such feature is handled in `triton.backends.ventus.textual`; this test compiles a
corpus that exercises the known features so a new one fails here, not in a user's
kernel. Compile-only: no simulator needed.
"""

import importlib
import sys
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
TARGET = GPUTarget("ventus", "ventus-gpgpu", 32)


@pytest.fixture
def ventus_backend(monkeypatch, tmp_path):
    isolated = tmp_path / "triton" / "backends"
    isolated.mkdir(parents=True)
    (isolated / "ventus").symlink_to(VENTUS_ROOT / "backend", target_is_directory=True)
    monkeypatch.setattr(triton_backends, "__path__", [str(isolated), *triton_backends.__path__])
    importlib.invalidate_caches()
    try:
        monkeypatch.setitem(triton_backends.backends, "ventus", triton_backends._discover_backends()["ventus"])
        yield
    finally:
        for name in ("triton.backends.ventus.driver", "triton.backends.ventus.compiler", "triton.backends.ventus.textual",
                     "triton.backends.ventus"):
            sys.modules.pop(name, None)
        importlib.invalidate_caches()


@triton.jit
def _vector_add(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(z_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)


@triton.jit
def _reduce(x_ptr, z_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    tl.store(z_ptr + pid, tl.sum(x, axis=0))


@triton.jit
def _softmax(x_ptr, z_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    tl.store(z_ptr + offs, e / tl.sum(e, axis=0), mask=offs < n)


CORPUS = {
    "vector_add": (_vector_add, {"x_ptr": "*fp32", "y_ptr": "*fp32", "z_ptr": "*fp32", "n": "i32"}, {"BLOCK": 32}, 1),
    "reduce": (_reduce, {"x_ptr": "*fp32", "z_ptr": "*fp32", "n": "i32"}, {"BLOCK": 32}, 1),
    "softmax": (_softmax, {"x_ptr": "*fp32", "z_ptr": "*fp32", "n": "i32"}, {"BLOCK": 32}, 1),
}


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_kernel_reaches_elf(ventus_backend, monkeypatch, tmp_path, name):
    fn, signature, constexprs, num_warps = CORPUS[name]
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / name))
    triton_compile(ASTSource(fn=fn, signature=signature, constexprs=constexprs), target=TARGET, options={"num_warps": num_warps})
    assert list((tmp_path / name).rglob("*.elf")), f"{name}: no ELF produced"
