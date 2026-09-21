import importlib
import runpy
import subprocess
import sys
from importlib.metadata import EntryPoint, EntryPoints
from pathlib import Path

import pytest
import setuptools

import triton.backends as triton_backends
from triton.backends import backends
from triton.backends.compiler import GPUTarget, Language
from triton.compiler.compiler import make_backend

REPO_ROOT = Path(__file__).resolve().parents[4]
VENTUS_ROOT = REPO_ROOT / "third_party/ventus"
TOOL_ROOT = Path("/home/weijiale/Code/cuda2rvv/ventus-env/install/bin")


@pytest.fixture
def setup_contract(monkeypatch):
    setup_kwargs = {}
    subprocess_run = subprocess.run
    original_sys_path = sys.path[:]

    def run(command, *args, **kwargs):
        if command[:3] == ["git", "submodule", "update"]:
            return subprocess.CompletedProcess(command, 0)
        return subprocess_run(command, *args, **kwargs)

    monkeypatch.setenv("TRITON_PLUGIN_DIRS", str(VENTUS_ROOT))
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: setup_kwargs.update(kwargs))
    monkeypatch.setattr(subprocess, "run", run)
    try:
        namespace = runpy.run_path(str(REPO_ROOT / "setup.py"))
    finally:
        sys.path[:] = original_sys_path
    return namespace, setup_kwargs


@pytest.fixture
def ventus_backend(monkeypatch, setup_contract, tmp_path):
    namespace, setup_kwargs = setup_contract
    external = namespace["BackendInstaller"].copy_externals()
    assert len(external) == 1
    descriptor = external[0]
    assert Path(descriptor.backend_dir) == VENTUS_ROOT / "backend"
    install_dir = Path(descriptor.install_dir)
    assert install_dir == REPO_ROOT / "python/triton/backends/ventus"
    ventus_entry_points = [
        value for value in setup_kwargs["entry_points"]["triton.backends"] if value.startswith("ventus =")
    ]
    assert ventus_entry_points == ["ventus = triton.backends.ventus"]

    isolated_backends = tmp_path / "triton/backends"
    isolated_backends.mkdir(parents=True)
    (isolated_backends / "ventus").symlink_to(descriptor.backend_dir, target_is_directory=True)
    monkeypatch.setattr(triton_backends, "__path__", [str(isolated_backends), *triton_backends.__path__])
    importlib.invalidate_caches()
    try:
        entry_points = EntryPoints((EntryPoint(name="ventus", value="triton.backends.ventus",
                                               group="triton.backends"), ))
        monkeypatch.setattr(triton_backends, "entry_points", lambda: entry_points)
        discovered = triton_backends._discover_backends()
        registration = discovered["ventus"]
        monkeypatch.setitem(backends, "ventus", registration)
        yield registration
    finally:
        for name in ("triton.backends.ventus.driver", "triton.backends.ventus.compiler", "triton.backends.ventus"):
            sys.modules.pop(name, None)
        importlib.invalidate_caches()


def test_external_backend_discovery(ventus_backend):
    assert (VENTUS_ROOT / "backend/name.conf").read_text().strip() == "ventus"
    target = GPUTarget("ventus", "ventus-gpgpu", 32)
    compatible = [entry.compiler for entry in backends.values() if entry.compiler.supports_target(target)]
    assert compatible == [ventus_backend.compiler]


def test_backend_registration_and_scaffolding(ventus_backend):
    target = GPUTarget("ventus", "ventus-gpgpu", 32)
    backend = make_backend(target)
    assert backend.binary_ext == "elf"
    options = backend.parse_options({"num_warps": 2, "num_stages": 3, "target_features": "+m,+f"})
    assert options.num_warps == 2
    assert options.num_stages == 3
    assert options.target_features == "+m,+f"
    assert options.target_triple == "riscv32"
    assert options.pointer_width == 32
    assert options.abi_revision == 1
    assert dict(options.tool_paths) == {name: str(TOOL_ROOT / name) for name in ("clang", "opt", "llc", "ld.lld")}
    assert all(Path(path).is_file() for path in dict(options.tool_paths).values())
    assert hash(options)
    assert not any(isinstance(value, GPUTarget) for value in options.__dict__.values())

    assert backend.pack_metadata(type("Metadata", (), {"num_warps": 2, "num_stages": 3, "shared": 0})()) == (2, 3, 0)
    codegen_fns = backend.get_codegen_implementation(options)
    assert codegen_fns["min_dot_size"](None, None) == (1, 1, 1)
    stages = {}
    backend.add_stages(stages, options, Language.TRITON)
    assert list(stages) == ["ttir", "ttgir", "llir", "elf"]
    assert all(callable(stage) for stage in stages.values())

    assert ventus_backend.driver.is_active() is False
    driver = ventus_backend.driver()
    with pytest.raises(RuntimeError, match="Ventus runtime is not implemented"):
        driver.get_current_target()


def test_tool_paths_are_configurable_and_immutable(ventus_backend, tmp_path):
    tool_paths = {}
    for name in ("clang", "opt", "llc", "ld.lld"):
        path = tmp_path / name
        path.touch()
        tool_paths[name] = str(path)

    backend = ventus_backend.compiler(GPUTarget("ventus", "ventus-gpgpu", 32))
    options = backend.parse_options({"tool_paths": tool_paths})
    assert dict(options.tool_paths) == tool_paths
    tool_paths["clang"] = "/changed/clang"
    assert dict(options.tool_paths)["clang"] == str(tmp_path / "clang")
    assert hash(options)


@pytest.mark.parametrize(
    "tool_paths",
    [
        {"clang": "relative/clang", "opt": "/tools/opt", "llc": "/tools/llc", "ld.lld": "/tools/ld.lld"},
        {"clang": "/tools/not-clang", "opt": "/tools/opt", "llc": "/tools/llc", "ld.lld": "/tools/ld.lld"},
        {"clang": "/tools/clang"},
    ],
)
def test_backend_rejects_invalid_tool_paths(ventus_backend, tool_paths):
    backend = ventus_backend.compiler(GPUTarget("ventus", "ventus-gpgpu", 32))
    with pytest.raises(ValueError, match="tool_paths"):
        backend.parse_options({"tool_paths": tool_paths})


def test_native_scaffold_is_observable():
    from triton._C import libtriton

    assert hasattr(libtriton, "ventus")
    assert libtriton.ventus.load_dialects(libtriton.ir.context()) is False
    assert libtriton.ventus.passes.register_passes() is True
    status = libtriton.ventus.registration_status()
    assert status["dialect"] is False
    assert status["passes"] is True


@pytest.mark.parametrize(
    "target",
    [
        GPUTarget("ventus", "wrong-arch", 32),
        GPUTarget("ventus", "ventus-gpgpu", 64),
        GPUTarget("cuda", "ventus-gpgpu", 32),
    ],
)
def test_backend_rejects_incompatible_targets(ventus_backend, target):
    assert not ventus_backend.compiler.supports_target(target)
    with pytest.raises(ValueError, match="unsupported Ventus target"):
        ventus_backend.compiler(target)


@pytest.mark.parametrize(
    "options",
    [
        {"unknown": True},
        {"num_warps": 0},
        {"num_stages": 0},
        {"target_features": ["+m"]},
        {"tool_paths": []},
    ],
)
def test_backend_rejects_unsupported_options(ventus_backend, options):
    backend = ventus_backend.compiler(GPUTarget("ventus", "ventus-gpgpu", 32))
    with pytest.raises((TypeError, ValueError)):
        backend.parse_options(options)
