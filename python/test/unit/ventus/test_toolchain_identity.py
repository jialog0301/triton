import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import triton
import triton._C.libtriton as libtriton


REPO_ROOT = Path(__file__).resolve().parents[4]
IDENTITY_PATH = REPO_ROOT / "third_party/ventus/toolchain/version.json"
LLVM_INFO_PATH = REPO_ROOT / "cmake/llvm-info.json"

REQUIRED_KEYS = {
    "identity_status",
    "release_valid",
    "release_blockers",
    "triton_commit",
    "triton_worktree_content_hash",
    "triton_consumer_llvm_hash",
    "triton_consumer_llvm_build",
    "triton_home",
    "triton_llvm_cache_path",
    "python_executable",
    "python_version",
    "python_triton_provenance",
    "libtriton_content_hash",
    "ventus_env_commit",
    "llvm_commit",
    "pocl_commit",
    "driver_commit",
    "spike_commit",
    "cyclesim_commit",
    "gpgpu_commit",
    "systemc_commit",
    "opencl_cts_commit",
    "testcases_commit",
    "dirty_components",
    "dirty_component_content_hashes",
    "ventus_install_prefix",
    "tool_binary_hashes",
    "runtime_build_manifest",
    "runtime_install_manifest",
    "runtime_binary_build_ids",
    "runtime_binary_runpaths",
    "target_triple",
    "mcpu",
    "pointer_width",
    "warp_size",
    "capability_record_version",
    "simulator_address_convention",
    "resource_unit_contract",
    "llvm_resource_transport",
    "completion_contract_version",
    "ventus_linker_script",
    "ventus_crt0_input",
    "ventus_libclc_input",
    "ventus_workitem_input",
    "ventus_kernel_entry_or_init",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_output(*args):
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_output_at(path, *args):
    return subprocess.run(
        ["git", *args], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _dirty_content_hash(path, excluded=()):
    excluded = set(excluded)
    pathspec = [".", *[f":(exclude){item}" for item in sorted(excluded)]]
    tracked = hashlib.sha256(
        subprocess.run(
            ["git", "diff", "HEAD", "--binary", "--", *pathspec],
            cwd=path,
            check=True,
            capture_output=True,
        ).stdout
    ).hexdigest()
    names = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=path,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    names = [name for name in names if name and name.decode() not in excluded]
    if not names:
        return tracked
    untracked_records = b"".join(
        hashlib.sha256((path / name.decode()).read_bytes()).hexdigest().encode()
        + b"  "
        + name
        + b"\n"
        for name in names
    )
    untracked = hashlib.sha256(untracked_records).hexdigest()
    return hashlib.sha256(f"{tracked}\n{untracked}\n".encode()).hexdigest()


def _elf_field(path, pattern):
    output = subprocess.run(
        ["readelf", "-n" if pattern == "Build ID" else "-d", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if pattern == "Build ID":
        match = __import__("re").search(r"Build ID:\s*([0-9a-f]+)", output)
    else:
        match = __import__("re").search(r"\((?:RUNPATH|RPATH)\).*\[([^]]+)\]", output)
    return match.group(1) if match else None


def test_toolchain_identity_matches_verified_environment():
    identity = json.loads(IDENTITY_PATH.read_text())
    llvm_info = json.loads(LLVM_INFO_PATH.read_text())

    assert REQUIRED_KEYS <= identity.keys()
    assert identity["target_triple"] == "riscv32"
    assert identity["mcpu"] == "ventus-gpgpu"
    assert identity["pointer_width"] == 32
    assert identity["warp_size"] == 32
    assert identity["supported_local_sizes"] == [[32, 1, 1], [64, 1, 1]]
    assert identity["mma_local_size"] == [32, 1, 1]
    assert identity["cta_to_work_group"] == "one_to_one"
    assert identity["grid_dimensions"] == 1
    assert identity["scalar_types"] == ["i1", "i32", "f32"]
    assert identity["identity_status"] == "pre_hardening"
    assert identity["release_valid"] is False
    assert set(identity["release_blockers"]) == {
        "runtime_provenance_uncoordinated",
        "llvm_resource_transport_unconsumed",
        "hard_coded_resource_consumption_present",
    }

    assert identity["triton_commit"] == _git_output("rev-parse", "HEAD")
    assert identity["triton_worktree_content_hash"] == _dirty_content_hash(
        REPO_ROOT, {"third_party/ventus/toolchain/version.json"}
    )
    assert identity["triton_consumer_llvm_hash"] == llvm_info["llvm_hash"]
    assert identity["triton_consumer_llvm_build"] == llvm_info["build_number"]
    assert Path(identity["triton_home"]).resolve() == Path(os.environ["TRITON_HOME"]).resolve()
    expected_cache = f"llvm-{llvm_info['llvm_hash'][:8]}-ubuntu-x64-{llvm_info['build_number']}"
    cache_path = Path(identity["triton_llvm_cache_path"])
    assert cache_path.name == expected_cache
    assert cache_path.resolve() == (Path(os.environ["TRITON_HOME"]) / ".triton/llvm" / expected_cache).resolve()
    assert llvm_info["llvm_hash"] in subprocess.run(
        [str(cache_path / "bin/clang"), "--version"], check=True, capture_output=True, text=True
    ).stdout

    assert Path(identity["python_executable"]).resolve() == Path(sys.executable).resolve()
    assert identity["python_version"] == sys.version
    assert Path(triton.__file__).resolve().is_relative_to(REPO_ROOT)
    assert Path(libtriton.__file__).resolve().is_relative_to(REPO_ROOT)
    assert identity["libtriton_content_hash"] == _sha256(libtriton.__file__)
    assert identity["python_triton_provenance"]["status"] == "editable"
    assert Path(identity["python_triton_provenance"]["source"]).resolve() == REPO_ROOT

    dirty_components = set(identity["dirty_components"])
    assert dirty_components == set(identity["dirty_component_content_hashes"])
    assert dirty_components == set(identity["dirty_component_review"])
    assert all(len(value) == 64 for value in identity["dirty_component_content_hashes"].values())

    ventus_root = Path(identity["ventus_install_prefix"]).parent
    component_commits = {
        ".": "ventus_env_commit",
        "llvm": "llvm_commit",
        "pocl": "pocl_commit",
        "driver": "driver_commit",
        "spike": "spike_commit",
        "cyclesim": "cyclesim_commit",
        "gpgpu": "gpgpu_commit",
        "systemc": "systemc_commit",
        "OpenCL-CTS": "opencl_cts_commit",
        "testcases": "testcases_commit",
    }
    for component, key in component_commits.items():
        assert identity[key] == _git_output_at(ventus_root / component, "rev-parse", "HEAD")
    dirty_paths = {
        "ventus-env": ".",
        "llvm": "llvm",
        "pocl": "pocl",
        "driver": "driver",
        "spike": "spike",
        "cyclesim": "cyclesim",
        "gpgpu": "gpgpu",
        "ocl-icd": "ocl-icd",
        "systemc": "systemc",
        "testcases": "testcases",
    }
    for component, relative_path in dirty_paths.items():
        assert identity["dirty_component_content_hashes"][component] == _dirty_content_hash(
            ventus_root / relative_path
        )

    required_tools = {"clang", "opt", "llc", "ld.lld", "spike"}
    assert required_tools <= identity["tool_binary_hashes"].keys()
    for tool in required_tools:
        record = identity["tool_binary_hashes"][tool]
        path = Path(record["path"])
        assert path.is_absolute() and path.is_file()
        assert record["sha256"] == _sha256(path)

    runtime_paths = {
        "libspike_main.so": "lib/libspike_main.so",
        "libVentusCycleSim.so": "lib/libVentusCycleSim.so",
        "libramulator.so": "lib/libramulator.so",
        "libspike_driver.so": "lib/libspike_driver.so",
        "libcyclesim_driver.so": "lib/libcyclesim_driver.so",
        "librtlsim_driver.so": "lib/librtlsim_driver.so",
        "libgvm_driver.so": "lib/libgvm_driver.so",
        "libauto_select_driver.so": "lib/libauto_select_driver.so",
        "libsystemc-2.3.4.so": "systemc/lib-linux64/libsystemc-2.3.4.so",
    }
    install_prefix = Path(identity["ventus_install_prefix"])
    assert set(identity["runtime_binary_hashes"]) == set(runtime_paths)
    assert set(identity["runtime_binary_build_ids"]) == set(runtime_paths) | {"spike"}
    assert set(identity["runtime_binary_runpaths"]) == set(runtime_paths) | {"spike"}
    for name, relative_path in runtime_paths.items():
        path = install_prefix / relative_path
        assert identity["runtime_binary_hashes"][name] == _sha256(path)
        assert identity["runtime_binary_build_ids"][name] == _elf_field(path, "Build ID")
        assert identity["runtime_binary_runpaths"][name] == _elf_field(path, "RUNPATH")
    spike_path = install_prefix / "bin/spike"
    assert identity["runtime_binary_build_ids"]["spike"] == _elf_field(spike_path, "Build ID")
    assert identity["runtime_binary_runpaths"]["spike"] == _elf_field(spike_path, "RUNPATH")
    backend = identity["ventus_backend_library"]
    backend_path = Path(backend["path"])
    assert backend["sha256"] == _sha256(backend_path)
    assert backend["build_id"] == _elf_field(backend_path, "Build ID")
    assert backend["runpath"] == _elf_field(backend_path, "RUNPATH")
    generation_material = (
        identity["dirty_component_content_hashes"]["llvm"]
        + "\n"
        + backend["sha256"]
        + "\n"
        + backend["build_id"]
        + "\n"
    ).encode()
    assert backend["generation_id"] == hashlib.sha256(generation_material).hexdigest()

    assert identity["runtime_build_manifest"]["coordinated"] is False
    assert identity["runtime_install_manifest"]["coordinated"] is False
    assert identity["runtime_build_manifest"]["generation_id"] is None
    assert identity["runtime_install_manifest"]["generation_id"] is None
    assert identity["runtime_build_manifest"]["status"] == "observed_not_proven"
    assert identity["runtime_install_manifest"]["status"] == "observed_not_proven"

    for key in (
        "ventus_linker_script",
        "ventus_crt0_input",
        "ventus_libclc_input",
        "ventus_workitem_input",
    ):
        record = identity[key]
        path = Path(record["path"])
        assert path.is_absolute() and path.is_file()
        assert record["sha256"] == _sha256(path)

    assert identity["capability_record_version"] == 1
    assert identity["simulator_address_convention"] == "physical_device_values"
    assert identity["completion_contract_version"] == 1
    assert identity["resource_unit_contract"] == {
        "lds": {"unit": "bytes", "aggregation": "per_cta"},
        "pds": {"unit": "bytes", "aggregation": "per_work_item"},
        "sgpr": {"unit": "32_bit_register_slots", "aggregation": "per_wavefront"},
        "vgpr": {"unit": "wavefront_wide_register_slots", "aggregation": "per_wavefront"},
    }
    assert identity["llvm_resource_transport"] == {
        "encoding": "target_endian_vres_v1",
        "magic": 0x53455256,
        "version": 1,
        "record_size": 24,
        "field_width": 32,
        "field_order": ["vgpr", "sgpr", "lds", "pds"],
        "consumer": None,
        "validation_status": "provider_validated_runtime_unconsumed",
        "hard_coded_resource_consumption_present": True,
        "acceptance_status": "rejected",
    }
