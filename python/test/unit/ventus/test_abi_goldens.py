import importlib.util
import os
import json
import pytest
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
ABI_DIR = REPO_ROOT / "third_party/ventus/test/abi"
CHECKER_PATH = ABI_DIR / "check_golden.py"
KERNELS = ("vector_add", "masked_copy", "barrier_local")


def _load_checker():
    spec = importlib.util.spec_from_file_location("ventus_abi_golden_checker", CHECKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_opencl_source_has_a_valid_llvm_golden():
    assert {path.stem for path in ABI_DIR.glob("*.cl")} == set(KERNELS)
    assert {path.stem for path in (ABI_DIR / "golden").glob("*.ll")} == set(KERNELS)

    checker = _load_checker()
    facts = {name: checker.check_golden(name) for name in KERNELS}
    for kernel_facts in facts.values():
        assert kernel_facts["target_triple"] == "riscv32"
        assert kernel_facts["target_datalayout"] == "e-m:e-p:32:32-i64:64-n32-S128-A5-G1"
        assert kernel_facts["calling_convention"] == "ventus_kernel"
        assert 1 in kernel_facts["signature_address_spaces"]

    assert 3 in facts["barrier_local"]["body_address_spaces"]
    assert facts["vector_add"]["argument_address_spaces"] == [1, 1, 1, 0]
    assert facts["masked_copy"]["argument_address_spaces"] == [1, 1, 0]
    assert facts["barrier_local"]["argument_address_spaces"] == [1, 1]
    assert facts["barrier_local"]["barrier_intrinsics"] == [
        "llvm.riscv.ventus.barrier"
    ]
    assert facts["vector_add"]["builtin_declarations"] == ["_Z13get_global_idj"]
    assert facts["masked_copy"]["builtin_declarations"] == ["_Z13get_global_idj"]
    assert facts["barrier_local"]["builtin_declarations"] == [
        "_Z12get_local_idj",
        "_Z13get_global_idj",
        "_Z14get_local_sizej",
    ]


def test_regenerated_goldens_match_normalized_abi_facts(tmp_path):
    checker = _load_checker()
    # Regenerating the goldens needs the pinned Ventus clang, which is an external
    # tool the suite does not own. Without it there is nothing to compare against,
    # and a missing configuration is not a backend defect -- say so and move on
    # (AGENTS.md documents the export).
    if "VENTUS_CLANG" not in os.environ:
        pytest.skip("VENTUS_CLANG is not set: export the pinned Ventus tool paths "
                    "(see AGENTS.md) to regenerate the ABI goldens")
    clang = Path(os.environ["VENTUS_CLANG"])
    for kernel in KERNELS:
        generated = tmp_path / f"{kernel}.ll"
        checker.generate_golden(clang, ABI_DIR / f"{kernel}.cl", generated)
        assert checker.extract_facts(generated, kernel) == checker.check_golden(kernel)


def test_pinned_link_inputs_supply_observed_abi_roles():
    identity = json.loads(
        (REPO_ROOT / "third_party/ventus/toolchain/version.json").read_text()
    )
    linker_script = Path(identity["ventus_linker_script"]["path"])
    assert "ENTRY(_start)" in linker_script.read_text()

    nm = identity["tool_binary_hashes"]["llvm-nm"]["path"]
    crt0 = identity["ventus_crt0_input"]["path"]
    workitem = identity["ventus_workitem_input"]["path"]
    libclc = identity["ventus_libclc_input"]["path"]
    symbols = subprocess.run(
        [nm, "-A", crt0, workitem, libclc],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert f"{crt0}: 00000000 T _start" in symbols
    for builtin in (
        "__builtin_riscv_global_id_x",
        "__builtin_riscv_workgroup_id_x",
        "__builtin_riscv_workitem_id_x",
        "__builtin_riscv_local_size_x",
    ):
        assert f" T {builtin}" in symbols
    assert f"{libclc}:" in symbols
    assert " U __builtin_riscv_workitem_id_x" in symbols
