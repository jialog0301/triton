import re
import hashlib
import json
import subprocess
from pathlib import Path

ABI_DIR = Path(__file__).resolve().parent
EXPECTED_DATALAYOUT = "e-m:e-p:32:32-i64:64-n32-S128-A5-G1"
EXPECTED_TRIPLE = "riscv32"
REPO_ROOT = ABI_DIR.parents[3]
IDENTITY = json.loads((REPO_ROOT / "third_party/ventus/toolchain/version.json").read_text())


def extract_facts(path, kernel):
    text = path.read_text()

    datalayout = re.search(r'^target datalayout = "([^"]+)"$', text, re.MULTILINE)
    triple = re.search(r'^target triple = "([^"]+)"$', text, re.MULTILINE)
    kernel_definition = re.search(
        rf"^(define\s+(?:[^@\n]*\s)?(ventus_kernel)\s+[^@\n]*@{re.escape(kernel)}\([^\n]+)$",
        text,
        re.MULTILINE,
    )
    assert datalayout, f"{path}: missing target datalayout"
    assert triple, f"{path}: missing target triple"
    assert kernel_definition, f"{path}: missing ventus_kernel definition"
    assert datalayout.group(1) == EXPECTED_DATALAYOUT
    assert triple.group(1) == EXPECTED_TRIPLE

    signature = kernel_definition.group(1)
    body_start = kernel_definition.end()
    body_end = text.find("\n}", body_start)
    assert body_end != -1, f"{path}: unterminated kernel body"
    kernel_body = text[body_start:body_end]
    signature_address_spaces = sorted({int(value) for value in re.findall(r"addrspace\((\d+)\)", signature)})
    body_address_spaces = sorted({int(value) for value in re.findall(r"addrspace\((\d+)\)", kernel_body)})
    barrier_intrinsics = sorted(set(re.findall(r"@((?:llvm\.)?riscv\.ventus\.barrier(?:\.with\.scope)?)", kernel_body)))
    builtin_declarations = sorted(
        set(re.findall(
            r"^declare\s+[^@\n]*@((?:__builtin_riscv_|_Z\d+get_)[^(]+)\(",
            text,
            re.MULTILINE,
        )))
    attribute_id = re.search(r"\)\s+[^\n]*#(\d+)", signature)
    assert attribute_id, f"{path}: kernel has no attribute group"
    attribute = re.search(rf"^attributes #{attribute_id.group(1)} = \{{([^\n]+)\}}$", text, re.MULTILINE)
    assert attribute, f"{path}: missing kernel attribute definition"
    kernel_attributes = attribute.group(1)
    assert '"target-cpu"="ventus-gpgpu"' in kernel_attributes
    assert '"target-features"=' in kernel_attributes
    argument_metadata = re.search(r"!kernel_arg_addr_space !(\d+)", signature)
    assert argument_metadata, f"{path}: missing kernel argument metadata"
    argument_spaces = re.search(rf"^!{argument_metadata.group(1)} = !\{{([^\n]+)\}}$", text, re.MULTILINE)
    assert argument_spaces, f"{path}: missing kernel argument address spaces"

    if kernel == "barrier_local":
        assert 3 in body_address_spaces, f"{path}: missing static AS3 local access"
        assert barrier_intrinsics == ["llvm.riscv.ventus.barrier"
                                      ], (f"{path}: unexpected barrier contract {barrier_intrinsics}")
        assert "@llvm.riscv.ventus.barrier(i32 1)" in kernel_body
        assert "declare void @llvm.riscv.ventus.barrier(i32 immarg)" in text

    return {
        "target_datalayout": datalayout.group(1),
        "target_triple": triple.group(1),
        "calling_convention": kernel_definition.group(2),
        "signature_address_spaces": signature_address_spaces,
        "body_address_spaces": body_address_spaces,
        "argument_address_spaces": [int(value) for value in re.findall(r"i32 (\d+)", argument_spaces.group(1))],
        "barrier_intrinsics": barrier_intrinsics,
        "builtin_declarations": builtin_declarations,
        "kernel_attributes": kernel_attributes,
    }


def check_golden(kernel):
    return extract_facts(ABI_DIR / "golden" / f"{kernel}.ll", kernel)


def generate_golden(clang, source, output):
    pinned = IDENTITY["tool_binary_hashes"]["clang"]
    assert clang.resolve() == Path(pinned["path"]).resolve(), f"un-pinned Ventus Clang: {clang}"
    assert hashlib.sha256(clang.read_bytes()).hexdigest() == pinned["sha256"]
    subprocess.run(
        [
            str(clang),
            "-target",
            "riscv32",
            "-mcpu=ventus-gpgpu",
            "-cl-std=CL2.0",
            "-S",
            "-emit-llvm",
            str(source),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
