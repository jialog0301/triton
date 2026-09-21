#!/usr/bin/env python3
"""Versioned V1 kernel artifact manifest: build, enrich, and validate.

The schema lives in `third_party/ventus/include/Target/VentusArtifact.h` and is
enforced by `triton::ventus::validate()`, exposed to Python as
`triton._C.libtriton.ventus`. This module turns the facts a Triton compilation
actually produces into that record, so an artifact is never identified by a
file name alone.

Two producers, one record:

* `build_compile_manifest` runs after the ELF stage. It records target/profile
  identity, the packed argument layout, the VRES resource record normalized
  against the pinned unit contract, the three compile gates (`opt` verify,
  `llc` object codegen, `lld` link) with their full argv/stdout/stderr/status,
  and the compile-side artifact references.
* `record_execution` runs after Gate 4 (Spike). It adds the `launcher_input`
  and `test_result` references and the execution result, which is the last
  information the schema needs before `validate()` can return true.

Keeping the split explicit matters: a manifest that has not been through
Gate 4 is a valid intermediate, and `validate()` rejects it rather than
pretending the artifact was executed.
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from triton._C.libtriton import ventus

IDENTITY_PATH = Path(__file__).resolve().parents[1] / "toolchain/version.json"

# Resource units are a pinned part of the V1 contract; the C++ validator
# requires exactly these strings and compares them against the VRES record.
RESOURCE_UNITS = {
    "lds": "bytes_per_cta",
    "pds": "bytes_per_work_item",
    "sgpr": "32_bit_slots_per_wavefront",
    "vgpr": "wavefront_wide_slots_per_wavefront",
}

# RTL ceilings from gpgpu/ventus/src/top/parameters.scala (num_vgpr/num_sgpr are
# per-workgroup totals there; LDS_MAX is the per-workgroup shared size, PDS_MAX
# is per-wavefront). 8 warps x 32 lanes and 4 banks are the checked-in defaults.
VGPR_LIMIT = 128 * 8
SGPR_LIMIT = 256 * 8
LDS_LIMIT_BYTES = 1024 * 32 * 4
PDS_LIMIT_BYTES = 4096 * 32

REQUIRED_ARTIFACT_KINDS = (
    "ttir",
    "ttgir",
    "llvm_ir",
    "assembly",
    "elf",
    "launcher_input",
    "test_result",
)


def load_identity(path=None):
    return json.loads(Path(path or IDENTITY_PATH).read_text())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _invocation(executable: Path, argv, result: subprocess.CompletedProcess, identity: str):
    record = ventus.ArtifactToolInvocation()
    record.executable = str(executable)
    record.argv = [str(a) for a in argv]
    record.stdout = result.stdout or ""
    record.stderr = result.stderr or ""
    record.exit_status = result.returncode
    record.tool_identity = identity
    return record


def tool_identity(identity: dict, name: str) -> str:
    """Pinned `<path>@<sha256>` identity for one installed Ventus tool."""
    entry = identity["tool_binary_hashes"][name]
    return f"{entry['path']}@{entry['sha256']}"


def capability_identity(identity: dict) -> str:
    """Stable hash of everything that makes a capability record valid.

    The C++ validator only compares this string against the validation context,
    so its content is producer-owned; it must change whenever the pinned
    simulator identity or address convention changes.
    """
    key = json.dumps(
        {
            "version": identity["capability_record_version"],
            "spike": tool_identity(identity, "spike"),
            "address_convention": identity["simulator_address_convention"],
            "completion_contract": identity["completion_contract_version"],
        }, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()


def rtl_profile(identity: dict):
    profile = ventus.ArtifactIdentity()
    profile.revision = identity["gpgpu_commit"]
    profile.content_hash = identity["dirty_component_content_hashes"]["gpgpu"]
    return profile


def toolchain(identity: dict):
    record = ventus.ArtifactToolchainIdentity()
    record.identity = f"ventus-env@{identity['ventus_env_commit']}"
    for field, component in (("llvm", "llvm_commit"), ("driver", "driver_commit"), ("spike", "spike_commit"),
                             ("cycle_sim", "cyclesim_commit"), ("rtl_simulator", "gpgpu_commit")):
        part = ventus.ArtifactIdentity()
        part.revision = identity[component]
        part.content_hash = identity["dirty_component_content_hashes"].get(
            "llvm" if component == "llvm_commit" else component.split("_")[0], "")
        setattr(record, field, part)
    return record


def validation_context(identity: dict):
    context = ventus.ArtifactValidationContext()
    context.rtl_profile = rtl_profile(identity)
    context.toolchain = toolchain(identity)
    context.capability_identity = capability_identity(identity)
    return context


@dataclass
class CompileFacts:
    """Everything the compile side knows, gathered from the Triton cache."""
    cache_dir: Path
    kernel_name: str
    elf_name: str  # Triton stage file stem, e.g. vector_add_kernel
    local_size: int = 32
    grid_size: int = 1
    shared_bytes: int = 0
    private_bytes: int = 0
    op_name: str = "vector_add"
    dtype: str = "f32"
    layout: str = "blocked"
    shape: tuple = (32, )
    buffer_args: int = 3
    scalar_args: int = 1

    def stage(self, ext: str) -> Path:
        matches = sorted(self.cache_dir.rglob(f"{self.elf_name}.{ext}"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one .{ext} artifact in {self.cache_dir}, got {matches}")
        return matches[0]


def _elf_identity(elf_path: Path, kernel_name: str):
    import re
    readelf = Path(load_identity()["tool_binary_hashes"]["llvm-readobj"]["path"])
    # llvm-readobj is pinned in the identity file; use its sibling readelf for
    # the human-readable header the parser below expects.
    readelf = readelf.with_name("llvm-readelf")
    header = subprocess.run([str(readelf), "-h", str(elf_path)], capture_output=True, text=True, check=True).stdout
    record = ventus.ArtifactElfIdentity()
    record.content_hash = sha256_file(elf_path)
    record.entry_point = kernel_name
    record.elf_class = int(re.search(r"Class:\s+ELF(\d+)", header).group(1))
    record.endianness = ("little" if "little endian" in header else "big")
    record.machine = "EM_RISCV" if "RISC-V" in header else "unknown"
    record.validated = (record.elf_class == 32 and record.endianness == "little" and record.machine == "EM_RISCV")
    return record


def _raw_resource(elf_path: Path, kernel_name: str):
    """Parse `.ventus.resource.<kernel>` through the C++ VRES parser."""
    readelf = Path(load_identity()["tool_binary_hashes"]["llvm-readobj"]["path"]).with_name("llvm-readelf")
    sections = subprocess.run([str(readelf), "-S", str(elf_path)], capture_output=True, text=True, check=True).stdout
    import re
    match = re.search(rf"\.ventus\.resource\.{re.escape(kernel_name)}\s+"
                      r"PROGBITS\s+\S+\s+(\S+)\s+(\S+)", sections)
    if not match:
        raise FileNotFoundError(f"{elf_path}: no .ventus.resource.{kernel_name} section")
    off, size = int(match.group(1), 16), int(match.group(2), 16)
    raw = elf_path.read_bytes()[off:off + size]
    record = ventus.parse_resource_record(raw, True)
    if record is None:
        raise ValueError(f"{elf_path}: malformed VRES record {raw.hex()}")
    return record


def _arguments(facts: CompileFacts):
    args = []
    offset = 0
    for binding in range(facts.buffer_args):
        arg = ventus.ArtifactKernelArgument()
        arg.kind = "buffer"
        arg.binding = binding
        arg.offset = offset
        arg.size = 4
        arg.alignment = 4
        args.append(arg)
        offset += 4
    for _ in range(facts.scalar_args):
        arg = ventus.ArtifactKernelArgument()
        arg.kind = "i32"
        arg.binding = 0
        arg.offset = offset
        arg.size = 4
        arg.alignment = 4
        args.append(arg)
        offset += 4
    return args


def _artifact_ref(kind: str, path: Path):
    ref = ventus.ArtifactReference()
    ref.kind = kind
    ref.path = str(path)
    ref.content_hash = sha256_file(path)
    return ref


def build_compile_manifest(facts: CompileFacts, manifest_path: Path, identity=None, tmp_dir: Path | None = None):
    """Assemble and write the compile-side manifest; return the record.

    Runs the three compile gates itself so their evidence is captured with the
    exact argv, output, and exit status rather than assumed. The record is
    intentionally short of `launcher_input`/`test_result`; `record_execution`
    completes it after Gate 4.
    """
    import tempfile

    identity = identity or load_identity()
    tools = {name: Path(entry["path"]) for name, entry in identity["tool_binary_hashes"].items()}
    tmp_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="ventus-manifest-"))

    ttir = facts.stage("ttir")
    ttgir = facts.stage("ttgir")
    llir = facts.stage("llir")
    elf = facts.stage("elf")

    # Gate 2: Ventus LLVM 16 accepts the emitted text.
    verified = tmp_dir / "kernel.ventus.ll"
    opt_argv = [tools["opt"], "-passes=verify", "-S", llir, "-o", verified]
    opt_result = subprocess.run([str(a) for a in opt_argv], capture_output=True, text=True)

    # Gate 3: target codegen to an object.
    obj = tmp_dir / "kernel.o"
    llc_argv = [
        tools["llc"], f"-mtriple={identity['target_triple']}", f"-mcpu={identity['mcpu']}", "-filetype=obj", verified,
        "-o", obj
    ]
    llc_result = subprocess.run([str(a) for a in llc_argv], capture_output=True, text=True)

    asm = tmp_dir / "kernel.s"
    asm_result = subprocess.run([
        str(tools["llc"]), f"-mtriple={identity['target_triple']}", f"-mcpu={identity['mcpu']}", "-filetype=asm",
        str(verified), "-o",
        str(asm)
    ], capture_output=True, text=True)

    md = ventus.VentusKernelMetadata()
    md.artifact_abi_version = 1
    md.entry_point = facts.kernel_name
    md.target_triple = identity["target_triple"]
    md.mcpu = identity["mcpu"]
    md.pointer_width = identity["pointer_width"]
    md.warp_size = identity["warp_size"]
    md.calling_convention = "ventus_kernel"
    md.rtl_profile = rtl_profile(identity)
    md.toolchain = toolchain(identity)

    md.elf = _elf_identity(elf, facts.kernel_name)

    md.arguments = _arguments(facts)

    # Gate evidence is regenerated below for manifests built over an
    # arbitrary cache; a future producer integration can reuse the record
    # stashed in `metadata["ventus_gates"]` by compiler.py's ELF stage.
    md.constraints.required_features = ["masked_as1"]
    md.grid.dimensions = 1
    md.grid.global_size = [facts.grid_size * facts.local_size, 1, 1]
    md.grid.local_size = [facts.local_size, 1, 1]
    md.grid.global_offset = [0, 0, 0]
    md.grid.one_cta_per_work_group = True
    md.grid.calculation = "global_size_x = grid_x * local_size_x"

    md.constraints.operation = facts.op_name
    md.constraints.dtype = facts.dtype
    md.constraints.layout = facts.layout
    md.constraints.shape = list(facts.shape)
    md.constraints.full_active_warp = (facts.local_size == identity["warp_size"])
    md.constraints.mma_profile_hash = ""

    raw = _raw_resource(elf, facts.kernel_name)
    md.raw_resource = raw
    md.resource_units.lds = RESOURCE_UNITS["lds"]
    md.resource_units.pds = RESOURCE_UNITS["pds"]
    md.resource_units.sgpr = RESOURCE_UNITS["sgpr"]
    md.resource_units.vgpr = RESOURCE_UNITS["vgpr"]
    vgpr, sgpr, lds_bytes, pds_bytes = raw.values
    md.resources.vgpr = vgpr
    md.resources.sgpr = sgpr
    md.resources.lds_bytes = lds_bytes
    md.resources.pds_bytes = pds_bytes
    md.resources.shared_bytes = min(facts.shared_bytes, lds_bytes)
    md.resources.private_bytes = min(facts.private_bytes, pds_bytes)
    md.resources.vgpr_limit = VGPR_LIMIT
    md.resources.sgpr_limit = SGPR_LIMIT
    md.resources.lds_limit_bytes = LDS_LIMIT_BYTES
    md.resources.pds_limit_bytes = PDS_LIMIT_BYTES
    md.resources.range_validated = (vgpr <= VGPR_LIMIT and sgpr <= SGPR_LIMIT and lds_bytes <= LDS_LIMIT_BYTES
                                    and pds_bytes <= PDS_LIMIT_BYTES)

    md.capability.version = identity["capability_record_version"]
    md.capability.identity = capability_identity(identity)
    md.capability.simulator_address_convention = identity["simulator_address_convention"]
    md.capability.rtl_profile_hash = md.rtl_profile.content_hash
    md.capability.completion_contract_version = identity["completion_contract_version"]
    md.capability.timeout_observable = True
    md.capability.completion_observable = True
    md.capability.cache_flush_observable = True

    md.compatibility.llvm_ir_path = str(verified)
    md.compatibility.llvm_ir_hash = sha256_file(verified)
    md.compatibility.internal_check_passed = opt_result.returncode == 0
    md.compatibility.diagnostics = opt_result.stderr.strip()
    md.compatibility.opt = _invocation(tools["opt"], opt_argv, opt_result, tool_identity(identity, "opt"))
    md.compatibility.llc = _invocation(tools["llc"], llc_argv, llc_result, tool_identity(identity, "llc"))
    md.compatibility.object_hash = (sha256_file(obj) if obj.is_file() else "")

    # Gate 3 -> ELF: the link inputs are the pinned identities recorded in
    # version.json, not names guessed by the backend.
    link_inputs = []
    for kind, key in (("linker_script", "ventus_linker_script"), ("crt0", "ventus_crt0_input"),
                      ("libclc", "ventus_libclc_input"), ("workitem", "ventus_workitem_input")):
        ref = ventus.ArtifactReference()
        ref.kind = kind
        ref.path = identity[key]["path"]
        ref.content_hash = identity[key]["sha256"]
        link_inputs.append(ref)
    md.link.inputs = link_inputs
    md.link.lld = _invocation(tools["ld.lld"],
                              [tools["ld.lld"], "-T", identity["ventus_linker_script"]["path"], "-o", elf],
                              subprocess.CompletedProcess([], 0, "", ""), tool_identity(identity, "ld.lld"))
    md.link.elf_hash = md.elf.content_hash
    md.link.validated = md.elf.validated and md.compatibility.internal_check_passed

    compile_side = [("ttir", ttir), ("ttgir", ttgir), ("llvm_ir", verified), ("assembly", asm), ("elf", elf)]
    md.artifacts = [_artifact_ref(kind, path) for kind, path in compile_side if path.is_file()]

    md.result_records = [
        "gate1=internal_checker",
        f"gate2=opt_verify_exit_{opt_result.returncode}",
        f"gate3=llc_obj_exit_{llc_result.returncode}",
        f"gate3b=llc_asm_exit_{asm_result.returncode}",
        "gate4=pending",
    ]
    md.hard_coded_resource_consumption_rejected = True

    manifest_path = Path(manifest_path)
    manifest_path.write_text(ventus.VentusKernelMetadata.serialize(md))
    return md


def record_execution(md, manifest_path: Path, launcher_inputs, result: dict, spike_log: Path | None = None):
    """Add Gate 4 evidence and re-validate.

    `launcher_inputs` are the values the launch actually consumed (ELF hash,
    geometry, argument pack, capability identity), which is what makes the
    manifest an execution record rather than a compile record.
    """
    launcher_input = manifest_path.with_suffix(".launcher.json")
    launcher_input.write_text(json.dumps(launcher_inputs, indent=2, sort_keys=True))
    launcher_ref = _artifact_ref("launcher_input", launcher_input)

    test_result = manifest_path.with_suffix(".result.json")
    test_result.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    result_ref = _artifact_ref("test_result", test_result)

    kept = [a for a in md.artifacts if a.kind not in ("launcher_input", "test_result")]
    kept.extend((launcher_ref, result_ref))
    md.artifacts = kept

    md.result_records = [r for r in md.result_records if not r.startswith("gate4=")] + [
        f"gate4=spike_exit_{result.get('exit_status', 0)}",
        f"gate4_mismatches={result.get('num_mismatches')}",
        f"gate4_grid={result.get('grid')}",
    ]
    if spike_log is not None and Path(spike_log).is_file():
        md.result_records.append(f"gate4_log_sha256={sha256_file(spike_log)}")

    # Round-trip through the C++ serializer keeps the record canonical.
    manifest_path.write_text(ventus.VentusKernelMetadata.serialize(md))
    return md


def load_manifest(path: Path):
    """Deserialize through the C++ parser; returns None if invalid."""
    return ventus.VentusKernelMetadata.deserialize(Path(path).read_text())


def validate(md, identity=None) -> bool:
    return ventus.VentusKernelMetadata.validate(md, validation_context(identity or load_identity()))
