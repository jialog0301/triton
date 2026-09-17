









import hashlib
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from triton._C.libtriton import ir, passes, llvm
from triton._C.libtriton import ventus
from triton.backends.compiler import BaseBackend, GPUTarget, Language


_TOOL_ROOT = Path("/home/weijiale/Code/cuda2rvv/ventus-env/install/bin")
_TOOL_NAMES = ("clang", "opt", "llc", "ld.lld")
_TOOL_PATHS = tuple((name, str(_TOOL_ROOT / name)) for name in _TOOL_NAMES)
_VENTUS_INSTALL = _TOOL_ROOT.parent

# Machine-verified ABI constants. See toolchain/version.json and the ABI
# goldens under third_party/ventus/test/abi.
_TARGET_TRIPLE = "riscv32"
_TARGET_MCPU = "ventus-gpgpu"
_TARGET_FEATURES = (
    "+32bit,+a,+m,+relax,+zdinx,+zfinx,+zhinx,+zve32f,+zve32x,+zvl32b,"
    "-64bit,-save-restore"
)

# Standard triton.compile option keys with no Ventus meaning in V1. They are
# accepted (so the generic compile path works unchanged) and ignored.
_IGNORED_OPTIONS = frozenset({
    "cluster_dims",
    "debug",
    "enable_fp_fusion",
    "extern_libs",
    "instrumentation_mode",
    "launch_cooperative_grid",
    "launch_pdl",
    "maxnreg",
    "sanitize_overflow",
})


@dataclass(frozen=True)
class VentusOptions:
    num_warps: int = 4
    num_stages: int = 3
    warp_size: int = 32
    target_features: str = _TARGET_FEATURES
    target_triple: str = _TARGET_TRIPLE
    pointer_width: int = 32
    abi_revision: int = 1
    # Fields the generic frontend reads directly; V1 carries fixed defaults.
    debug: bool = False
    sanitize_overflow: bool = False
    arch: str = "ventus-gpgpu"
    supported_fp8_dtypes: tuple[str, ...] = ()
    deprecated_fp8_dtypes: tuple[str, ...] = ()
    allowed_dot_input_precisions: tuple[str, ...] = ("ieee", )
    default_dot_input_precision: str = "ieee"
    max_num_imprecise_acc_default: int = 0
    tool_paths: tuple[str, ...] = _TOOL_PATHS

    def __post_init__(self):
        if isinstance(self.num_warps, bool) or self.num_warps <= 0 or self.num_warps & (self.num_warps - 1):
            raise ValueError("num_warps must be a positive power of two")
        if isinstance(self.num_stages, bool) or not isinstance(self.num_stages, int) or self.num_stages <= 0:
            raise ValueError("num_stages must be a positive integer")
        if self.warp_size != 32:
            raise ValueError("V1 targets the fixed 32-lane Ventus warp size")
        if self.target_triple != _TARGET_TRIPLE:
            raise ValueError(f"target_triple must be {_TARGET_TRIPLE!r}")
        if self.pointer_width != 32:
            raise ValueError("pointer_width must be 32 for riscv32")
        if not isinstance(self.target_features, str):
            raise TypeError("target_features must be a string")
        try:
            tool_paths = dict(self.tool_paths)
        except (TypeError, ValueError) as exc:
            raise ValueError("tool_paths must map tool names to absolute paths") from exc
        if set(tool_paths) != set(_TOOL_NAMES):
            raise ValueError(f"tool_paths must contain exactly {list(_TOOL_NAMES)}")
        for name in _TOOL_NAMES:
            path = Path(tool_paths[name])
            if not path.is_absolute() or path.name != name:
                raise ValueError(f"tool_paths[{name!r}] must be an absolute path ending in {name!r}")
        object.__setattr__(self, "tool_paths", tuple((name, tool_paths[name]) for name in _TOOL_NAMES))

    def hash(self):
        key = "_".join(
            [f"{name}-{val}" for name, val in sorted(self.__dict__.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


# The consumer LLVM's calling-convention numbering differs from the pinned
# Ventus LLVM 16 (its CC 104 is `amdgpu_cs_chain`, which the Ventus parser
# rejects). The `ventus_kernel` keyword is therefore attached to the emitted
# textual IR here; `opt -passes=verify` in the elf stage then validates that
# the Ventus toolchain accepts the result.
_KERNEL_DEFINE = re.compile(r"^define\s+(?:[^@\n]*?)\bvoid\s+@{name}\s*\(", re.M)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _apply_ventus_kernel_convention(text: str, kernel_name: str) -> str:
    pattern = re.compile(_KERNEL_DEFINE.pattern.format(name=re.escape(kernel_name)),
                         re.M)
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise RuntimeError(
            f"Ventus: expected exactly one definition of kernel {kernel_name!r} "
            f"in the emitted LLVM IR, found {len(matches)}")
    return pattern.sub(f"define ventus_kernel void @{kernel_name}(", text)


# Second consumer-side-only IR feature on the same textual boundary. MLIR's
# `llvm.or` carries a `disjoint` flag and core `applyLinearLayout` sets it for
# provably non-overlapping bit ranges, so every LinearLayout-based index
# computation emits `or disjoint`. The pinned Ventus LLVM 16 parser predates
# the keyword (added in LLVM 17) and rejects it. Dropping it is
# semantics-preserving: `disjoint` is a promise about the operands, not an
# instruction, a value, or any part of the computed result.
_OR_DISJOINT = re.compile(r"\bor\s+disjoint\s+")


def _strip_or_disjoint(text: str) -> str:
    return _OR_DISJOINT.sub("or ", text)


class VentusBackend(BaseBackend):

    binary_ext = "elf"

    def __init__(self, target):
        if not self.supports_target(target):
            raise ValueError(f"unsupported Ventus target: {target}")
        self.target = target

    @staticmethod
    def supports_target(target: GPUTarget):
        return target == GPUTarget("ventus", "ventus-gpgpu", 32)

    def parse_options(self, options):
        supported = {"num_warps", "num_stages", "warp_size", "target_features", "tool_paths"}
        unknown = set(options) - supported - _IGNORED_OPTIONS
        if unknown:
            raise ValueError(f"unsupported Ventus options: {sorted(unknown)}")
        return VentusOptions(**{k: v for k, v in options.items() if k in supported})

    def pack_metadata(self, metadata):
        return metadata.num_warps, metadata.num_stages, metadata.shared

    def get_codegen_implementation(self, options):
        # `tl.dot` is outside the V1 scope; report the honest 1x1x1 lower
        # bound so any accidental use fails the shape assert up front.
        return {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}

    def get_module_map(self):
        return {}

    def load_dialects(self, context):
        ventus.load_dialects(context)

    def hash(self):
        identity = Path(__file__).resolve().parents[1] / "toolchain/version.json"
        return hashlib.sha256(identity.read_bytes()).hexdigest()

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        passes.common.add_inliner(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        passes.ttir.add_convert_to_ttgpuir(
            pm, "ventus", options.num_warps, options.warp_size, 1)
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "make_ttgir")
        metadata["num_warps"] = options.num_warps
        metadata["num_stages"] = options.num_stages
        # V1 kernels use no shared memory; the allocation analysis lands with
        # barrier/local-memory support (barrier_local milestone).
        metadata["shared"] = 0
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        # The entry kernel name is needed later to attach the `ventus_kernel`
        # calling convention on the textual IR.
        entry_name = mod.get_entry_func_name()
        metadata["name"] = entry_name
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.convert.add_scf_to_cf(pm)
        ventus.passes.add_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.convert.add_reconcile_unrealized_casts(pm)
        pm.run(mod, "make_llir")
        # MLIR LLVM dialect -> textual LLVM IR. Optimization is owned by the
        # external Ventus toolchain; the consumer side only attaches the ABI
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        if llvm_mod is None:
            raise RuntimeError("Ventus: failed to translate module to LLVM IR")
        ventus.finalize_llir(llvm_mod)
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return _strip_or_disjoint(
            _apply_ventus_kernel_convention(ret, entry_name))

    @staticmethod
    def _run_tool(argv):
        result = subprocess.run(argv, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Ventus tool failed ({result.returncode}): {' '.join(argv)}\n"
                f"stderr:\n{result.stderr}"
            )
        return result

    @staticmethod
    def _run_gate(argv):
        """Run one compile gate, recording evidence instead of discarding it.

        Returns (CompletedProcess, evidence-dict). The evidence is stashed in
        `metadata["ventus_gates"]` so the producer-side manifest
        (`backend/manifest.py`) records the argv/stdout/stderr/status that
        actually produced this artifact rather than re-running the tools.
        """
        result = subprocess.run([str(a) for a in argv], capture_output=True,
                                text=True)
        evidence = {
            "executable": str(argv[0]),
            "argv": [str(a) for a in argv],
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_status": result.returncode,
            "tool_identity": f"{argv[0]}@{_sha256_file(Path(argv[0]))}",
        }
        if result.returncode != 0:
            raise RuntimeError(
                f"Ventus tool failed ({result.returncode}): {' '.join(evidence['argv'])}\n"
                f"stderr:\n{result.stderr}"
            )
        return result, evidence

    @staticmethod
    def make_elf(src, metadata, options):
        tools = dict(options.tool_paths)
        lib = _VENTUS_INSTALL / "lib"
        linker_script = lib / "ldscripts/ventus/elf32lriscv.ld"
        # Link inputs verified by python/test/unit/ventus/test_abi_goldens.py:
        # crt0.o provides _start, riscv32clc.o the work-item builtins,
        # libworkitem.a the __builtin_riscv_* implementations.
        link_inputs = [lib / "crt0.o", lib / "riscv32clc.o", lib / "libworkitem.a"]
        for path in [linker_script, *link_inputs]:
            if not path.is_file():
                raise FileNotFoundError(f"Ventus link input missing: {path}")

        with tempfile.TemporaryDirectory(prefix="triton-ventus-") as tmp:
            tmp = Path(tmp)
            checked_ll = tmp / "kernel.ventus.ll"
            ll = tmp / "kernel.ll"
            obj = tmp / "kernel.o"
            elf = tmp / "kernel.elf"

            # The boundary to the Ventus LLVM 16 toolchain is checked textual
            # IR: `opt -passes=verify` fails loudly on malformed IR before the
            # backend sees it. Gate evidence is kept for the artifact manifest.
            ll.write_text(src)
            _, opt_evidence = VentusBackend._run_gate([
                tools["opt"], "-passes=verify", "-S", str(ll),
                "-o", str(checked_ll),
            ])
            _, llc_evidence = VentusBackend._run_gate([
                tools["llc"], f"-mtriple={options.target_triple}",
                f"-mcpu={_TARGET_MCPU}", "-filetype=obj",
                str(checked_ll), "-o", str(obj),
            ])
            # `riscv32clc.o` is a 22 MB object (all OpenCL builtins, not an
            # archive), so an unpruned link pulls the whole builtins library
            # into the image. `--gc-sections` drops everything unreachable,
            # and `-u` roots the kernel because nothing in the image
            # references it: the runtime looks it up by symbol name, so a
            # plain gc-sections link would silently delete it.
            link_argv = [
                tools["ld.lld"], "-T", str(linker_script),
                "--gc-sections", "-u", metadata["name"],
                str(link_inputs[0]), str(obj), str(link_inputs[1]),
                str(link_inputs[2]), "-o", str(elf),
            ]
            _, lld_evidence = VentusBackend._run_gate(link_argv)

            metadata["ventus_gates"] = {
                "llvm_ir_hash": _sha256_file(checked_ll),
                "object_hash": _sha256_file(obj),
                "elf_hash": _sha256_file(elf),
                "opt": opt_evidence,
                "llc": llc_evidence,
                "lld": lld_evidence,
                "link_inputs": [
                    {"kind": kind, "path": str(path),
                     "content_hash": _sha256_file(path)}
                    for kind, path in (("linker_script", linker_script),
                                       ("crt0", link_inputs[0]),
                                       ("libclc", link_inputs[1]),
                                       ("workitem", link_inputs[2]))
                ],
            }
            return elf.read_bytes()

    def add_stages(self, stages, options, language):
        if language != Language.TRITON:
            raise ValueError(f"Ventus does not support input language {language}")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["elf"] = lambda src, metadata: self.make_elf(src, metadata, options)
