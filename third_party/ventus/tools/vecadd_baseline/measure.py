#!/usr/bin/env python3
"""Measure the Triton and OpenCL arms of the same vector-add workload.

Both arms run on the same simulator (`--backend spike|cyclesim|...`) with the same
shape: `n` elements, a work-group of `--local` work-items, `ceil(n / local)`
work-groups, and a tail guard. The Triton arm is this backend's kernel; the
OpenCL arm is `vecadd_baseline` (see `kernel.cl`), built by this script if needed.

The comparable number is the kernel's own model time. The cycle simulator logs

    kernel <id> <name> initialized @T0ns
    kernel <id> <name> finished @T1ns

so the reported figure is `T1 - T0`, in nanoseconds of model time (10 ns per
clock cycle, see README 4.1). That window is used for both arms because a
driver's total simulated time also covers pre-kernel setup and the host copies
after the kernel, which a reference launcher and an OpenCL runtime do not bracket
identically.

Each arm runs in its own process as a subprocess: both simulators write their log
through spdlog to the process's real stderr (so Python-level redirection cannot
capture it), and neither can be opened twice in one process anyway.

Usage:
    measure.py --n 1024 --local 32 --backend cyclesim
"""

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
VENTUS_ROOT = REPO / "third_party/ventus"
BASELINE_DIR = Path(__file__).resolve().parent
INSTALL = Path("/home/weijiale/Code/cuda2rvv/ventus-env/install")

# Resource declarations the OpenCL arm sends, from `pocl_ventus.cc`
# (`ldssize`/`pdssize`/`sgpr_usage`/`vgpr_usage`). The Triton arm is launched
# with `force_resources` so both arms declare the same numbers: register
# declarations steer the model's warp scheduling, so leaving our VRES-derived
# values in place would compare two different occupancy decisions rather than two
# implementations.
OPENCL_DECLARED = {"lds": 0x1000, "pds": 0x1000, "sgpr": 64, "vgpr": 64}

_KERNEL_WINDOW = re.compile(
    r"kernel \d+ \S+ initialized\b[^\n]*@(\d+)ns.*?"
    r"kernel \d+ \S+ finished @(\d+)ns", re.DOTALL)


def parse_kernel_window(output: str) -> int | None:
    """Kernel model time in ns, from the cycle simulator's own log lines.

    The device logs through `spdlog::stdout_color_mt("ventus")`, so these lines
    arrive on stdout, not stderr.
    """
    m = _KERNEL_WINDOW.search(output)
    return int(m.group(2)) - int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# OpenCL arm
# --------------------------------------------------------------------------- #

def build_opencl_arm() -> Path:
    exe = BASELINE_DIR / "vecadd_baseline"
    src = BASELINE_DIR / "main.cc"
    if exe.is_file() and exe.stat().st_mtime >= src.stat().st_mtime:
        return exe
    subprocess.run([
        str(INSTALL / "bin" / "clang++"), "-O2", "-std=c++11", str(src), "-o",
        str(exe), f"-I{INSTALL / 'include'}", f"-L{INSTALL / 'lib'}", "-lOpenCL",
    ], check=True, capture_output=True)
    return exe


def ventus_env(backend: str) -> dict:
    """Environment the Ventus OpenCL runtime needs (see rodinia's runner)."""
    env = dict(os.environ)
    env["PATH"] = f"{INSTALL / 'bin'}:{env['PATH']}"
    env["LD_LIBRARY_PATH"] = str(INSTALL / "lib")
    env["OCL_ICD_VENDORS"] = str(INSTALL / "lib" / "libpocl.so")
    env["POCL_DEVICES"] = "ventus"
    env["VENTUS_BACKEND"] = backend
    if backend == "cyclesim":
        env["VENTUS_CYCLESIM_LOG_LEVEL"] = "info"
    return env


def run_opencl_arm(n: int, local: int, backend: str) -> dict:
    exe = build_opencl_arm()
    # The Ventus OpenCL runtime writes its intermediate files (object0.cl,
    # *.riscv, *.vmem, vecadd_0.log, ...) into the current directory, so run it
    # in a scratch directory instead of wherever the caller happens to stand.
    workdir = Path(tempfile.mkdtemp(prefix="ventus-opencl-"))
    proc = subprocess.run(
        [str(exe), str(BASELINE_DIR / "kernel.cl"), str(n), str(local)],
        capture_output=True, text=True, env=ventus_env(backend), check=False,
        cwd=str(workdir))
    return {
        "exit_status": proc.returncode,
        "result_line": next((line for line in proc.stdout.splitlines()
                             if line.startswith("RESULT ")), None),
        "kernel_model_ns": parse_kernel_window(proc.stdout + proc.stderr),
        "workdir": str(workdir),
        "stderr_tail": (proc.stdout + proc.stderr)[-400:],
    }


# --------------------------------------------------------------------------- #
# Triton arm (runs in its own process; see the module docstring)
# --------------------------------------------------------------------------- #

def _triton_child(n: int, local: int, backend: str, cache: Path) -> int:
    """Compile and launch the Triton kernel, printing one JSON line."""
    import triton
    import triton.backends as triton_backends
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource
    from triton.compiler.compiler import compile as triton_compile

    isolated = cache / "backends"
    isolated.mkdir(parents=True)
    (isolated / "ventus").symlink_to(VENTUS_ROOT / "backend",
                                     target_is_directory=True)
    triton_backends.__path__ = [str(isolated), *triton_backends.__path__]
    importlib.invalidate_caches()
    triton_backends.backends["ventus"] = \
        triton_backends._discover_backends()["ventus"]

    @triton.jit
    def vector_add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(z_ptr + offs, x + y, mask=mask)

    triton_compile(ASTSource(
        fn=vector_add_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "z_ptr": "*fp32",
                   "n": "i32"},
        constexprs={"BLOCK": local},
    ), target=GPUTarget("ventus", "ventus-gpgpu", 32), options={"num_warps": 1})

    launcher = importlib.import_module("triton.backends.ventus.launcher")
    result = launcher.run_vector_add(
        launcher.LaunchSpec(elf=next(cache.rglob("*.elf")), n_elements=n,
                            local_size=local, driver=backend,
                            force_resources=True, **OPENCL_DECLARED))
    print(json.dumps({
        "grid": result["grid"],
        "num_mismatches": result["num_mismatches"],
        "driver_total_ns": result["simulated_time_ns"],
        "declared": OPENCL_DECLARED,
    }))
    return 0


def run_triton_arm(n: int, local: int, backend: str) -> dict:
    cache = Path(tempfile.mkdtemp(prefix="ventus-measure-cache-"))
    env = ventus_env(backend)
    env["PYTHONPATH"] = str(REPO / "python")
    env.setdefault("TRITON_HOME", str(REPO / ".triton-home"))
    env["TRITON_CACHE_DIR"] = str(cache / "triton")
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--arm", "triton",
         "--n", str(n), "--local", str(local), "--backend", backend,
         "--cache", str(cache)],
        capture_output=True, text=True, env=env, cwd=str(REPO), check=False)
    payload = next((line for line in proc.stdout.splitlines()
                    if line.startswith("{")), None)
    out = json.loads(payload) if payload else {"error": proc.stdout[-200:]}
    out["exit_status"] = proc.returncode
    out["kernel_model_ns"] = parse_kernel_window(proc.stdout + proc.stderr)
    out["stderr_tail"] = (proc.stdout + proc.stderr)[-400:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--local", type=int, default=32)
    ap.add_argument("--backend", default="spike",
                    choices=["spike", "cyclesim", "rtlsim", "gvm", "auto"])
    ap.add_argument("--skip-triton", action="store_true")
    ap.add_argument("--skip-opencl", action="store_true")
    ap.add_argument("--arm", default="both", choices=["both", "triton", "opencl"],
                    help="internal: run one arm in this process")
    ap.add_argument("--cache", default=None, help="internal")
    args = ap.parse_args()

    if args.arm == "triton":
        return _triton_child(args.n, args.local, args.backend,
                             Path(args.cache))

    print(f"# n={args.n} local={args.local} backend={args.backend}"
          f" grid={(args.n + args.local - 1) // args.local}")
    if not args.skip_triton:
        t = run_triton_arm(args.n, args.local, args.backend)
        print(f"triton   grid={t.get('grid')} "
              f"mismatches={t.get('num_mismatches')} "
              f"kernel_model_ns={t.get('kernel_model_ns')} "
              f"driver_total_ns={t.get('driver_total_ns')}")
        if t.get("error"):
            print(f"  error: {t['error']}")
    if not args.skip_opencl:
        o = run_opencl_arm(args.n, args.local, args.backend)
        print(f"opencl   exit={o['exit_status']} {o['result_line']}")
        print(f"         kernel_model_ns={o['kernel_model_ns']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
