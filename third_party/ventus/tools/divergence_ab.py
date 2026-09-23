#!/usr/bin/env python3
"""A/B the P3 software-predicate and hardware-divergence arms of `tl.where`.

The same kernel is compiled twice, once with `predication="software"` (the
data-plane `llvm.select`) and once with `predication="hardware"` (the `cond_br`
diamond the `select-to-branch` pass builds). Both must compute the same values;
the point of the measurement is what the divergence costs, so the two arms are
run on the same simulator with the same shape.

The cycle simulator logs

    kernel <id> <name> initialized @T0ns
    kernel <id> <name> finished @T1ns

and the reported figure is `T1 - T0` (10 ns per cycle; README 4.1). Each arm
runs in its own process: both timing simulators can only be opened once per
process, and the spdlog output goes to the process's real stdout anyway. The
disassembly op counts come from the linked ELF, so they do not depend on which
driver ran.

Usage:
    divergence_ab.py --backend cyclesim --n 1024 --block 32
"""

import argparse
import collections
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
INSTALL = Path("/home/weijiale/Code/cuda2rvv/ventus-env/install")
OBJDUMP = INSTALL / "bin" / "llvm-objdump"

KERNEL = "where_kernel"
SIGNATURE = {"x_ptr": "*fp32", "z_ptr": "*fp32"}
# Divergence-carrying instructions the Ventus LLVM generates from control flow.
DIVERGENT_OPS = ("setrpc", "vbeq", "vblt", "join")

_KERNEL_WINDOW = re.compile(r"kernel \d+ \S+ initialized\b[^\n]*@(\d+)ns.*?"
                            r"kernel \d+ \S+ finished @(\d+)ns", re.DOTALL)


def parse_kernel_window(output: str) -> int | None:
    m = _KERNEL_WINDOW.search(output)
    return int(m.group(2)) - int(m.group(1)) if m else None


def ventus_env(backend: str) -> dict:
    env = dict(os.environ)
    env["PATH"] = f"{INSTALL / 'bin'}:{env['PATH']}"
    env["LD_LIBRARY_PATH"] = str(INSTALL / "lib")
    if backend == "cyclesim":
        env["VENTUS_CYCLESIM_LOG_LEVEL"] = "info"
    return env


def disassemble_counts(elf: Path) -> dict:
    """Opcode histogram of the kernel body, from the linked ELF.

    The kernel entry is a `T` symbol, but `--disassemble-symbols` matches the
    full symbol and gives the body between entry and the next label; a plain
    histogram over every instruction after the entry line is enough to count
    the divergence-carrying ops.
    """
    out = subprocess.run([str(OBJDUMP), "-d", str(elf)], capture_output=True, text=True, check=True).stdout
    counts = collections.Counter()
    inside = False
    for line in out.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+[0-9a-f]{2}(?: [0-9a-f]{2})+\s+(\S+)", line)
        if not m:
            m = re.match(r"\s*([0-9a-f]+):\s+[0-9a-f]{2}\s+(\S+)", line)
        if not m:
            if f"<{KERNEL}>:" in line:
                inside = True
            elif re.match(r"^[0-9a-f]+ <", line):
                inside = False
            continue
        if inside:
            counts[m.group(2)] += 1
    return dict(counts)


def _child(arm: str, n: int, block: int, backend: str, cache: Path) -> int:
    import triton
    import triton.backends as triton_backends
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler.compiler import compile as triton_compile

    isolated = cache / "backends"
    isolated.mkdir(parents=True, exist_ok=True)
    (isolated / "ventus").symlink_to(VENTUS_ROOT / "backend", target_is_directory=True)
    triton_backends.__path__ = [str(isolated), *triton_backends.__path__]
    importlib.invalidate_caches()
    triton_backends.backends["ventus"] = triton_backends._discover_backends()["ventus"]

    @triton.jit
    def where_kernel(x_ptr, z_ptr, BLOCK: tl.constexpr):
        # An alternating per-lane mask so every warp really diverges: a
        # threshold like `offs < n` only splits the one boundary warp.
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        z = tl.where(offs % 2 == 0, x * 2.0, 0.0)
        tl.store(z_ptr + offs, z)

    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    triton_compile(
        ASTSource(fn=where_kernel, signature=dict(SIGNATURE), constexprs={"BLOCK": block}),
        target=GPUTarget("ventus", "ventus-gpgpu", 32),
        options={"num_warps": 1, "predication": arm})

    elf = next((cache / "triton").rglob("*.elf"))
    launcher = importlib.import_module("triton.backends.ventus.launcher")
    result = launcher.run_where(
        launcher.LaunchSpec(elf=elf, n_elements=n, local_size=32, kernel_name=KERNEL, elements_per_program=block,
                            driver=backend))
    print(
        json.dumps({
            "arm": arm,
            "grid": result["grid"],
            "num_mismatches": result["num_mismatches"],
            "driver_total_ns": result["simulated_time_ns"],
            "ops": disassemble_counts(elf),
        }))
    return 0


def run_arm(arm: str, n: int, block: int, backend: str) -> dict:
    cache = Path(tempfile.mkdtemp(prefix=f"ventus-div-{arm}-"))
    env = ventus_env(backend)
    env["PYTHONPATH"] = str(REPO / "python")
    env.setdefault("TRITON_HOME", str(REPO / ".triton-home"))
    proc = subprocess.run([
        sys.executable,
        str(Path(__file__).resolve()), "--arm", arm, "--n",
        str(n), "--block",
        str(block), "--backend", backend, "--cache",
        str(cache)
    ], capture_output=True, text=True, env=env, cwd=str(REPO), check=False)
    payload = next((line for line in proc.stdout.splitlines() if line.startswith("{")), None)
    out = json.loads(payload) if payload else {"error": proc.stdout[-400:]}
    out["exit_status"] = proc.returncode
    out["kernel_model_ns"] = parse_kernel_window(proc.stdout + proc.stderr)
    out["output_tail"] = (proc.stdout + proc.stderr)[-400:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024, help="elements the grid covers (a multiple of --block)")
    ap.add_argument("--block", type=int, default=32, help="tile size in elements")
    ap.add_argument("--backend", default="spike", choices=["spike", "cyclesim", "rtlsim", "gvm", "auto"])
    ap.add_argument("--arm", default="both", choices=["both", "software", "hardware"], help="internal")
    ap.add_argument("--cache", default=None, help="internal")
    args = ap.parse_args()

    if args.arm != "both":
        return _child(args.arm, args.n, args.block, args.backend, Path(args.cache))

    print(f"# kernel={KERNEL} n={args.n} block={args.block} backend={args.backend} "
          f"grid={(args.n + args.block - 1) // args.block}")
    print(f"{'arm':9s} {'mismatch':>8s} {'kernel_ns':>10s} {'total_ns':>10s}  " +
          "  ".join(DIVERGENT_OPS) + "  insns")
    for arm in ("software", "hardware"):
        r = run_arm(arm, args.n, args.block, args.backend)
        if r.get("error") or r.get("exit_status") not in (0, None):
            print(f"{arm:9s} error: {r.get('error')}\n{r.get('output_tail')}")
            continue
        ops = r.get("ops", {})
        print(f"{arm:9s} {r.get('num_mismatches'):>8} {str(r.get('kernel_model_ns')):>10} "
              f"{str(r.get('driver_total_ns')):>10}  " +
              "  ".join(f"{ops.get(op, 0):>{len(op)}}" for op in DIVERGENT_OPS) +
              f"  {sum(ops.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
