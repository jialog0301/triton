#!/usr/bin/env python3
"""Compile the reference vector-add kernel with the Ventus backend and write
every compilation stage to a directory.

    .venv/bin/python third_party/ventus/tools/dump_ir.py --out-dir /tmp/ventus-out

Writes <out-dir>/vector_add_kernel.{source,ttir,ttgir,llir,elf} .
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402
from triton.compiler.compiler import compile as triton_compile  # noqa: E402

TARGET = GPUTarget("ventus", "ventus-gpgpu", 32)


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, x + y, mask=mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="/tmp/ventus-out", help="directory receiving the stage artifacts")
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--num-warps", type=int, default=1)
    args = parser.parse_args()

    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    kernel = triton_compile(
        ASTSource(fn=vector_add_kernel, signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "z_ptr": "*fp32", "n": "i32"},
                  constexprs={"BLOCK": args.block}),
        target=TARGET,
        options={"num_warps": args.num_warps},
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stage in ("source", "ttir", "ttgir", "llir", "elf"):
        payload = kernel.asm[stage]
        path = out_dir / f"{kernel.name}.{stage}"
        path.write_bytes(payload if isinstance(payload, bytes) else payload.encode())
        print(path)


if __name__ == "__main__":
    main()
