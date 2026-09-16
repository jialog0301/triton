# Ventus Reference Launcher

This is the first Triton-side smoke launcher for the Ventus Spike driver. It
uses the installed `libspike_driver.so` and does not modify Ventus driver or
Spike sources.

The launcher currently supports a vector-add style kernel whose argument
buffer contains device addresses in command-line order:

```text
--input a.bin:64 --input b.bin:64 --output c.bin:64 --expected c.ref:64
```

The input and expected files are raw bytes. The output file is written after
the kernel completes. `--grid` is a work-group count; `--local` must be
`16,1,1`.

This tool implements the **legacy-8x2** launch shape only: 8 lanes per warp
and 2 warps per work-group, giving a 16-lane local size. Its `warp_count` is
derived as `local[0] / 8`, so it cannot describe a V1 kernel, whose warp is 32
lanes. V1 Triton kernels belong on the Python reference launcher:

```bash
third_party/ventus/backend/launcher.py --elf kernel.elf --n 100 \
  --profile v1-32 --json result.json
```

That launcher selects a named profile (`legacy-8x2`, `v1-32`, `v1-64`), reads
the kernel's own `.ventus.resource.<kernel>` record, and refuses a profile
that disagrees with the kernel's `num_warps` or local size. Passing a V1
kernel to this C++ tool is rejected up front rather than launched with 8-lane
warps, which would silently compute the wrong lanes.

The driver currently fixes its internal kernel load/run base at `0x80000000`.
Therefore `--entry` is recorded in the execution log and must be the entry
address expected by the generated kernel metadata. The launcher rejects an
entry outside the RV32 address range, but cannot override the installed
driver's internal run-base behavior without changing that driver.

The checked-in Ventus example `spike/gpgpu-testcase/vsw_testcase/vecadd.riscv`
was built for the legacy `numw=2,numt=8` launch profile (16 lanes), which is
what this tool targets. Note that the driver additionally only accepts entry
metadata `0x800000b8`, so the ELF must have been built for that base.

## Build

```bash
make -C third_party/ventus/reference_launcher
```

Set `VENTUS_INSTALL_PREFIX` if the Ventus installation is not at the default
`/home/weijiale/Code/cuda2rvv/ventus-env/install`.

## Run

```bash
LD_LIBRARY_PATH="$VENTUS_INSTALL_PREFIX/lib" \
  third_party/ventus/reference_launcher/ventus_spike_smoke \
  --elf kernel.riscv --entry 0x800000b8 --grid 1,1,1 --local 16,1,1 \
  --lds-size 4096 --pds-size 4096 --args-log run.json \
  --input a.bin:64 --input b.bin:64 --output c.bin:64 --expected c.ref:64 \
  --timeout-ms 60000 --log spike.log
```

The process exits nonzero for invalid resources, missing files, a driver error,
a timeout, or an output mismatch. The JSON-like log is intentionally simple
and deterministic for this first smoke-test stage; it records the ELF SHA-256,
all launch parameters, the command line, and the result comparison.
