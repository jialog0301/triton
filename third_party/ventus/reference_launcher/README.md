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
the kernel completes. `--grid` is a work-group count; `--local` is the number
of lanes in one work-group and must currently be `32` or `64`.

The driver currently fixes its internal kernel load/run base at `0x80000000`.
Therefore `--entry` is recorded in the execution log and must be the entry
address expected by the generated kernel metadata. The launcher rejects an
entry outside the RV32 address range, but cannot override the installed
driver's internal run-base behavior without changing that driver.

The checked-in Ventus example `spike/gpgpu-testcase/vsw_testcase/vecadd.riscv`
was built for the legacy `numw=2,numt=8` launch profile (16 lanes). It is not a
valid positive test for this V1 launcher, which intentionally accepts only
`[32,1,1]` and `[64,1,1]`. Use a Triton-generated ELF with the V1 profile, or
keep the legacy test on its original driver path.

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
  --elf kernel.riscv --entry 0x800000b8 --grid 1,1,1 --local 32,1,1 \
  --lds-size 4096 --pds-size 4096 --args-log run.json \
  --input a.bin:64 --input b.bin:64 --output c.bin:64 --expected c.ref:64 \
  --timeout-ms 60000 --log spike.log
```

The process exits nonzero for invalid resources, missing files, a driver error,
a timeout, or an output mismatch. The JSON-like log is intentionally simple
and deterministic for this first smoke-test stage; it records the ELF SHA-256,
all launch parameters, the command line, and the result comparison.
