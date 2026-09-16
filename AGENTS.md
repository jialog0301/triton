# Working on Triton

## Build and Testing Guidelines
- Before running tests for native/compiler changes, run `make` in the triton directory to rebuild triton. DO NOT RUN `make` if you only changed Python code or code in `python/triton_kernels`.
- For compiler changes, add tests in `python/test/` (pytest) or test (lit). Keep GPU-only tests in `python/test/unit/` or `python/test/gluon/`, name them `test_<feature>_<condition>`, and avoid creating new test files unless requested.
- Run pytest with `-s --tb=short`. Run a single test with `pytest file.py::test_name`.
- The build dir is given by `BUILD_DIR := $(shell PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')`
- Run lit from the build dir:  `cd BUILD_DIR; ninja triton-opt; lit -v test/<path>.mlir` (example: `lit -v test/TritonNvidiaGPU/tmem_layouts.mlir`).
- Lit tests can be run locally (no GPU required).
- Compiler crashes sometimes print an MLIR reproducer (external_resources / mlir_reproducer). Save the full MLIR + {-# ... #-} metadata to `/tmp/<file>.mlir`, then run `triton-opt /tmp/<file>.mlir --run-reproducer` to reproduce locally.

### Ventus backend runtime tests (Spike)

The Ventus V1 tests execute the compiled Triton ELF on the instruction-level
Ventus Spike simulator through `third_party/ventus/backend/launcher.py` +
`libspike_driver.so`; they need no GPU. Run the backend suite (pytest):

```bash
source .venv/bin/activate
export TRITON_HOME="$PWD/.triton-home"
export PYTHONPATH="$PWD/python"
# ABI-golden regeneration requires the pinned Ventus tool paths:
export VENTUS_CLANG=.../ventus-env/install/bin/clang
export VENTUS_OPT=.../ventus-env/install/bin/opt
export VENTUS_LLC=.../ventus-env/install/bin/llc
export VENTUS_LLD=.../ventus-env/install/bin/ld.lld
python -m pytest python/test/unit/ventus/ -q
```

`test_vector_add_on_spike` and `test_vector_add_manifest_gate4` are the M1
Gate 4 (Spike execution) baseline: the manifest is only accepted by
`ventus.VentusKernelMetadata.validate()` after `launcher_input`/`test_result`
are recorded. Native libtriton changes (e.g. `third_party/ventus/triton_ventus.cc`
bindings) require a rebuild first: `cd "$(BUILD_DIR)"; ninja triton`.

## C++ Guidelines
- In C++, never put side-effecting code in `assert`. Assertions may be compiled out, so perform mutations and other required computation before the assertion and assert only the resulting condition. This guideline does not apply to Python `assert` statements.
