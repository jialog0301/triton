#!/usr/bin/env python3
"""Reference launcher for the V1 Ventus Triton backend (Spike execution).

Runs a Triton-compiled Ventus ELF on the instruction-level Ventus Spike
simulator via the installed `libspike_driver.so`, without OpenCL/POCL.

This is the test-only execution baseline for the M1 kernel matrix: it is NOT
a physical device driver. Spike is chosen because V1 declares it the
mandatory M1/M2 execution baseline (the RTL/rtlsim path and CycleSim stay
capability-gated), and because the Ventus driver's spike_device runs a
work-group per Spike invocation by tuning a `--gpgpuarch` config string.

ABI facts encoded here (machine-verified against the Ventus LLVM 16 toolchain
and the POCL launch path, see docs/plans/2026-08-20-triton-ventus-iree-
implementation.md and third_party/ventus/toolchain/README.md):

* Kernel arguments are loaded by the kernel prologue from the device address
  passed in a0 (`lowerKernArgParameterPtr` -> X10). The crt0 passes a0 from
  `KNL_ARG_BASE` (offset 4) of a 64-byte kernel metadata word buffer that
  lives in spike-backed device memory (classic POCL layout, see
  pocl_ventus.cc). Pointers and scalar values are packed little-endian 32-bit.
* The same buffer also holds the OpenCL-style launch info at fixed offsets
  (KNL_ENTRY 0, KNL_ARG_BASE 4, KNL_WORK_DIM 8, KNL_GL_SIZE_* 12/16/20,
  KNL_LC_SIZE_* 24/28/32, KNL_GL_OFFSET_* 36/40/44).
* A `meta_data` record (spike_main/spike_main.h) drives the spike_device
  invocation: wf_size = threads/warp, wg_size = warps/work-group, kernel_size=
  work-group counts per axis, ldsSize, pdsSize, pdsBaseAddr, knlbase (the
  metadata buffer addr), sgpr/vgpr usage.
* spike_device.run() builds a separate `sim` per work-group and drives the
  ELF loaded at 0x80000000 (entry `_start`), matching crt0.S's `jalr t1` to
  the kernel in KNL_ENTRY. Buffer allocation starts at 0x90000000
  (ARGBASEADDR).
"""

import ctypes
import json
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

INSTALL = Path("/home/weijiale/Code/cuda2rvv/ventus-env/install")
LLVM_NM = INSTALL / "bin" / "llvm-nm"

# Launchable device drivers. All five export the same `vt_*` API (checked with
# llvm-nm), so a launch picks one by name:
#   spike     instruction-level functional simulation (the M1 baseline)
#   cyclesim  cycle-level SystemC GPGPU + ramulator DRAM. This is the timing
#             source of truth: AGENTS.md requires performance numbers to come
#             from here, spike being functional-only.
#   rtlsim    Verilator RTL simulation
#   gvm       the original GVM model
#   auto      the driver's own runtime selection (`libventus_driver.so`)
DRIVERS = {
    "spike": "libspike_driver.so",
    "cyclesim": "libcyclesim_driver.so",
    "rtlsim": "librtlsim_driver.so",
    "gvm": "libgvm_driver.so",
    "auto": "libauto_select_driver.so",
}
DEFAULT_DRIVER = "spike"
# Retained for callers that bind to the module-level default directly.
DRIVER_SO = INSTALL / "lib" / DRIVERS[DEFAULT_DRIVER]

# The cycle simulator keeps its timing state in its own library. Every device's
# `vt_dump_perf` is a no-op (`return 0` in spike/cyclesim/rtlsim/gvm), so the
# simulated time is read through the simulator's C API instead; a simulated
# clock period of 10 ns follows from `cyclesim/src/parameters.h:42`.
CYCLESIM_SO = INSTALL / "lib" / "libVentusCycleSim.so"
# `ventus_cyclesim_init` calls `sc_set_time_resolution`, and SystemC refuses a
# second simulation in one process: "(E514) set time resolution failed:
# simulation running", followed by an abort. One cyclesim launch per process is
# therefore a hard constraint, enforced below with a Python error rather than a
# SIGABRT from inside the model. The flag lives on `sys` because the backend
# discovery tests purge `triton.backends.ventus.*` from `sys.modules` (which
# would reset a module global while the SystemC context would survive).
_CYCLESIM_OPENED_ATTR = "_triton_ventus_cyclesim_opened"

# Kernel metadata buffer layout (device words), see libclc ventus.h:
KNL_ENTRY = 0
KNL_ARG_BASE = 4
KNL_WORK_DIM = 8
KNL_GL_SIZE_X = 12
KNL_GL_SIZE_Y = 16
KNL_GL_SIZE_Z = 20
KNL_LC_SIZE_X = 24
KNL_LC_SIZE_Y = 28
KNL_LC_SIZE_Z = 32
KNL_GL_OFFSET_X = 36
KNL_GL_OFFSET_Y = 40
KNL_GL_OFFSET_Z = 44
KNL_MAX_METADATA_SIZE = 64

# `.ventus.resource.<kernel>` = magic "VRES", version, record size, then four
# little-endian uint32 fields in the order the Ventus backend serializes them.
RESOURCE_MAGIC = b"VRES"
RESOURCE_FIELDS = ("vgpr", "sgpr", "lds", "pds")
RESOURCE_SECTION_PREFIX = ".ventus.resource."

# Ventus spike driver buffer limits (spike_main/spike_device.cc):
# `char logfilename[64]` receives `--log=<path>.log`, so the path (plus
# suffixes) must fit in 64 chars. This is the binding constraint on staging.
DRIVER_PATH_LIMIT = 64 - len("--log=") - len(".log") - 1   # 53

# Launch profiles. The named vocabulary is shared with
# `third_party/ventus/reference_launcher/ventus_spike_smoke.cpp` and
# `docs/plans/2026-09-15-ventus-launch-profile-design.md`, so both launchers
# describe a launch the same way. Each tool realizes the shapes it can
# actually launch: this one launches V1 Triton kernels (32-lane warps), while
# the C++ smoke tool covers the legacy 16-lane OpenCL shape.
#
# `lanes_per_warp * warps_per_workgroup == local_size_x`, and the driver's
# `wf_size`/`wg_size` come from the first two fields. `vector_length` mirrors
# spike_device.cc: `vlen = num_thread * 32` with `elen = 32`.
V1_LANES_PER_WARP = 32


@dataclass(frozen=True)
class VentusLaunchProfile:
    lanes_per_warp: int
    warps_per_workgroup: int
    local_size_x: int
    local_size_y: int
    local_size_z: int
    vector_length: int
    lds_size: int
    pds_size: int
    sgpr_usage: int
    vgpr_usage: int


def _profile(lanes_per_warp, warps_per_workgroup, lds_size, pds_size,
             sgpr_usage, vgpr_usage):
    return VentusLaunchProfile(
        lanes_per_warp=lanes_per_warp,
        warps_per_workgroup=warps_per_workgroup,
        local_size_x=lanes_per_warp * warps_per_workgroup,
        local_size_y=1,
        local_size_z=1,
        vector_length=lanes_per_warp * 32,
        lds_size=lds_size,
        pds_size=pds_size,
        sgpr_usage=sgpr_usage,
        vgpr_usage=vgpr_usage,
    )


# The complete named vocabulary, including the legacy shape this launcher
# does not launch (the C++ smoke tool owns it). Kept here so the two launchers
# cannot drift on what a profile name means.
LAUNCH_PROFILE_VOCABULARY = {
    "legacy-8x2": _profile(8, 2, 0x1000, 0x1000, 64, 64),
    "v1-32": _profile(V1_LANES_PER_WARP, 1, 0x1000, 0x1000, 64, 64),
    "v1-64": _profile(V1_LANES_PER_WARP, 2, 0x1000, 0x1000, 64, 64),
}

# Profiles this launcher can launch: V1 kernels with the fixed 32-lane warp.
BUILTIN_PROFILES = {
    name: profile for name, profile in LAUNCH_PROFILE_VOCABULARY.items()
    if profile.lanes_per_warp == V1_LANES_PER_WARP
}


class _MetaData(ctypes.Structure):
    """`driver_metadata_t` as the device drivers declare it.

    spike/gvm/rtlsim end at `pdsBaseAddr`; cyclesim appends a `kernel_name`
    pointer that it turns into a `std::string` for its task/kernel names. Left
    unset, that field is whatever follows the struct in memory, which cyclesim
    reads: the name came out as garbage, and a null one aborted the model with
    `basic_string: construction from null is not valid`. Passing a longer struct
    is harmless to the drivers that stop earlier.
    """

    _fields_ = [
        ("kernel_id", ctypes.c_uint64),
        ("kernel_size", ctypes.c_uint64 * 3),
        ("wf_size", ctypes.c_uint64),
        ("wg_size", ctypes.c_uint64),
        ("metaDataBaseAddr", ctypes.c_uint64),
        ("ldsSize", ctypes.c_uint64),
        ("pdsSize", ctypes.c_uint64),
        ("sgprUsage", ctypes.c_uint64),
        ("vgprUsage", ctypes.c_uint64),
        ("pdsBaseAddr", ctypes.c_uint64),
        ("kernel_name", ctypes.c_char_p),
    ]


class _Driver:
    """ctypes binding to the Ventus spike device driver shared object."""

    def __init__(self, so: Path):
        if so.name == DRIVERS["cyclesim"]:
            if getattr(sys, _CYCLESIM_OPENED_ATTR, False):
                raise RuntimeError(
                    "the cycle simulator can only be opened once per process "
                    "(SystemC rejects a second simulation); run each cyclesim "
                    "launch in its own process")
            setattr(sys, _CYCLESIM_OPENED_ATTR, True)
        self.lib = ctypes.CDLL(str(so))
        self.dev = ctypes.c_void_p()
        f = self.lib
        f.vt_dev_open.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        f.vt_dev_open.restype = ctypes.c_int
        f.vt_dev_close.argtypes = [ctypes.c_void_p]
        f.vt_dev_close.restype = ctypes.c_int
        f.vt_buf_alloc.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                   ctypes.POINTER(ctypes.c_uint64),
                                   ctypes.c_int, ctypes.c_uint64,
                                   ctypes.c_uint64]
        f.vt_buf_alloc.restype = ctypes.c_int
        f.vt_copy_to_dev.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                     ctypes.c_void_p, ctypes.c_uint64,
                                     ctypes.c_uint64, ctypes.c_uint64]
        f.vt_copy_to_dev.restype = ctypes.c_int
        f.vt_copy_from_dev.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                       ctypes.c_void_p, ctypes.c_uint64,
                                       ctypes.c_uint64, ctypes.c_uint64]
        f.vt_copy_from_dev.restype = ctypes.c_int
        f.vt_upload_kernel_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                            ctypes.c_int]
        f.vt_upload_kernel_file.restype = ctypes.c_int
        f.vt_start.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_uint64]
        f.vt_start.restype = ctypes.c_int
        f.vt_ready_wait.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        f.vt_ready_wait.restype = ctypes.c_int

        if f.vt_dev_open(ctypes.byref(self.dev)) != 0:
            raise RuntimeError("vt_dev_open failed")
        self._buffers = []

    def alloc(self, size: int) -> int:
        v = ctypes.c_uint64()
        if self.lib.vt_buf_alloc(self.dev, size, ctypes.byref(v), 0, 0, 0) != 0:
            raise RuntimeError("vt_buf_alloc failed")
        self._buffers.append(v.value)
        return v.value

    def to_dev(self, vaddr: int, data: bytes):
        buf = ctypes.create_string_buffer(bytes(data), len(data))
        if self.lib.vt_copy_to_dev(self.dev, vaddr,
                                   ctypes.cast(buf, ctypes.c_void_p),
                                   len(data), 0, 0) != 0:
            raise RuntimeError("vt_copy_to_dev failed")

    def from_dev(self, vaddr: int, size: int) -> bytes:
        buf = ctypes.create_string_buffer(size)
        if self.lib.vt_copy_from_dev(self.dev, vaddr,
                                     ctypes.cast(buf, ctypes.c_void_p),
                                     size, 0, 0) != 0:
            raise RuntimeError("vt_copy_from_dev failed")
        return buf.raw

    def upload(self, elf: Path):
        # spike_device::set_filename sprintf()s the path into a fixed
        # `new char[128]`, and builds `--log=<path>.log` into char[64]. Both
        # silently overflow on a deep cache path, so the launcher stages the
        # ELF at a short path first (see stage_elf).
        if self.lib.vt_upload_kernel_file(self.dev, str(elf).encode(), 0) != 0:
            raise RuntimeError("vt_upload_kernel_file failed")

    def launch(self, metadata, timeout_s: int = 120):
        if self.lib.vt_start(self.dev, ctypes.byref(metadata), 0) != 0:
            raise RuntimeError("vt_start failed")
        # The cyclesim device ignores this timeout (`(void)timeout;` in
        # cyclesim_device/ventus.cpp) and spins until the model goes idle, so a
        # launch that never idles cannot be bounded from here -- cap it
        # externally. spike honours the value.
        if self.lib.vt_ready_wait(self.dev, timeout_s * 1000) != 0:
            raise RuntimeError("vt_ready_wait failed")

    def close(self):
        self.lib.vt_dev_close(self.dev)


class _CyclesimTime:
    """Read `ventus_cyclesim_get_time()` for the handle a driver opened.

    `vt_dev_open` in `cyclesim_device/ventus.cpp` stores exactly the
    `ventus_cyclesim_t *` that `ventus_cyclesim_init` returned, and that is what
    `vt_ready_wait`/`ventus_cyclesim_get_time` take. The driver loads the
    simulator by absolute path, so dlopening the same path here yields the same
    already-loaded instance and the same handle stays valid.
    """

    def __init__(self, so: Path = CYCLESIM_SO):
        self.lib = ctypes.CDLL(str(so))
        self.lib.ventus_cyclesim_get_time.argtypes = [ctypes.c_void_p]
        self.lib.ventus_cyclesim_get_time.restype = ctypes.c_uint64

    def ns(self, dev) -> int:
        """Simulated time at this instant, in nanoseconds of model time."""
        return int(self.lib.ventus_cyclesim_get_time(dev))


@dataclass
class LaunchSpec:
    """One Spike execution: an ELF plus launch geometry and arguments."""
    elf: Path
    n_elements: int
    local_size: int = 32
    ptr_dtype: str = "f32"
    scalar: int = 0                  # n (tail) for the vector_add kernel
    work_dim: int = 1
    num_warps: int = 1
    sgpr: int = 64
    vgpr: int = 64
    lds: int = 0
    pds: int = 0x1000
    timeout_s: int = 120
    keep_log: bool = False
    profile: str = "v1-32"           # named launch profile (see BUILTIN_PROFILES)
    # Entry symbol to launch. The ABI below (x_ptr, y_ptr, z_ptr, n) and the
    # `z[i] = x[i] + y[i]` reference are what the launcher actually requires,
    # so any elementwise-add kernel that covers [0, n) qualifies -- 1-D or a
    # 2-D tile whose flattened offsets cover the same range.
    kernel_name: str = "vector_add_kernel"
    driver: str = DEFAULT_DRIVER      # device driver, see DRIVERS

    def __post_init__(self):
        self.elf = Path(self.elf)
        if not self.elf.is_file():
            raise FileNotFoundError(self.elf)
        if self.driver not in DRIVERS:
            raise ValueError(
                f"unknown driver {self.driver!r}; expected one of "
                f"{sorted(DRIVERS)}")
        if self.profile not in LAUNCH_PROFILE_VOCABULARY:
            raise ValueError(
                f"unknown launch profile {self.profile!r}; "
                f"expected one of {sorted(LAUNCH_PROFILE_VOCABULARY)}")
        if self.profile not in BUILTIN_PROFILES:
            # A known profile this launcher cannot realize: `legacy-8x2` is an
            # 8-lane shape owned by the C++ smoke tool.
            raise ValueError(
                f"launch profile {self.profile!r} is not launchable by this "
                f"V1 launcher; use {sorted(BUILTIN_PROFILES)}")
        if self.local_size not in (32, 64):
            raise ValueError("V1 launch local_size must be 32 or 64")


def _kernel_entry(elf: Path, name: str) -> int:
    out = subprocess.run([str(LLVM_NM), str(elf)], capture_output=True,
                         text=True, check=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == "T" and parts[2] == name:
            return int(parts[0], 16)
    raise KeyError(f"kernel {name!r} not found in {elf} (T symbol)")


def _resource_record(elf: Path, kernel: str) -> dict:
    """Read the backend-serialized `.ventus.resource.<kernel>` record.

    Returns {vgpr, sgpr, lds, pds} from the four little-endian uint32 fields,
    or an empty record if the section is absent. POCL reads the same section
    through hard-coded offsets; V1 keeps the record as the sole transport and
    the launcher validates it against the compiled artifact.
    """
    out = subprocess.run(
        [_readelf(), "-S", str(elf)], capture_output=True, text=True,
        check=True).stdout
    import re
    # `[10] .ventus.resource.<k> PROGBITS <addr> <off> <size> ...`
    sec = re.search(rf"{re.escape(RESOURCE_SECTION_PREFIX + kernel)}\s+"
                    r"PROGBITS\s+\S+\s+(\S+)\s+(\S+)", out)
    if not sec:
        return {}
    off, size = int(sec.group(1), 16), int(sec.group(2), 16)
    data = elf.read_bytes()[off:off + size]
    if len(data) < 4 + 16 or data[:4] != RESOURCE_MAGIC:
        return {}
    # magic(4) version(1) record_size(1) pad(2); fields start at offset 8.
    fields = struct.unpack_from("<IIII", data, 8)
    return dict(zip(RESOURCE_FIELDS, fields))


def _readelf() -> str:
    return str(INSTALL / "bin" / "llvm-readelf")


def stage_elf(elf: Path, staging_dir: Path | None = None) -> Path:
    """Copy `elf` to a path short enough for the Ventus spike driver.

    `spike_device::set_filename` (spike_main/spike_device.cc:342) formats the
    ELF path into `new char[128]` and the log argument into `char[64]` as
    `--log=<path>.log`. A Triton cache path is ~165 characters and a pytest
    `tmp_path` ~67, so either overflows those buffers and aborts the process
    before Spike starts. The binding constraint is the smaller one:

        len("--log=") + len(path) + len(".log") + NUL <= 64  =>  len(path) <= 53

    `staging_dir` is accepted for callers that own an already-short directory;
    otherwise a short hash-keyed directory under the system temp dir is used,
    which also keeps concurrent launches of different ELFs apart.
    """
    if staging_dir is None:
        import hashlib
        import tempfile
        key = hashlib.sha256(elf.read_bytes()).hexdigest()[:8]
        staging_dir = Path(tempfile.gettempdir()) / f"vt{key}"
    staged = staging_dir / "k.elf"
    if len(str(staged)) > DRIVER_PATH_LIMIT:
        raise ValueError(
            f"staged ELF path {str(staged)!r} exceeds the Ventus driver's "
            f"{DRIVER_PATH_LIMIT}-character limit; pass a shorter staging_dir")
    staging_dir.mkdir(parents=True, exist_ok=True)
    if not staged.is_file() or staged.read_bytes() != elf.read_bytes():
        staged.write_bytes(elf.read_bytes())
    return staged


def run_vector_add(spec: LaunchSpec, launch_dir: Path | None = None) -> dict:
    """Launch the standard vector_add kernel and return result + evidence.

    Kernel ABI (from the TTGIR golden): x_ptr, y_ptr, z_ptr, n.
    `launch_dir` is the staging directory for the short-path ELF; it defaults
    to the driver-safe hash-keyed temp dir used by `stage_elf`.
    """
    if spec.scalar == 0:
        spec.scalar = spec.n_elements
    n = spec.n_elements
    profile = BUILTIN_PROFILES[spec.profile]
    if spec.local_size != profile.local_size_x:
        raise ValueError(
            f"local_size {spec.local_size} disagrees with profile "
            f"{spec.profile!r} (local_size_x={profile.local_size_x})")
    # A profile declares how many warps the launch has; the kernel was
    # compiled for `num_warps` warps. Mismatching them would launch lanes the
    # kernel never accounted for (Triton distributes the block over
    # num_warps x warp_size lanes), so the results would alias rather than
    # fail. Refuse instead.
    if spec.num_warps != profile.warps_per_workgroup:
        raise ValueError(
            f"kernel num_warps {spec.num_warps} disagrees with profile "
            f"{spec.profile!r} (warps_per_workgroup="
            f"{profile.warps_per_workgroup})")
    local = profile.local_size_x
    grid = (n + local - 1) // local
    nb = n * 4
    entry = _kernel_entry(spec.elf, spec.kernel_name)
    resources = _resource_record(spec.elf, spec.kernel_name)

    # The compiled kernel's own resource record wins; the profile value is the
    # fallback for kernels without the section. LDS and PDS are floored at the
    # crt0 stack contract: crt0.S gives every warp 1 KiB of local memory
    # (`sp = wid * 1024 + CSR_LDS`) and spike_device::run reserves the private
    # segment, so a launch below those floors would alias warp stacks. The
    # compiled value is still carried in the evidence for launch-time checks.
    lds = max(resources.get("lds", spec.lds), profile.lds_size)
    pds = max(resources.get("pds", spec.pds), profile.pds_size)

    driver = _Driver(INSTALL / "lib" / DRIVERS[spec.driver])
    # Spike's driver formats the ELF path into fixed-size buffers, so a
    # Triton cache path or a pytest tmp_path overflows it; stage at a short
    # path. Log collection happens after the launch (see below).
    staged = stage_elf(spec.elf, launch_dir)
    try:
        x = [float(i + 1) for i in range(n)]
        y = [float(2 * i) for i in range(n)]
        xa = driver.alloc(nb)
        ya = driver.alloc(nb)
        za = driver.alloc(nb)
        driver.to_dev(xa, struct.pack(f"{n}f", *x))
        driver.to_dev(ya, struct.pack(f"{n}f", *y))
        driver.to_dev(za, bytes(nb))

        argbuf = struct.pack("<IIII", xa, ya, za, spec.scalar)
        arga = driver.alloc(len(argbuf))
        driver.to_dev(arga, argbuf)

        knl = bytearray(KNL_MAX_METADATA_SIZE)
        struct.pack_into("<I", knl, KNL_ENTRY, entry)
        struct.pack_into("<I", knl, KNL_ARG_BASE, arga)
        struct.pack_into("<I", knl, KNL_WORK_DIM, spec.work_dim)
        struct.pack_into("<I", knl, KNL_GL_SIZE_X, grid * local)
        struct.pack_into("<I", knl, KNL_GL_SIZE_Y, 1)
        struct.pack_into("<I", knl, KNL_GL_SIZE_Z, 1)
        struct.pack_into("<I", knl, KNL_LC_SIZE_X, local)
        struct.pack_into("<I", knl, KNL_LC_SIZE_Y, 1)
        struct.pack_into("<I", knl, KNL_LC_SIZE_Z, 1)
        knla = driver.alloc(KNL_MAX_METADATA_SIZE)
        driver.to_dev(knla, knl)
        # Private memory (PDS) is a per-work-item size in the metadata but a
        # per-launch range in the address space: the documented convention is
        #     pdsSize * wf_size * wg_size * grid
        # starting at `pdsBaseAddr` (cyclesim/src/task.cpp sizes that range and
        # then asserts it was allocated exactly; the hardware gives each
        # work-group its own CSR_PDS offset derived from it). spike ignores the
        # address entirely -- spike_device::run hard-codes a 0x10000000 segment
        # per work-group, which POCL mirrors -- so sizing this for a single
        # work-group is invisible there and only showed up when the grid of a
        # launch grew beyond one.
        pds_addr = driver.alloc(pds * profile.lanes_per_warp *
                                profile.warps_per_workgroup * grid)

        md = _MetaData()
        md.kernel_id = 0
        md.kernel_size[0] = grid
        md.kernel_size[1] = 1
        md.kernel_size[2] = 1
        md.wf_size = profile.lanes_per_warp
        md.wg_size = profile.warps_per_workgroup
        md.metaDataBaseAddr = knla
        md.ldsSize = lds
        md.pdsSize = pds
        md.sgprUsage = resources.get("sgpr", spec.sgpr)
        md.vgprUsage = resources.get("vgpr", spec.vgpr)
        md.pdsBaseAddr = pds_addr
        md.kernel_name = spec.kernel_name.encode()

        t0 = time.monotonic()
        driver.upload(staged)
        driver.launch(md, timeout_s=spec.timeout_s)
        elapsed = time.monotonic() - t0
        # Read the simulated time before closing the device: the handle is the
        # simulator instance (see _CyclesimTime).
        simulated_ns = (_CyclesimTime().ns(driver.dev)
                        if spec.driver == "cyclesim" else None)
        z = struct.unpack(f"{n}f", driver.from_dev(za, nb))
    finally:
        driver.close()

    expect = [a + b for a, b in zip(x, y)]
    mismatches = [(i, got, exp)
                  for i, (got, exp) in enumerate(zip(z, expect))
                  if abs(got - exp) > 1e-6]

    result = {
        "kernel": spec.kernel_name,
        "entry": hex(entry),
        "n": n,
        "local_size": local,
        "grid": grid,
        "num_warps": spec.num_warps,
        "profile": spec.profile,
        "lanes_per_warp": profile.lanes_per_warp,
        "warps_per_workgroup": profile.warps_per_workgroup,
        "vector_length": profile.vector_length,
        # The driver-facing values actually handed to `vt_start`; a profile
        # that did not reach the driver would still show up as a mismatch in
        # the spike log, but these make the wiring assertable directly.
        "driver_wf_size": md.wf_size,
        "driver_wg_size": md.wg_size,
        "driver_lds_size": md.ldsSize,
        "driver_pds_size": md.pdsSize,
        "elf": str(spec.elf),
        "elf_sha256": _sha256(spec.elf),
        "staged_elf": str(staged),
        "resource_record": resources,
        "elapsed_s": round(elapsed, 4),
        # Which device produced this result: the functional answer comes from
        # any driver, the timing answer only from cyclesim.
        "driver": spec.driver,
        "simulated_time_ns": simulated_ns,
        "num_mismatches": len(mismatches),
        "first_mismatches": mismatches[:8],
        "z_first8": z[:8],
    }
    # spike_device writes the instruction log next to the *staged* ELF.
    if spec.keep_log:
        log_path = Path(str(staged) + ".log")
        if log_path.is_file():
            result["spike_log"] = log_path.read_text(errors="replace")
    return result


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--local", type=int, default=32)
    ap.add_argument("--profile", default="v1-32",
                    choices=sorted(BUILTIN_PROFILES))
    ap.add_argument("--driver", default=DEFAULT_DRIVER, choices=sorted(DRIVERS))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    spec = LaunchSpec(elf=args.elf, n_elements=args.n, local_size=args.local,
                      profile=args.profile, driver=args.driver)
    result = run_vector_add(spec)
    print(json.dumps(result, indent=2))
    ok = result["num_mismatches"] == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())