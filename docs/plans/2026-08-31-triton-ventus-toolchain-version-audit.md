# Triton/Ventus 工具链身份与版本审计

## 1. 结论与适用范围

本文记录 2026-08-31 当前工作设施的源码、Python、LLVM、Ventus 安装物和
simulator/driver 身份。它是 V1 工具链身份、缓存隔离和发布复现的事实依据；V1
功能范围仍以 [Triton-for-Ventus V1 Scope](2026-08-28-triton-for-ventus-v1-scope.md)
为准，Ventus ABI/设施说明仍以
[Ventus Environment Facility Snapshot](2026-08-25-ventus-env-facility-snapshot.md)
为准。

当前环境**不能作为可发布的统一工具链身份**：当前 Python venv 中 pip Triton 的构建
来源未知；共享 Triton LLVM stable symlink 当前指向与目标 checkout consumer pin 不同的
revision；当前 checkout 尚无 native `libtriton` build；Ventus simulator/driver 缺少一次
统一生成的 build/install manifest。当前环境只可用于审计或明确标记的实验。

本文是 narrative audit。Task 1 生成并验证的
`third_party/ventus/toolchain/version.json` 才是 milestone/release 的最终 machine-readable
identity source；它必须保存本文要求的完整 hash、build-id、路径和 manifest 字段。

## 2. Triton Checkout、Python 与 Consumer LLVM

### 2.1 当前目标 checkout

| 项目 | 审计值 |
| --- | --- |
| 路径 | `/home/weijiale/Code/cuda2rvv/triton` |
| commit | `310241f824a4d8c80305f1b60d773c5e2f098003` |
| describe | `legacy-backend-6142-g310241f824` |
| branch | `main` |
| commit date | `2026-08-13` |
| consumer LLVM pin | `cmake/llvm-info.json`: `b010a18d2b648cab83c83967ff26b8fde11acdc6`, build `1` |
| workflow producer pin | `cmake/llvm-build-info.json`: `941a04e69ee8fe4c7a162b2f1e215aa8df867534`, build `1` |
| native build | 当前没有 native `libtriton` build |

`cmake/llvm-info.json` 是此 checkout 下载/消费 Triton LLVM 的 pin。
`cmake/llvm-build-info.json` 是 workflow 生产 LLVM 包的 pin，不是本地 consumer pin，不能
据此选择本 checkout 的 LLVM cache。

### 2.2 当前运行 Python 与 package provenance

| 项目 | 审计值 |
| --- | --- |
| Python | `/home/weijiale/Code/learn/triton/.venv/bin/python3` |
| Python version | `3.11.15` |
| CPython build | Clang-built CPython |
| pip | `26.2.1` |
| installed Triton | pip Triton `3.7.1`，位于该 venv 的 `site-packages` |
| pip `libtriton.so` SHA-256 | `89448c9bb54bc2fb1e7ef234603ddaef6a29ba4dbcabc6c3c4ada4442d4a0bb9` |
| package origin metadata | 无 `direct_url.json`；pip package provenance unknown |
| venv 所在目录的 checkout（context only） | `/home/weijiale/Code/learn/triton`，commit `a2d75c0e4d7d81f5085053a76d9b971fc6e706e7` |
| context checkout consumer LLVM | `ce3529423abda3fc4ad0b542daa50af21e539a29` |
| context checkout cache | `~/.triton/llvm/llvm-ce352942-ubuntu-x64-1` |

venv 的目录位置和相邻 learn checkout 只是上下文，不能证明 pip Triton `3.7.1` 由
`a2d75c0e...` checkout 构建。当前常规环境中 `PYTHONPATH` 未设置。import mismatch 只在
显式设置 `PYTHONPATH=/home/weijiale/Code/cuda2rvv/triton/python` 后复现：解释器组合了当前
checkout 的 Python source 与缺失/未由当前 checkout 构建的 native module resolution，随后
import 失败。不得把该复现描述为默认环境已自动混用两个 checkout。

共享链接 `~/.triton/llvm/llvm-ubuntu-x64` 当前解析到
`~/.triton/llvm/llvm-ce352942-ubuntu-x64-1`；该 identity 与目标 checkout 的 consumer pin
`b010a18d...` 不匹配，因此 symlink 当前值不能作为目标 checkout 实际 LLVM identity 的
证据。正常 `build_helpers` 流程按 `cmake/llvm-info.json` 构造 revisioned
`llvm-b010a18d-ubuntu-x64-1` 目录，检查其 `version.txt`，必要时校验并下载 archive，将该
revisioned 路径作为 `LLVM_SYSPATH` 传给 CMake，之后才重写 stable symlink。因此不能仅因
symlink 当前指向 `ce352942...` 就断言正常构建必然消费该目录，也不能在未做兼容测试时
声称两个 LLVM package 在语义上 incompatible；可确认的是版本 identity mismatched。

修复策略二选一：为目标 checkout 建立 dedicated venv；或在经过清理和核验的既有 venv
中仔细重装当前 checkout。worktree 必须使用隔离的 `TRITON_HOME`，例如
`/home/weijiale/Code/cuda2rvv/triton/.worktrees/triton-ventus-v1/.triton-home`，用于并发
worktree 的 cache 隔离、可复现性和避免 stable symlink 相互改写；必须核验其中实际
revisioned directory/`version.txt` 对应 `b010a18d...`。构建 Triton 时不得把用户 override
`LLVM_SYSPATH` 指向 Ventus LLVM 16。

## 3. Host 工具

| 工具 | 版本/状态 |
| --- | --- |
| CMake | `3.31.10` |
| Ninja | `1.13.0.git.kitware.jobserver-pipe-1` |
| GCC/G++ | `13.3.0` |
| system `clang`/`opt`/`llc` | LLVM `18.1.3` |
| system `ld.lld` | absent |

system LLVM tools 不是 Ventus tools。生成 Ventus artifact 必须使用下述 install prefix
的绝对路径，不能依赖 `/usr/bin` 中的 `clang`、`opt`、`llc` 或 linker。

## 4. Ventus-env 源码身份

superproject `/home/weijiale/Code/cuda2rvv/ventus-env` 位于
`7e9790708d58ebf697d74fa8dadbaafa1232ca1d`，当前 dirty。子模块快照为：

| Component | Commit | 当前状态 |
| --- | --- | --- |
| LLVM | `d4f2063fe81cbbefda34da60d4cd5c46bce3d231` | clean |
| POCL | `0c9a0bf6c51a922cd483e11281ddad70667b681b` | dirty，一处 source change |
| Driver | `86a4860f6112e2f016d0f86c2653a75337f4f1f2` | dirty，log-level changes |
| GPGPU/RTL | `f5853809f114192b99657e7021d1e50dfe961fe1` | 仅 untracked docs |
| CycleSim | `335ba24d2c7763c87e8c9bfe889c9074763a32f7` | dirty，多处 sources/tests |
| Spike | `abe4323aa551fc5e2f80c46322cc3feb6be49a6d` | dirty，`vftta_vv.h` |
| OCL ICD | `2faf7063c91fdb0f2471f98bf5b2c49c8f907423` | dirty/build artifacts |
| SystemC | `5fc1469339775b54b1fcc2020ac744a58be5b50e` | dirty/generated build artifacts |
| Testcases | `e4630621280554f691580b03b266e0c95b7db47d` | dirty/generated/build artifacts |
| OpenCL CTS | `a08f46ba93e23283df625599eb1cd4507fb21ec1` | clean |
| Rodinia | `b9890b87cb83b108f5cabf3b84568e627299d1dc` | submodule pin |

对 dirty component，commit 只能标识基线，不能标识实际参与构建的源码。实验和 release
manifest 必须额外记录 source diff 或规范化 patch 的 content hash；必要时还应记录生成物
和测试输入 hash。

## 5. Ventus 安装工具链与已验证能力

安装前缀是 `/home/weijiale/Code/cuda2rvv/ventus-env/install`：

- `clang` 报告 LLVM/Clang `16.0.0`，commit `d4f2063...`；`opt`/`llc` 报告
  `16.0.0git`；`ld.lld` 报告 `16.0.0`。
- `llvm-lit` 是 `llvm/build` 中的 `16.0.0dev`，不在 install prefix 中。
- registered targets 是 `AMDGPU`、`X86`、`RISCV`。
- LLVM build 为 `Release`、assertions `OFF`、host compiler 为 GCC/G++，install prefix
  精确为 `/home/weijiale/Code/cuda2rvv/ventus-env/install`。
- `ventus-gpgpu` Clang target 已验证为 `riscv32`、ABI `ilp32`，features 为
  `+m,+a,+zfinx,+zdinx,+zve32f,+zve32x,+zvl32b,+zhinx`。
- barrier lit 和 `kernel_args` 的 `llc` checks 已通过。

Ventus Clang 是 ABI oracle，用于生成和比较 Ventus LLVM 16 可接受的 ABI/IR；它不是
构建 Triton 的 LLVM dependency。始终使用 install-prefix 的绝对 binaries，绝不能使用
`/usr/bin` 的 `clang`、`llc` 或 `opt`。

## 6. Spike、CycleSim 与 Driver 安装物审计

### 6.1 Spike

安装的 `spike --help` 报告 `1.1.1-dev`；源码基线 commit 是 `abe4323...`，但当前
`vftta_vv.h` dirty。installed/build Spike binary SHA-256 相同，均为
`2d28a9f4bbb1104f5d960c760ab00de44faf514644194083887f6e993b8fdf9f`，时间戳为 Aug 10；
`vftta` source 在 Aug 25 修改，因此安装的 Spike **不包含**当前 MMA source patch。

M3 前必须重建 Spike 并记录 patch hash。M1/M2 必须明确决定是否把 clean-commit binary
作为 baseline，并 reset 或 quarantine 该 patch；绝不能用当前 dirty source state 给 Aug 10
安装 binary 贴身份标签。

### 6.2 CycleSim 与 Driver

- CycleSim build `libVentusCycleSim.so` SHA-256 是
  `0028340e64281d44a4b7f1c3fecf31e82ddf373d51f5a99d2c718de80e2d8a53`；install SHA-256
  是 `509109a13a76428c50d0c82cb1e20abd2e0332fcc4b7453ff4184744b4dc0b63`。二者 build-id
  均为 `4fa07406ffe5ac9edc52db956959eddacc1048fc`。
- Driver build `libcyclesim_driver.so` SHA-256 是
  `70859e1165b87603380b3ac8b0143841092b80d33a00f27f20966039b193df7a`；install SHA-256
  是 `65f951fc9ba2c026dfdf56633bf2243976c8e5cc049fe8ce75b5911e2de55cba`。二者 build-id
  均为 `ff16feb05ef32d06da64d8e967e43c681e85835b`。
- 已观察到的 byte difference 是 CMake install RUNPATH relocation：CycleSim 从包含 build
  directory 的 absolute path
  `/home/weijiale/Code/cuda2rvv/ventus-env/cyclesim/build/dependencies/ramulator2`
  改为 `$ORIGIN`；driver 从 install absolute path
  `/home/weijiale/Code/cuda2rvv/ventus-env/install/lib` 改为 `$ORIGIN`。CycleSim 两侧仍包含
  SystemC install RUNPATH。以上不是 stripping 证据，也不是 source-code mismatch 证据。
- Driver install 时间是 2026-08-17，CycleSim core install 时间是 2026-08-21。时间戳只
  证明发生了不同 install events，且当前没有统一 generation manifest；它们不证明两者
  来自不兼容的源码 generation。

identity 必须同时保存 build/install content hash、build-id 和 RUNPATH。release 或 M3 前
仍应从协调的一组 pins 完整 rebuild/install simulator-driver set，并生成统一 manifest；
当前缺少该 manifest，只可作为 experimental identity。

## 7. LLVM 版本边界

严禁把 Triton 所需的 LLVM 24-ish API/object code 与 Ventus LLVM 16 link 到同一进程。
V1 固定边界为：

```text
Triton process + consumer LLVM b010a18d...
  -> producer-local MLIR LLVM Dialect/native IR
  -> Triton Ventus Backend: VentusLLVM16Compatibility
  -> kernel.ventus.ll
  -> subprocess: absolute Ventus LLVM 16 opt
  -> subprocess: absolute Ventus LLVM 16 llc
  -> kernel object
  -> subprocess: absolute Ventus LLVM 16 ld.lld
  -> Ventus ELF
```

跨版本边界不得传 LLVM bitcode。textual LLVM IR 也不是无条件兼容格式。M1 由 Triton
Ventus Backend 拥有 first-class `VentusLLVM16Compatibility` stage（checker API 可命名为
`LLVM16CompatibilityChecker`）：正式输入是 serialization 前 producer-local MLIR LLVM
Dialect/native IR，正式输出是 `kernel.ventus.ll`，即 tested Ventus LLVM 16-compatible
textual subset。它不是 generic arbitrary version translator，不允许 string replacement；
应优先约束 producer translation，serialization 后只能进行 structured、allowlisted
normalization，且不能静默删除 semantic attributes 或 module flags。

contract allowlist 包括 required `riscv32` triple/data layout、RV32/32-bit device pointers、
`ventus_kernel`、`i1/i32/f32`、合法 AS1/AS3、受限只读 AS4、backend-only AS5、已测试
builtin/barrier signatures，以及 ordinary scalar CFG/phi/load/store/GEP/allowlisted calls。
denylist 包括未由 Ventus LLVM 16 contract 测试/支持的 attributes、intrinsics、metadata/
module flags，scalable vectors，i64 device pointer/address，atomics、EH/`invoke`、unsupported
calls、arbitrary `addrspacecast`、AS4 write、user AS5 和 unsupported memory order/scope。
未知 feature 必须 warning 并 fail closed。该检查基于 tested contract，不表示每个 newer
LLVM textual feature 在语义上必然不兼容。

M1 必须分别通过：(1) internal compatibility checker；(2) absolute Ventus LLVM 16 `opt`
parse/verify；(3) absolute Ventus LLVM 16 `llc -mcpu=ventus-gpgpu` target codegen to object。
syntax parse/verify 与 target/codegen semantics 是不同证据，`opt -verify` 单独不足。
Task 12 ELF/Spike execution 是第四个 gate。manifest 保存 `kernel.ventus.ll` content hash、
internal diagnostics、absolute argv、stdout/stderr、exit status、tool identity 和 object hash。
tests 至少包含 OpenCL/Clang golden normalized semantic/ABI fact comparison、Triton-shaped
vector add，以及 unknown attribute、unsupported intrinsic、scalable vector、i64 device
pointer/address、`atomicrmw`、`invoke`/EH、illegal AS cast、AS4 write、user AS5、unsupported
metadata/module flag negatives。
V1 进一步统一为：所有 `addrspacecast` 都拒绝，包括 otherwise well-typed cast；atomic
denylist 必须分别覆盖 `atomicrmw`、`cmpxchg`、atomic load、atomic store、`fence` 和任意
ordering/scope。

Gate 3 object 到 ELF 的 producer-local owner 是
`third_party/ventus/backend/compiler.py`。它必须用 absolute `VENTUS_LLD` argv，显式传入
Task 1 pinned linker script、kernel object、crt0、libclc/workitem inputs、kernel entry/init
handling 和 ELF output；捕获 argv/stdout/stderr/status。具体 installed object/archive 名称
由 Task 1 和 ABI goldens 实测固定，不能在 backend 中猜测。manifest 保存所有 input
identity/hash、object hash、ELF hash 和 link validation。Tasks 11、18、19、20、20B 及未来
compiler path 都必须逐 module 走 `kernel.ventus.ll`、Gates 1-3 和该 link stage，禁止绕过。
shared compiled `ventus-mlir` library 只能在两侧 revision 有意对齐后引入。

## 8. 新窗口进入已验证 Worktree 环境

以下命令是 `feature/triton-ventus-v1` 的权威 shell-entry procedure。每次打开新终端或
新 OpenCode 窗口时执行；不需要重新创建 venv、重新安装依赖或重新构建 Triton。

```bash
cd /home/weijiale/Code/cuda2rvv/triton/.worktrees/triton-ventus-v1
source .venv/bin/activate

export TRITON_HOME="$PWD/.triton-home"

export VENTUS_ROOT=/home/weijiale/Code/cuda2rvv/ventus-env
export VENTUS_INSTALL_PREFIX="$VENTUS_ROOT/install"

export VENTUS_CLANG="$VENTUS_INSTALL_PREFIX/bin/clang"
export VENTUS_OPT="$VENTUS_INSTALL_PREFIX/bin/opt"
export VENTUS_LLC="$VENTUS_INSTALL_PREFIX/bin/llc"
export VENTUS_LLD="$VENTUS_INSTALL_PREFIX/bin/ld.lld"
export VENTUS_OBJDUMP="$VENTUS_INSTALL_PREFIX/bin/llvm-objdump"
export VENTUS_READOBJ="$VENTUS_INSTALL_PREFIX/bin/llvm-readobj"
export VENTUS_NM="$VENTUS_INSTALL_PREFIX/bin/llvm-nm"
export VENTUS_SPIKE="$VENTUS_INSTALL_PREFIX/bin/spike"
```

正常 Triton build shell 只设置上述 cache 和 Ventus 工具的显式绝对路径，不把 Ventus
`bin`/`lib` 注入全局环境。当前已验证环境为：

- Worktree commit `310241f824a4d8c80305f1b60d773c5e2f098003`，branch
  `feature/triton-ventus-v1`。
- Python `.venv/bin/python` 为 Python 3.11.15；editable Triton 为
  `3.8.0+git310241f8`。
- `python/triton/_C/libtriton.so` SHA-256 为
  `0d3cbe7101537ae7485f74edfb4d1c88f3fb4db90cfeb73f2e3caaa860b72a77`。
- Worktree-local LLVM cache 解析到
  `.triton-home/.triton/llvm/llvm-b010a18d-ubuntu-x64-1`。

### 8.1 进入后立即验证

```bash
python -c '
import sys
import triton
import triton._C.libtriton as libtriton

print("python:", sys.executable)
print("triton version:", triton.__version__)
print("triton source:", triton.__file__)
print("libtriton:", libtriton.__file__)
'

readlink -f "$TRITON_HOME/.triton/llvm/llvm-ubuntu-x64"
"$TRITON_HOME/.triton/llvm/llvm-ubuntu-x64/bin/clang" --version
cat "$TRITON_HOME/.triton/llvm/llvm-b010a18d-ubuntu-x64-1/version.txt"

"$VENTUS_CLANG" --version
"$VENTUS_OPT" --version
"$VENTUS_LLC" --version
"$VENTUS_LLD" --version

"$VENTUS_CLANG" \
  -target riscv32 \
  -mcpu=ventus-gpgpu \
  -### \
  -x c /dev/null \
  -c

git status --short --branch
```

验证结果必须满足：

- Python、Triton source 和 `libtriton.so` 均位于当前 Worktree，不得指向
  `/home/weijiale/Code/learn/triton/.venv`。
- Triton consumer LLVM 输出 commit
  `b010a18d2b648cab83c83967ff26b8fde11acdc6`。
- Ventus Clang 输出 commit
  `d4f2063fe81cbbefda34da60d4cd5c46bce3d231`。
- Ventus target dry-run 包含 `-triple riscv32`、`-target-cpu ventus-gpgpu` 和
  `-target-abi ilp32`。
- Git branch 是 `feature/triton-ventus-v1`，且没有非预期改动。

任一身份不满足时必须停止执行计划并报告，不能通过修改 `PYTHONPATH`、全局 `PATH` 或
`LLVM_SYSPATH` 临时掩盖。

### 8.2 首次重建或环境损坏时

只有 `.venv`、`.triton-home` 或当前 checkout native build 缺失/损坏时才执行：

```bash
cd /home/weijiale/Code/cuda2rvv/triton/.worktrees/triton-ventus-v1

/home/weijiale/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3 \
  -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
python -m pip install -r python/test-requirements.txt

export TRITON_HOME="$PWD/.triton-home"
python -m pip install -e . --no-build-isolation -v
```

该构建必须解析当前 checkout 的 `cmake/llvm-info.json` consumer pin
`b010a18d2b648cab83c83967ff26b8fde11acdc6`。不得把
`LLVM_SYSPATH` 指向 Ventus LLVM 16。

### 8.3 Ventus Runtime 测试子 Shell

Ventus runtime tests 若确实依赖 loader/PATH 环境，使用单独 subshell，退出后自动恢复：

```bash
(
  export PATH="$VENTUS_INSTALL_PREFIX/bin:$PATH"
  export LD_LIBRARY_PATH="$VENTUS_INSTALL_PREFIX/lib:$VENTUS_INSTALL_PREFIX/systemc/lib-linux64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  # Run only the selected Ventus runtime test here.
)
```

绝不能在主 Triton build shell 中 source `ventus-env/env.sh`。常规 Triton 开发只使用上述
显式变量和绝对路径；也不要设置用户 override
`LLVM_SYSPATH=$VENTUS_INSTALL_PREFIX` 来构建 Triton。

## 9. Release Identity Rule

一个可发布或可作为 milestone gate 的 toolchain identity 必须绑定：

1. Triton commit、`cmake/llvm-info.json` consumer LLVM hash/build、实际解析的
   `TRITON_HOME` revisioned cache 路径/`version.txt`，以及 Python
   executable/version/build、pip metadata/provenance status 和 native `libtriton` content
   hash；unknown provenance 必须显式编码，不能从 venv 路径推断。
2. Ventus superproject/submodule commits；每个 dirty component 的 reviewed patch/content
   hash，或证明 clean 的状态。
3. Ventus LLVM/Clang/`opt`/`llc`/`ld.lld` 的绝对路径、版本、target/features 和 binary
   content hash。
4. Spike、CycleSim、driver、SystemC 及相关 runtime library 的 source identity、build
   manifest、install manifest、content hash 和 build-id；build/install 不同必须显式记录。
5. target/ABI/profile、artifact ABI、golden/test identity，`kernel.ventus.ll` hash，以及
   internal checker、absolute Ventus LLVM 16 `opt` parse/verify、absolute Ventus LLVM 16
   `llc` object codegen 三个独立结果；object/linker/runtime-input hashes、absolute LLD
   command evidence 和 ELF hash；Gate 4 launcher input/manifest hash、Spike binary identity、
   execution result/status 和 test-result hash。任一必需字段不一致时，artifact 必须在
   compile 或 launch 前拒绝。

M1/M2 milestone、release 和 required CI 必须实际 provision 并运行上述 pinned Spike/
runtime identity。缺失、identity mismatch 或 required suite skip 是 hard environment/gate
failure，不能形成 green acceptance。仅明确 non-milestone 的 convenience local run 可省略
runtime suite并报告 non-green status；CycleSim 仍是唯一 capability-gated optional baseline。

不得用 branch、`latest`、单独的 dirty commit、共享 cache symlink 名称或源码状态替代实际
消费的 binary/artifact identity。本文的 narrative values 也不能替代 Task 1
`version.json` 的 machine-readable final identity。

## 10. 当前阻塞项

| 阻塞项 | 影响 | 解除条件 |
| --- | --- | --- |
| pip Triton provenance unknown，显式 current-source `PYTHONPATH` import 失败 | 不能证明 Python/native 与目标 source 一致 | dedicated venv 或正确重装当前 checkout；记录 metadata 和完整 `libtriton` hash |
| 共享 LLVM symlink 指向 `ce352942...` | 与目标 consumer pin `b010a18d...` identity mismatched，不能作为实际消费证据 | worktree 私有 `TRITON_HOME`；核验 revisioned `b010a18d...` directory/`version.txt` 和 CMake `LLVM_SYSPATH` |
| 当前 checkout 无 native `libtriton` | 不能证明 source/native 一致 | 从当前 checkout 构建并记录 library hash |
| Ventus dirty components 未绑定 patch hash | commit 不能唯一标识源码 | clean checkout 或 reviewed diff/content hash |
| installed Spike 不含当前 MMA patch | M3 MMA simulator identity 错配 | 重建并记录 source patch/binary hash；M1/M2 隔离该 patch |
| M1/M2 pinned Spike/runtime 缺失或 required suite 被 skip | milestone/release Gate 4 无有效 execution evidence | provision matching binaries/inputs and run mandatory Spike; local convenience skip 不计 acceptance |
| V1 linker script/crt0/libclc/workitem concrete installed names 未固定 | object 不能形成可复现 ELF | Task 1/ABI goldens 发现并 hash 绝对路径；`compiler.py` 用 absolute `VENTUS_LLD` argv |
| CycleSim/driver build/install content hashes 因 RUNPATH relocation 不同 | 只记录单侧 hash 会丢失实际 artifact identity | 记录双侧完整 hash、相同 build-id 和 RUNPATH |
| simulator-driver install events 时间不同且无统一 manifest | release/M3 不能证明协调安装过程 | 从协调 pins rebuild/install 并生成统一 manifest |
| LLVM 24-ish 与 Ventus LLVM 16 API/IR 差异 | in-process link/bitcode 不安全；parse success 也不能证明 target codegen | producer-side compatibility checker + hashed `kernel.ventus.ll` + absolute `opt` parse/verify + absolute `llc` object codegen；禁止字符串替换 |
