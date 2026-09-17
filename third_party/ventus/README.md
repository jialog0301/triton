# Ventus Triton 后端

本目录是 Triton 的 Ventus GPGPU 后端（`backend/` 为 Python 插件，`lib/` + `include/` 为
MLIR/LLVM 下降低源，`toolchain/` 记录工具链身份与边界事实，`reference_launcher/` 与
`backend/launcher.py` 负责在 Spike 上执行，`test/` 存放 ABI golden）。

## 1. 定位与范围

- **唯一目标：Ventus GPGPU**（Ventus-only）。不追求向标准 RVV 平台（香山、Ara 等）的可移植性，
  该目标已由 `AGENTS.md` 明确作废；`路线分析.md` 中"转向标准 RVV 1.0 / 纯软件掩码"的提案
  仅作为背景资料，不再约束技术选型。
- **硬件计划：上 FPGA。交付物：论文级研究。** 因此性能结论必须有真实测量支撑：
  Spike 只是功能仿真，cycle 级数值以已安装的 `libVentusCycleSim.so` + `libramulator.so` +
  `libcyclesim_driver.so` 为准——`launcher.py` 现已按名选择驱动（`LaunchSpec.driver`，
  见 §2 与 §8）。
- **Ventus 专有硬件特性是要实现的目标，不是要规避的限制**：分歧硬件（`vbranch`/`join`、
  硬件掩码栈）、寄存器与内存布局、LDS/PDS 共享内存、M4K8N4 MMA（待 M3 契约）。
- 工具链（`ventus-env` 内的 LLVM 16 fork）**可以修改**。`toolchain/README.md` 所述的
  "checked textual LLVM IR boundary" 是当前实现细节，不是永久约束。
- 继续遵守：不在 Triton 核心散落 `isa<Ventus*>`；不做掩盖硬件能力缺失的静默回退。

## 2. 现状（已验证）

| 环节 | 状态 | 证据 |
|---|---|---|
| 编译 `ttir→ttgir→llir→elf` | 完成 | `python/test/unit/ventus/test_pipeline_vector_add.py` |
| Spike 执行 | 完成 | `test_vector_add_on_spike`：指令日志出现 `vfadd`/`vsw12.v`，grid=4，`num_mismatches=0`，驱动 `wf_size=32`/`wg_size=1` |
| 版本化 manifest | 完成 | `test_vector_add_manifest_gate4`；编译 gate 记录 argv/stdout/stderr/exit/tool 哈希 |
| 索引生成（布局无关） | 完成 | 手写 1-D 公式已删除，改用核心 `populateMakeRangeOpToLLVMPattern`（`ttg::toLinearLayout` + `emitIndices`）；2-D tile 与 2-warp 均在 Spike 上 `num_mismatches=0` |
| 单测 | 28 项：27 通过 | `python/test/unit/ventus/`；唯一红是 `test_toolchain_identity` 的外部 gpgpu 组件漂移（见 2.1） |
| cycle 级测量 | 打通（基线数字待补） | `LaunchSpec.driver` 可选 spike / cyclesim / rtlsim / gvm / auto；cyclesim 已能跑 grid>1 并回报模型时间：n=32/64/100 → 15830/20780/31910 ns（10 ns 时钟，见 §4.1） |
| ELF 体积 | 26 KB | `--gc-sections -u <kernel>` 裁剪前为 14 MB |
| 后端可用性 | 目标 `GPUTarget("ventus","ventus-gpgpu",32)`；`triton.compile` 可用 | `driver.py` 仍是 stub，`kernel[grid](...)` 尚不可用 |

固定链接输入：`crt0.o`（提供 `_start`）、`riscv32clc.o`（work-item 内建）、`libworkitem.a`、
`ldscripts/ventus/elf32lriscv.ld`，全部按绝对路径调用。

### 2.1 与上游的关系（已 rebase 到 origin/main）

`feature/triton-ventus-v1` 已 **rebase 到 `origin/main`**：`0 落后 / 4 领先`，我们的 4 个提交
直接坐在最新上游之上。`main` 分支仍停在旧的 `af6b189cf5`（按既有习惯不动它）。

rebase **零 git 冲突**——我们与上游的文件交集只有 `.gitignore`、`AGENTS.md`、`CMakeLists.txt`，
且多为纯追加。此前用 cherry-pick 试过的 3 个提交在 rebase 时被 git 自动识别为"已应用"并跳过，
印证了 patch 等价的判断。

由此一次拿到了先前**无法单独 cherry-pick** 的两条布局修复：

| 提交 | 内容 | 为什么单独 pick 拿不到 |
|---|---|---|
| `92ff4362da` | Fix operand layouts when absorbing view conversions (#11758) | 依赖中途重构（`canUseResultEncoding` 返回值由 `bool` 变为 `std::optional<SmallVector<OpOperand *>>`） |
| `51593ac6b6` | Relax broadcast layouts (#11759) | 给 `DialectInferLayoutInterface` 加方法，牵动 NVIDIA/AMD/Gluon 三份实现 |

#### rebase 暴露了 git 看不见的破坏面

git 层面零冲突，但**编译时**我们的 `TritonGPUToLLVM.cpp` 撞上两处核心 API 变更：

| 变更 | 来源 | 我们的修法 |
|---|---|---|
| `TargetInfoBase::getMulhiFuncName` 被删除 | `0f77b09309 [GPU] Expose unsigned mulhi to LLVM optimization` (#11768) | 删掉该 override |
| `populateElementwiseOpToLLVMPatterns` 去掉 `targetInfo` 参数 | 同期 elementwise 重构 | 调用处去掉该实参 |

**结论：git 冲突 ≠ 破坏面。** 我们对核心 API 的使用（`TargetInfoBase` override、`populate*`
签名、`packLLElements` 等）只在编译时才暴露漂移，所以每次升级都必须**构建 + 跑全套测试**，
不能只看 rebase 是否干净。

#### 若只想要上游某个提交：先数"中间漂移"

```bash
git log --oneline <我们的基点>..<目标提交>^ -- <该补丁触及的文件>
```

为 0 才有资格单独 cherry-pick；不为 0 说明补丁期望的行上下文已被上游改动，冲突**与自己的改动
无关**（我们根本没碰那些文件）。实测：`7a7f10458e` 触及的 5 个文件漂移 0 次 → 干净；
`51593ac6b6` 的 11 个文件漂移 1–11 次 → 必冲突，只能走 rebase。

**两种操作的冲突面方向相反**：cherry-pick 的冲突大小正比于"补丁文件在中间被上游改动的次数"
（我们无法控制）；rebase 的冲突面是"我们改过的文件 ∩ 上游改过的文件"，我们只有 3 个。
**因此要取上游 core 里的东西，rebase 才是正路。**

## 3. 关键实现决策（含理由）

1. **`ventus_kernel` 调用约定在文本层注入。** 该约定在 Ventus LLVM 16 中是 CC 104，但消费端
   LLVM 的 104 是 `amdgpu_cs_chain` 并打印该关键字，Ventus 解析器拒收。因此 C++ finalize
   不设 CC，由 `make_llir` 用确定性正则改写入口函数的 `define` 行。
2. **谓词访存使用 `llvm.cond_br` 菱形，不用 `llvm.masked.*`。** masked 内建在两版 LLVM 间
   参数个数不同（消费端删除了对齐参数），且翻译器用自身的内建表校验声明，故无解。
3. **链接必须 `--gc-sections -u <kernel>`。** `riscv32clc.o` 是 22 MB 单体对象（非归档），
   不裁剪则产物 14 MB；而内核无任何引用者（运行时按符号名查找），漏掉 `-u` 会被静默删除。
4. **两类 lowering 是后端私有**：全局 `tt.load/tt.store`（核心只处理 local/scratch）与
   浮点 elementwise（核心只注册整数集）。见 `VentusLoadStoreOpToLLVM.cpp`。
5. **寄存器只上报、不分配。** VGPR/SGPR 由 Ventus LLVM 16 的 `llc` 分配；Triton 侧只能通过
   布局（`sizePerThread`/`warpsPerCTA`）影响压力，并读 VRES 记录上报。
6. **索引生成已 LinearLayout 化（2026-09-17）**：`tt.make_range` 不再由后端私有公式降低，
   改走核心 `populateMakeRangeOpToLLVMPattern`，即 `ttg::toLinearLayout(tensorTy)` +
   `emitIndices(...)`；同时注册核心 `populateViewOpToLLVMPatterns`（`expand_dims`/`broadcast`/
   `splat`/…），后端不再自带 splat lowering。三条硬门槛（rank 必须为 1、编码必须恰是
   `BlockedEncodingAttr`、公式里 `warp * 32` 隐含 1-D）随之消失。见第 6 节。

## 4. 硬件资源模型（RTL 事实）

来源：`gpgpu/ventus/src/top/parameters.scala`、`toolchain/version.json` 的
`resource_unit_contract`、以及产物中的 VRES 记录。

| 资源 | 单位 | 聚合粒度 | 上限（RTL 默认 4 bank、8 warp） | 备注 |
|---|---|---|---|---|
| VGPR | wavefront-wide register slots | per wavefront | 1024（`num_vgpr = 128 * num_warp`） | wavefront 宽，随 warp 数缩放 |
| SGPR | 32-bit register slots | per wavefront | 2048（`num_sgpr = 256 * num_warp`） | 同上 |
| **LDS** | bytes | **per CTA（work-group）** | 131072（128 KB，`sharemem_size`） | **片上共享内存**；`LDS_BASE = 0x70000000`，LLVM 侧为 `addrspace(3)` |
| PDS | bytes | per work-item（上限按 per-wavefront 记录） | 131072（`4096 * 32`） | 每个 work-item 的数据空间；驱动接收 `pdsSize`/`pdsBaseAddr` |

**LDS（Local Data Share）就是这颗 GPGPU 的"共享内存"**：同一 work-group 内所有 work-item 可见，
属于片上 SRAM，不占 global 内存带宽，因此是 tiling / reduction / MMA 操作数暂存的基石。
ABI golden 中它的形态是模块级 `addrspace(3)` 全局量：

```llvm
@barrier_local.values = internal unnamed_addr addrspace(3) global [64 x float] undef, align 4
call void @llvm.riscv.ventus.barrier(i32 1)
```

驱动启动参数（`launcher.py` 注释与 `meta_data` 记录）：`wf_size`、`wg_size`、`kernel_size`、
`ldsSize`、`pdsSize`、`pdsBaseAddr`、`knlbase`、`sgprUsage`、`vgprUsage`。

VRES v1 资源记录：24 字节，字段序 `[vgpr, sgpr, lds, pds]`，由 Ventus LLVM 16 后端依据
`ventus_kernel` 约定自动生成，落在 `.ventus.resource.<kernel>` 段；`backend/manifest.py` 解析后
填入 `md.resources`。**注意 LDS/PDS 的声明值尚需与真实需求一致**，当前 `metadata["shared"]` 仍写死 0。

### 4.1 驱动与 cycle 级测量（2026-09-17 打通）

**验证路径的决策（2026-09-17）：正确性验证继续用 spike。** spike 快、无版本依赖、无需
nocache/时钟等前提，因此它是唯一的功能回归目标；cyclesim 与 rtlsim 只作为**按需测量**手段
（`LaunchSpec.driver` 显式选择），不进日常验证回路。理由见 §4.2。

`launcher.py` 按名选择驱动（`LaunchSpec.driver`，`DRIVERS` 表）：`spike`（功能）、`cyclesim`
（**计时真值**）、`rtlsim`、`gvm`、`auto`。五者导出同一套 `vt_*` API（`llvm-nm` 核对过），
因此切换只是换一个 `.so`；结果里新增 `driver` 与 `simulated_time_ns` 字段。

计时数字取自模拟器自身：所有设备的 `vt_dump_perf` 都是空实现（`return 0`），所以 launcher 直接
dlopen `libVentusCycleSim.so` 调 `ventus_cyclesim_get_time(dev)`（`vt_dev_open` 返回的句柄就是
`ventus_cyclesim_t *`）。时钟周期 10 ns（`cyclesim/src/parameters.h:42`），故
`cycles = simulated_time_ns / 10`。实测（同一 ELF，`num_warps=1`，`mismatches=0`）：

| n | grid | simulated_time_ns |
|---|---|---|
| 32 | 1 | 15830 |
| 64 | 2 | 20780 |
| 100 | 4 | 31910 |
| 128 | 4 | 31940 |

**两条 spike 掩盖的元数据事实**（都只在 grid>1 暴露，见 §8）：cyclesim 按
`pdsSize * wf_size * wg_size * grid` 在 `pdsBaseAddr` 处圈定私有内存（并在结束时断言该范围确实按此
大小分配过），而 `spike_device::run` 忽略该地址、给每个 work-group 固定 `0x10000000`；cyclesim 的
`driver_metadata_t` 比 spike 多一个尾部 `kernel_name` 指针，不填就是野指针（表现为内核名乱码，
grid>1 时构造 `std::string` 直接 abort）。

**一个进程只能开一次 cyclesim**：`ventus_cyclesim_init` 会调 `sc_set_time_resolution`，第二次报
`(E514) ... simulation running` 并终止进程。launcher 用 `sys` 上的进程级标志提前抛 Python 异常
（模块级全局会被后端发现测试的清 `sys.modules` 重置），因此测试里只有一次 cyclesim 启动。

### 4.2 RTL 模型（Verilator）当前能跑到什么程度（2026-09-17 实测）

`driver="rtlsim"` 走同一套 `vt_*` API，模型时间同法可读（`ventus_rtlsim_get_time`）。单位同样是
**ns**：Verilator 无 `timescale` 指令、用默认 1 ns timeunit，`HALF_CYCLE_TIME = 5`
（`ventus_rtlsim_impl.cpp:27`）→ 10 ns/cycle，与 cyclesim 的时钟一致。

同一 ELF（vector_add，`num_warps=1`）实测：

| grid | 结果 | wall |
|---|---|---|
| 1 | 通过，`mismatches=0`，模型时间 34075（内核 finish @9075） | 0.2 s |
| 2 | 通过，`mismatches=0` | 0.2 s |
| 3 / 4 | **abort**：`Assertion failed: UNDEFINED INSTRUCTION @ SM 0 warp 1 PC 0x90004000` | — |

grid=3/4 的失败是**多波分发**问题：第 3 个 work-group 从 metadata 缓冲区地址（`CSR_KNL`）开始取指，
而 RTL 的 `num_sm = 2`（`gpgpu/ventus/src/top/parameters.scala:7`），一波只放得下 2 个 CTA。
**所以 RTL 目前的可跑上限是 grid ≤ 2**，我们现有测试用的 grid=4 形状还跑不了。

**关键是这份 RTL 是 nocache 变体**：`gpgpu/sim-verilator-nocache` 去掉了 L1 D-cache 与 L2
（保留 L1 I-cache，日志自报 "you are running no-cache version RTL"）。因此"谁更接近真实硬件"要分轴：

| 轴 | 更保真 | 说明 |
|---|---|---|
| 核心微架构（流水线、warp 调度、寄存器/记分牌、分歧） | **RTL** | 硬件描述本体；cyclesim 是 SystemC 性能模型 |
| 内存层级（L1D/L2/DRAM 计时） | **cyclesim** | 带 ramulator；装着的 RTL 去掉了 cache。缓存版 `gpgpu/sim-verilator/` 在树里，但**没有**装成 `libVentusRTL.so` |

**数字不可直接比**：RTL 报的是"内核 finish + 仿真收尾"，cyclesim 报的是自身结束时刻，同一 kernel
RTL 9075 ns vs cyclesim 15830 ns 不能当模型偏差——要得到模型校准结论，必须先统一测量窗口
（内核 start→finish）。这是 P0 的下一项。

rtlsim 也是**一进程一次**：第二次 `vt_dev_open` 因 driver 里
`spdlog::stdout_color_mt("ventus")` 重复注册而 `terminate`（driver 侧一行可修：已存在就取旧的）。

**因此 RTL 不参与日常验证**（见 §4.1 的决策）：装着的模型是 nocache 变体、grid 上限 2、没有版本记录
（`gpgpu` 树正在被实时编辑），任何一条都足以让"每次提交跑 RTL"失去意义。套件里保留
`test_vector_add_on_cyclesim` / `test_vector_add_on_rtl` 两个小测试，只用于守护驱动接线本身
（各 ~0.2 s）；功能正确性的权威仍是 spike。

### 4.3 跨实现对照：Triton vs OpenCL（2026-09-17 建立）

工具：`tools/vecadd_baseline/`（`kernel.cl` + `main.cc` 主机程序 + `measure.py` 双臂驱动器）。
两臂跑**同一形状、同一模拟器**：n 个元素、`local` 个 work-item 一个 work-group、
`ceil(n/local)` 个 work-group、尾部用 `gid < n` 兜住；输入同为 `a[i]=i+1`、`b[i]=2i`。
OpenCL 臂经 POCL 的 ventus device 走 `libventus_driver.so`（auto_select），因此
`VENTUS_BACKEND` 与 Triton 臂选同一个模拟器（`spike`/`cyclesim`/`rtlsim`）：

```bash
# POCL 的调用约定与 rodinia 的 runner 一致
PATH=$V/bin:$PATH LD_LIBRARY_PATH=$V/lib OCL_ICD_VENDORS=$V/lib/libpocl.so \
POCL_DEVICES=ventus VENTUS_BACKEND=cyclesim ./vecadd_baseline kernel.cl 1024 32

# 一次跑两臂（各自一个进程，见下）
.venv/bin/python third_party/ventus/tools/vecadd_baseline/measure.py --n 1024 --local 32 --backend cyclesim
```

**首个数字**（n=1024、local=32、grid=32，两臂 `num_mismatches=0`）：

| 臂 | 模拟器 | 内核模型时间 | 折算 cycles（10 ns/cycle） |
|---|---|---|---|
| Triton | cyclesim | 183015 ns | 18301.5 |
| OpenCL (POCL) | cyclesim | 181465 ns | 18146.5 |
| Triton / OpenCL | spike（仅功能） | — | 两臂均通过 |

差 1550 ns（≈0.85%），**但这个差目前不能当"代码质量差异"**——两臂的驱动侧资源声明不同：
POCL 在 `pocl_ventus.cc` 里硬编码 `ldssize=0x1000`、`pdssize=0x10000000` 等，而我们的 launcher
用产物里的 VRES 记录（vgpr=7、sgpr=8）并把 pds 按 grid 摊开。要做**受控对照**，必须先把
`pdsSize`/`ldsSize`/`sgprUsage`/`vgprUsage` 在两臂对齐（我们这边可用 `LaunchSpec` 的同名字段覆盖），
而这正对应 `version.json` 里 `hard_coded_resource_consumption_present` 那条 release blocker。

其余限制：spike 只做功能、没有 cycle 数；RTL 覆盖不了 grid=32（§4.2 的 grid ≤ num_sm=2）；两臂各自的
"总模拟时间"窗口不同（`measure.py` 因此统一取模拟器日志里的 `initialized→finished` 内核窗口）。

## 5. 与 NVIDIA / AMD 的 pass 对照（MMA 视角）

核心的 `accelerate-matmul` 是 NVIDIA 专属（`AccelerateMatmul.cpp` 只有 `getMMAVersionSafe` 与
`dyn_cast<NvidiaMmaEncodingAttr>`），因此 **AMD 自写了 1767 行的 `AccelerateAMDMatmul.cpp`**。
第三个后端做 MMA 的官方姿势就是如此。

| pass | 归属 | 我们的做法 | 规模参考 |
|---|---|---|---|
| `accelerate-matmul` | 核心(NVIDIA) / AMD 自写 | **必须自写** `VentusAccelerateMatmul`（选 tile、挂 MMA 与 dot_operand 编码） | 200–400 行 |
| `optimize-dot-operands` | 核心版无厂商分支 | **先直接用核心版**，不够再 fork | 0 |
| dot 操作数搬运 | NVIDIA ldmatrix/TMA、AMD buffer_load | 从 LDS 取操作数（依赖 LDS 支持） | 100–200 行 |
| dot lowering | nvidia ≈2600 行 / AMD ≈2000 行 | 自写，只做 M4K8N4 一条路 | 300–600 行（inline asm 路线） |
| FMA fallback | AMD `FMA.cpp` 126 行 | **建议先做**：MMA 未就绪前保证 `tt.dot` 语义正确 | ≤100 行 |
| LDS 分配 + membar | 核心 `add_allocate_shared_memory` + `ModuleMembarAnalysis` | 直接用核心，仅补 `TargetInfoBase` 钩子 | 0 |
| 布局类核心 pass（coalesce / remove-layout-conversions / reduce-data-duplication / optimize-thread-locality） | 核心 | **不用重写**（前提：编码可被 LinearLayout 表达） | 0 |
| TMA / mbarrier / tmem / cluster / warp-spec / tcgen05 | 各自专属 | **不需要**（Ventus 无对应硬件） | 0 |

## 6. LinearLayout 的使用方式

**现状（2026-09-17）：第 1 步已完成。** 索引生成（`tt.make_range`）与视图类
（`expand_dims`/`broadcast`/`splat`/…）都走核心 populate，后端不再持有任何布局公式：

| 原后端私有实现 | 现状 |
|---|---|
| `VentusMakeRangeOpConversion`（rank==1 门槛 + `dyn_cast<BlockedEncodingAttr>` 门槛 + 手写 `r + lane*spt + warp*32*spt`） | 删除；改用核心 `populateMakeRangeOpToLLVMPattern`（`ttg::toLinearLayout` + `emitIndices`） |
| `VentusSplatOpConversion`、`VentusConstantSplatOpConversion` | 删除；核心 `populateViewOpToLLVMPatterns` 覆盖（且支持指针 bitcast 与位宽打包，是超集） |
| 无 `expand_dims`/`broadcast` lowering | 由 `populateViewOpToLLVMPatterns` 提供，这是 2-D tile 的必要条件 |

留下的后端私有 lowering 只剩两类，且都与布局无关：全局 `tt.load/tt.store`
（逐元素、`unpackLLElements`）与浮点 elementwise。

#### 实测：迁移后的能力与证据（2026-09-17）

| 场景 | 布局特征 | 结果 |
|---|---|---|
| 1-D vector add（`num_warps=1`） | blocked、仅 lane/register basis | Spike `num_mismatches=0`（原有基线） |
| 2-D tile `4x8`（`tl.arange(M)[:,None]`/`[None,:]`） | rank-2 blocked `order=[1,0]`、两个 `#ttg.slice` 子编码 | 编译通过 + Spike `num_mismatches=0`（`test_2d_tile_vector_add_on_spike`） |
| 1-D `num_warps=2`（`v1-64`，N=128 全活跃） | blocked + warp basis | Spike `num_mismatches=0`（`test_vector_add_two_warps_on_spike`） |

2-D tile 的 LLIR 里能直接读到 LinearLayout 的产物（lane 3–4 位 → dim0，lane 0–2 位 → dim1，
再 `m*8 + n`）：

```llvm
%10 = or i32 0, %9          ; 原样是 `or disjoint`，见下
%11 = and i32 %10, 24       ; dim0: lane>>3
%12 = lshr i32 %11, 3
%22 = and i32 %21, 7        ; dim1: lane&7
%28 = mul i32 %17, 8
%29 = add i32 %28, %27      ; offs = m*8 + n
```

#### 迁移带出的第二个文本边界（已解决）

`applyLinearLayout` 对可证明位不相交的操作数用 `or disjoint`（MLIR `llvm.or` 的 disjoint 标志）。
该关键字是 LLVM 17 才有的，Ventus LLVM 16 的 parser 直接报 `expected type`。修法在
`backend/compiler.py`：`_strip_or_disjoint()` 在写文本 IR 时去掉该关键字（语义等价——disjoint 是
操作数承诺，不是指令/取值的一部分），仍由 elf 阶段的 `opt -passes=verify` 把关。详见
`toolchain/README.md` 的 "Textual IR Boundary Facts"。

#### 下一步（仍是第 2、3 步）

1. ~~**改造现有 lowering（零核心改动）**~~：已完成，见上表。
2. **MMA 编码属性**：新增 `VentusMmaEncodingAttr`，放在**核心** `TritonGPUAttrDefs.td`（NVIDIA
   与 AMD 的 MMA 编码都在核心：`NvidiaMmaEncodingAttr:1272`、`AMDMfmaEncodingAttr:902`、
   `AMDWmmaEncodingAttr:1075`），并带 `MmaEncodingTrait`。
   - `MmaEncodingTrait` 要求**两个**方法：`getRepOrderForOperand` 与 `dotOperandToLinearLayout`
     （后者带默认实现 `report_fatal_error("this MMA layout is not a valid dot-operand parent")`
     —— 即**不实现也能编译通过，运行时才炸**，属静默陷阱）；
   - 另需实现 `DistributedEncodingTrait`：`getRepOrder`、`getTotalElemsPerThread`、
     `getElemsPerThread`、`toLinearLayout`、`getLinearLayout`、`basesPerDim`、`orderPerDim`、`getOrder`；
   - `toLinearLayout` 实现在核心 `LinearLayoutConversions.cpp`，照 `NvidiaMmaEncodingAttr::toLinearLayout`
     （`:959`）或 `AMDMfmaEncodingAttr::444` 的写法；
   - 完成后核心的 allocation / convert-layout / dot-operand 优化会**自动**理解该编码。
3. **LDS 布局**：核心现已把 shared 布局降级改为接口分派（`dyn_cast<SharedEncodingTrait>`，
   `LinearLayoutConversions.cpp:1251`），因此**可以给 Ventus 实现自己的 shared 编码属性**
   （`SharedEncodingTrait` 的方法为 `getAlignment` + `toLinearLayout`）。起步阶段更省事的选择
   仍是核心 `SharedLinearEncodingAttr`（`LinearLayoutConversions.cpp:1238`）。

#### 迁移前的等价性实测（保留作为判据记录）

迁移**之前**用 `triton._C.libtriton.linear_layout` 的 `LinearLayout.from_bases(...).apply({...})`
对手写公式 `r + lane*spt + warp*32*spt` 做了逐点比对，四种配置**零不等点**——这是"迁移不改变语义"
的机器证据（公式现已删除，本表仅作为迁移决策的依据保留）：

| 配置 | bases | 不等点 |
|---|---|---|
| `spt=1, tpw=32, warps=1` | `register [], lane [[1],[2],[4],[8],[16]], warp []` | 0 |
| `spt=2` | `register [[1]], lane [[2],…,[32]]` | 0 |
| `spt=4` | `register [[1],[2]], lane [[4],…,[64]]` | 0 |
| `warps=2` | `lane [[1],…,[16]], warp [[32]]` | 0 |

**曾被卡住的能力边界（现已解锁）**：2-D tile kernel 过去编译失败——

```
failed to legalize 'tt.make_range' : () -> tensor<4xi32,
  #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1,1],
            threadsPerWarp = [4,8], warpsPerCTA = [1,1], order = [1,0]}>}>
```

原因就是后来被删掉的两道硬门槛（`rank != 1`、`dyn_cast<BlockedEncodingAttr>`）。核心侧
`SliceEncodingAttr::toLinearLayout`（`LinearLayoutConversions.cpp:1048`）本就存在，所以这次
只需改索引生成的来源。**迁移面确实只到索引生成**：load/store 走 `unpackLLElements`，与布局和
rank 无关；`expand_dims`/`broadcast` 是 2-D tile 的第二个前置条件，由核心 view pattern 一并解决。

工具函数：`LinearLayout::identity1D`、`sublayout`、`invertAndCompose`、`basesPerDim`
（`include/triton/Tools/LinearLayout.h:341-769`）；调试时 `llvm::errs() << ll`。

## 7. 计划与优先级

| 序 | 内容 | 依赖 | 论文价值 |
|---|---|---|---|
| **P0** | 测量闭环：~~驱动可选 + cycle 级数字~~（§4.1）、~~OpenCL(POCL) 基线~~（§4.3 已建立，首个受控数字待补：先对齐两臂的资源声明）。*已降级*：RTL 多波分发、cyclesim↔RTL 模型校准——验证路径已定 spike（§4.1） | — | 使后续所有结论可证 |
| **P1** | 布局/向量化/占用率：~~LinearLayout 化索引~~（已完成，见第 6 节）、`sizePerThread`、`num_warps`、coalesce；去除逐元素标量访存 | — | "Triton 生成 vs OpenCL/手写"主结果 |
| **P2** | LDS + barrier：`add_allocate_shared_memory` + membar + 实现 `storeDShared`/`loadDShared`；barrier 走文本注入或 inline asm | 核心基建已备 | 支撑 tiling/reduction/MMA |
| **P3** | 分歧硬件（`vbranch`/`join`/掩码栈）与现有软件谓词路径做 A/B | 工具链（新内建/CC） | **论文核心差异化**（软件谓词一臂已实现） |
| **P4** | LLVM 版本对齐 → 一等 Ventus 内建，移除文本边界与相关 hack | 工具链（大工程） | 工程债清理，P3/P5 的前提 |
| **P5** | M4K8N4 MMA：编码属性 + accelerate-matmul + dot lowering | **M3 的指令/寄存器布局契约** | 性能上界 |

## 8. 陷阱清单

- **MMA 操作数寄存器布局必须与 LinearLayout 严格一致**，否则是静默算错而非崩溃。先用单个已知
  tile 的微基准在 cyclesim 上验证映射，再放大。
- `MmaEncodingTrait::dotOperandToLinearLayout` 的默认实现是 `report_fatal_error`——**不实现能编译
  通过，运行时才炸**。P5 必须实现它。
- shared 布局**不再**是硬编码 else-if 链（该结论在旧版本成立）：`58100bd95b` 之后改为
  `SharedEncodingTrait` 接口分派，树外 shared 编码可正常扩展。
- 与上游 patch 等价的 cherry-pick 在将来 rebase 时会被自动跳过；**但依赖中途重构的提交不能单独
  cherry-pick**（见 2.1），强行为之等于在旧 API 上手工移植语义。
- 消费端 LLVM 的翻译器按自身内建表校验声明：Ventus 专有内建无法直接声明（`masked.*`、
  `llvm.riscv.ventus.barrier` 皆属此类），须走文本注入或 `llvm.inline_asm`。
- **`or disjoint` 不能跨到 Ventus LLVM 16**（LLVM 17 才有该关键字）。任何走 `emitIndices` 的
  lowering 都会带上它，`backend/compiler.py::_strip_or_disjoint()` 在文本层去掉。同类风险：消费端
  LLVM 若引入新的标志/关键字（如 `icmp samesign`），会在 elf 阶段的 `opt -passes=verify` 才暴露。
- `--gc-sections` 必须配 `-u <kernel>`，否则内核被静默删除（已有测试双向守护）。
- **PDS 是"每 work-item 的大小、每次 launch 的范围"**：`pdsBaseAddr` 处必须按
  `pdsSize * wf_size * wg_size * grid` 连续分配（`cyclesim/src/task.cpp:50` 如此计算并断言）。
  spike 忽略该地址，所以只按单 work-group 分配在 spike 上永远看不出问题——只有 grid>1 的
  cyclesim 才暴露。
- **`driver_metadata_t` 各设备不同构**：cyclesim 尾部多一个 `kernel_name` 指针，spike/gvm/rtlsim
  没有。漏填会让 cyclesim 打印乱码内核名，并在 grid>1 时以
  `basic_string: construction from null is not valid` abort。launcher 的 `_MetaData` 按最长形态声明。
- **cyclesim 一个进程只能跑一次**（SystemC 二次初始化：`(E514) set time resolution failed`），
  且驱动忽略 `vt_ready_wait` 的 timeout（`(void)timeout;`），失控的启动只能用外部超时兜住。
- **rtlsim 同样一进程一次**，但原因是 driver 重复注册 spdlog logger（`logger with name 'ventus'
  already exists` → terminate）。launcher 已提前抛异常。
- **RTL 的 grid 上限 = `num_sm` = 2**：超过一波的 grid 会在模型内 abort
  （`UNDEFINED INSTRUCTION ... PC 0x90004000`，第 3 个 work-group 从 metadata 缓冲区取指）。多波分发
  修好之前，RTL 只能验证 grid ≤ 2 的形状。
- **装着的 `libVentusRTL.so` 是 nocache 变体**（L1D/L2 被移除）。它的 cycle 数只代表核心微架构，
  不能当 cache-accurate 的硬件性能数字；引用时须写明变体。
- **不同模拟器的"总时间"窗口不同**（内核 finish 时刻 vs 仿真收尾），跨模拟器比数字前先统一窗口。
- `version.json` 尚未纳入版本控制，新克隆会缺该文件（`VentusBackend.hash()` 与身份测试都依赖它）。

## 9. 未决 / 外部依赖

- **M3**：M4K8N4 的 source/generated-RTL/testbench 记录；Spike 中 dirty 的 `vftta_vv.h` 实验需按新
  Chisel 设计重建。P5 的关键路径。
- `driver.py` 仍是 stub：Triton 原生启动路径不可用，目前只能 `triton.compile` + 参考启动器。
- 启动形态：`supported_local_sizes = [[32,1,1],[64,1,1]]`，`mma_local_size = [32,1,1]`，grid 1 维；
  `v1-64`（`num_warps=2`）profile 已端到端验证（`test_vector_add_two_warps_on_spike`，N=128）。
- `metadata["shared"]` 目前写死 0，需在 P2 换成真实分配结果。
