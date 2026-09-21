# Triton-for-Ventus V1 范围规范

## 1. 状态与事实来源

**状态：normative proposed specification；compiler integration not yet implemented。**

本文是 Triton-for-Ventus 第一版本的权威范围规范（authoritative V1
specification）。实现事实以 [Ventus Environment Facility Snapshot](2026-08-25-ventus-env-facility-snapshot.md)
为准；Triton/Python/LLVM cache、Ventus installed binaries 和 release identity 以
[Triton/Ventus Toolchain Version Audit](2026-08-31-triton-ventus-toolchain-version-audit.md)
为准；持久目标架构以 [Shared Ventus MLIR Design](2026-08-20-triton-ventus-iree-shared-mlir-design.md)
为准；具体任务以 [Implementation Plan](2026-08-20-triton-ventus-iree-implementation.md)
为执行参考。documentation-sync design/plan 是已完成的历史同步记录，不属于持久
architecture authority chain。
Architecture/design 文档描述目标设计，facility snapshot 描述已检查 checkout
中的实现事实。当前 `ventus-env` 含 dirty Spike、Driver、CycleSim、POCL 及
generated/build artifacts；未固定、未审查的
本地改动不是稳定 ABI 或 release facility。

已完成的 Ventus RTL compatibility review 表明：M1/M2 的 32-lane scalar-SIMT、
AS1/AS3、full work-group barrier 和 ordinary FP32 lowering 方向与检查到的设施基本
兼容，但仍须完成本文规定的 runtime、resource 和 bounded-divergence hardening；MMA
仍是 proposed/unfrozen，不能由“RTL 中存在 `VFTTA_VV`”推导为已支持。

## 2. V1 定义与端到端闭环

V1 是一个可选择的 Ventus Triton backend，能够把受支持的 Triton kernel 编译为
Ventus ELF，并通过参考 launcher 按 milestone gate 执行和验证：M1/M2 必须在 Spike
通过，M3/M4 必须同时在 Spike 和 RTL simulation 通过；CycleSim 仅在报告精确能力时
运行，否则记录 capability-gated skip。V1 的算子主线是 correctness-first；性能专用
路径只在有明确 target contract 时启用。

```text
Triton Python DSL
  -> TTIR
  -> TTGIR
  -> conceptual TritonGPUToVentus / shared Ventus MLIR stages
  -> producer-specific adapter
  -> VentusLLVM16Compatibility
  -> kernel.ventus.ll + pinned Ventus LLVM tool invocation
  -> Ventus LLVM (-mcpu=ventus-gpgpu)
  -> ELF + resource metadata + versioned artifact manifest
  -> reference launcher / Ventus C driver API
  -> Spike / CycleSim / RTL simulation
```

上图中的 MLIR stages 是概念 pipeline，不表示 V1 已有或要求一个 shared compiled
C++ MLIR library。由于检查到的 Ventus LLVM 是 LLVM 16 fork，而当前 Triton checkout
的 consumer LLVM 由 `cmake/llvm-info.json` 固定为 `b010a18d...`（
`cmake/llvm-build-info.json` 只是 workflow producer pin），V1 唯一允许的 concrete
boundary 是 producer-specific adapter 加 textual LLVM IR 和 absolute-path Ventus LLVM
16 subprocess invocation；不得 link 两侧 LLVM object code，也不得跨边界交换 LLVM
bitcode。`VentusLLVM16Compatibility` 是 Triton Ventus Backend 拥有的 first-class
producer-side stage：其正式输入是 serialization 前 producer-local MLIR LLVM Dialect/native
LLVM IR，正式输出是 `kernel.ventus.ll`，即经过检查的 Ventus LLVM 16-compatible textual
LLVM IR subset。它不是任意 LLVM version translator，不能用字符串替换修补 IR；实现必须
优先约束 MLIR LLVM Dialect/native translation 的输出，serialization 后只允许结构化、
allowlisted normalization。`kernel.ventus.ll` 还必须分别通过 absolute-path Ventus LLVM 16
`opt` parse/verify 和 `llc -mcpu=ventus-gpgpu` object codegen；`opt -verify` 单独不足以证明
target/codegen semantics。只有两边 revision 有意对齐后，才可引入 shared compiled C++
MLIR library。Triton 负责 operator DSL、TTIR/TTGIR、layout 和 kernel artifact；Ventus LLVM
负责 ABI lowering、uniform/varying、SIMT reconvergence、register allocation、
instruction selection、resource collection 和 ELF；launcher/driver 负责参数打包、
launch geometry、内存和同步。V1 不把 Triton 变成完整模型 runtime，也不要求
IREE 成为 Triton 编译的前置依赖；预编译 artifact 可由后续 IREE HAL 导入。

## 3. 固定目标 Profile

V1 必须显式记录并验证以下 profile，不得由 runtime 静默覆盖：

| 项目 | V1 契约 |
| --- | --- |
| target triple | `riscv32`（可表现为 `riscv32-unknown-unknown`） |
| target CPU | `ventus-gpgpu` |
| pointers | RV32，32-bit device pointers |
| warp | `warp_size = 32` lanes |
| mapping | `1 CTA = 1 Ventus work-group` |
| supported launch geometry | general path local size exactly `[32,1,1]` or `[64,1,1]`；MMA exactly `[32,1,1]` |
| scalar types | `i1`、`i32`、`f32` |
| grid | 仅支持 one-dimensional grid；`program_id(0)` 是主 grid ID |
| geometry ceiling | 默认 RTL 的 8 warps/256 work-items 仅是 validation ceiling，不扩大 V1 支持范围 |

1D launch mapping 固定为：`local_size = [32,1,1]` 或 `[64,1,1]`，
`global_size = [grid[0] * local_size[0],1,1]`，work-group 数量为 `grid[0]`；
`program_id(0) = work_group_id(0)`，`local_id(0)` 的范围为
`0 <= local_id(0) < local_size[0]`，其余两个 local/global 维度均为固定的
one-dimensional 值。MMA 的 `local_size` 必须严格为 `[32,1,1]`；general path
才允许 `[32,1,1]` 或 `[64,1,1]`。默认 RTL 的 8-warps/256-work-items 只用于
验证资源上限，不表示 V1 接受更大的 launch geometry。

代码生成采用 per-logical-thread scalar semantics；不得预先生成显式 `<32 x T>`
SIMT values、`SETRPC`、`JOIN` 或 vector branches，这些由 Ventus LLVM 负责。

## 4. 支持的 Backend Components

V1 backend 至少包括以下可独立测试的组件：

- `VentusTargetInfo`：注册 `GPUTarget`、目标 profile、warp/work-group 合法性和 feature 查询。
- `TritonGPUToVentus`：TTGIR 到 shared target level 的 pass 注册和 lowering pipeline。
- `SPMDOpToLLVM`：`program_id`、work-item ID、grid/work-group 维度和相关 libclc/builtin 接口。
- `TritonGPUToLLVM`：复用通用 elementwise、shape/view、control-flow 和 pointer lowering。
- `MemoryOpToLLVM`：AS1 global、AS3 static shared、mask 和地址空间验证。
- `VentusLLVM16Compatibility`（checker API 可命名为 `LLVM16CompatibilityChecker`）：接收
  serialization 前 producer-local MLIR LLVM Dialect/native IR，按 tested contract 检查并
  生成 `kernel.ventus.ll`，保存 deterministic diagnostics、tool output 和 content hash。
- `VentusELFLink`：由 `third_party/ventus/backend/compiler.py` producer-locally 拥有，使用
  absolute pinned `VENTUS_LLD`、linker script、crt0、libclc/workitem 和 kernel entry/init
  contract，把 Gate 3 object 链接为 hashed Ventus ELF。
- `BarrierLowering`：full work-group barrier，要求所有 participating warps 控制流 uniform。
- `ReductionLowering`：shared-memory/barrier based 的 correctness-first `sum`/`max`。
- `VentusMmaLowering`：只处理 M3 冻结后、本规范定义的固定 FP32 MMA profile。
- `VentusArtifact`：ELF、entry point、ABI、资源、feature 和参数 manifest 的生成/校验。
- `reference_launcher`：共享 argument packing、launch metadata 和 artifact 校验的参考实现。

普通算术、内存、循环、分支和 reduction 不新增重复的 Ventus-specific generic
operations；优先使用 Triton/MLIR standard dialect，target-specific 语义仅通过
必要的 attributes、interfaces 或受控 MMA lowering 保留。

### 4.1 Ventus LLVM 16 Compatibility Contract

checker 的 allowlist 是经过 OpenCL/Clang golden、Ventus `opt` 和 Ventus `llc` 实测的
contract，而不是“所有较新 LLVM textual feature 都语义不兼容”的声明。M1 allowlist 至少
包括：required `riscv32` target triple 和 pinned RV32 data layout、32-bit device pointers、
`ventus_kernel` calling convention、`i1/i32/f32`、合法 AS1/AS3 访问、受限只读 AS4、仅
backend spill/private-resource 使用的 AS5、已冻结 signature 的 ID/builtin/barrier calls，
以及 ordinary scalar CFG、`phi`、load/store、GEP 和 allowlisted calls。

checker 必须拒绝：Ventus LLVM 16 contract 未支持或未经测试的新 attributes、intrinsics、
metadata/module flags，scalable vectors，64-bit device pointer/address，atomics（包括
`atomicrmw`、`cmpxchg`、atomic load/store、`fence` 及任意 ordering/scope）、EH/`invoke`、
unsupported calls，所有 `addrspacecast`（包括 otherwise well-typed cast），AS4 write、
user-visible AS5。未知 feature 必须 warning 加 fail-closed error；
不能静默删除 semantic attributes/module flags。诊断必须稳定包含 category、operation 或
IR location、observed feature、required contract 和 action。syntax parse success 与
target/codegen semantic acceptance 必须分别记录，不能互相替代。

M1 compatibility tests 至少包括一个 OpenCL/Clang golden 的 normalized semantic/ABI fact
comparison（不是 byte-identical IR）、一个 Triton-shaped positive vector add，以及以下
negative matrix：unknown attribute、unsupported intrinsic、scalable vector、i64 device
pointer/address、`atomicrmw`、`cmpxchg`、atomic load、atomic store、`fence`/unsupported
ordering/scope、`invoke`/EH、otherwise well-typed `addrspacecast`、AS4 write、user AS5、
unsupported metadata/module flag。V1 不允许任何 `addrspacecast`。

### 4.2 Producer Pipeline No-Bypass Invariant

Task 11、18、19、20、20B 以及未来所有 Ventus Triton compiler path 对每个 module 都必须：
生成并 hash `kernel.ventus.ll`，依次运行 Gates 1-3，通过
`third_party/ventus/backend/compiler.py` 定义的 V1 object-to-ELF stage 链接，并把 gate/link
证据写入 manifest。任何直接绕过 checker、`opt`、`llc` 或 V1 ELF stage 的路径都不属于
V1 accepted backend。

## 5. SPMD、Layout 与数据语义

### 5.1 SPMD Builtins

V1 支持 one-dimensional grid，以及与其对应的 `program_id(0)`、work-group ID、
local work-item ID、work-group size 和 grid size。初始实现可沿用检查到的
`__builtin_riscv_*`/libclc 表达；稳定 LLVM ID intrinsics 是后续 hardening，不应
伪造为已存在的事实。

### 5.2 TTGIR Layouts

支持范围为：

- `BlockedEncodingAttr`：主 layout；固定 warp 32，work-group size compile-time specialization。
- `SliceEncodingAttr`：仅用于合法的 blocked/layout decomposition。
- `SharedEncodingAttr`：仅用于 static AS3 shared tile，并须通过 RTL bank/访问验证。
- `DotOperandEncodingAttr`：仅当能准确映射到 blocked 或固定 MMA fragment 时使用。
- `VentusMmaEncodingAttr`：窄范围、显式描述固定 RTL MMA fragment/lane mapping 的 target attribute；
  不代表通用矩阵布局，不得扩展为任意 shape 的通用 encoding。
- `VentusDotOperandEncodingAttr`：窄范围描述 profile A 的 A/B operand fragment；只接受固定 dtype、
  shape、warp 和 active-lane 条件，不能承载 masked 或 low-precision dot。

不支持 CTA cluster、warp specialization、TMA/`cp.async` 或依赖隐式 NVIDIA
lane convention 的 layout。

## 6. V1 Kernel Semantics

### 6.1 Elementwise、Shape 与 Pointer

支持 scalar `i1`/`i32`/`f32` arithmetic、compare/select、broadcast、reshape/view、
index/shape calculation、pointer arithmetic，以及可证明合法的 predication。
支持 masked load/store 到 LLVM address space 1（AS1 global memory），包括边界 mask；
mask 必须在 lowering 中保留为内存访问语义，不能被错误地当作全 warp active。
AS4 constant 只允许作为经过 ABI/golden 验证的只读寻址约定，不是 V1 独立 constant
memory 分配、上传或一致性保证。AS5 private/PDS 只用于 backend spill/private-resource
约定，不构成 general user-visible private-memory allocation 或任意 AS5 pointer 支持；
AS4/AS5 的非法参数、cast、load/store pattern 必须在 Ventus LLVM 前诊断。

普通 CFG 可交给 Ventus LLVM 做 divergent branch/reconvergence lowering，但 V1 只接受
经过验证并有静态或运行时界限的 divergent control flow。检查到的 SIMT stack depth、
overflow 行为和深层嵌套安全性尚未形成 V1 contract；超出已验证嵌套/循环界限的 kernel
必须诊断 unsupported，不能把未观察到 overflow 当作硬件保证。

### 6.2 Shared Memory 与 Barrier

支持静态 LLVM AS3（shared/local）work-group objects，以及其编译期确定的 static
shared byte accounting、读写和 full work-group barrier。AS3 对象的大小和总量必须
进入 manifest 并在 launch 前校验；不支持 dynamic local/shared kernel arguments 或
host-managed LDS upload，subgroup-only barrier 也不属于 V1。shared layout 必须在
RTL 上验证 bank/replay 行为；CycleSim 不足以证明 bank cost。

### 6.3 Reduction

V1 只承诺 non-empty、fixed/static extent 的 correctness-first `sum` 和 `max`
reduction，支持 per-warp 以及通过 static AS3 + full work-group barrier 的 1-2 warp
reduction。masked value 使用 reduction identity：`sum` 使用 `+0.0`，按固定、可
复现的 reduction order 与 reference 比较，并记录 FP32 tolerance；`max` 使用
`-inf`。为保持与 reference tests 一致，V1 对 reduction 输入中的 NaN 采用
propagate-NaN policy；`max(-0,+0)` 和 `max(+0,-0)` 均必须返回 `+0`，其余
signed-zero 结果也采用该 canonical tie rule；正负 infinity 按 IEEE `max`
语义处理。reference 使用 deterministic ordered reduction，以固定输入顺序和上述
tie/NaN policy 作为比较基准。不得依赖尚未验证的 shuffle、ballot 或 native RVV reduction；
需要跨 warp 时使用可审计的 shared-memory 阶段。空 extent、dynamic extent 和未定义
NaN/特殊值语义的 reduction 必须诊断 unsupported。
检查到 RTL decode/FPU 中存在 native `FMAX`/`VFMAX`/FP compare 路径，但这不能证明其
NaN propagation 和 signed-zero tie behavior 等同于上述 normative policy；`max` lowering
可按需要显式组合 classify/compare/select/canonicalization，直到 differential tests 证明
某个 native sequence 完全等价。

### 6.4 普通 FMA 参考/回退路径

ordinary blocked FP32 FMA 只覆盖 V1 accepted set 中明确支持的 fixed/static blocked
matmul/dot patterns；它是 reference/fallback，不是一般 matrix 主路径或 general
`tl.dot` path。
它使用 `BlockedEncodingAttr` 和普通 multiply-add，不要求 target-specific instruction，
用于建立独立于 MMA 的数值基线；arbitrary `tl.dot` shapes/layouts 仍必须诊断
unsupported，不能借 fallback 宣称 general `tl.dot`。fallback 发生在 MMA 选择之前，
或通过一个显式、已测试的 rematerialization 将合法输入重新形成为 `BlockedEncodingAttr`
后再 lowering；不得从已经形成的 MMA fragment 隐式反向转换，除非存在明确测试过的
fragment-to-blocked conversion。

一个具体的 V1 accepted pattern 是 static 2D FP32 blocked matmul：每个 program
计算 compile-time `M=N=K=32` 的 output tile，使用 `BlockedEncodingAttr`、
full active lanes 和 general-path `local_size=[32,1,1]`，A/B/C/D 均为 FP32，
无 dynamic extent、arbitrary layout 或 MMA fragment。该 `32x32x32` kernel 是
ordinary FMA 的 canonical test shape；其他 static blocked extents 只有在拥有
对应 lowering 和 correctness test 时才加入 accepted set，未列出的 `tl.dot`
shape/layout 必须诊断 unsupported。

## 7. 固定 RTL MMA Vertical Slice

V1 MMA **不是 general Triton `tl.dot` support**，而是一个固定 vertical slice。
`VentusMmaProfileA` 在 M3 freeze 前只是 proposed profile：

```text
(M,K,N) = (4,8,4)
logical M = constructor DimM, logical K = constructor DimN,
logical N = constructor DimK
labeled logical order = (M,K,N); constructor positional order = (M,N,K)
D logical [M,N] = A logical [M,K] x B logical [K,N] + C logical [M,N]
A [M,K] = [4,8], B [K,N] = [8,4], C/D [M,N] = [4,4]
physical vftta B fragment = B^T [N,K] = [4,8]
vftta consumes row-oriented A [M,K] and B^T [N,K] fragments:
  D[m,n] = sum_k A[m,k] * B^T[n,k] + C[m,n]
```

冻结后该 profile 的 canonical operation name 为 `VentusMmaProfileA`，指令为
`vftta.vv`，dtype 固定为 FP32，使用完整 active warp（32 lanes），kernel 必须是
`[32,1,1]` 的 1 warp。mandatory architectural operand binding 固定为旧值
`vd=C[M,N]` accumulator、`vs1=A[M,K]`、`vs2=physical B^T[N,K]`，执行后新值
`vd=D[M,N]`；A/B/C/D 的 lane-fragment layout、operand order、累加语义、寄存器约束和
clobber 必须在 contract manifest 中固定；V1 不支持 instruction-level masked
MMA，MMA tile 外的边界处理必须由 kernel 级 shape guard、padding 或 ordinary FMA
完成。

当前 Chisel source 的 `VFTTA_VV` 默认参数在标注维度后描述 logical `M4K8N4`；
检查到的 CycleSim 实现是不同的更小 operation，且未发现稳定 LLVM intrinsic、
lane-fragment ABI、masking contract 或 resource contract。因此在 M3 contract freeze
前不得把 CycleSim 行为当作已对齐 RTL
事实，`VentusMmaProfileA` 也不得标记为已冻结。M3 必须使 mandatory Spike 与 RTL 的
shape、lane mapping、结果语义和 active warp contract 对齐；CycleSim 只是 capability-gated
functional/cycle experiment，不能提供独立 RTL architectural/performance proof。RTL 结果必须先存在，
并作为 freeze/acceptance 的必要输入。若某 backend 明确不可用，只能 capability-gated
skip，不能以 skip 替代 RTL 结果。只有通过下述 source/generated consistency gate 的
pinned generated RTL/testbench 才是 source of truth；freeze 后 simulator、compiler 或
profile 变更都必须更新 pinned revision/content hash、golden 和 capability record。

更精确地说，检查到的 Chisel source profile 与 checked-in generated Verilog 目前不能
视为同一 source of truth：`gpgpu/ventus/src/top/parameters.scala` 的默认
`num_thread=32` 和 `tc_dim=Seq(4,8,4)`，传入
`TensorCoreFP32(vl, DimM, DimN, DimK, ...)` 后，对应逻辑
`(M,K,N)=(4,8,4)`，即 `M4K8N4`。映射必须写作 logical `M=DimM`、logical
`K=DimN`、logical `N=DimK`；labeled logical order 是 `(M,K,N)`，constructor
positional order 是 `(M,N,K)`，任何 manifest、test 或诊断都不得保存无标签 tuple。
然而本次检查的 `gpgpu/driver/rtl/GPGPU_top.v` 看起来来自更旧/更小配置，
只呈现 4 个 dot units 和 4-element dot。M3 必须先证明 pinned Chisel source、生成参数、
generated Verilog 和 testbench 完全一致，之后该生成物才可成为 RTL source-of-truth
golden；在此之前不得以 checked-in Verilog 证明 profile A。

当前 source wrapper 的 lane mapping 候选事实为：A/B physical lanes `0..31`，C/D
meaningful lanes `0..15`；`TensorCoreFP32` 先将全部 32 个 output lanes 置零，而
`vTCexe` 对全部 32 lanes 置 writeback mask，因此当前 wrapper 表现为 upper lanes
`16..31` 写零。M3 必须通过与 pinned generated RTL 一致的 testbench 验证并冻结这一
行为；本文不把它推广为所有 Ventus 参数配置的普遍保证。

full-active、unmasked、`e32` 和 fixed width 32 是 compiler contract requirements；
检查到的硬件路径并非全部独立 enforce 这些前置条件。因此 legality verifier 和 launcher
必须主动拒绝不满足条件的 artifact，不能依赖硬件自然 trap 或忽略非法状态。

M3 manifest 必须逐项冻结：logical `M=DimM`、`K=DimN`、`N=DimK`，labeled logical
order `(M,K,N)` 与 constructor positional order `(M,N,K)`；三显式 operand、
tied-destination 的 `vftta.vv vd, vs2, vs1` assembly/semantic contract，mandatory binding
旧 `vd=C[M,N]`、`vs1=A[M,K]`、`vs2=physical B^T[N,K]`、新 `vd=D[M,N]`，以及 legacy
CycleSim 四寄存器 fixture 的处置；exact FP32 multiplication/accumulation tree、每级 rounding、
NaN、signed zero、exception，以及 `fflags` production、跨 lane/pipeline aggregation 和
architectural writeback；inactive/unused output lane behavior；encoded `vm` mask-control bit
（注释 notation，不是第四个 assembly operand）；architectural `vl`
versus fixed hardware width；register range、alignment 和 overlap；latency、issue、
ready/valid backpressure；以及 dot/add pipeline 和其他 resource units。任一字段未冻结、
source/generated RTL identity 不一致、或 legacy fixture 未隔离时，M4 必须保持 disabled。

### MMA Prototype Boundary

在没有稳定 intrinsic 前，允许两种受控原型，二选一并记录 compiler/toolchain identity：

1. 添加 LLVM intrinsic，例如 `llvm.riscv.ventus.vftta.vv`，由 Ventus LLVM 做合法性、
   register/resource 和 assembly lowering。
2. 在最终 LLVM IR/assembly 边界使用受控 inline asm，固定约束、clobber、operand order
   和 `vftta.vv` spelling；不得从一般 Triton inline asm 任意注入指令。

intrinsic/asm 方案都必须有 LLVM IR、assembly、ELF、mandatory Spike/RTL differential
golden，以及 capability-gated CycleSim result/skip；未满足这些条件时编译器必须选择
ordinary FMA 或给出 capability error。

### MMA Legality 与回退规则

只有在 M3 已冻结且同时满足以下条件才可选择 profile A：`f32` A/B/C/D、精确
`(M,K,N)=(4,8,4)` shape、logical `D=A x B + C` 与 physical `B^T` convention
如上、operand binding 明确为旧 `vd=C[M,N]`、`vs1=A[M,K]`、
`vs2=B^T[N,K]`、新 `vd=D[M,N]` 并通过 fragment golden
验证、`warp_size=32`、full active warp、local size `[32,1,1]`、合法的 `VentusMmaEncodingAttr` 和
`VentusDotOperandEncodingAttr`、无 instruction-level mask、满足寄存器/LDS/resource
限制，且 artifact target/toolchain feature 明确包含已冻结 profile A。以下任一情况必须
reject-with-diagnostic 或（仅在 fixed/static blocked pattern 中）回退 ordinary FMA：任意 shape、dynamic shape、非 FP32、
masked MMA、非 32-lane warp、非法 fragment mapping、资源超限、缺少 simulator/RTL
capability，或 intrinsic/asm contract 版本不匹配。诊断中的 required shape 必须使用
`required_shape=M4K8N4`（或等价的明确 `(M,K,N)=(4,8,4)` 表达）。编译器不得静默
改变结果语义。

## 8. ABI、ELF、Resource 与 Artifact

V1 artifact 必须携带 versioned manifest，而不是只依赖 ELF 文件名。manifest 至少包含：

- `artifact_abi_version`、`target_triple`、`mcpu`、pointer width、`warp_size`，以及
  pinned RTL profile identity（profile revision/content hash）。
- `toolchain_identity`、Ventus/LLVM/driver/simulator revision 或内容 hash；比较时必须
  区分 pinned RTL contract 与 simulator implementation，二者不匹配时拒绝 MMA artifact。
- ELF content hash、entry-point symbol、endianness/class/machine 校验结果。
- `ventus_kernel` calling convention、packed argument offsets、size/alignment、buffer binding。
- grid 计算规则、work-group size、global/local/offset metadata。
- required features（包括 `VentusMmaProfileA`）、dtype/layout/operation/shape constraints。
- static shared bytes、private bytes、VGPR、SGPR、PDS/resource limits。
- raw `.ventus.resource.<kernel>` 四个 target-endian `uint16` 字段的解析结果及其版本/来源。
- resource-unit contract：LDS 以 bytes；PDS 的分配/上限单位；SGPR/VGPR 是
  per-wavefront、per-CTA 还是 total resident allocation，及相应 aggregation/range rule。
- compiler intermediate artifact references：TTIR、TTGIR、LLVM IR、assembly、ELF 和 test result。
- `kernel.ventus.ll` content hash、internal compatibility-check result，以及 absolute Ventus
  LLVM 16 `opt`/`llc` 的完整 argv、stdout、stderr、exit status 和 tool identity。
- Gate 3 object hash，以及 producer-local `compiler.py` 使用 absolute `VENTUS_LLD` 的完整
  argv、stdout/stderr/status；linker script、crt0、libclc、workitem、kernel entry/init 等输入
  的绝对路径/identity/hash；ELF output hash。具体安装文件名由 Task 1/ABI goldens 核验，
  不得由 backend 猜测。

现有 raw resource section 只有 `VGPR, SGPR, LDS bytes, PDS bytes`，没有 magic、version、
record size 或 target identity，且当前 POCL path 不消费它并使用 hard-coded values。
V1 必须提供 versioned manifest/schema 和 canonical argument packer；launcher/runtime
必须显式解析并做 resource range validation，在 launch 时拒绝 ABI、resource、profile
或 manifest mismatch。发布前必须隔离或移除当前 hard-coded resource path，不能把 raw
section 误认为已经是一个统一稳定对象；资源版本化、range checking、消费路径和 ABI
version 是发布条件。

检查到的通用 ELF loader 会读取 `PT_LOAD` segments，但不验证 artifact ABI、target、
entry symbol、content hash 或 manifest identity；现有 `vt_dev_caps` 仅暴露少量且各 backend
不一致的参数，不能承担 V1 capability record。当前 runtime 的部分 resource 值为
hard-coded，字段单位/聚合域不一致；RTL simulation virtual memory 未完成，返回地址实际
采用 simulator physical-address convention；completion/cache flush 还包含固定 drain
steps 或缺少 timeout/flush completion observability。以上均是 M1 必须实现和测试的
要求，不是现有 runtime support 声明：reference launcher 必须验证 ELF/artifact identity，
使用独立 versioned capability record，显式记录 physical-address simulator convention，
并提供 deterministic completion、timeout、cache-flush status 和 failure diagnostics。

## 9. Reference Launcher 与验证产物

提供 test-only `reference_launcher`，从同一 manifest 读取 argument packing、ELF
entry point、work-group/grid geometry、AS3 bytes 和 target capability，连接现有
Ventus C driver API。保留以下可复现 artifacts：

- source Triton kernel 和 compile options。
- TTIR/TTGIR snapshots，尤其是 layout、mask 和 MMA legality decision。
- lowered MLIR/LLVM IR、assembly、ELF 和 manifest。
- `kernel.ventus.ll`、internal checker diagnostics、absolute `opt` parse/verify diagnostics、
  absolute `llc` object-codegen diagnostics 及其 content hash。
- object-to-ELF link command/diagnostics、linker/runtime input hashes、ELF hash；Gate 4 还保存
  launcher input/manifest hash、pinned Spike binary identity、execution result/status 和
  test-result hash。
- POCL/OpenCL ABI golden、argument buffer 和 launch metadata golden。
- Spike correctness result、CycleSim result/skip record、RTL result/trace 及 toolchain
  hash；M3/M4 MMA 的 RTL result 不得标记为 unavailable 或 skip。
- 资源、skip/fallback 原因和数值 tolerance report。

launcher 先校验 target、pinned RTL profile、ABI、feature、shape、resource 和 symbol；
不兼容 artifact 必须在 launch 前失败，且 canonical argument packer 是唯一参数布局
来源。Spike 是 V1 basic kernels 的 mandatory execution baseline。便利性本地测试只有在
明确未请求/provision runtime suite 时可报告 non-green、non-acceptance skip；M1/M2
milestone、release 和 required CI 缺少或错配 pinned Spike/runtime 必须 hard fail，不能以
skip 获得 green acceptance。CycleSim 才允许 capability-gated skip。`rtlsim`/`gvm` 的
非 V1 baseline 能力记录不能替代 mandatory Spike。M3 的 `vftta.vv` 必须
完成 mandatory Spike/RTL differential alignment；CycleSim 只能在精确能力可用时加入，
只能对明确不可用的 non-RTL simulator 做
capability-gated skip，RTL 结果和 RTL/source-of-truth contract 不能 skip。V1 不定义
physical driver。

## 10. Capability Diagnostics

诊断必须 deterministic、可 grep，并包含 operation、shape/dtype、layout、target、
required feature、observed capability 和 action。例如：

```text
unsupported Ventus MMA: profile=VentusMmaProfileA required_shape=M4K8N4
observed_shape=M16K16N16 action=unsupported=no_general_tl_dot
```

对 `unsupported`、`fallback`、`artifact-mismatch`、`resource-exceeded`、`simulator-skip`
和 `contract-version-mismatch` 分别编码；禁止以普通 generic failure 掩盖 MMA、ABI
或 simulator contract 不匹配。

## 11. Deliverable Kernel Matrix

| Kernel class | V1 result |
| --- | --- |
| fill/copy | compile + execute |
| vector add、multiply、fused elementwise | compile + execute |
| boundary masked AS1 copy/load/store | compile + execute |
| broadcast、reshape/view、select/ReLU | compile + execute |
| per-warp and 1-2 warp `sum`/`max` | compile + execute with tolerance |
| static AS3 store -> barrier -> load | compile + execute |
| mandatory canonical blocked FP32 FMA `32x32x32`（及经显式 lowering/test 纳入的额外 static pattern） | compile + execute reference/fallback correctness |
| proposed `VentusMmaProfileA` contract | M3 contract freeze only；不宣称 Triton compile/execute |
| exact-profile Triton `tl.dot` -> FP32 `vftta.vv` | M4 compile + mandatory Spike/RTL differential execute + capability-gated CycleSim；M3 contract required |
| illegal/arbitrary MMA or `tl.dot` shape/dtype/mask/layout | deterministic diagnostic; FMA fallback only for supported blocked pattern |

M1 的 Task 12 execution matrix 必须逐项覆盖 fill、copy、vector add、multiply、fused
elementwise、masked AS1 copy/load/store、broadcast、reshape/view、select 和 ReLU。测试可使用
一个 parameterized basic-kernel file 加现有 vector-add/masked-copy dedicated files，不要求
每个 operation 单独建文件。

## 12. Milestones

- **M1 Basic Backend**：注册 `riscv32/ventus-gpgpu`，固定 32-lane/1D grid/1 CTA per
  work-group 和 `[32,1,1]`/`[64,1,1]` general geometry，完成 `i1/i32/f32`、SPMD builtins、
  Blocked layout、elementwise、shape/pointer lowering、masked AS1 memory、versioned
  manifest/schema、canonical argument packer、ELF/resource parser（含 range validation）、
  独立 capability record、resource-unit contract、ELF identity validation、AS4/AS5 restriction
  diagnostics、bounded divergent-control-flow validation、simulator physical-address convention
  和 deterministic completion/timeout/cache-flush observability；完成 producer-side
  `VentusLLVM16Compatibility`，生成并保存 `kernel.ventus.ll`。每个 M1 compile artifact
  必须依次通过三个独立 gate：(1) internal compatibility checker；(2) absolute-path Ventus
  LLVM 16 `opt` parse/verify；(3) absolute-path Ventus LLVM 16 `llc` target codegen to object。
  Gate 3 后必须由 `compiler.py` 以 absolute `VENTUS_LLD`、pinned linker script、crt0、
  libclc/workitem inputs 和 kernel entry/init contract 把 object 链接为 ELF；这是 Gate 4
  Spike 前置条件。`opt -verify` 单独不构成 M1 acceptance。reference launcher 在
  launch-time reject ABI/resource/capability mismatch。M1 basic kernels 必须在 Spike 运行，
  不能依赖 hard-coded resources。开始 M1 前必须通过 toolchain audit gate：当前 checkout
  使用隔离 `TRITON_HOME` 并实际解析到 `cmake/llvm-info.json` 的 `b010a18d...` consumer
  LLVM revisioned directory/`version.txt`；隔离用于并发与复现，不能只以当前 stable
  symlink 作为实际消费证据。Python/package provenance unknown 时必须显式记录，不得从
  venv 所在 checkout 推断，并记录完整 native library、Ventus compiler、Spike、driver 和
  simulator binary hashes/build-id；不得把 `llvm-build-info.json` 当作 consumer pin。
- **M2 Shared/Reduction**：完成 static AS3 work-group objects、static shared byte accounting、
  full work-group barrier、1-2 warp correctness-first `sum`/`max` 和 fixed/static blocked
  FP32 FMA patterns；继续要求 manifest/schema、canonical packer、resource parsing/range
  validation、launch-time mismatch rejection，且隔离/移除 hard-coded resource consumption；
  basic kernels 在 Spike 运行，CycleSim smoke test 按 capability 报告。
- **M3 MMA Contract Alignment**：contract-only milestone。先从 pinned Chisel source
  重新生成 RTL，把 source pin/hash、effective parameters、generated Verilog 和 testbench
  纳入一个 verified generation record；只有确认
  expected 16 dot units、8-element reductions 与 profile A 一致后，生成 RTL 才可作为
  source of truth。随后冻结 profile A 的 labeled logical/constructor dimensions、lane
  fragments/writeback、mandatory operand binding、active warp、mask/`vl`/`e32`、
  tied-destination instruction syntax、exact FP semantics、`fflags` production/aggregation/writeback、
  register constraints、latency/backpressure、resource 和 simulator contract；补齐 mandatory
  Spike/RTL differential goldens 和 capability-gated CycleSim result/skip，但不在本 milestone 宣称 Triton `tl.dot`
  compile/execute。
- **M4 Triton `tl.dot` -> `vftta`**：只实现能证明为 profile A 的 Triton `tl.dot` pattern，
  生成 `vftta.vv`，完成 compile、artifact/metadata、mandatory Spike/RTL differential
  和 capability-gated CycleSim
  execute；对其他 pattern 发出诊断或仅在 accepted blocked set 中走 ordinary FMA；
  不得宣称 general `tl.dot`；M3 manifest 上述任一字段缺失或 mismatch 时 M4 必须 disabled。

## 13. Acceptance Criteria

V1 通过的必要条件：

1. 所有 artifact 明确记录 `riscv32`、`ventus-gpgpu`、RV32 pointers、warp 32、pinned RTL profile 和 ABI/toolchain identity；manifest/schema 已版本化。release identity 必须满足 toolchain audit 的 source、revisioned cache/`version.txt`、binary、build/install manifest、RUNPATH 和 dirty-patch hash 规则；最终身份以 Task 1 machine-readable `version.json` 为准。当前 unknown pip provenance 和缺少统一 simulator-driver install manifest 的环境不能作为 release identity。
2. M1 的每个 `kernel.ventus.ll` 均通过 internal compatibility checker、absolute Ventus
   LLVM 16 `opt` parse/verify、absolute Ventus LLVM 16 `llc` object codegen 三个独立 gate；
   manifest 保存 IR hash、命令、输出和 deterministic diagnostics；Gate 3 object 必须经
   定义的 V1 ELF stage 链接并记录 link/ELF evidence。Task 12 ELF/Spike
   execution 是第四个 gate，不能由前三者替代。
3. M1/M2 kernel matrix 在 Spike 上编译、启动并与 CPU/POCL reference 对齐；CycleSim
   仅在 capability 可用时执行并允许明确 gated skip。M1/M2 milestone/release/required CI
   缺少 pinned Spike/runtime 是 hard failure；本地 convenience skip 不是 green acceptance。
   M3/M4 的 Spike 和 RTL 均 mandatory。
   Gate 4 evidence 必须包括 ELF hash、launcher input/manifest hash、pinned Spike binary
   identity、execution result/status 和 test-result hash。
4. TTGIR layout、AS1/AS3、mask、barrier、argument packing、ELF symbol/resource 和 manifest 有 golden tests；LLVM 16 contract tests 包含 OpenCL/Clang normalized fact comparison、Triton-shaped vector add 和完整 negative matrix。
5. `sum`/`max` 不依赖未经证明的 shuffle/ballot；barrier 不发生 deadlock，资源超限在 launch 前诊断。
6. canonical argument packer、`.ventus.resource` parsing/range validation 和 launch-time ABI/resource rejection 生效；hard-coded resource consumption 已隔离或移除。
   resource units/aggregation、independent capability record、ELF identity、simulator
   physical-address convention 和 deterministic completion/timeout/cache-flush diagnostics
   也必须完成验证；AS4/AS5 restriction 与 bounded-divergence negative tests 必须通过。
7. ordinary blocked FP32 FMA 的 mandatory canonical accepted case 是 `32x32x32`；
   其他 static blocked pattern 只有具备显式 lowering 和 correctness tests 才能进入
   accepted set。它仅作为 reference/fallback，不依赖 MMA，也不是 general `tl.dot` path。
8. proposed profile A 仅在 M3 source/parameters/generated RTL/testbench identity 一致并
   freeze 后、`(M,K,N)=(4,8,4)`、规定 fragment/writeback convention、full active/
   unmasked/e32/fixed-width 和其他 legality 条件满足时生成 `vftta.vv`。
9. M3 的 Spike/RTL MMA differential alignment 完成，且 Spike 与 RTL 结果
   已验证、均不可 skip；只有 CycleSim 等非 RTL simulator backend 可按 capability
   gating skip。
10. 中间产物、launcher 输入、诊断、fallback 和 skip reason 可复现。
11. M3 manifest 已记录 explicit logical/constructor mapping、mandatory operand binding、
    tied-destination syntax、exact FP tree/rounding/NaN/exceptions、`fflags`
    production/aggregation/writeback、unused lanes、mask/`vl`、register constraints、
    latency/backpressure 和 resource units；swapped `vs1`/`vs2` 与 non-tied accumulator
    negative tests 必须通过；
    任一字段缺失时 M4 保持 disabled。

## 14. 明确排除项

V1 明确不包括：

- general arbitrary MMA shapes，亦不包括 general Triton `tl.dot` support；V1 MMA 是固定 vertical slice。
- FP16、BF16、FP8、低精度或 low-bit MMA/算术。
- instruction-level masked MMA。
- shuffle、ballot、未经验证的 subgroup collectives。
- atomics、async copy、`cp.async`、CTA cluster、warp specialization。
- dynamic shared-memory/local arguments、物理 hardware driver。
- full attention、完整 model graph、TorchInductor full-model ownership、serving、KV-cache policy。

这些能力可在后续版本以独立 contract 加入；不得通过扩大 V1 attribute、inline asm
或 fallback 的含义而隐式纳入。
