# 独立 Triton 编译后端实现指南

本文总结实现一个独立 Triton 硬件编译后端时的推荐分层、实施顺序、接口契约和常见风险。PPU 后端是本文的主要参考对象，但本文不要求新后端复制 PPU 对 CUDA target、NVVM 和 `.cubin` 的兼容性设计。

## 1. 目标与范围

一个完整的 Triton 后端不是单独的 `compiler.py`，而是一套从 Triton 前端到设备执行的闭环：

```text
Triton frontend
    -> backend discovery / target selection
    -> TTIR
    -> target-specific TTGIR
    -> target-specific LLVM IR
    -> external toolchain
    -> device binary
    -> runtime module loading
    -> signature-specific launcher
    -> kernel execution
```

PPU 当前的主要编译流程是：

```text
TTIR -> TTGIR -> LLIR -> hgbin -> HGGC load/launch
```

其中：

- `TTIR` 主要使用 Triton 通用优化；
- `TTGIR` 开始体现目标硬件的 layout、矩阵计算和异步内存语义；
- `LLIR` 将 target-specific IR 降低为 LLVM IR、intrinsic 或 inline assembly；
- `hgbin` 由 `llvm-irformatter` 和 `ppu-llc` 生成；
- HGGC runtime 负责加载 binary、查找 kernel 和启动 kernel。

## 2. 推荐实现顺序

实现时不建议一开始就开发完整的矩阵指令、异步拷贝和高级调度。应先建立可验证的最小闭环，再逐层增加硬件能力。

### 阶段 0：明确硬件和 ABI

在写代码前先确定：

- target 名称和架构命名方式；
- warp/wave size；
- block、grid、cluster 的映射方式；
- global、shared、local 等地址空间；
- kernel 参数 ABI；
- shared memory 和 scratch memory 的传递方式；
- LLVM DataLayout 和目标 triple；
- 外部编译器的输入格式和输出 binary 格式；
- runtime 的 module、function、launch API。

这一阶段的产物不是代码，而是一份 ABI 表。后续 compiler、lowering、launcher 和 runtime 都必须遵守同一份表。

### 阶段 1：打通插件发现和最小运行闭环

首先实现：

```text
backend/compiler.py
backend/driver.py
backend/__init__.py
```

完成以下能力：

1. 后端能够被 Triton entry point 或 in-tree 机制发现；
2. `supports_target()` 能准确识别自己的 target；
3. `parse_options()` 能生成稳定的 options 对象；
4. `add_stages()` 能注册 `ttir -> ttgir -> llir -> binary`；
5. 最简单的 elementwise kernel 能生成 binary；
6. runtime 能加载 binary 并找到 kernel；
7. launcher 能启动 kernel 并验证结果。

这一阶段可以暂时只支持普通 load、store 和 elementwise 运算。不要在基本执行闭环未验证前投入大量时间优化矩阵计算。

**完成标准：** 一个最小 kernel 能通过编译、缓存、加载、启动和结果校验。

### 阶段 2：复用通用 Triton TTIR/TTGIR

尽量复用 Triton 的公共 pass：

```text
inliner
canonicalizer
CSE
symbol DCE
coalesce
remove layout conversions
thread locality optimization
LICM
loop scheduling
prefetch
```

此阶段只实现目标硬件必需的基础能力：

- `program_id`、thread ID 和 warp ID；
- 基础 load/store；
- 基础地址空间转换；
- 基础 elementwise lowering；
- 基本 reduction 或 shuffle；
- shared memory 分配。

**完成标准：** 常见 elementwise、broadcast、reduction 和简单 transpose kernel 正确执行。

### 阶段 3：实现目标 layout 和 allocation

当基础算子稳定后，再实现目标硬件的：

- shared memory encoding；
- register/thread distribution；
- dot operand encoding；
- layout conversion；
- shared memory swizzle；
- allocation、offset 和对齐计算。

PPU 的 `PPUAIUSharedEncodingAttr`、`PPUMmaEncodingAttr` 说明：如果 target-specific layout 被核心 allocation、pipeliner、dot operand 等多个通用 pass 使用，那么它可能需要进入 Triton 核心，或者通过明确的 target interface 暴露给核心 pass。

优先选择以下方式之一：

```text
TargetLayoutInterface
TargetAllocationInterface
TargetMmaInterface
```

不要在核心代码中大量散落类似以下判断：

```cpp
if (isa<PPUMmaEncodingAttr>(...)) {
  ...
}
```

**完成标准：** layout conversion 的结果稳定，shared memory 不越界，矩阵操作数的 register 分布符合硬件要求。

### 阶段 4：实现矩阵计算单元

实现目标硬件的矩阵计算支持：

1. matmul acceleration pass；
2. dot operand layout；
3. MMA/AIU version 选择；
4. instruction shape 和 tile shape；
5. operand promotion 和 accumulator 类型；
6. FP8、BF16、FP16、FP4 或 INT8 等低精度类型；
7. 矩阵 load/store。

建议先支持一个硬件版本和一种稳定的数据类型，再逐步扩展。不要在同一个 pass 中同时引入多个硬件代际、scaled dot 和所有低精度类型。

**完成标准：** `tl.dot` 或等效矩阵 kernel 能正确执行，并能通过独立的 dtype、shape 和边界测试。

### 阶段 5：实现异步拷贝和软件流水线

如果硬件支持异步数据移动，应形成清晰的 IR lowering 链：

```text
通用 load / tt.aiu_load
    -> target async copy op
    -> commit / wait
    -> local load
    -> target instruction
```

PPU 的 AIU lowering 是这一模式的例子：将高层 AIU load 转换为异步 global-to-local copy、commit、wait 和 local load。

只有在同步版本的矩阵计算正确后，再加入：

- async copy；
- multi-buffering；
- latency assignment；
- loop scheduling；
- software pipelining；
- prefetch。

**完成标准：** 异步路径和同步路径结果一致，依赖关系正确，不能出现过早消费、重复等待或 buffer 覆盖。

### 阶段 6：补齐高级硬件特性

最后再实现：

- multi-CTA 或 cluster；
- warp specialization；
- 特殊 barrier/fence；
- tensor memory；
- descriptor 或 TMA 类机制；
- 高级 instruction scheduling；
- cooperative grid；
- PDL 或程序化同步。

每加入一个特性，都要同时检查 compiler metadata、LLVM lowering、launcher 配置和 runtime API 是否完整支持。

### 阶段 7：工程化和性能优化

功能闭环稳定后补齐：

- 编译缓存失效；
- IR dump 和 IR override；
- debug information；
- 外部工具复现命令；
- profiling hook；
- sanitizer；
- benchmark 接口；
- Gluon 支持；
- package/wheel 安装；
- CI 和多架构测试。

## 3. 分层实现参考

### 3.1 Target 和 backend discovery

推荐使用真正独立的 target：

```python
GPUTarget("my_backend", arch, warp_size)
```

并让后端只接受自己的 target：

```python
@staticmethod
def supports_target(target):
    return target.backend == "my_backend"
```

PPU 当前的：

```python
target.backend == "cuda"
```

会导致 PPU 和 NVIDIA backend 竞争同一 target，必须依赖 `ppu-smi` 特判消歧。这是兼容性方案，不是新后端的推荐方案。

独立后端还应拥有：

- 独立 compiler entry point；
- 独立 driver entry point；
- 独立 runtime probe；
- 独立 target name；
- 独立 binary extension。

不要通过修改 Triton 核心增加大量：

```python
if _is_my_device():
    ...
```

优先让 target identity 和 `is_active()` 自然完成选择。

### 3.2 Compiler 和 options

后端 options 建议使用不可变 dataclass：

```python
@dataclass(frozen=True)
class MyBackendOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    warp_size: int = 32
    arch: str = None
    my_llc_options: Optional[str] = None
    extern_libs: dict = None
```

需要实现或提供的实际接口包括：

```text
supports_target
parse_options
add_stages
load_dialects
get_codegen_implementation
get_module_map
pack_metadata
binary_ext
hash
```

其中 `binary_ext`、`pack_metadata` 和 `get_target_name` 在当前 Triton 版本中并不全部体现在 `BaseBackend` 的 abstract API 中，但运行时会实际依赖它们。

所有 stage 都应使用同一个已经解析的 target 配置。不要在不同 stage 中混用：

```text
options.arch
self.target.arch
capability
```

尤其要检查 `TRITON_OVERRIDE_ARCH` 是否同时影响 TTGIR、LLIR 和最终 assembler。

### 3.3 编译 stage

常见的最小 stage 为：

```text
ttir -> ttgir -> llir -> binary
```

每个阶段的职责应保持清晰：

| 阶段 | 主要职责 |
|---|---|
| TTIR | Triton 语义和通用高层优化 |
| TTGIR | layout、线程分布、矩阵、异步内存、循环调度 |
| LLIR | 地址空间、目标指令、LLVM ABI、kernel metadata |
| binary | 调用外部工具链并返回 bytes |

最终 binary 阶段不应再进行复杂的 IR 语义变换。它主要负责临时文件、工具调用、错误处理、日志和产物清理。

### 3.4 Dialect 和 target-specific IR

当现有 Triton IR 无法表达硬件语义时，使用两级 target dialect：

```text
Triton high-level op
    -> target high-level op
    -> target low-level op
    -> LLVM intrinsic / inline asm
```

适合独立 dialect 的场景：

- 特殊异步执行；
- 特殊矩阵 load/store；
- 独特同步或内存语义；
- LLVM dialect 无法准确表达的设备指令；
- 需要跨多个 lowering pass 保留的硬件语义。

如果差异只是 tile、layout、warp distribution 或 MMA 版本，优先使用 encoding attribute 和 target interface，不要无条件增加新 operation。

### 3.5 LLVM lowering

LLVM lowering 至少要解决：

- thread/program/warp ID；
- barrier、shuffle、ballot 和 reduction；
- global/shared/local 地址空间；
- load/store 和 atomic；
- dot/matrix 指令；
- elementwise 和 conversion；
- kernel 参数 ABI；
- scratch 和 shared memory；
- debug metadata；
- target-specific function attributes。

建议将硬件差异集中到 `TargetInfo` 或等价抽象中，例如：

```text
TargetCapabilities
TargetInfo
TargetLayout
TargetMemoryModel
TargetInstructionBuilder
```

如果使用 inline assembly，需要明确维护以下契约：

```text
MLIR op
    -> LLVM inline asm / intrinsic
    -> IR formatter
    -> external compiler
    -> device instruction
```

指令 mnemonic、寄存器约束、向量/矩阵 operand 布局、dtype 编码和地址空间都必须由 lowering、formatter 和 assembler 共同定义。

### 3.6 DataLayout、地址空间和 kernel metadata

LLVM DataLayout 不是可选细节。它决定：

- 指针宽度；
- 各地址空间指针宽度；
- integer/float/vector 对齐；
- GEP、load/store 和 kernel 参数 ABI。

新后端应明确设置自己的：

```text
DataLayout
TargetTriple
TargetCPU / features
```

除 DataLayout 外，还要统一处理：

- `reqntid` 或等价线程配置；
- shared memory 大小；
- global/profile scratch 大小和对齐；
- attention 或其他专用 hint；
- register/resource 使用量。

PPU 使用了部分 NVVM metadata 作为兼容手段。真正独立的后端最好定义自己的 target metadata，而不是继续依赖 `nvvm.annotations`、`nvvm.reqntid` 等 CUDA 约定。

### 3.7 外部工具链

外部工具链通常包含：

```text
LLVM IR
    -> optional IR formatter
    -> assembler/compiler
    -> device binary
```

工具查找建议遵循：

```text
用户显式路径
    > 安装包自带工具
    > SDK 默认路径
    > PATH
```

工具版本必须进入 backend hash。至少应包含：

- assembler/compiler 版本；
- 目标架构；
- 影响代码生成的外部配置。

编译失败时应记录：

- 返回码；
- stderr；
- 完整复现命令；
- 输入 IR 或 IR 路径；
- target 架构；
- 工具版本。

不要使用无法安全处理空格、引号和特殊字符的字符串拼接来构造命令。优先使用参数列表调用 subprocess；如果外部工具确实要求 shell，必须明确处理转义和日志展示。

### 3.8 Driver、binary loading 和 launcher

编译器和 runtime 必须共同遵守以下协议：

```text
binary bytes
    -> load module
    -> lookup function by kernel name
    -> query resource usage
    -> return module/function/resource metadata
```

当前 Triton runtime 通常期待 `load_binary()` 返回：

```text
(module, function, n_regs, n_spills, n_max_threads)
```

signature-specific launcher 需要处理：

- grid 和 block 配置；
- stream；
- function handle；
- packed metadata；
- device pointer；
- scalar 参数；
- constexpr 参数过滤；
- tensor descriptor 展开；
- global/profile scratch；
- launch hooks。

`pack_metadata()` 的返回顺序必须与 launcher 的 C 解包顺序严格一致。它们虽然分属 compiler 和 driver 文件，但实际上是一个跨文件 ABI。

### 3.9 Language extension 和 libdevice

后端 language 扩展通常包括：

- 硬件变量，例如 `num_warps`、`num_threads`、timer 或 SM ID；
- 特殊 builtin，例如异步 copy、matrix load 和 barrier；
- `@core.extern` 形式的 libdevice wrapper。

推荐让设备符号使用后端自己的命名空间，例如：

```text
__mybackend_sin
__mybackend_exp
```

如果复用其他后端的符号，必须明确记录其兼容边界和链接要求。

libdevice 文件内容应参与 options hash。替换同一路径下的 bitcode 后，必须能够使 kernel cache 失效。

### 3.10 构建、安装和发现

构建系统需要同时接入：

```text
Python package
CMake native libraries
pybind module
backend entry point
language extension package
external tools and bitcode
```

建议将 native library 拆分为：

```text
MyBackendDialect
MyBackendTransforms
MyBackendGPUToLLVM
MyBackendRuntimeBinding
MyBackendPlugin
```

每个 library 应尽量有独立测试，避免所有逻辑集中在一个 plugin target 中。

## 4. 必须保持一致的跨层契约

| 契约 | 编译侧 | Runtime 侧 |
|---|---|---|
| target identity | `GPUTarget`、内部 target | device probe、driver 选择 |
| kernel name | LLVM `define`、metadata | module function lookup |
| warp size | TTGIR、TargetInfo | block dimension、thread mapping |
| num warps | options、metadata | block 配置 |
| num CTAs | TTGIR、metadata | grid/cluster 配置 |
| shared memory | allocation、`ttg.shared` | launch shared memory |
| scratch | LLVM metadata | host allocation、kernel 参数 |
| argument ABI | LLVM function signature | launcher 参数数组 |
| descriptor ABI | lowering 参数展开 | launcher descriptor 展开 |
| DataLayout | LLVM module | assembler、device ABI |
| binary format | final compiler 输出 | runtime module loader |
| toolchain version | backend hash | 实际生成 binary 的工具 |

任何一个字段在两侧定义不一致，都可能出现：

- 编译成功但 module 加载失败；
- function lookup 失败；
- 参数错位；
- shared memory 配置错误；
- kernel launch 被 runtime 拒绝；
- kernel 执行结果错误。

## 5. 关键注意事项

### 5.1 不要复制 PPU 的 CUDA 兼容包袱

以下做法是 PPU 当前为了复用 Triton CUDA 生态而采用的兼容方案，新后端不应默认复制：

- 使用 `GPUTarget("cuda", ...)`；
- 通过 `ppu-smi` 在多个 backend 之间消歧；
- 使用 `torch.cuda` 作为设备接口；
- 依赖 NVVM dialect 和 NVVM metadata；
- 将自己的 binary 映射成 `.cubin`；
- 在 Triton 核心中增加设备特判；
- 把 warp size 硬编码为 32。

### 5.2 统一架构表示

架构信息至少有三种常见表示：

```text
用户输入：sm90
内部 capability：90
外部工具参数：sm_90
```

应在 options 或 target configuration 中统一解析，之后所有 stage 使用同一个对象。架构 override 必须同时影响 TTGIR、LLIR、assembler 和 runtime validation。

### 5.3 不要把 capability 判断散落在各个 pass

不推荐到处使用：

```cpp
if (capability == 80) { ... }
else if (capability == 89) { ... }
```

建议定义统一目标能力描述：

```text
compute capability
warp size
MMA version
AIU version
address spaces
async copy support
cluster support
FP8/FP4 support
```

新增硬件代际时，只增加 target capability 描述和对应 lowering，而不是修改大量互相独立的条件分支。

### 5.4 warp size 必须贯穿全链路

warp size 会影响：

- TTGIR 线程分布；
- `num_threads`；
- warp ID 计算；
- shuffle 和 reduction；
- launcher blockDim；
- runtime 最大线程检查。

如果设备不是 32-wide warp，不能只修改 options。PPU 中类似 `thread_id / 32` 的硬编码都应在新后端中避免。

### 5.5 binary 生成成功不等于 kernel 可执行

必须分别验证：

```text
IR 可被 formatter 接受
binary 可被 runtime 加载
kernel symbol 可以查找到
参数 ABI 正确
launch geometry 正确
shared/scratch 配置正确
结果正确
```

每一层失败都应有独立的诊断信息。

### 5.6 缓存必须覆盖所有代码生成输入

缓存 key 至少应覆盖：

- Triton 版本；
- source hash；
- backend hash；
- target 架构；
- options；
- 外部 compiler 版本；
- libdevice 内容；
- 影响编译结果的环境变量。

只记录工具路径而不记录工具版本、只记录 bitcode 路径而不记录 bitcode 内容，都会造成 stale binary。

### 5.7 外部工具路径错误要尽早、清晰地失败

应在编译开始前检查：

- 工具是否存在；
- 是否为普通文件；
- 是否可执行；
- SDK 环境变量是否为空；
- runtime 动态库是否可被加载。

错误信息应列出实际检查过的路径，而不是暴露底层 `TypeError` 或模糊的 `FileNotFoundError`。

### 5.8 PPU 中的已知风险

参考 PPU 实现时，尤其注意以下问题：

1. PPU compiler 仍将 `supports_target()` 绑定到 `cuda`，会与 NVIDIA backend 竞争。
2. `ppu-smi` 特判属于仓库级消歧逻辑，不是通用插件协议。
3. PPU 同时使用 `options.arch`、`self.target.arch` 和 capability，架构 override 需要重点验证。
4. `get_irformatter()` 和 `get_ppu_llc()` 对 `PPU_SDK` 未设置的情况应提供更清晰的错误。
5. PPU 依赖 `nvvm.annotations`、NVVM dialect 和部分 CUDA 风格地址空间，这些依赖会降低后端独立性。
6. PPU 的 `.hgbin` 在 Triton core 中存在 `.cubin` 兼容别名，新后端不应将这种兼容逻辑作为默认设计。

## 6. 验收和调试清单

### 插件和构建

- [ ] `triton.backends.my_backend.compiler` 可以导入。
- [ ] `triton.backends.my_backend.driver` 可以导入。
- [ ] 只发现一个具体 compiler 和 driver 类。
- [ ] CMake native library 和 pybind module 构建成功。
- [ ] language extension 和 libdevice 安装成功。

### 编译器

- [ ] target 选择不会与其他后端冲突。
- [ ] TTIR 阶段可以正常运行。
- [ ] TTGIR 中的 layout 和 thread distribution 正确。
- [ ] LLIR 中没有未处理的 Triton dialect。
- [ ] DataLayout、地址空间和 kernel ABI 正确。
- [ ] external compiler 可以接受 LLIR。
- [ ] binary stage 返回正确的 bytes。

### Runtime

- [ ] runtime library 能被链接和动态加载。
- [ ] binary 能被 module loader 加载。
- [ ] kernel name 可以查找。
- [ ] resource metadata 可以读取。
- [ ] launcher 的参数格式与 kernel ABI 一致。
- [ ] grid、block、shared memory 和 scratch 配置一致。

### 正确性

- [ ] elementwise。
- [ ] broadcast。
- [ ] reduction。
- [ ] load/store 边界和 mask。
- [ ] 不同 dtype。
- [ ] 不同 `num_warps`。
- [ ] 不同 grid size。
- [ ] 非对齐和尾块场景。
- [ ] 矩阵计算。
- [ ] 异步拷贝和流水线。

### 工程化

- [ ] cache key 覆盖工具链和外部库。
- [ ] 编译错误包含复现命令和 stderr。
- [ ] 支持 IR dump 或 override。
- [ ] 支持 line info 和 debug info。
- [ ] 支持 benchmark 和 profiling。
- [ ] 多架构构建和测试通过。

## 7. 最终原则

一个独立 Triton 后端的最小正确闭环是：

```text
独立 target
    + 独立 compiler
    + 独立 driver
    + target-specific IR
    + LLVM lowering
    + external toolchain
    + binary loader
    + launcher ABI
    = 可执行后端
```

推荐始终遵循以下顺序：

```text
先打通执行闭环
    -> 再支持通用算子
    -> 再实现 layout
    -> 再实现矩阵计算
    -> 再实现异步流水线
    -> 最后加入高级硬件特性和性能优化
```

PPU 后端最值得借鉴的是其完整的端到端组织方式：Python backend、MLIR dialect、target pass、LLVM lowering、外部编译器、缓存、runtime loader 和动态 launcher 相互配合。真正独立的新后端则应在此基础上使用自己的 target identity、runtime ABI、metadata、工具链和 binary 格式，减少对 CUDA/NVVM 兼容层的依赖。
