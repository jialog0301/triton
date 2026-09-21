# NVIDIA 训练与 Ventus 推理架构结论

> **状态说明：** 本文保留最初的模型部署架构讨论。涉及 Triton 角色、
> Ventus MLIR 层和实施顺序的结论，已由
> `2026-08-20-triton-ventus-iree-shared-mlir-design.md` 和
> `2026-08-23-triton-for-ventus-learning-roadmap.md` 更新。当前设施事实以
> `2026-08-25-ventus-env-facility-snapshot.md` 为准。

## 1. 目标与边界

本文整理以下目标下的关键架构结论：

- 使用同一份 PyTorch `nn.Module` 定义模型。
- 在 NVIDIA GPU 上完成训练。
- 将训练后的权重部署到 Ventus GPGPU 上执行推理。
- 最终支持 Transformer/LLM 的完整 prefill 和 decode。
- 第一阶段以 FP32 为必需基线；FP16/BF16 是需要独立端到端验证的后续能力。
- Ventus 已具备 LLVM 后端和基础设备 Runtime。

这里的“同一套代码”应定义为共享模型语义，而不是共享目标相关 kernel：

- 共享模型结构、权重、配置、tokenizer 和数值语义。
- 不强求 NVIDIA 与 Ventus 共享 kernel 调度、设备 Runtime、内存布局或硬件指令实现。
- 训练程序和部署程序可以使用不同的编译链。

## 2. 核心结论

1. Triton 是以 MLIR 为编译器核心构建的 GPU kernel 编译器，但完整 Triton 还包括 Python DSL、JIT、缓存、LLVM 代码生成和 GPU Runtime。
2. 当前官方 Triton 主仓库内置的完整硬件后端是 NVIDIA CUDA 和 AMD HIP/ROCm。Ventus 不是现有内置后端。
3. Ventus 是基于 RISC-V ISA 的 SIMT GPGPU，不应被建模为普通 RVV CPU 后端。其 Triton/IREE lowering 必须保留 thread、warp、work-group、地址空间和同步语义。
4. IREE 不是 PyTorch 的子系统。IREE 是独立的模型编译器和 Runtime；PyTorch 只是可能的模型来源之一。
5. vLLM 和 SGLang 位于 LLM serving 与请求调度层，IREE 位于模型编译与设备执行层，Triton或 TileLang 位于单 kernel 生成层。
6. 当前正式选择 MLIR/IREE 作为完整模型主链：PyTorch/torch-mlir 负责前端，StableHLO 负责模型契约，IREE 负责模型编译与部署，共享 Ventus MLIR target support 负责薄目标/ABI 契约，Ventus LLVM 负责机器代码，Ventus Driver API 负责连接模拟后端。
7. Triton 不承担第一阶段完整模型主链，但可并行实现原生 Ventus operator backend，用于 ABI、MLIR、LLVM、算子调优和 ELF Artifact 验证。
8. 第一阶段不应实现 Triton-to-Linalg、TVM Relax/TIR 主链、TileLang Ventus backend 或 vLLM/SGLang serving 集成；Triton Ventus backend 作为独立算子支线，不替代 IREE Flow/Stream/HAL。

## 3. 各组件的准确定位

### 3.1 PyTorch

PyTorch负责模型开发和 NVIDIA 训练：

- 定义 `nn.Module`。
- Autograd 和优化器。
- 分布式训练。
- 保存权重。
- 使用 `torch.export` 或其他导出机制生成部署输入。

训练侧可以继续使用 `torch.compile`、TorchInductor 和 Triton。Ventus 部署时不应要求存在 Python 或 PyTorch Runtime。

### 3.2 Triton

Triton主要是 GPU kernel 编译器：

```text
Python Triton DSL
  -> TTIR
  -> TTGIR
  -> 目标相关 GPU IR
  -> MLIR LLVM Dialect
  -> LLVM IR
  -> 目标二进制
```

TTIR 表达 tile 级计算语义，TTGIR 通过 layout encoding 表达 tensor 到 thread、warp 和 CTA 的分布。Triton适合优化单个 GEMM、Attention、Reduction 或融合 kernel，但不负责完整模型的请求调度、KV cache policy 和在线服务。

### 3.3 TileLang

TileLang 与 Triton 处于相近的 kernel DSL 层，但编译基础不同：

- Triton 主要使用 MLIR 的 TTIR/TTGIR。
- TileLang 主要建立在 TVM TIR/TIRX 上。

TileLang显式表达 tile、shared memory、fragment、并行循环和软件流水，适合新硬件早期的高性能 kernel bring-up。它可以替换 Triton承担 Ventus 核心 kernel 的开发，但不能替换 IREE 的模型编译、内存规划和 HAL Runtime。

### 3.4 IREE

IREE 是完整模型编译器和轻量级部署 Runtime：

```text
模型输入 IR
  -> IREE Flow
  -> IREE Stream
  -> IREE HAL
  -> 目标 Codegen
  -> 目标 executable
  -> IREE Runtime
```

主要职责包括：

- 图级融合和 dispatch 划分。
- Tensor 到 buffer 的转换。
- Buffer 生命周期与内存规划。
- 异步依赖和设备提交。
- 目标 executable 生成与打包。
- HAL 设备抽象。

IREE 的直接上层通常是 StableHLO、TOSA、Linalg 或其他受支持的模型 IR。PyTorch 模型需要先经过图捕获和导出转换，才能进入 IREE。

### 3.5 vLLM 和 SGLang

vLLM 和 SGLang 主要解决在线 LLM serving：

- 请求队列与 continuous batching。
- Prefill/decode 调度。
- KV cache 分配和分页。
- Prefix cache。
- Sampling。
- OpenAI-compatible API。
- 多设备和模型服务。

它们可以把 IREE 编译后的 Ventus 模型作为底层执行器，但不会替代 IREE Codegen、Ventus LLVM 后端或设备 Runtime。

## 4. 最新总体架构

```text
                           PyTorch nn.Module
                                  |
                 +----------------+----------------+
                 |                                 |
                 v                                 v
          NVIDIA Training                  Ventus Inference Export
                 |                                 |
   PyTorch/CUDA/cuDNN/cuBLAS                  torch.export
   NCCL/Inductor/Triton                            |
                 |                                 v
                 v                    ExportedProgram / ATen FX Graph
             Checkpoint                              |
        state_dict/safetensors                       v
                 |                         torch-mlir FxImporter
                 |                                   |
                 |                                   v
                 |                          Torch MLIR Dialect
                 |                                   |
                 |                   +---------------+----------------+
                 |                   |                                |
                 |          torch-mlir generic passes       Ventus frontend passes
                 |          ReduceOpVariants                Backend legal-op policy
                 |          DecomposeComplexOps             DecomposeOnTorch
                 |          MaximizeValueSemantics          Rewrite/Fuse custom ops
                 |          Shape/dtype refinement          ConvertTorchToCustomCall
                 |                   |                                |
                 |                   +---------------+----------------+
                 |                                   |
                 |                                   v
                 |                         Torch-to-StableHLO
                 |                                   |
                 |                                   v
                 |              StableHLO primitives + @ventus.* custom_call
                 |                                   |
                 +----------- Weight Packaging ------+
                                                     |
                                   Frontend/Compiler StableHLO Contract
                                                     |
                                                     v
                                          IREE Input Conversion
                                                     |
                                                     v
                                                IREE Flow
                                      graph optimization and fusion
                                       dispatch partition/workload
                                                     |
                                                     v
                                               IREE Stream
                                      resource lifetime/allocation/reuse
                                         asynchronous dependencies
                                                     |
                                                     v
                                                 IREE HAL
                                  Ventus executable/buffer binding/dispatch
                                                     |
                                                     v
                                      IREE Ventus Dispatch Codegen
                                                     |
                    +--------------------------------+-----------------------------+
                    |                                |                             |
                    v                                v                             v
          Generic structured path          Ventus custom calls          Optional kernel DSL
          Linalg/SCF/Vector/MemRef       GEMM/Attention/RMSNorm        TileLang or Triton
                    |                                |                  TIR or TTIR/TTGIR
                    +--------------------------------+-----------------------------+
                                                     |
                                                     v
                                         Ventus-specific lowering
                                    thread/warp/work-group/address spaces
                                      barrier/shared memory/kernel ABI
                                                     |
                                                     v
                                  MLIR LLVM Dialect + Ventus intrinsics
                                                     |
                                                     v
                                           Native LLVM IR
                                                     |
                                                     v
                                         Ventus LLVM Backend
                                                     |
                                                     v
                           Binary + constants + weights + dispatch metadata
                                                     |
                                                     v
                                  IREE HAL Ventus Driver + Ventus Runtime
                                                     |
                                                     v
                                             Ventus GPGPU
```

职责分配如下：

- PyTorch 管模型定义和训练。
- IREE 管完整推理程序、dispatch、buffer、executable 和设备执行。
- IREE Ventus Codegen 和 Ventus 专用 kernel library 负责第一阶段 kernel；TileLang或 Triton 只作为后期可选的外部高性能 kernel 生成器。
- Ventus LLVM Backend 管目标指令选择和机器代码生成。
- Ventus Runtime 管设备内存、binary 加载、kernel launch 和同步。
- vLLM或 SGLang 在需要在线服务时管理请求和 KV cache policy。

### 4.1 前端设计采用 ByteIR 模式

前端不自行实现 PyTorch importer，而是复用 `torch.export` 和 torch-mlir：

```text
PyTorch nn.Module
  -> torch.export.ExportedProgram
  -> torch-mlir FxImporter
  -> Torch MLIR Dialect
  -> Torch/Ventus frontend passes
  -> StableHLO primitives + stablehlo.custom_call @ventus.*
```

各组件职责为：

- `torch.export` 捕获 functional ATen FX graph、参数、buffer、shape constraint 和函数签名。
- torch-mlir 负责 FX 导入、Torch Dialect、通用 decomposition 和 Torch-to-StableHLO。
- Ventus frontend 只负责 backend legal-op policy、少量模型特定 decomposition、custom-op 规范化和缺失转换。
- StableHLO 是前端与模型编译器之间的公开契约。
- `stablehlo.custom_call @ventus.*` 保留适合由 Ventus 专用 kernel 执行的粗粒度语义。

LLM 权重不应全部冻结为大型 MLIR literal。前端需要保留参数名称与模型签名，打包阶段将 `state_dict` 或 safetensors 转为独立的常量归档，并维护参数到归档 offset 的映射。

### 4.2 Custom call 决策

Torch op 按以下策略处理：

```text
Ventus有专用实现
  -> 保留/融合高层语义
  -> stablehlo.custom_call @ventus.*

torch-mlir已有正确转换
  -> StableHLO primitive

可由基础算子表达
  -> Torch Dialect decomposition
  -> StableHLO primitive

均不支持
  -> AOT编译错误，或在带PyTorch Runtime的JIT模式下做图级fallback
```

第一批 custom-call 建议限制为：

- `ventus.gemm` 或 `ventus.gemv`。
- `ventus.rms_norm`。
- `ventus.rope`。
- `ventus.prefill_attention`。
- `ventus.decode_attention`。
- `ventus.kv_cache_update`。

编译期常量放在结构化 attribute 中；position、sequence length 和 block table 等运行时数据作为显式 operand，不应隐藏在 opaque 字符串中。

## 5. 硬件相关 IR 的位置

硬件相关 IR 应位于通用 kernel codegen 之后、LLVM Dialect 之前：

```text
StableHLO/Linalg
  计算什么
       |
SCF/Vector/GPU 或 TTGIR
  如何分块和并行
       |
Ventus-specific IR
  如何映射到 Ventus 执行模型
       |
LLVM Dialect
  指针、控制流、调用和 ABI
       |
原生 LLVM IR
       |
Ventus LLVM Backend
  具体指令选择和寄存器分配
```

适合进入 Ventus-specific IR 的信息包括：

- Thread、lane、warp 和 work-group ID。
- Global、shared/local 和 private 地址空间。
- Work-group barrier 和 warp-level synchronization。
- Shuffle、reduction 和 atomic。
- Ventus 特殊 load/store 或矩阵指令。
- Kernel ABI、work-group size 和 shared memory metadata。

不应进入该层的信息包括：

- 完整 Transformer 模型结构。
- Tokenizer 和采样策略。
- KV cache 的 serving policy。
- IREE Flow/Stream 的模型级依赖。
- 训练反向传播语义。

原型阶段可以直接从通用 IR lowering 到 LLVM Dialect 与 Ventus builtin。只有当 Ventus 特有语义需要跨多个 pass 分析和优化时，再建立独立的 VentusGPU Dialect。

## 6. Triton 与 TileLang 的选择

### 6.1 选择 Triton 的条件

- 希望 TorchInductor 自动生成 Ventus kernel。
- 希望复用现有 Triton kernel 生态。
- 希望 IREE 与 kernel 编译器都围绕 MLIR 工作。
- 希望研究 TTGIR layout 到 Ventus SIMT 的映射。
- 愿意实现 TTIR/TTGIR、layout、shared memory 和 barrier 的 Ventus lowering。

可能的编译链：

```text
TTIR
  -> TTGIR
  -> TritonVentusGPU
  -> LLVM Dialect
  -> Ventus LLVM
```

### 6.2 选择 TileLang 的条件

- 第一阶段只需要有限数量的 LLM 核心 kernel。
- 希望显式控制 tile、shared memory 和 pipeline。
- 希望更快进行 Ventus 性能 bring-up。
- 可以接受 IREE/MLIR 和 TileLang/TVM 两套 IR 栈。
- 不要求 Inductor 自动生成任意 Ventus kernel。

可能的编译链：

```text
TileLang DSL
  -> TVM TIR/TIRX
  -> Ventus-specific TIR passes
  -> Ventus LLVM
  -> Ventus binary
```

### 6.3 当前最终选择

当前已经确定采用 MLIR/IREE 主链：

- IREE 作为完整模型主链。
- IREE Ventus 通用 Codegen 支持基础算子。
- VentusGPU Dialect 表达 thread、warp、work-group、地址空间、shared memory、barrier、atomic 和特殊指令。
- VentusGPU-to-LLVM 将硬件语义降低为 LLVM Dialect 和 Ventus LLVM intrinsics。
- Ventus HAL Driver 适配现有 Ventus Runtime。
- Triton、TVM 和 TileLang 仅在后期有明确性能收益时，以外部 kernel 或独立 codegen 路径接入。

这样可以避免第一阶段同时维护 MLIR、TIR 和 Triton 三套 kernel IR 体系。

## 7. IREE 与 TileLang 的集成边界

由于 IREE 使用 MLIR，TileLang 使用 TVM TIR/TIRX，第一阶段不建议强制共享中间 IR。推荐共享以下稳定边界：

- Kernel ABI。
- Ventus LLVM intrinsic。
- 地址空间编号。
- Binary 格式。
- Kernel metadata。
- Ventus Runtime API。

推荐方式是预编译 kernel：

```text
TileLang kernel
  -> TileLang Ventus backend
  -> Ventus binary + metadata
  -> IREE executable 打包
  -> IREE HAL dispatch
```

Metadata 至少应包含：

- Kernel symbol。
- 参数类型、顺序和对齐要求。
- Buffer binding。
- Grid 计算规则。
- Work-group size。
- Shared memory 大小。
- 目标架构和功能要求。

长期可以研究 IREE dispatch 到 TileLang/TIR 的自动生成，但不应成为第一版的依赖。

## 8. Prefill、Decode 与 KV Cache

### 8.1 分开编译

LLM 推理应至少拆成两个模块：

- Prefill：处理 prompt，计算密集，以 GEMM 和 Attention 为主。
- Decode：每步处理一个 token，带宽和 KV cache 访问敏感，以 GEMV 和 decode attention 为主。

Host Runtime 负责自回归循环和采样，不建议第一阶段把整个生成循环编译进设备程序。

### 8.2 第一阶段 Shape 策略

- `batch_size = 1`。
- 模型维度、head 数和 head dimension 固定。
- KV cache 的最大长度固定。
- Decode 的当前位置是动态 scalar。
- Prefill 使用有限 sequence-length bucket，例如 128、512 和 2048。

### 8.3 KV Cache

第一阶段采用静态预分配的连续 KV cache：

```text
K/V cache:
[num_layers, batch, num_kv_heads, max_seq_len, head_dim]
```

待单请求推理稳定后，再引入分页 KV cache、block table 和 slot mapping，以便接入 vLLM或 SGLang。

## 9. IREE 与 vLLM/SGLang 的关系

关键边界为：

```text
vLLM/SGLang：
决定本轮执行哪些请求，管理分页KV cache和采样。

IREE：
执行一次明确的prefill或decode模型调用。

TileLang/Triton：
生成该模型调用中的高性能kernel。
```

长期 serving 栈可以是：

```text
客户端请求
  -> vLLM或SGLang
  -> IREE Ventus Model Executor
  -> IREE HAL
  -> IREE通用kernel + TileLang优化kernel
  -> Ventus LLVM
  -> Ventus Runtime
  -> Ventus GPGPU
```

如果目标是标准高吞吐 LLM 服务，可优先考虑 vLLM；如果目标强调共享 prompt prefix、Agent、RAG 或复杂生成流程，可优先考虑 SGLang。两者都不应在设备执行链尚未稳定时接入。

## 10. 不推荐的主链

第一阶段不推荐：

```text
PyTorch
  -> Inductor
  -> Triton
  -> Triton-to-Linalg
  -> IREE
  -> Ventus
```

原因是该路线先把高层计算变成 GPU tile/SPMD 程序，再尝试恢复成 Linalg，之后 IREE 又重新进行 tiling 和 distribution，可能导致：

- Triton layout 和调度信息丢失。
- 重复优化和重复 lowering。
- 复杂 pointer arithmetic、atomic、shared memory 和 barrier 难以转换。
- Inductor 生成的任意 Triton kernel 难以保证覆盖。
- 故障定位跨越过多 IR 层。

Triton-to-Linalg 更适合作为兼容、语义迁移或编译器研究路径。

## 11. 分阶段实施建议

### 阶段 1：Ventus 执行基础设施

- 建立 IREE Ventus target 和 HAL adapter。
- 封装现有 Ventus Runtime。
- 统一 kernel ABI、地址空间和 binary metadata。
- 跑通 elementwise kernel。

验收标准：能够加载 binary、传输 buffer、启动 work-group，并与 CPU reference 对齐。

### 阶段 2：Transformer 基础算子

- MatMul/GEMM。
- GEMV。
- Elementwise。
- Reduction。
- RMSNorm。
- Softmax。
- RoPE。
- Reshape/transpose。

验收标准：固定 shape 的单个 Transformer block 与 PyTorch reference 数值对齐。

### 阶段 3：完整 Prefill

- 支持 sequence-length bucket。
- 支持多层模型和连续 KV cache。
- 优化 GEMM 和 prefill attention。

验收标准：完整 prefill logits 和 KV cache 与 PyTorch CUDA reference 对齐。

### 阶段 4：完整 Decode

- 单 token decode。
- 动态 position。
- KV cache 更新。
- Host 侧生成循环。

验收标准：逐 token logits、生成 token 和 KV cache 与 reference 对齐。

### 阶段 5：Ventus 关键 Kernel 优化

- 基于性能分析确定通用 IREE Codegen 的瓶颈。
- 优先优化 GEMM、GEMV、RMSNorm、RoPE、prefill attention 和 decode attention。
- 首先使用 VentusGPU Dialect、Ventus intrinsic 和预编译 Ventus kernel library 实现专用路径。
- 仅在收益明确时评估 Triton或 TileLang 外部 kernel，并通过稳定 ABI 打包进 IREE executable。

验收标准：关键 kernel 可独立验证，IREE 可正确选择和调用，并获得明确性能收益。

### 阶段 6：Serving

- 将 KV cache 改为分页布局。
- 定义 block table 和 slot mapping ABI。
- 接入 vLLM或 SGLang 中的一个。
- 增加 continuous batching 和 prefix cache。

验收标准：多请求并发推理正确，吞吐和延迟具有稳定基线。

### 阶段 7：可选 Triton 路线

- 评估 TorchInductor 生成 kernel 的需求。
- 评估 TTGIR 到 Ventus 的映射成本。
- 仅在自动生成 Ventus kernel 成为核心需求时开发 Triton Ventus 后端。

## 12. 主要风险

### 模型导出覆盖

PyTorch 到 IREE 的导出链可能遇到不支持的算子、mutation、动态 shape 和自定义 op。应优先选择一个固定模型并建立可导出的推理 wrapper。

### FP16/BF16 支持

需要确认 Ventus ISA、LLVM 后端、Runtime 和内存路径对 FP16/BF16 的实际支持范围，包括算术、转换、累加精度和原子操作。

### Kernel ABI 不一致

IREE、VentusGPU-to-LLVM 和 Ventus Runtime 必须统一参数布局、地址空间、work-group size、shared memory 和 metadata。ABI 应先稳定，再进行大规模 kernel 开发。后期接入 Triton或 TileLang 时，也必须遵守同一 ABI。

### 双 IR 栈维护

IREE/MLIR 与 TileLang/TVM 的组合会增加调试成本。应以 binary + ABI 为清晰边界，避免第一阶段建设复杂的 MLIR/TIR 双向转换。

### LLM 范围过大

完整 prefill + decode 涉及大量算子和 Runtime 语义。即使最终目标不变，也必须通过 elementwise、Transformer block、prefill、decode、serving 的顺序逐步验证。

### Serving 过早接入

在单请求模型执行和 KV cache 尚未稳定前接入 vLLM或 SGLang，会混合设备、编译、模型和调度问题，显著增加故障定位难度。

## 13. 与 TVM 的区别

### 13.1 两套系统覆盖的层次相似

如果把本文系统完整实现，它与 TVM 的能力范围高度重叠：

```text
本文系统：
PyTorch Export -> Torch/StableHLO -> IREE Flow/Stream/HAL
               -> Linalg/Vector/Ventus lowering -> Ventus Runtime

TVM路线：
PyTorch/ONNX frontend -> Relax -> TIR -> Target Codegen -> TVM Runtime
```

两者都可以承担：

- 模型导入和图级 IR。
- 算子分解、融合和子图划分。
- Kernel tiling、并行映射和目标 lowering。
- AOT binary、常量与 metadata 打包。
- 独立于 PyTorch 的部署 Runtime。
- 自定义设备后端接入。

因此，问题不是 TVM 能否实现目标，而是采用哪套编译器基础设施更匹配项目的长期边界。

### 13.2 核心抽象对照

| 层次 | 本文 IREE/MLIR 路线 | TVM 路线 |
|---|---|---|
| PyTorch 图捕获 | `torch.export` | PyTorch frontend/export importer |
| 前端 IR | Torch MLIR Dialect | Relax importer 内部表示 |
| 公开模型契约 | StableHLO | Relax/TVMScript，或 frontend 内部契约 |
| 模型图 IR | IREE Flow | Relax |
| 资源与异步计划 | IREE Stream | Relax/TIR/Runtime 中的相关机制 |
| 设备抽象 | IREE HAL | TVM Target/DeviceAPI/Runtime Module |
| 结构化计算 | Linalg | TIR PrimFunc |
| 调度系统 | Linalg Transform/目标 pipeline | TIR Schedule/MetaSchedule |
| Kernel DSL | 可选 Triton/TileLang | TIR/TileLang 更原生 |
| 目标代码生成 | MLIR LLVM -> Ventus LLVM | TVM LLVM Codegen -> Ventus LLVM |
| 部署 Runtime | IREE Runtime/HAL driver | TVM Runtime/DeviceAPI |

### 13.3 IREE/MLIR 路线的优势

- StableHLO 是成熟的模型交换契约，ByteIR 已验证 PyTorch -> Torch Dialect -> StableHLO 的工程模式。
- torch-mlir、StableHLO、Linalg、SCF、Vector、GPU 和 LLVM Dialect 可以形成统一 MLIR pipeline。
- IREE Flow/Stream/HAL 明确分离图 dispatch、资源生命周期、异步执行和设备抽象。
- IREE 以 AOT、独立部署和多种设备 executable 为核心设计目标。
- 如果后续开发 Triton Ventus 后端，Triton 与 IREE 都位于 MLIR 生态，复用硬件 Dialect 和 LLVM lowering 更自然。
- StableHLO custom call 适合作为普通 structured codegen 与专用 LLM kernel 之间的稳定边界。

### 13.4 TVM 路线的优势

- Relax + TIR + Target + Runtime 是一套完整、统一的模型到设备编译栈。
- TIR Schedule 能显式表达 split、reorder、bind、cache、vectorize、tensorize 和 pipeline，适合新硬件 bring-up。
- MetaSchedule 为目标相关 tile、线程映射、vectorization 和 memory hierarchy 提供系统化搜索基础。
- TVM 对自定义 Target、intrinsic、DeviceAPI 和 Runtime Module 的接入路径成熟。
- 如果 TileLang 是核心 kernel DSL，TileLang 与 TVM 同属 TIR/TIRX 体系，可避免 IREE/MLIR + TileLang/TVM 的双 IR 栈。
- 对固定模型、固定硬件和显式 kernel 调优，TVM 路线可能更直接。

### 13.5 为什么不能简单说“不能直接使用 TVM”

可以直接使用 TVM，而且它曾是合理的备选方案。采用 TVM 时，架构可以改为：

```text
PyTorch nn.Module
  -> torch.export / 模型导入
  -> TVM Relax
  -> 图优化、融合、算子/外部kernel选择
  -> TIR / TileLang
  -> Ventus Target + intrinsic lowering
  -> LLVM IR
  -> Ventus LLVM Backend
  -> TVM Ventus DeviceAPI/Runtime Module
  -> Ventus Runtime
```

但“直接使用 TVM”仍然不等于零开发。至少需要完成：

- 验证或补齐 `torch.export`/目标模型到 Relax 的算子覆盖。
- 实现 Ventus TVM Target 与 feature 描述。
- 定义 thread、warp、work-group 和 storage scope 映射。
- 实现 Ventus intrinsic、地址空间、barrier 和 kernel ABI lowering。
- 接入 Ventus LLVM 后端、binary 格式和 linker。
- 实现 TVM Ventus DeviceAPI、Module loader、内存与 launch 接口。
- 为 GEMM、Attention 和 KV cache 建立 TIR/TileLang schedule。
- 处理 LLM 权重打包、prefill/decode 和 serving ABI。

TVM 可以省去自研图编译器和通用调度基础设施，但不能省去 Ventus 目标后端、Runtime 适配和高性能 kernel。

### 13.6 两条路线的主要代价

#### IREE/MLIR 路线

- IREE 与 Ventus LLVM 的版本和 MLIR commit 对齐成本高。
- IREE 的 Ventus Codegen/HAL 仍需新建。
- 如果高性能 kernel 主要用 TileLang，会同时维护 MLIR 和 TVM TIR 两套 IR。
- IREE 的通用 GPU codegen 未必天然匹配 Ventus，需要目标特定策略。

#### TVM 路线

- PyTorch -> StableHLO -> IREE 的 ByteIR 模式不能直接照搬，需要使用或完善 PyTorch -> Relax 前端。
- 如果希望 StableHLO 成为公共模型契约，需要额外维护 StableHLO -> Relax 导入路径。
- 如果后续重用 Triton/TTGIR，会形成 TVM TIR + MLIR 的另一种双 IR 栈。
- TVM Runtime 与现有 Ventus Runtime 仍需适配，不能仅靠 LLVM target 完成部署。
- 在线 LLM serving 同样需要后续对接 vLLM/SGLang，TVM 本身不替代该层。

### 13.7 已完成的选型结论

本项目已选择 IREE/MLIR 主链，理由是：

- StableHLO 是明确的前端/编译器公共契约。
- 希望沿用 ByteIR 风格的 torch-mlir 前端。
- 需要 Flow/Stream/HAL 的清晰 AOT 与异步执行抽象。
- Ventus 是需要显式建模 thread、warp、work-group、地址空间和同步的 GPGPU。
- 已有 Ventus LLVM 后端，MLIR LLVM Dialect 是自然的目标边界。
- VentusGPU Dialect 可以隔离通用结构化计算、硬件执行语义和 LLVM 指令选择。
- 后续若开发 Triton Ventus 后端，可复用 MLIR/LLVM 生态与部分硬件契约。

TVM 路线仍具有以下优势，但本项目不将其作为模型编译主链：

- TileLang 是主要高性能 kernel 开发语言。
- 项目强调 TIR 显式 schedule、MetaSchedule 和硬件协同调优。
- 希望模型图、kernel IR、target 和 Runtime 都留在 TVM 生态。
- PyTorch/目标模型到 Relax 的前端覆盖已经满足需求。
- 团队更熟悉 TVM/TIR。

本项目采用以下边界：

```text
模型编译主链：
torch-mlir -> StableHLO -> IREE -> VentusGPU -> LLVM

可选外部kernel：
TVM/TileLang或Triton -> Ventus binary -> IREE executable
```

不允许 IREE Flow/Stream/HAL 与 TVM Relax/Runtime 同时承担模型主链职责，以避免重复的图 IR、Runtime 和资源规划体系。

### 13.8 TVM 原型的后续定位

不再以 IREE/TVM 双原型决定主链。TVM 原型只在后期出现以下需求时开展：

- 需要评估 TileLang 编写某个关键 kernel 的生产效率。
- IREE 通用和 Ventus 专用 MLIR Codegen 均无法达到性能目标。
- 希望利用 TIR Schedule 或 MetaSchedule 搜索某类 kernel。
- 已有可独立编译的 TileLang/TIR kernel 值得复用。

该原型的输出只能是符合 Ventus kernel ABI 的 binary 和 metadata，并通过 `stablehlo.custom_call` 与 IREE executable 接入；不得引入第二套模型图和 Runtime 主链。

## 14. 最终推荐

当前主线已经确定为 MLIR/IREE 方案：

```text
训练：
PyTorch nn.Module
  -> PyTorch CUDA/Inductor/Triton
  -> NVIDIA GPU

推理：
同一 nn.Module + 同一权重
  -> model.eval() 推理 wrapper
  -> torch.export
  -> torch-mlir FxImporter
  -> Torch MLIR Dialect
  -> Ventus frontend passes
  -> StableHLO + @ventus.* custom_call
  -> IREE Flow/Stream/HAL
  -> Linalg/SCF/Vector/GPU
  -> VentusGPU Dialect
  -> VentusGPU-to-LLVM
  -> LLVM Dialect + Ventus intrinsics
  -> Native LLVM IR
  -> Ventus LLVM Backend
  -> Ventus binary + model metadata
  -> IREE HAL Ventus Driver
  -> Ventus Runtime
  -> Ventus GPGPU

在线服务（后期）：
vLLM或SGLang
  -> IREE Ventus Model Executor
```

第一阶段只实现 MLIR/IREE 主链，不引入 TVM/TileLang/Triton。最先应稳定的公共接口是：

- StableHLO 版本和 `@ventus.*` custom-call schema。
- VentusGPU Dialect 的执行模型。
- Ventus kernel ABI。
- 地址空间模型。
- LLVM intrinsic/builtin。
- Binary 和 metadata 格式。
- Runtime 的内存与 launch API。

在主链稳定之后，再按性能数据决定是否接入：

- Triton Ventus backend。
- TileLang/TIR Ventus backend。
- 预编译 Ventus kernel library。

这些路径必须通过 StableHLO custom call、Ventus kernel ABI、binary metadata 和 IREE HAL 接入，不改变模型主链。

## 15. 最终架构决策

### 15.1 主链

```text
PyTorch nn.Module
  -> torch.export.ExportedProgram
  -> torch-mlir FxImporter
  -> Torch MLIR Dialect
  -> torch-mlir generic passes
  -> Ventus frontend passes
  -> Torch-to-StableHLO
  -> StableHLO + stablehlo.custom_call @ventus.*
  -> IREE Input Conversion
  -> IREE Flow
  -> IREE Stream
  -> IREE HAL
  -> IREE Ventus Codegen
  -> Linalg/SCF/Vector/MemRef/GPU
  -> VentusGPU Dialect
  -> VentusGPU-to-LLVM
  -> MLIR LLVM Dialect
  -> Native LLVM IR
  -> Ventus LLVM Backend
  -> Model Package
  -> IREE HAL Ventus Driver
  -> Ventus Runtime
```

### 15.2 组件职责

| 组件 | 职责 |
|---|---|
| PyTorch | 模型定义、训练、权重生成 |
| `torch.export` | 捕获 functional ATen graph、shape constraints 和 graph signature |
| `torch-mlir` | FX 导入、Torch Dialect、通用 decomposition、Torch-to-StableHLO |
| Ventus frontend | legal-op policy、模型特定 decomposition、custom call、缺失转换 |
| StableHLO | 前端与编译器之间的稳定模型契约 |
| IREE Flow | 图优化、融合、dispatch 划分 |
| IREE Stream | buffer/resource 生命周期、分配、复用和异步依赖 |
| IREE HAL | device、executable、buffer binding 和 dispatch 抽象 |
| IREE Ventus Codegen | 通用结构化算子到 Ventus 执行模型的 lowering |
| VentusGPU Dialect | thread/warp/work-group、地址空间、同步和特殊指令语义 |
| VentusGPU-to-LLVM | Ventus 硬件语义到 LLVM Dialect/intrinsic |
| Ventus LLVM | 指令选择、寄存器分配、汇编和 binary 生成 |
| Ventus HAL Driver | IREE HAL 到现有 Ventus Runtime 的适配 |
| Ventus Runtime | 内存、binary 加载、kernel launch 和同步 |
| Triton/TileLang/TVM | 后期可选的外部高性能 kernel 生成路径 |

### 15.3 第一阶段范围

第一阶段只验证以下闭环：

```text
固定 PyTorch MLP/Transformer block
  -> torch.export
  -> torch-mlir
  -> StableHLO
  -> IREE CPU 验证
  -> IREE Ventus Codegen
  -> VentusGPU Dialect
  -> Ventus LLVM
  -> IREE HAL Ventus Driver
  -> Ventus Runtime
```

第一阶段明确不做：

- TVM Relax/TIR 主链。
- Triton-to-Linalg。
- Triton Ventus backend。
- TileLang Ventus backend。
- Paged KV cache。
- vLLM/SGLang serving。
- 多卡、量化和任意 dynamic shape。

第一阶段的成功标准是：固定 shape、FP16/BF16 的基础模型计算能够在 Ventus 上运行，并且输出与 PyTorch reference 数值一致。

### 15.4 后续性能路径

主链正确后，针对 GEMM、GEMV、RMSNorm、RoPE、Prefill Attention、Decode Attention 和 KV cache 进行性能分析。

只有当 IREE 通用 Codegen 无法满足性能目标时，才接入专用 kernel：

```text
StableHLO custom call @ventus.gemm
  -> 预编译 Ventus kernel
  -> 或 Triton Ventus backend
  -> 或 TileLang/TIR Ventus backend
  -> IREE HAL executable
```

专用 kernel 不应反向改变 StableHLO 模型语义，也不应成为第一阶段模型导入和基本正确性的前置依赖。
