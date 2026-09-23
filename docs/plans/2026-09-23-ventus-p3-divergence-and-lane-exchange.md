# Ventus P3 实施计划：分歧双路 A/B 与 warp 内数据交换

> **执行方式：** 逐任务执行（建议用 `superpowers:executing-plans`）；每个任务先写失败测试、再实现、再验证、再提交。
> **上位文档：** `third_party/ventus/README.md` §7（计划表）/ §4.5；`ventus-env/llvm/llvm/docs/VentusLLVMInputContract.md`；`ventus-env/gpgpu/docs/Ventus-GPGPU-doc.md`。

**Goal:** 在 Ventus 后端把"分歧"做成可切换的两条 lowering 路径（硬件 vbranch 分歧 vs 软件 select 谓词）并给出 A/B 数字；同时把 warp 内 lane 间数据交换用 LDS 打通，使 `tt.reduce` / warp 内 `tt.gather` / `tl.histogram` 可用。

**Architecture:** 分歧双路在 Triton lowering 层做开关（`VentusOptions.predication`），不改硬件语义；warp 内交换复用 P2 的 `global_smem`（`addrspace(3)`）+ barrier，按硬件文档指定的"memory + barrier 或 atomic"模型实现，不依赖不存在的 shuffle。

**Tech Stack:** MLIR/LLVM 16（Triton 后端 C++ + Python 管线）、Ventus LLVM fork、Spike / cyclesim / rtlsim 三个模拟器。

---

## 0. 前置事实（已实测，2026-09-23）

1. **硬件无 lane 间数据交换指令**（`Ventus-GPGPU-doc.md:46`）：不支持 shuffle/gather/permute；数据依赖用 **memory + barrier 或 atomic** 显式操作。可用同步原语只有 `barrier`（block 级，`:60/:73`）与 `barriersub`（LLVM 有 `llvm.riscv.ventus.subgroup.barrier` → `barriersub` 指令，但未列入 V1 provider 契约）。
2. **分歧硬件已在跑**：masked `vector_add` 在 spike 上的日志含 `setrpc 3807 / vbeq 1692 / join 2115`。来源是 Triton 把 masked 访存降成 `llvm.cond_br` 菱形（`VentusLoadStoreOpToLLVM.cpp:51/76`），Ventus LLVM 再自动生成 `vbeq/setrpc/join`。
3. **软件谓词一臂已实现**：`tl.where` → `LLVM::SelectOp`（`lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp:480/498`，经 `TritonGPUToLLVM.cpp:615` 注册），实测 LLIR 只有 `select`、无 `cond_br`。缺的是"同一个 op 可切换走哪条臂"。
4. **warp 内交换全是响亮失败**：`VentusTargetInfo::shuffleXor/Up/Idx/permute` 一律 `unsupported`，`warpReduce` 返回 false（`third_party/ventus/lib/TritonGPUToLLVM.cpp:382-403`）。
5. **核心已为 reduce/gather 预留 scratch 机制**：`AllocateSharedMemory` 经 `ReduceOpHelper::getScratchSizeInBytes` / `GatherLoweringHelper::getScratchSizeInBytes` 给 op 打 `allocation.offset`（`lib/Conversion/TritonGPUToLLVM/GatherOpToLLVM.cpp:92` 用 `getSharedMemoryBase`）。但 **reduce 的 lane 归约不产生 scratch**，需 Ventus 侧自行预留（见 Task 3.1）。
6. **有界分歧契约**：provider 最多 2 层嵌套分歧区域/循环，更深要拒绝；SIMT-stack 深度/overflow 不是硬件保证（`VentusLLVMInputContract.md:82-85`）。

---

## P3.1 分歧双路 lowering + 开关

### Task 1.1：加 `predication` 选项并让两臂都可编译

**Files:**
- Modify: `third_party/ventus/backend/compiler.py`（`VentusOptions`，约 `40-65`；`make_llir`，约 `204-238`）
- Modify: `third_party/ventus/triton_ventus.cc`（用现有 `passes.def` 机制暴露新 pass）
- Modify: `third_party/ventus/include/Target/TritonVentusToLLVM.h`（声明新 pass 工厂）

**Step 1：加选项字段**

```python
# compiler.py, VentusOptions
    # "software": tl.where/arith.select lowers to LLVM select (current).
    # "hardware": arith.select lowers to an llvm.cond_br diamond so the
    #             Ventus LLVM emits vbeq/setrpc/join (README 4.2, P3 A/B).
    predication: str = "software"
```

在 `__post_init__` 校验取值只能是 `"software"`/`"hardware"`。

**Step 2：写失败测试（先只断言选项被接受）**

新增 `python/test/unit/ventus/test_p3_divergence.py`：

```python
import importlib
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import compile as triton_compile
from pathlib import Path

TARGET = GPUTarget("ventus", "ventus-gpgpu", 32)

@triton.jit
def where_kernel(x_ptr, z_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    z = tl.where(offs < n, x * 2.0, 0.0)
    tl.store(z_ptr + offs, z)

def _compile(tmp_path, predication):
    import os
    os.environ["TRITON_CACHE_DIR"] = str(tmp_path / "cache")
    src = ASTSource(fn=where_kernel,
                    signature={"x_ptr": "*fp32", "z_ptr": "*fp32", "n": "i32"},
                    constexprs={"BLOCK": 32})
    return triton_compile(src, target=TARGET,
                          options={"num_warps": 1, "predication": predication})

def test_predication_option_is_accepted(tmp_path):
    _compile(tmp_path, "software")
```

**Step 3：跑测试确认失败**（选项未知 → `parse_options` 丢弃/报错）

Run: `python -m pytest python/test/unit/ventus/test_p3_divergence.py -q -s --tb=short`
Expected: FAIL（`predication` 未被接受或被忽略）

**Step 4：让 `make_llir` 读取选项**（先占位，不改 lowering）

在 `make_llir` 里按 `options.predication` 决定是否运行新 pass（下一 Task 实现）：
```python
        if options.predication == "hardware":
            ventus.passes.add_select_to_branch(pm)   # 新 pass，先注册空实现
```

**Step 5：跑测试确认通过**；re-run
Run: `python -m pytest python/test/unit/ventus/test_p3_divergence.py -q -s --tb=short`
Expected: PASS

**Step 6：Commit**
```bash
git add third_party/ventus/backend/compiler.py third_party/ventus/triton_ventus.cc \
        third_party/ventus/include/Target/TritonVentusToLLVM.h \
        python/test/unit/ventus/test_p3_divergence.py
git commit -m "feat(ventus): add the predication option and its pass hook"
```

### Task 1.2：臂 A —— `arith.select` → `llvm.cond_br` 菱形

**Files:**
- Create: `third_party/ventus/lib/TritonVentusToLLVM/SelectToBranch.cpp`
- Modify: `third_party/ventus/lib/TritonVentusToLLVM/CMakeLists.txt`（加源文件）
- Modify: `third_party/ventus/include/Target/TritonVentusToLLVM.h`（`createVentusSelectToBranchPass`）
- Modify: `third_party/ventus/triton_ventus.cc`（`passes.def("add_select_to_branch", ...)`）
- Test: `python/test/unit/ventus/test_p3_divergence.py`

**Step 1：写失败测试（断言两臂 LLIR 的形态差异）**

```python
def _llir(tmp_path, predication):
    _compile(tmp_path, predication)
    return next((tmp_path / "cache").rglob("*.llir")).read_text()

def test_select_arm_is_data_plane(tmp_path):
    ll = _llir(tmp_path, "software")
    assert "select i1" in ll
    assert "br i1" not in ll          # 纯数据选择，无控制流分歧

def test_branch_arm_is_control_plane(tmp_path):
    ll = _llir(tmp_path, "hardware")
    assert "br i1" in ll              # 菱形被建出来了
```

**Step 2：跑测试确认失败**

Run: `python -m pytest python/test/unit/ventus/test_p3_divergence.py -q -s --tb=short`
Expected: `test_branch_arm...` FAIL（无 `br i1`）

**Step 3：实现 pass（最小）**

新 pass 在 `arith.select`（condition 为 `i1`、可分歧）上建菱形：
```cpp
// SelectToBranch.cpp：把 arith.select 改成
//   cf.cond_br %cond, ^then, ^else ; then: yield trueVal ; else: yield falseVal
// 只在 pass 被显式加入管线时运行（options.predication == "hardware"）。
```
关键点：`arith.select` 的两个操作数都是已求值的 SSA 值，把它改成分支**不会**改变语义，但会让 Ventus LLVM 的分歧分析把它当控制流分歧处理，从而生成 `vbeq/join`。这是"同一语义、两种 codegen"的对照基础。

**Step 4：跑测试确认通过**；re-run（同 Step 2 命令，Expected: PASS）

**Step 5：spike 真值一致（两臂都要 0 mismatch）**

在 `test_p3_divergence.py` 加：
```python
import importlib
def _run_spike(tmp_path, predication, n=32):
    _compile(tmp_path, predication)
    launcher = importlib.import_module("triton.backends.ventus.launcher")
    elf = next((tmp_path / "cache").rglob("*.elf"))
    return launcher.run_vector_add(
        launcher.LaunchSpec(elf=elf, n_elements=n, local_size=32,
                            kernel_name="where_kernel", elements_per_program=32))

def test_where_arms_agree_on_spike(tmp_path):
    assert _run_spike(tmp_path, "software")["num_mismatches"] == 0
    assert _run_spike(tmp_path, "hardware")["num_mismatches"] == 0
```
Run: 同上。Expected: PASS（注意 spike 可一进程多次；cyclesim/rtlsim 不行）

**Step 6：Commit**
```bash
git add third_party/ventus/lib/TritonVentusToLLVM/SelectToBranch.cpp \
        third_party/ventus/lib/TritonVentusToLLVM/CMakeLists.txt \
        third_party/ventus/include/Target/TritonVentusToLLVM.h \
        third_party/ventus/triton_ventus.cc python/test/unit/ventus/test_p3_divergence.py
git commit -m "feat(ventus): lower arith.select to a cond_br diamond for the hardware arm"
```
> 构建：`cd "$(BUILD_DIR)"; ninja triton`（改 C++ 必须重编）。

### Task 1.3：A/B 取数

**Files:**
- Create: `third_party/ventus/tools/divergence_ab.py`（复用 `tools/vecadd_baseline/measure.py` 的 driver/`force_resources` 约定）
- Modify: `third_party/ventus/README.md` §4.6（新增小节）

**Step 1：写测量脚本**（同一 ELF 的两个变体，分别 cyclesim 一次/进程）
**Step 2：跑并记录**
```bash
python third_party/ventus/tools/divergence_ab.py --kernel where --n 1024 --block 32
```
Expected: 表：`software` vs `hardware` 的 `num_mismatches`（都应 0）、`simulated_time_ns`、指令数（spike log 里 `setrpc/vbeq/join` 计数）。
**Step 3：把结果写进 README §4.6**（含形状、两臂数字、`setrpc/join` 计数）。
**Step 4：Commit**
```bash
git add third_party/ventus/tools/divergence_ab.py third_party/ventus/README.md
git commit -m "feat(ventus): measure the software-predicate and hardware-divergence arms"
```

---

## P3.2 有界分歧契约 + producer 校验

### Task 2.1：检测并拒绝 >2 层分歧嵌套

**Files:**
- Create: `third_party/ventus/lib/TritonVentusToLLVM/BoundedDivergence.cpp`
- Modify: `third_party/ventus/lib/TritonVentusToLLVM/CMakeLists.txt`
- Modify: `third_party/ventus/include/Target/TritonVentusToLLVM.h`
- Modify: `third_party/ventus/triton_ventus.cc`
- Modify: `third_party/ventus/backend/compiler.py`（在 `make_llir` 的 `add_scf_to_cf` **之前**运行）
- Test: `python/test/unit/ventus/test_p3_divergence.py`

**Step 1：写失败测试**
```python
@triton.jit
def nested_if_kernel(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    a = tl.where(x > 0.0, 1.0, 0.0)
    b = tl.where(x > 1.0, a, 0.0)
    c = tl.where(x > 2.0, b, 0.0)
    if tl.sum(c) > 0.0:        # 一个真实 scf.if
        tl.store(x_ptr + offs, c)

def test_three_nested_divergent_regions_are_rejected(tmp_path):
    with pytest.raises(Exception, match="divergent"):
        _compile_kernel(tmp_path, nested_if_kernel, ...)
```
> 先确认"三层"在 TTGIR 里确实表现为 3 层嵌套分歧（`scf.if`/`cf`），必要时用更明确的 3 层 `if` kernel。

**Step 2：跑确认失败**（当前不拒绝）
**Step 3：实现 pass**：遍历 kernel 内 `scf.if`/`scf.for`/`cf.cond_br` 的嵌套深度，**保守假设每层都分歧**，`maxDepth > 2` → 报 `unsupported`（与其它 `unsupported(...)` 一致的诊断文本，含 `"divergent"`）。
**Step 4：跑确认通过**（规则 kernel 仍能编译、运行）
**Step 5：Commit**
```bash
git add third_party/ventus/lib/TritonVentusToLLVM/BoundedDivergence.cpp \
        third_party/ventus/... python/test/unit/ventus/test_p3_divergence.py
git commit -m "feat(ventus): reject divergent control flow past the validated nesting bound"
```

---

## P3.3 warp 内数据交换（LDS 路线）

### Task 3.1（前置，最高风险）：为 warp 交换预留 LDS scratch

**背景：** `tt.reduce` 的 lane 归约在核心不产生 scratch（§0.5），所以 `allocation.offset` 可能为 0 甚至没有 `global_smem`。必须先解决"scratch 从哪来"。

**Files:**
- Modify: `third_party/ventus/lib/TritonVentusToLLVM/TritonGPUToLLVM.cpp`（`createSharedMemoryGlobal` 约 `280-300`；新增 `warpScratchOffset()` 辅助）
- Modify: `third_party/ventus/lib/TritonVentusToLLVM/TritonGPUToLLVM.cpp`（`make_llir` 的 `metadata["shared"]` 计算，在 `compiler.py`）

**Step 1：决定并实现预留方案（推荐）**
在 `createSharedMemoryGlobal` 里把全局大小设为 `ttg.shared + kPerWarpScratch * numWarps`，其中 `kPerWarpScratch = 32 * 4`（32 lane × 最大 4 字节元素）；把 `ttg.shared` 之后的这段作为 warp 交换区。若 `ttg.shared == 0` 但有 reduce/gather 需要交换，也要创建该全局（不能沿用"0 字节就不建"的旧逻辑）。
并把 `metadata["shared"]` 相应加上这段（让 launcher 的 floor/断言一致）。

**Step 2：写失败测试**
```python
def test_reduce_reserves_warp_scratch(tmp_path):
    k = _compile_kernel(tmp_path, reduce_kernel, ...)   # 见 Task 3.2
    assert k.metadata.shared >= 32 * 4                  # 至少一个 warp 的交换区
```
**Step 3：跑确认失败**
**Step 4：实现并跑通**
**Step 5：Commit** `feat(ventus): reserve a warp-exchange scratch region in global_smem`

### Task 3.2：`tt.reduce` 的 warp 内归约走 LDS

**Files:**
- Modify: `third_party/ventus/lib/TritonVentusToLLVM/TritonGPUToLLVM.cpp`（`VentusTargetInfo::warpReduce` 约 `399-404`；辅助函数区 `emitSharedStore`/`emitSharedLoad` 约 `216-260`）
- Test: `python/test/unit/ventus/test_reduce_lane_exchange.py`（新增；见 AGENTS.md 关于新测试文件的说明）

**实现要点（覆盖 `warpReduce`，不再走 `shuffleXor`）：**
1. 取 warp 交换区基址：`getStackPointer(rewriter, funcOp)` + `warpScratchBase + wid * 32 * elemBytes`。
2. 每 lane 把每个 operand 的 acc 写入 `scratch[lane]`。
3. **一次** `targetInfo.barrier(loc, rewriter, AddrSpace::Local)`（reduce 是 block-uniform 控制流，CTA barrier 安全；`reg` 之间如需复用 scratch 再 barrier）。
4. 每 lane 从 scratch 读出需要归约的 lane 值并按 combine 组合（用 `op.getCombineOp()`，注意 V1 的 `sum`/`max` identity 与 NaN tie policy，见 `2026-08-28-...-scope.md` §6.3）。
5. 返回 `true`（接管归约，核心不再落 `shuffleXor`）。

**Step 1：写失败测试（先只要求编过）**
```python
@triton.jit
def reduce_kernel(x_ptr, z_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    s = tl.sum(x, axis=0)            # 跨 lane 归约
    tl.store(z_ptr, s)

def test_sum_over_warp_on_spike(...):
    ...  # n=128, BLOCK=32, num_warps=1, grid=4
    assert result["num_mismatches"] == 0
```
**Step 2：跑确认失败**（当前 `unsupported: shuffle xor`）
**Step 3：实现 `warpReduce`**
**Step 4：跑通 + 与 Host 参考逐元素比对（含 NaN/±0 的 policy）**
**Step 5：Commit** `feat(ventus): reduce across a warp through shared memory`

### Task 3.3：warp 内 `tt.gather` 走 LDS

**Files:**
- Modify: `third_party/ventus/lib/TritonVentusToLLVM/TritonGPUToLLVM.cpp`（`shuffleIdx` 两个重载 约 `388-393`）
- Test: `python/test/unit/ventus/test_reduce_lane_exchange.py`

**要点：** `shuffleIdx(val, lane)` 用同一 warp 交换区：写 `scratch[lane]` → barrier → 读 `scratch[srcLane]`；实现后 `lib/Conversion/TritonGPUToLLVM/GatherOpToLLVM.cpp:340` 的 warp 内分支即可工作。
**Steps：** 同 3.2（失败测试 → 实现 → spike 0 mismatch → commit）。

### Task 3.4（优化）：用 `barriersub` 替代 CTA barrier

**Files:**
- Modify: `third_party/ventus/lib/TritonVentusToLLVM/TritonGPUToLLVM.cpp`（`warpSync` 约 `360-362`；交换路径的 barrier 调用）
- Test: 现有 reduce/gather 测试 + 一个固定 cycle 数下限

**前提：** 先在 spike / cyclesim / RTL 上验证 `barriersub`（`barrier x0,x0,N` 的 subgroup 版本；LLVM 有 `llvm.riscv.ventus.subgroup.barrier`，但要按 P2 的方式发 inline asm，因为消费端 LLVM 会拒该内建）。**验证不了就不要切**（契约里它不在 V1）。
**Steps：** 先写一个"只验 barriersub 能编能跑"的测试 → 通过后再替换 → 比 cycle。

---

## P3.4 `tl.histogram`（共享原子）

**Files:**
- Modify: `third_party/ventus/lib/TritonVentusToLLVM/TritonGPUToLLVM.cpp`（原子 lowering 入口）
- Test: `python/test/unit/ventus/test_reduce_lane_exchange.py`

**要点：** `HistogramOpToLLVM.cpp:129` 用 `getSharedMemoryBase` + 共享原子；Ventus 的 `A` 扩展有 `amoadd` 等（`VentusInstrInfoA.td`），需确认能作用于 `addrspace(3)`。若不支持 → 退回 LDS + barrier 的非原子实现。
**Steps：** 失败测试 `tl.histogram` → 实现 → spike 0 mismatch → commit。

---

## P3.5 A/B 汇总与 README

**Files:**
- Modify: `third_party/ventus/README.md`（把 §7 的 P3 行换成任务表引用本文；新增 §4.6 分歧 A/B、§4.7 warp 交换）

**Steps：** 汇总 P3.1/P3.3 的数字；更新 §7 表；`test_toolchain_identity` 若要绿需按 `toolchain/README.md` 重录身份快照（非本阶段必需）。

---

## 判据 / 命令汇总

```bash
source .venv/bin/activate
export TRITON_HOME="$PWD/.triton-home"
export PYTHONPATH="$PWD/python"
export VENTUS_CLANG=/home/weijiale/Code/cuda2rvv/ventus-env/install/bin/clang
export VENTUS_OPT=/home/weijiale/Code/cuda2rvv/ventus-env/install/bin/opt
export VENTUS_LLC=/home/weijiale/Code/cuda2rvv/ventus-env/install/bin/llc
export VENTUS_LLD=/home/weijiale/Code/cuda2rvv/ventus-env/install/bin/ld.lld

# C++ 改动后
cd "$(BUILD_DIR)"; ninja triton

python -m pytest python/test/unit/ventus/test_p3_divergence.py -q -s --tb=short
python -m pytest python/test/unit/ventus/test_reduce_lane_exchange.py -q -s --tb=short
python -m pytest python/test/unit/ventus/ -q --tb=short
```

**P3 完成的定义：**
1. `predication="software"` 与 `"hardware"` 两臂对同一 `tl.where` 核都编过、spike 都 0 mismatch、cyclesim 给出 cycle 数（§4.6 有表）。
2. >2 层分歧嵌套被确定性拒绝，有负例测试。
3. `tt.reduce`（warp 内 + 跨 warp）与 warp 内 `tt.gather` 在 spike/cyclesim 上 0 mismatch，且不依赖任何 shuffle。
4. `tl.histogram` 通或明确记录为不支持（响亮失败）。
5. README §7/§4.x 更新。

## 范围外（不在 P3）

- P4：LLVM 版本对齐、把 inline asm/文本 hack 换成一等内建（P3 会新增内建需求，正好给 P4 铺路）。
- P5：M4K8N4 MMA。
- 工具链侧 SIMT-stack 深度/overflow 的硬件建模（当前契约明确它不是保证；P3 只做 producer 侧拒绝）。

## 风险与依赖

1. **Task 3.1 的 scratch 预留是最不稳的一环**：`global_smem` 目前"0 字节就不建"，而 reduce 的 lane 归约没有 allocation。必须先落地预留方案，否则 `getStackPointer` 会 `assert(globalBase)` 失败。先用一个最小 reduce 核验证。
2. **`barriersub` 可靠性未证**：不在 V1 契约里，Task 3.4 必须先在三个模拟器上拿到证据，否则一律用 CTA `barrier`。
3. **两臂的等价性**：把 `select` 改成分支不能改变任何值/副作用（AGENTS.md 的块模型约束）；只允许在 condition 是 `i1` 且两个臂已求值时做。
4. **RTL 限制**：grid ≤ 2、nocache 变体、SIMT-stack 行为未建模——RTL 只用于微架构旁证，功能真值仍以 spike 为准。
5. **新测试文件**：AGENTS.md 倾向不新建测试文件；本计划提议新增 `test_p3_divergence.py` / `test_reduce_lane_exchange.py`，如需保持单文件请把用例并入 `test_pipeline_vector_add.py`。
