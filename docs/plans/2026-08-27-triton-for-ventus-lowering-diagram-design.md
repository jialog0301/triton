# Triton-for-Ventus Lowering Architecture Diagram Design

## Purpose

Create a technical architecture diagram focused exclusively on the Triton
operator compilation path for Ventus. The diagram should explain the backend
lowering architecture rather than repeat the complete-model IREE deployment
view.

## Layout

Use a left-to-right pipeline with three visual regions:

1. Triton input: Triton Python DSL, TTIR, and TTGIR with layouts.
2. Central lowering region: a large `TritonGPUToVentus` container that occupies
   roughly half of the canvas.
3. Ventus output: MLIR LLVM Dialect, native LLVM IR, Ventus LLVM, ELF and
   simulator execution.

The central region contains six focused modules:

- TargetInfo, backend options, and capability validation.
- SPMD builtin lowering.
- TTGIR layout decomposition into per-thread scalar semantics.
- Masked memory operations and address-space lowering.
- Shared-memory allocation and work-group barrier lowering.
- Kernel ABI, target attributes, resource metadata, and artifact emission.

## Supporting Information

Add three compact information bands below the main pipeline:

- Reuse boundary: generic TritonGPUToLLVM lowering versus Ventus-specific code.
- Execution mapping: Triton program/CTA, warp, thread and memory spaces mapped
  to Ventus work-group, warp, work-item and address spaces.
- Initial profile and exclusions: RV32, warp size 32, FP32, blocked layouts,
  ordinary FMA, and unsupported advanced features.

## Visual Semantics

- Purple: Triton IR and layout stages.
- Orange: Ventus-specific adapter and lowering responsibilities.
- Blue: MLIR/LLVM compiler stages.
- Green: ELF artifacts and successful execution outputs.
- Grey: validation references and capability boundaries.
- Red dashed elements: unsupported or future capabilities.

The diagram will be delivered as editable Draw.io XML plus PNG and editable SVG
exports. Structural validation must report no dangling edges, overlaps, or
edge-through-node issues.
