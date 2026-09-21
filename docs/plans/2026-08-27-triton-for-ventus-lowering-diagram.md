# Triton-for-Ventus Lowering Diagram Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate a Draw.io architecture diagram that explains the TritonGPUToVentus lowering path in detail.

**Architecture:** Use a left-to-right compiler pipeline with a large central lowering container. Keep the Triton input and Ventus output concise, and use supporting bands for reuse boundaries, execution mapping, validation, and the initial target profile.

**Tech Stack:** Draw.io XML, draw.io desktop CLI, drawio-skill validation scripts

---

### Task 1: Author The Lowering Diagram

**Files:**
- Create: `docs/plans/triton-for-ventus-lowering-architecture.drawio`

**Step 1: Create the Draw.io XML**

Add the Triton input pipeline, central `TritonGPUToVentus` lowering modules,
Ventus LLVM output pipeline, execution mapping, reuse boundary, validation
references, and initial capability profile.

**Step 2: Validate the XML**

Run:

```bash
python3 /home/weijiale/.agents/skills/drawio-skill/skills/drawio-skill/scripts/validate.py docs/plans/triton-for-ventus-lowering-architecture.drawio --score
```

Expected: `0 error(s), 0 warning(s)` and no overlaps, crossings, or
edge-through-vertex findings.

### Task 2: Export The Diagram

**Files:**
- Create: `docs/plans/triton-for-ventus-lowering-architecture.png`
- Create: `docs/plans/triton-for-ventus-lowering-architecture.drawio.svg`

**Step 1: Export the PNG preview**

Run:

```bash
drawio -x -f png --width 2000 -o docs/plans/triton-for-ventus-lowering-architecture.png docs/plans/triton-for-ventus-lowering-architecture.drawio
```

Expected: draw.io reports a successful source-to-output conversion.

**Step 2: Export the editable SVG**

Run:

```bash
drawio -x -f svg -e --embed-svg-images -o docs/plans/triton-for-ventus-lowering-architecture.drawio.svg docs/plans/triton-for-ventus-lowering-architecture.drawio
```

Expected: draw.io reports a successful source-to-output conversion.

**Step 3: Re-run structural validation**

Run the validator from Task 1 again after any layout correction.

Expected: `0 error(s), 0 warning(s)`.
