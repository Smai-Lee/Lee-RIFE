# ComfyUI-Lee-RIFE

Fast, **GPU-batched** RIFE frame interpolation for ComfyUI — a self-contained node
with the weights bundled, no extra dependencies.

## What it does

* Runs RIFE (`rife49.pth`, architecture v4.7) **on the GPU in batches** instead of
one frame-pair at a time.
* Enables RIFE's **real bidirectional ensemble** (a forward and a time-reversed
pass, averaged) for cleaner motion.
* `fp16` by default (visually identical to `fp32` for interpolation, roughly 2×
faster on modern NVIDIA cards); `fp32` selectable.
* Automatically halves the batch on out-of-memory, so it scales from small cards
to big ones.

## Requirements

* ComfyUI
* An NVIDIA GPU with a CUDA build of PyTorch (the same one ComfyUI already uses).
* **No extra pip installs** — it only uses what ComfyUI provides.

## Install

1. Copy the `ComfyUI-Lee-RIFE` folder into `ComfyUI/custom_nodes/`
(or `git clone` it there).
2. Make sure `models/rife49.pth` is present (it ships with this repo;
see `models/HOW-TO-ADD-WEIGHTS.txt` if it is missing).
3. Restart ComfyUI completely (the process, not just the browser tab).

The node appears as **“RIFE GPU Fast (Lee)”** under the **Lee/RIFE** category.

## Usage

Wire your frames into `frames`. For doubling 15 fps → 30 fps, set `multiplier = 2`
and set your video-output node to 30 fps.

|Parameter|Meaning|
|-|-|
|`multiplier`|Output frames per input frame (2 = double the frame rate).|
|`ensemble`|Bidirectional pass (better quality, \~2× the work). On by default.|
|`scale\_factor`|Internal flow scale; leave at `1.0` unless resolution is very high.|
|`precision`|`fp16` (fast) or `fp32` (bit-faithful to the original node).|
|`batch\_size`|Frame-pairs per GPU batch. Higher = better GPU use; lowers itself on OOM.|

**Output count:** for `multiplier = m` and `N` input frames you get `m × N`
frames (e.g. 81 frames at 15 fps → 162 frames at 30 fps, same duration).

On the first run you will see a console line like
`\[ComfyUI-Lee-RIFE] loaded rife49.pth (arch 4.7) on cuda:0 / torch.float16`,
confirming it is on the GPU.

## Credits \& license

This node's own code is **MIT** (see `LICENSE`), © 2026 Smai-Lee.

It builds on:

* **RIFE / Practical-RIFE** (Zhewei Huang et al.) — the model and architecture (MIT).
* **ComfyUI-Frame-Interpolation** (Fannovel16) — `rife_arch.py` is vendored from it (MIT).

Full attribution and the upstream license texts are in **`THIRD-PARTY-NOTICES.md`**.
Please cite the RIFE paper if you use this in research (citation in the notices file).

