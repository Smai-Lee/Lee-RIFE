# nodes.py  --  ComfyUI-Lee-RIFE
#
# GPU-batched RIFE frame interpolation as a self-contained ComfyUI node.
#
# Why batched: the common RIFE node pushes frame-pairs through ONE AT A TIME,
# copies each pair CPU<->GPU and flushes the CUDA cache every few frames. On a
# fast card the RIFE forward is tiny, so the GPU idles between those serial
# steps (low GPU utilisation). This node batches the pairs and skips the
# per-frame flush, keeping the GPU busy. It also enables RIFE's real
# bidirectional "ensemble" (averaging a forward and a time-reversed pass).
#
# The architecture (rife_arch.py) and the weights (models/rife49.pth) come from
# RIFE / Practical-RIFE and ComfyUI-Frame-Interpolation, both MIT-licensed.
# See THIRD-PARTY-NOTICES.md.

import os
import torch

from .rife_arch import IFNet

try:
    from comfy.model_management import get_torch_device, soft_empty_cache
except Exception:  # allows importing/testing outside ComfyUI
    def get_torch_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def soft_empty_cache(*a, **k):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

# RIFE checkpoint -> architecture version
CKPT_ARCH = {
    "rife40.pth": "4.0", "rife41.pth": "4.0", "rife42.pth": "4.2",
    "rife43.pth": "4.3", "rife44.pth": "4.3", "rife45.pth": "4.5",
    "rife46.pth": "4.6", "rife47.pth": "4.7", "rife48.pth": "4.7",
    "rife49.pth": "4.7",
}


def _available_ckpts():
    found = [n for n in CKPT_ARCH if os.path.isfile(os.path.join(MODELS_DIR, n))]
    # Fallback so the node still registers even before the weights are in place.
    return found or ["rife49.pth"]


class LeeRIFEGPU:
    _cache = {}  # (ckpt_name, dtype_str) -> loaded model

    @classmethod
    def INPUT_TYPES(s):
        ckpts = _available_ckpts()
        default = "rife49.pth" if "rife49.pth" in ckpts else ckpts[0]
        return {
            "required": {
                "frames": ("IMAGE",),
                "ckpt_name": (ckpts, {"default": default}),
                "multiplier": ("INT", {"default": 2, "min": 1, "max": 16}),
                "ensemble": ("BOOLEAN", {"default": True}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
                "precision": (["fp16", "fp32"], {"default": "fp16"}),
                "batch_size": ("INT", {"default": 16, "min": 1, "max": 256}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "interpolate"
    CATEGORY = "Lee/RIFE"

    def _get_model(self, ckpt_name, dtype):
        key = (ckpt_name, str(dtype))
        if key in self._cache:
            return self._cache[key]
        path = os.path.join(MODELS_DIR, ckpt_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{ckpt_name} not found in {MODELS_DIR}. "
                f"Place the RIFE weights file there (see models/HOW-TO-ADD-WEIGHTS.txt)."
            )
        arch = CKPT_ARCH.get(ckpt_name, "4.7")
        model = IFNet(arch_ver=arch)
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd)
        model.eval().to(get_torch_device())
        if dtype == torch.float16:
            model.half()
        print(f"[ComfyUI-Lee-RIFE] loaded {ckpt_name} (arch {arch}) on "
              f"{get_torch_device()} / {dtype}")
        self._cache[key] = model
        return model

    @torch.no_grad()
    def interpolate(self, frames, ckpt_name, multiplier, ensemble,
                    scale_factor, precision, batch_size):
        device = get_torch_device()
        dtype = torch.float16 if precision == "fp16" else torch.float32
        model = self._get_model(ckpt_name, dtype)

        N = frames.shape[0]
        if N < 2 or multiplier < 2:
            return (frames,)

        # ComfyUI IMAGE is (N,H,W,C) in [0,1]; RIFE wants (N,C,H,W).
        # Stays on CPU here; only the per-batch chunks are moved to the GPU.
        seq = frames.permute(0, 3, 1, 2).contiguous()
        img0_all = seq[:-1]
        img1_all = seq[1:]
        P = img0_all.shape[0]
        scale_list = [8 / scale_factor, 4 / scale_factor,
                      2 / scale_factor, 1 / scale_factor]

        # One batched pass per intermediate timestep k/multiplier.
        mids = []  # list over k, each tensor (P,C,H,W) on CPU / float32
        for k in range(1, multiplier):
            t = k / multiplier
            parts = []
            s = 0
            chunk = int(batch_size)
            while s < P:
                a = img0_all[s:s + chunk].to(device, dtype)
                b = img1_all[s:s + chunk].to(device, dtype)
                try:
                    r = model(a, b, timestep=t, scale_list=list(scale_list),
                              training=False, fastmode=True, ensemble=ensemble)
                except RuntimeError as e:
                    del a, b
                    if "out of memory" not in str(e).lower():
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if chunk == 1:
                        raise
                    chunk = max(1, chunk // 2)  # auto-shrink and retry
                    continue
                parts.append(r.float().clamp(0, 1).cpu())
                del a, b, r
                s += chunk
            mids.append(torch.cat(parts, 0))

        # Interleave originals + midpoints, pad to multiplier*N output frames.
        seq_cpu = seq.float().cpu()
        out = []
        for i in range(N - 1):
            out.append(seq_cpu[i])
            for k in range(multiplier - 1):
                out.append(mids[k][i])
        out.append(seq_cpu[N - 1])
        while len(out) < multiplier * N:
            out.append(seq_cpu[N - 1])

        result = torch.stack(out, 0).permute(0, 2, 3, 1).contiguous()  # (M*N,H,W,C)
        soft_empty_cache()
        return (result[..., :3],)


NODE_CLASS_MAPPINGS = {"LeeRIFEGPU": LeeRIFEGPU}
NODE_DISPLAY_NAME_MAPPINGS = {"LeeRIFEGPU": "RIFE GPU Fast (Lee)"}
