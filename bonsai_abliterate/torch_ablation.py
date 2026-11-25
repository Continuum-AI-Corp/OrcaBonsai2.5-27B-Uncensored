"""The same runtime projection, for PyTorch on the fp16 expansion of the pack.

The pack's packed matmul has kernels for Metal and CPU only, so on an NVIDIA GPU the
model runs as the fp16 expansion written by ``scripts/expand_pack.py`` -- the pack's
exact ternary values in an ordinary ``transformers`` checkpoint. This module wraps that
model's residual writers with forward hooks that apply

    y <- y - alpha (y . r) r

to every writer's output, accumulated in float32, exactly as ``ablation.py`` does for
MLX. The weights are not touched; removing the hooks restores the published model.

The writers are the same 129 modules: ``embed_tokens``, 64 ``mlp.down_proj``,
48 ``linear_attn.out_proj`` and 16 ``self_attn.o_proj``.
"""
from __future__ import annotations

RESIDUAL_WRITERS = ("mlp.down_proj", "self_attn.o_proj", "linear_attn.out_proj")
EMBEDDING = "model.embed_tokens"


def install(model, direction, alpha: float = 1.0, include_embedding: bool = True):
    """Register the projection on every residual writer. Returns the hook handles."""
    import torch

    d = torch.as_tensor(direction, dtype=torch.float32).reshape(-1)
    d = d / d.norm().clamp_min(1e-12)
    cache = {}

    def unit(device):
        if device not in cache:
            cache[device] = d.to(device)
        return cache[device]

    def hook(_module, _inputs, y):
        r = unit(y.device)
        yf = y.float()
        yf = yf - alpha * (yf @ r).unsqueeze(-1) * r
        return yf.to(y.dtype)

    handles, names = [], []
    for name, module in model.named_modules():
        if name.endswith(RESIDUAL_WRITERS) or (include_embedding and name.endswith(EMBEDDING)
                                                and name.count(".") <= 2):
            handles.append(module.register_forward_hook(hook))
            names.append(name)
    return handles, names


def remove(handles) -> None:
    for h in handles:
        h.remove()
