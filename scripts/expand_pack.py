#!/usr/bin/env python3
"""Expand the ternary MLX pack into an ordinary fp16 Hugging Face checkpoint.

This is what the direction was estimated on, and what the evaluation numbers in the
README were cross-checked against. It is a container change, not a requantization: each
ternary weight is decoded to exactly ``{-s, 0, +s}``, the Hadamard fold is undone with the
pack's own sign vectors, and the result is stored in fp16. The pack's quantized matmul
has no CUDA kernel, so this is the way to put the model on an NVIDIA GPU under PyTorch.

    python scripts/expand_pack.py --pack /path/to/Ternary-Bonsai-2.5-27B-mlx-2bit \\
        --out /path/to/Bonsai-2.5-27B-unfolded-fp16

Needs ``mlx`` (``mlx-cpu`` on x86 Linux) only for ``mx.dequantize`` and
``mx.hadamard_transform``; no matmul runs, so the CPU backend is fast enough.

Two layout conventions differ between the pack (mlx-lm's sanitized form) and
``transformers``, and both are undone here:

* the decoder RMSNorm weights (``input_layernorm``, ``post_attention_layernorm``,
  ``q_norm``, ``k_norm``, ``model.norm``) are stored as ``1 + w`` in the pack, and as
  ``w`` by ``transformers``' zero-centred norm;
* ``linear_attn.conv1d.weight`` is ``[C, K, 1]`` in the pack and ``[C, 1, K]`` in
  ``transformers``.

Only the language model is written. The vision tower is left in the pack.
"""
from __future__ import annotations

import sys
import json
import math
import argparse
from pathlib import Path

import numpy as np

SHIFTED_NORMS = (".input_layernorm.weight", ".post_attention_layernorm.weight",
                 ".q_norm.weight", ".k_norm.weight", "model.norm.weight")
PREFIX = "language_model."
SHARD_BYTES = 4 << 30


def parse():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    a = parse()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bonsai_abliterate.pack import resolve_pack

    import mlx.core as mx
    from safetensors.numpy import save_file

    pack = resolve_pack(a.pack)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((pack / "config.json").read_text())
    had = json.loads((pack / "hadamard.json").read_text())
    block = int(had["prism.hadamard.block_size"])
    folded = set(had["prism.hadamard.weight_names"]) | set(had["prism.hadamard.inverse_weight_names"])

    weights = mx.load(str(pack / "model.safetensors"))
    names = sorted(k for k in weights if k.startswith(PREFIX))

    def unfold(stem):
        w = mx.dequantize(weights[stem + ".weight"], weights[stem + ".scales"],
                          weights[stem + ".biases"], group_size=128, bits=2)
        w = w.astype(mx.float32)
        if stem + ".weight" in folded:
            shape = w.shape
            w = mx.hadamard_transform(w.reshape(-1, block), scale=1 / math.sqrt(block))
            w = w.reshape(shape) * weights[stem + ".signs"]
        return np.asarray(w.astype(mx.float16))

    shard, shard_bytes, index, n_shard = {}, 0, {}, 0

    def flush():
        nonlocal shard, shard_bytes, n_shard
        if not shard:
            return
        n_shard += 1
        fname = f"model-{n_shard:05d}.safetensors"
        save_file(shard, str(out / fname), metadata={"format": "pt"})
        for k in shard:
            index[k] = fname
        print(f"[write] {fname}  {len(shard)} tensors", flush=True)
        shard, shard_bytes = {}, 0

    done = set()
    for k in names:
        stem, _, leaf = k.rpartition(".")
        hf = k[len(PREFIX):]
        if leaf in ("scales", "biases", "signs"):
            continue
        if stem + ".scales" in weights:
            if stem in done:
                continue
            done.add(stem)
            arr = unfold(stem)
        else:
            arr = np.asarray(weights[k].astype(mx.float32))
            if any(hf.endswith(s) for s in SHIFTED_NORMS) and arr.ndim == 1:
                arr = arr - 1.0
            if hf.endswith("conv1d.weight") and arr.ndim == 3 and arr.shape[-1] == 1:
                arr = np.moveaxis(arr, 2, 1)
            arr = arr.astype(np.float16)
        shard[hf] = np.ascontiguousarray(arr)
        shard_bytes += arr.nbytes
        if shard_bytes >= SHARD_BYTES:
            flush()
    flush()

    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": index}, indent=1))
    text = dict(config["text_config"])
    text.update({"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text",
                 "torch_dtype": "float16", "dtype": "float16",
                 "tie_word_embeddings": False})
    (out / "config.json").write_text(json.dumps(text, indent=2))
    for extra in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                  "generation_config.json"):
        if (pack / extra).exists():
            (out / extra).write_bytes((pack / extra).read_bytes())
    print(f"[done] {out}  {len(index)} tensors in {n_shard} shards")


if __name__ == "__main__":
    main()
