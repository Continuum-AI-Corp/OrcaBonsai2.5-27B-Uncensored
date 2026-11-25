#!/usr/bin/env python3
"""Export the abliteration as a rank-1 LoRA adapter for llama.cpp.

The published ternary GGUFs stay byte-identical. llama.cpp builds a LoRA into the graph
as two extra matmuls on the activation and never merges it into the base weights, so a
rank-1 adapter is an exact, reversible way to carry the edit:

    W' = W - r (r^T W)        is rank 1, so        A = r^T W,  B = -r,  Delta_W = B @ A

Load it with `--lora`, dial it with `--lora-scaled adapter.gguf:0.5`, drop it to return
to the published behaviour. Nothing is requantized.

## The adapter is tied to the weights it was exported from

`A = r^T W` contains W. An adapter exported from Bonsai 2 and loaded on Bonsai 2.5 does
not project anything: it adds `-r (r^T W_2)` to `W_2.5`, a rank-1 perturbation computed
from somebody else's matrix. It loads without a warning, because the tensor names are
identical. Always export against the exact GGUF you serve.

## W comes from the GGUF itself

There is no unfolded fp16 checkpoint for Bonsai 2.5, and none is needed. The weights are
exactly ternary, so the matrix the base matmul computes with is fully determined by the
GGUF: decode the codes to `{-s, 0, +s}`, undo the Hadamard fold with the sign vectors the
GGUF declares in `prism.hadamard.*`, and you have W in the plain hidden basis. See
`bonsai_abliterate/unfold.py`. This runs on numpy alone, so the adapter can be rebuilt on
the GPU box that serves the model -- no Mac, no MLX.

Three properties of the fork decide what the adapter must look like, all checked against
its source and measured on 2.5:

* **The adapter is authored in the unfolded basis, with no Hadamard applied.** The fork's
  `build_lora_mm` rotates the activation only for the base matmul and hands the LoRA
  branch the unrotated activation.
* **`ssm_out`'s `lora_a` is written in tiled value-head order.** On a
  `prism.hadamard.gdn_v_grouped` pack the stored columns are grouped, but the LoRA branch
  sees the activation tiled, as the linear-attention block produced it; the base path
  permutes it, the LoRA path does not. In file order the 48 linear-attention writers
  silently do nearly nothing (AdvBench at scale 1: 87% refused, against 4% when tiled).
  See ``bonsai_abliterate.unfold.lora_input_order``.
* **No `token_embd`.** The fork rejects a LoRA on a Hadamard latent lookup table, and
  rejects the whole adapter with it. ``--embedding`` writes it anyway.

    python scripts/export_gguf_lora.py \
        --base-gguf /path/to/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf \
        --out       bonsai2.5-abliterate-lora.gguf

Then:

    llama-cli -m Ternary-Bonsai-2.5-27B-PTQ1_0.gguf --lora bonsai2.5-abliterate-lora.gguf

with PrismML's llama.cpp fork -- stock llama.cpp cannot open PTQ1_0 or PQ2_0, since their
tensor type ids are private to that fork.

`--checkpoint` still accepts an unfolded fp16 HF checkpoint, for anyone reproducing the
Bonsai 2 adapter the original way.

## Strength

`--lora` applies the adapter at scale 1.0, which is the projection exactly: llama.cpp
computes `scale = adapter_scale * alpha / rank` with `rank = lora_b->ne[0]`, and this
writes `alpha = 1` against rank 1. It is the same operator as a full weight
orthogonalisation, not an approximation of one. Larger scales over-project; see the
README for what each scale did on 2.5.

## Checks

`--check --pack <mlx pack>` decodes one tensor of each writer family twice -- with
`unfold.py` and with the pack's own `runtime/codec.py` + MLX -- and requires them to
agree. `--verify` re-reads every exported pair and confirms `B @ A` reproduces
`-r (r^T W)` and that the direction's leakage collapses.
"""
from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bonsai_abliterate.gguf_min import Gguf
from bonsai_abliterate.unfold import Hadamard, lora_input_order, unfolded

# GGUF residual-writer suffix -> how to find the same tensor in the HF checkpoint.
WRITERS = {
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
}
EMBED_GGUF = "token_embd.weight"
EMBED_HF = "model.language_model.embed_tokens.weight"
DEFAULT_DIRECTION = Path(__file__).resolve().parents[1] / "directions" / "refusal_dir.safetensors"


def parse():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-gguf", required=True,
                    help="the ternary GGUF the adapter will be loaded against; W is read "
                         "from it unless --checkpoint is given")
    ap.add_argument("--checkpoint", default=None,
                    help="optional unfolded fp16 HF checkpoint to read W from instead "
                         "(how the Bonsai 2 adapter was originally made)")
    ap.add_argument("--direction", default=str(DEFAULT_DIRECTION))
    ap.add_argument("--out", default="bonsai2.5-abliterate-lora.gguf")
    ap.add_argument("--embedding", action="store_true",
                    help="also write token_embd (129 sites). Off by default: the fork "
                         "refuses a LoRA on a prism.hadamard latent lookup table and "
                         "fails to load the whole adapter")
    ap.add_argument("--pack", default=None,
                    help="the MLX pack, needed only by --check (for its runtime/codec.py)")
    ap.add_argument("--check", action="store_true",
                    help="cross-check the GGUF decoder against the pack's own codec first")
    ap.add_argument("--verify", action="store_true",
                    help="after writing, confirm B @ A == -r (r^T W) on every site")
    return ap.parse_args()


def load_direction(path):
    from safetensors.numpy import load_file

    d = load_file(path)["direction"].astype(np.float32).reshape(-1)
    return d / np.linalg.norm(d)


class Checkpoint:
    """Read one tensor at a time from a sharded HF checkpoint.

    Per tensor rather than per shard on purpose: the checkpoint carries a bf16 vision
    tower alongside the fp16 language model, and numpy has no bf16, so loading a whole
    shard raises "data type 'bfloat16' not understood" even when every tensor we want
    from it is fp16. It is also a great deal lighter on memory.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]

    def __getitem__(self, name):
        from safetensors import safe_open

        with safe_open(str(self.root / self.index[name]), framework="np") as f:
            return f.get_tensor(name).astype(np.float32)


class FromGguf:
    """W in the plain basis, decoded and unfolded from the base GGUF itself."""

    def __init__(self, base: Gguf):
        self.base = base
        self.had = Hadamard(base)

    def __getitem__(self, gguf_name):
        return unfolded(self.base, gguf_name, self.had)


def gguf_to_hf(name: str) -> str | None:
    if name == EMBED_GGUF:
        return EMBED_HF
    for suffix, hf_suffix in WRITERS.items():
        if name.endswith(suffix) and name.startswith("blk."):
            layer = name.split(".")[1]
            return f"model.language_model.layers.{layer}.{hf_suffix}"
    return None


def run_check(base: Gguf, pack_dir: str | None) -> None:
    """Decode a tensor of each writer family with unfold.py and with the pack's codec."""
    if not pack_dir:
        raise SystemExit("--check needs --pack pointing at the MLX pack (for runtime/codec.py)")
    sys.path.insert(0, str(Path(pack_dir) / "runtime"))
    import mlx.core as mx
    from codec import transcode
    from runtime import fwht

    had = Hadamard(base)
    names = ["blk.0.ffn_down.weight", "blk.0.ssm_out.weight", "blk.3.attn_output.weight"]
    for name in names:
        shape = base.shape_out_in(name)
        words, sc, bi = transcode(base.raw(name), shape, base.tensors[name]["type_name"])
        rot = mx.dequantize(mx.array(words), mx.array(sc), mx.array(bi),
                            group_size=128, bits=2).astype(mx.float32)
        ref = np.asarray(fwht(rot, had.block, mx.array(had.signs[shape[1]]), inverse=True))
        ours = unfolded(base, name, had)
        rel = float(np.linalg.norm(ours - ref) / max(np.linalg.norm(ref), 1e-30))
        print(f"[check]   {name:<28} rel = {rel:.3e}")
        if rel > 1e-5:
            raise SystemExit(f"unfold.py disagrees with the pack's codec on {name}")
    print("[check] decoder matches the pack's own codec")


def main():
    a = parse()
    base = Gguf(a.base_gguf)
    arch = base.kv.get("general.architecture")
    print(f"[base] {a.base_gguf}")
    print(f"[base] arch={arch}  tensors={len(base.tensors)}  gguf v{base.version}")
    if base.kv.get("prism.hadamard.version") is None:
        print("[base] WARNING: no prism.hadamard.* metadata -- is this really the folded pack?")

    if a.check:
        run_check(base, a.pack)

    if a.checkpoint:
        ckpt = Checkpoint(a.checkpoint)
        weight = lambda gguf_name, hf_name: ckpt[hf_name]
        print(f"[W] from checkpoint {a.checkpoint}")
    else:
        src = FromGguf(base)
        weight = lambda gguf_name, hf_name: src[gguf_name]
        print(f"[W] decoded and unfolded from the base GGUF (block {src.had.block})")

    r = load_direction(a.direction)
    print(f"[dir] {a.direction}  {r.size} dims")

    targets = []
    for name in base.tensors:
        hf = gguf_to_hf(name)
        if hf is None:
            continue
        if name == EMBED_GGUF and not a.embedding:
            continue
        targets.append((name, hf))
    targets.sort()

    by_kind = {}
    for name, _ in targets:
        kind = EMBED_GGUF if name == EMBED_GGUF else name.split(".", 2)[2]
        by_kind[kind] = by_kind.get(kind, 0) + 1
    print(f"[targets] {len(targets)} sites: {by_kind}")

    import gguf

    writer = gguf.GGUFWriter(a.out, arch)
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora_alpha", 1.0)

    t0 = time.time()
    worst = 0.0
    for name, hf in targets:
        W = weight(name, hf)
        if name == EMBED_GGUF:
            # The embedding is looked up, not multiplied: a row IS a residual vector, so
            # the edit is W - (W r) r^T. llama.cpp also stores this one's factors the
            # opposite way round from every other tensor.
            lora_a = (W @ r).reshape(-1, 1).astype(np.float32)   # (n_vocab, 1)
            lora_b = (-r).reshape(-1, 1).astype(np.float32)      # (n_embd, 1)
            if a.verify:
                edited = W + lora_a @ lora_b.T
                worst = max(worst, float(np.abs(edited @ r).max() / np.abs(W @ r).max()))
        else:
            # W is [out, in]; the projection removes r from the output side.
            if W.shape[0] != r.size:
                raise SystemExit(f"{name} has {W.shape[0]} output rows, direction has "
                                 f"{r.size} -- not a residual writer?")
            lora_a = (r @ W).reshape(1, -1).astype(np.float32)   # (1, n_in)
            lora_b = (-r).reshape(-1, 1).astype(np.float32)      # (n_out, 1)
            if a.verify:
                edited = W + lora_b @ lora_a
                worst = max(worst, float(np.abs(r @ edited).max() / np.abs(r @ W).max()))
            order = lora_input_order(base, name)
            if order is not None:
                lora_a = np.ascontiguousarray(lora_a[:, order])
        writer.add_tensor(f"{name}.lora_a", lora_a)
        writer.add_tensor(f"{name}.lora_b", lora_b)
    print(f"[W] {len(targets)} sites in {time.time() - t0:.0f}s")
    if a.verify:
        print(f"[verify] worst remaining leakage along r, relative to before: {worst:.3e}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = Path(a.out).stat().st_size
    print(f"[done] {a.out}  ({size/1e6:.1f} MB, {2*len(targets)} tensors)")
    print()
    print("Use it against the base pack with PrismML's llama.cpp fork:")
    print(f"  llama-cli -m {Path(a.base_gguf).name} --lora {Path(a.out).name}")
    print()
    print("Scale 0 should reproduce the published model; if it does not, the adapter is")
    print("not what you think it is. Judge strength on many prompts, not one.")


if __name__ == "__main__":
    main()
