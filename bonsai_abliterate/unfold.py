"""Read a residual writer's real weights straight out of the ternary GGUF.

The Bonsai 2 adapter was authored from an unfolded fp16 checkpoint that was never
published. Bonsai 2.5 has no such checkpoint at all -- and it does not need one. The
pack's weights are exactly ternary, so the matrix the model multiplies by is fully
determined by the GGUF itself:

    W = dequant(codes, scale) . H . diag(signs)

``dequant`` maps the 2-bit / trit codes ``{0, 1, 2}`` to ``{-s, 0, +s}`` per 128-wide
group, ``H`` is the normalised blockwise Walsh-Hadamard transform (block 1024) and
``signs`` is the explicit +-1 vector the GGUF carries in ``prism.hadamard.sign_values``.
That is the inverse of the fold the pack applies to its activations, so the result is
the weight in the plain hidden basis -- the basis the refusal direction lives in and the
basis the fork hands the LoRA branch.

Reading W this way is *more* faithful than a fp16 checkpoint, not less: it is the exact
matrix the base matmul computes with, with no fp16 rounding of a pre-quantization weight
in between. Nothing here needs MLX, Metal or the pack's runtime -- numpy is enough, so
the adapter can be rebuilt on the same GPU box that serves the GGUF.

Both block layouts are decoded here independently of the pack's ``runtime/codec.py``;
``scripts/export_gguf_lora.py --check`` compares the two on a real tensor.
"""
from __future__ import annotations

import numpy as np

from .gguf_min import Gguf

GROUP = 128


def _decode_pq2(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PQ2_0, 34 bytes per group: fp16 scale, then 128 2-bit codes, 16 per LE word."""
    scales = data[:, :2].copy().view("<f2").astype(np.float32).reshape(-1)
    words = data[:, 2:].copy().view("<u4")                       # (blocks, 8)
    lanes = np.arange(16, dtype=np.uint32)
    codes = (words[..., None] >> (2 * lanes)) & 3                # (blocks, 8, 16)
    return codes.reshape(-1, GROUP).astype(np.int8), scales


def _decode_ptq1(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PTQ1_0, 28 bytes per group: 26 bytes of base-3 trits, then the fp16 scale.

    The trits are unpacked in three stages -- 16 bytes x 5, 8 bytes x 5, 2 bytes x 4 --
    with the ``(b * 3^t mod 256) * 3 >> 8`` extraction ggml uses, which yields 128 codes
    in the order the fork lays them out.
    """
    scales = data[:, 26:28].copy().view("<f2").astype(np.float32).reshape(-1)
    pieces = []
    for lo, hi, count in ((0, 16, 5), (16, 24, 5), (24, 26, 4)):
        packed = data[:, lo:hi].astype(np.uint16)
        for trit in range(count):
            pieces.append((((packed * 3 ** trit) & 255) * 3) >> 8)
    return np.concatenate(pieces, axis=1).astype(np.int8), scales


DECODERS = {"PQ2_0": (34, _decode_pq2), "PTQ1_0": (28, _decode_ptq1)}


def dequantize(gguf: Gguf, name: str) -> np.ndarray:
    """The stored (still Hadamard-folded) matrix as float32, shape [out, in]."""
    t = gguf.tensors[name]
    rows, cols = gguf.shape_out_in(name)
    raw = gguf.raw(name)
    if t["type_name"] in DECODERS:
        size, decode = DECODERS[t["type_name"]]
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, size)
        codes, scales = decode(data)
        # Affine container: scale = s, bias = -s, so a code c decodes to (c - 1) s.
        w = (codes.astype(np.float32) - 1.0) * scales[:, None]
        return w.reshape(rows, cols)
    if t["type"] == 0:
        return np.frombuffer(raw, dtype="<f4").reshape(rows, cols).copy()
    if t["type"] == 1:
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(rows, cols)
    raise NotImplementedError(f"{name}: tensor type {t['type_name']}")


def fwht(x: np.ndarray, block: int) -> np.ndarray:
    """Normalised Sylvester Walsh-Hadamard transform over each ``block`` of the last axis."""
    shape = x.shape
    y = x.reshape(-1, block).astype(np.float32, copy=True)
    h = 1
    while h < block:
        y = y.reshape(-1, block // (2 * h), 2, h)
        a, b = y[:, :, 0, :].copy(), y[:, :, 1, :]
        y[:, :, 0, :] += b
        y[:, :, 1, :] = a - b
        h *= 2
    y = y.reshape(shape)
    y *= 1.0 / np.sqrt(block)
    return y


class Hadamard:
    """The fold the GGUF declares in its ``prism.hadamard.*`` metadata."""

    def __init__(self, gguf: Gguf):
        kv = gguf.kv
        if kv.get("prism.hadamard.version") is None:
            raise ValueError("no prism.hadamard.* metadata -- not a Prism folded pack")
        if kv.get("prism.hadamard.sign_mode") != "explicit":
            raise ValueError("only explicit sign vectors are supported")
        self.block = int(kv["prism.hadamard.block_size"])
        self.signs = {}
        values = np.asarray(kv["prism.hadamard.sign_values"], dtype=np.float32)
        offset = 0
        for width in kv["prism.hadamard.sign_widths"]:
            v = values[offset:offset + width]
            if v.size != width or not np.isin(v, (-1.0, 1.0)).all():
                raise ValueError("invalid sign vector in prism.hadamard.sign_values")
            self.signs[int(width)] = v
            offset += width
        if offset != values.size:
            raise ValueError("trailing values in prism.hadamard.sign_values")
        self.folded = set(kv["prism.hadamard.weight_names"])
        self.inverse = set(kv.get("prism.hadamard.inverse_weight_names", []))

    def applies_to(self, name: str) -> bool:
        return name in self.folded or name in self.inverse


def lora_input_order(gguf: Gguf, name: str) -> np.ndarray | None:
    """Column order the fork's LoRA branch sees for ``name``, or None if it is the file's.

    The fork hands the LoRA branch the activation as the graph produced it, before any
    Hadamard handling. For ``ssm_out`` on a ``prism.hadamard.gdn_v_grouped`` pack that is
    the *tiled* value-head order ``[hd, nk, rep]``, while the stored weight's columns are
    *grouped* ``[hd, rep, nk]`` -- the base path permutes the activation to match
    (``build_hadamard_input``), the LoRA branch does not. So ``lora_a`` for ``ssm_out``
    has to be written in tiled order, or it multiplies the wrong features and the 48
    linear-attention writers are silently left un-ablated.

    Returns ``idx`` with ``A_tiled = A_grouped[idx]``.
    """
    if ".ssm_out." not in name or not gguf.kv.get("prism.hadamard.gdn_v_grouped", False):
        return None
    arch = gguf.kv["general.architecture"]
    n_v = int(gguf.kv[f"{arch}.ssm.time_step_rank"])
    n_k = int(gguf.kv[f"{arch}.ssm.group_count"])
    n_in = gguf.shape_out_in(name)[1]
    hd, rep = n_in // n_v, n_v // n_k
    r = np.arange(rep)[:, None, None]
    k = np.arange(n_k)[None, :, None]
    h = np.arange(hd)[None, None, :]
    # tiled flat index h + hd*(k + nk*r)  <-  grouped flat index h + hd*(r + rep*k)
    return (h + hd * (r + rep * k)).reshape(-1)


def unfolded(gguf: Gguf, name: str, had: Hadamard | None = None) -> np.ndarray:
    """``name``'s weight in the plain hidden basis, float32, shape [out, in].

    Folded projections compute ``W_f . H . S x``, and the embedding's lookup returns
    ``row . H . S``, so both unfold the same way: transform each row, then multiply by
    the sign vector of the row's width.
    """
    had = had or Hadamard(gguf)
    w = dequantize(gguf, name)
    if not had.applies_to(name):
        return w
    signs = had.signs[w.shape[1]]
    w = fwht(w, had.block)
    w *= signs
    return w
