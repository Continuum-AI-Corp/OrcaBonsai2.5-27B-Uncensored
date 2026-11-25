# OrcaRouter Ternary Bonsai 2.5 27B Uncensored

**Runtime-uncensored Ternary Bonsai 2.5 27B — without modifying or re-quantizing the original weights by **[`OrcaRouter research team`](https://www.orcarouter.ai)**.**

This repository applies refusal-direction ablation to Ternary Bonsai 2.5 27B entirely **at runtime**:

* [`prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit) — Apple Silicon / MLX
* [`prism-ml/Ternary-Bonsai-2.5-27B-GGUF`](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF) — llama.cpp, on NVIDIA GPUs, CPUs and Metal

The original Bonsai packs remain **bit-identical**.

* **27B parameters**
* **~1.76 bits/weight**
* **0 modified weights**
* **0 re-quantization**
* **0 additional weight quantization error**
* **Runtime-adjustable ablation strength**
* **129 residual intervention sites** (128 under llama.cpp, see [GGUF / llama.cpp](#gguf--llamacpp))
* **Apple Silicon / MLX, llama.cpp, and PyTorch on NVIDIA GPUs**

You supply the original Ternary Bonsai 2.5 pack. This repository supplies the refusal direction and the runtime that applies it.

> **Keep the weights compressed. Change the behavior at inference.**

---

## How it works

Traditional abliteration modifies model weights by orthogonalizing matrices that write into the residual stream against a learned refusal direction `r`:

```text
W ← W - r(rᵀW)
```

For a conventional BF16 or FP16 model, the resulting matrix can simply be saved as a new checkpoint.

Ternary Bonsai 2.5 is different.

Its weights are aggressively quantized. The affine container stores:

```text
scale = s
bias  = -s
```

so the 2-bit codes:

```text
{0, 1, 2}
```

decode to:

```text
{-s, 0, +s}
```

The model's quality at approximately **1.76 bits per weight** is the result of quantization-aware training.

Orthogonalizing one of these matrices produces a dense, full-precision matrix. Saving that result back into the ternary representation would therefore require **re-quantization**.

And simply re-quantizing the edited weights does not reproduce the quantization-aware training process that produced the original model.

So we don't edit the weights.

### Runtime ablation

Instead, the equivalent projection is applied when each residual contribution is produced:

```text
y ← y - α · dot(y, r) · r
```

where:

```text
y = residual contribution
r = normalized refusal direction
α = intervention strength
```

At `alpha=1`, the component of each residual write parallel to the refusal direction is removed.

Conceptually:

```text
     Original Ternary Bonsai 2.5
              27B
               │
               │
        bit-identical weights
               │
               ▼
      ┌──────────────────┐
      │  packed matmul   │
      └────────┬─────────┘
               │
               │ y
               ▼
      ┌──────────────────┐
      │ Runtime Ablation │
      │                  │
      │ y ← y - α(y·r)r │
      └────────┬─────────┘
               │
               ▼
         residual stream
               │
               ▼
            output
```

The same operator ships in three forms, one per runtime:

| Runtime | Where it runs | How the projection is applied | Sites |
| --- | --- | --- | ---: |
| MLX (`run.py`) | Apple Silicon; x86 CPU for checks | wraps each residual writer, compiled, fp32 accumulation | 129 |
| llama.cpp (`gguf/`) | NVIDIA / AMD / CPU / Metal, PrismML fork | rank-1 LoRA, `A = rᵀW`, `B = -r` | 128 |
| PyTorch (`torch_ablation.py`) | NVIDIA, on the fp16 expansion | forward hooks on each residual writer, fp32 accumulation | 129 |

The original ternary weights are never modified.

---

# Quickstart

## NVIDIA GPU / llama.cpp

The adapter is in this repository — nothing to build on our side. You need PrismML's
llama.cpp fork, because stock llama.cpp cannot open the ternary `PTQ1_0` / `PQ2_0` types:

```bash
git clone https://github.com/PrismML-Eng/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build -j --target llama-server llama-cli
```

Then run the published GGUF with the adapter:

```bash
hf download prism-ml/Ternary-Bonsai-2.5-27B-GGUF \
  Ternary-Bonsai-2.5-27B-PTQ1_0.gguf --local-dir .

llama-server -m Ternary-Bonsai-2.5-27B-PTQ1_0.gguf -ngl 99 \
  --lora gguf/bonsai2.5-abliterate-lora.gguf
```

`--lora-scaled gguf/bonsai2.5-abliterate-lora.gguf:0` is the published model, `:1` the full
projection. With `--lora-init-without-apply`, each request picks its own strength:

```json
{"messages": [...], "lora": [{"id": 0, "scale": 1.0}]}
```

## Apple Silicon

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the model:

```bash
python run.py \
  --pack /path/to/Ternary-Bonsai-2.5-27B-mlx-2bit \
  "your prompt"
```

`--pack` is the directory holding `config.json` and `runtime/`. If you downloaded with
`huggingface_hub`, that is the **snapshot** directory, not the `models--...` cache entry
above it:

```bash
python -c "from huggingface_hub import snapshot_download; \
print(snapshot_download('prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit'))"
```

A cache entry is accepted too and its newest snapshot is resolved for you.

By default:

```text
alpha = 1
```

which enables the full runtime projection.

## NVIDIA GPU / PyTorch

For research — hidden states, hooks, other directions — the pack can be expanded into an
ordinary fp16 checkpoint and run under `transformers` with the same 129-site projection:

```bash
python scripts/expand_pack.py \
  --pack /path/to/Ternary-Bonsai-2.5-27B-mlx-2bit \
  --out  /path/to/Bonsai-2.5-27B-unfolded-fp16

python scripts/check_torch.py \
  --checkpoint /path/to/Bonsai-2.5-27B-unfolded-fp16
```

See [Running on NVIDIA GPUs](#running-on-nvidia-gpus).

---
## Evaluation Results

| Benchmark | n | Base (refusal) | Ablated (refusal) | Δ (pp) |
|---|---:|---:|---:|---:|
| AdvBench | 100 | 99.0% | 4.0% | −95.0 |
| JailbreakBench | 100 | 96.0% | 3.0% | −93.0 |
| StrongREJECT | 150 | 96.7% | 7.3% | −89.4 |
| HarmBench | 150 | 99.3% | 9.3% | −90.0 |
| MaliciousInstruct | 100 | 97.0% | 0.0% | −97.0 |
| SimpleSafetyTests¹ | 50 | 76.0% | 8.0% | −68.0 |
| ForbiddenQuestions | 150 | 67.3% | 4.0% | −63.3 |

Over-refusal on benign prompts: **XSTest-safe 5.6% → 0.0%**, **JBB-benign 23.0% → 0.0%**.
Capability: **MMLU 77.3% → 76.7%**, **GSM8K 87.3% → 90.0%**, **CMMLU 72.6% → 72.2%** — all within noise.

Full tables with capability retention, sample sizes and methodology are under
[Evaluation](#evaluation).

# Verify the ablation

Run:

```bash
python scripts/selfcheck.py \
  --pack /path/to/Ternary-Bonsai-2.5-27B-mlx-2bit
```

`selfcheck.py` measures how much of the residual stream lies along the refusal direction before and after intervention.

A correctly instrumented setup should drive the remaining component to approximately:

```text
~1e-6 of the residual norm
```

If the number barely changes, the residual writers were probably not wrapped correctly.

The self-check also verifies that all expected intervention sites are present.

Expected:

```text
129 residual writers
```

On the Bonsai 2.5 pack, loaded through its own runtime, `install` wraps:

```text
mlp.down_proj 64   linear_attn.out_proj 48   self_attn.o_proj 16   model.embed_tokens 1   = 129
```

On an NVIDIA GPU, `scripts/check_torch.py` runs the same measurement on the fp16
expansion, with the same 129 writers. In bf16 the floor is ~1e-4 rather than ~1e-6 —
that is bf16's precision, not a missed writer. Measured on Bonsai 2.5, harmful prompt,
last token, `|<h, r>| / ||h||` at block outputs:

```text
layer    before      after
    8    7.1e-03    2.3e-05
   16    9.8e-03    4.3e-05
   24    8.7e-02    1.4e-04
   32    2.5e-01    1.1e-04
   40    4.4e-01    1.1e-04
   48    3.5e-01    9.9e-05
   56    2.7e-01    1.2e-04
```

The "before" column is also the evidence that the direction transfers to 2.5: in the
middle of the network it carries up to 44% of the residual norm on a harmful prompt.

(The very last hidden state is taken after the final RMSNorm, whose elementwise weight
does not preserve orthogonality, so it reads ~2e-3 even after a perfect ablation —
measure block outputs, as `selfcheck.py` does.)

Run `python scripts/test_ablation.py` for a model-free numerical check of the compiled correction, including FP16/FP32 outputs, changing directions, and alpha values 0, 1, and 2.

---

# Original vs. uncensored

Because the intervention happens entirely at runtime, the same model pack can be run with or without ablation.

### Original behavior

```bash
python run.py \
  --pack /path/to/Ternary-Bonsai-2.5-27B-mlx-2bit \
  --alpha 0 \
  "your prompt"
```

### Full runtime ablation

```bash
python run.py \
  --pack /path/to/Ternary-Bonsai-2.5-27B-mlx-2bit \
  --alpha 1 \
  "your prompt"
```

There is no second checkpoint.

```text
alpha = 0
     │
     └── Original model behavior

alpha = 1
     │
     └── Full refusal-direction projection
```

This also makes A/B testing straightforward because both configurations use the **same underlying weights**.

---

# Evaluation

Refusal is judged by a rule-based opening-phrase classifier (`caveat` = answered but
wrapped in a disclaimer) — indicative, not an LLM-judge / publication-grade number.
Thinking off, greedy decoding, 64-token budget on the refusal sets. Base and ablated
rows are the same weights in the same process: the published `PTQ1_0` GGUF under
PrismML's llama.cpp fork on one H100, with the adapter at scale 0 against scale 1.

Everything here is reproducible with `scripts/eval_gguf.py`, which draws each prompt set
from its public source with a fixed seed.

## Safety (thinking off)

| Benchmark | n | Base | This Model | Caveat |
|---|---|---|---|---|
| AdvBench | 100 | 99.0% | 4.0% | 29.0% |
| JailbreakBench | 100 | 96.0% | 3.0% | 23.0% |
| StrongREJECT | 150 | 96.7% | 7.3% | 21.3% |
| HarmBench | 150 | 99.3% | 9.3% | 17.3% |
| MaliciousInstruct | 100 | 97.0% | 0.0% | 18.0% |
| SimpleSafetyTests¹ | 50 | 76.0% | 8.0% | 18.0% |
| ForbiddenQuestions | 150 | 67.3% | 4.0% | 26.0% |

## Over-refusal on benign prompts

| Benchmark | n | Base | This Model |
|---|---|---|---|
| XSTest-safe | 250 | 5.6% | 0.0% |
| JBB-benign | 100 | 23.0% | 0.0% |

The projection does not only stop refusals on harmful prompts. The published pack turns
down nearly a quarter of JailbreakBench's *benign* prompts; ablated, it turns down none.

## Capability retention

| Benchmark | n | Base | This Model | Δ |
|---|---|---|---|---|
| MMLU | 300 | 77.3% | 76.7% | −0.6 |
| GSM8K² | 150 | 87.3% | 90.0% | +2.7 |
| CMMLU | 500 | 72.6% | 72.2% | −0.4 |

Every movement is within noise at these sample sizes — one question on GSM8K is 0.7
points. That is the result the runtime approach is for: the weights are bit-identical,
so there is no requantization to pay for.

¹ Understated on both sides. This set is mostly self-harm and abuse prompts. The
published model often answers with a crisis redirect or a flat "No, you should not…",
which the opening-phrase list does not count as a refusal, so its 76% is a floor.
Ablated, some replies open "It is important to clarify…" and push back in the body,
which the classifier scores as compliance, so the 8% is a floor too.

² Zero-shot, thinking off, 512-token budget, last number in the reply. 10–12% of
replies on both sides hit the budget before finishing, so this is a floor set partly by
that budget rather than the model's full GSM8K accuracy; the two columns are comparable
with each other, not with published GSM8K numbers.

## The exact projection agrees

The llama.cpp numbers above are the 128-site adapter. The same direction applied as the
exact 129-site projection — PyTorch hooks on the fp16 expansion, `scripts/check_torch.py`,
same prompts, same classifier — gives:

| Benchmark | n | Base | llama.cpp adapter (128) | PyTorch projection (129) |
|---|---|---|---|---|
| AdvBench | 100 | 99.0% | 4.0% | 3.0% |

The two paths are independent implementations — a rank-1 LoRA inside the fork's ternary
kernels, and output hooks on a dense fp16 model — and they land within one prompt of each
other. The missing embedding site does not show.

---

# Why not release modified weights?

Because modifying the weights defeats one of the most interesting properties of Bonsai.

A conventional abliteration performs:

```text
W' = W - r(rᵀW)
```

But `W'` is no longer ternary.

It contains arbitrary floating-point values.

To store it in the original pack, we would need something approximately equivalent to:

```text
ternary(
    W - r(rᵀW)
)
```

That introduces a new quantization step.

The original Bonsai model, however, achieved its compression through **quantization-aware training**, not through naïve post-training conversion of arbitrary dense matrices.

Instead we preserve:

```text
W
```

exactly and transform its output:

```text
y = Wx

y' = y - α(y·r)r
```

The stored model therefore remains untouched.

---

# 129 intervention sites

One important implementation detail is that **wrapping only `o_proj` is not enough**.

Ternary Bonsai 2.5 keeps the Qwen3.8-27B hybrid architecture:

```text
48 linear-attention layers
16 full-attention layers
64 MLP blocks
```

Each of these can write into the residual stream.

The runtime therefore wraps:

| Residual writer        |   Count |
| ---------------------- | ------: |
| `mlp.down_proj`        |      64 |
| `linear_attn.out_proj` |      48 |
| `self_attn.o_proj`     |      16 |
| `model.embed_tokens`   |       1 |
| **Total**              | **129** |

In other words:

```text
64 + 48 + 16 + 1 = 129
```

Wrapping only:

```text
self_attn.o_proj
```

would intercept only 16 of these sites.

`run.py` prints the number of wrapped residual writers.

`selfcheck.py` warns when it does not detect the expected:

```text
129
```

The 48 `linear_attn.out_proj` writers are also where the llama.cpp path is easiest to get
silently wrong — see [The `ssm_out` column order](#the-ssm_out-column-order).

---

# Do not touch the Hadamard transform

Another important detail is the basis in which the projection is applied.

The Bonsai pack keeps projections in a rotated basis on their **input dimension**.

Its runtime compensates for that rotation on the activation side.

The refusal projection in this repository operates on the **outputs** of those projections.

Those outputs are already in the normal hidden basis.

Likewise, the embedding output is un-rotated by the pack's own `Packed` implementation.

Therefore the refusal direction is simply an ordinary:

```text
5120-dimensional vector
```

No additional Hadamard rotation should be applied to the direction.

Doing so would project against the wrong basis.

---

# Use the model's bundled runtime

The pack must be loaded using **its own bundled runtime**.

An ordinary MLX loader may appear to load the model successfully while silently producing incorrect computation because it does not apply the activation transformations required by the stored weights.

This repository handles that through:

```text
bonsai_abliterate.pack
```

including the detail that:

```text
vision_artifact.load_vl_model
```

is the loader that accepts the:

```text
schema_version: 2
```

configuration used by these packs.

If outputs look completely wrong before ablation is even enabled, verify the pack-loading path first.

The 2.5 pack also ships an optional MTP draft head (`mtp/`). `run.py` does not load it.
A speculative decoder that does would still be governed by the ablated target model —
every drafted token is verified by it — but the draft acceptance rate may drop.

---

# Tuning `alpha`

The intervention does not have to be binary.

`alpha` controls how strongly the refusal direction is removed.

| `alpha` | Effect                                     |
| ------: | ------------------------------------------ |
|     `0` | Projection disabled; original behavior     |
|   `0.5` | Partial ablation                           |
|   `0.7` | Moderate ablation                          |
|   `1.0` | Full projection; default                   |
|  `>1.0` | Over-projection; may degrade model quality |

For example:

```bash
python run.py \
  --pack /path/to/Ternary-Bonsai-2.5-27B-mlx-2bit \
  --alpha 0.7 \
  "your prompt"
```

Under llama.cpp the same knob is the adapter scale, `--lora-scaled adapter.gguf:0.7`, or
the per-request `"lora": [{"id": 0, "scale": 0.7}]`.

Measured on Bonsai 2.5 through the llama.cpp adapter, same setup as
[Evaluation](#evaluation):

| strength | AdvBench refusal | MMLU (n = 300) |
| ---: | ---: | ---: |
| 0 | 99.0% | 77.3% |
| 0.5 | 96.0% | — |
| 0.7 | 86.0% | — |
| 1.0 | 4.0% | 76.7% |
| 1.5 | 0.0% | 75.3% |
| 2.0 | 0.0% | 73.0% |

The response is not linear. Refusal barely moves until about 0.7 and then collapses
between 0.7 and 1.0, so partial strengths are a narrow band rather than a smooth dial.
Above 1.0 the last few refusals go, and capability starts to pay for it: 1.5 costs about
2 MMLU points, 2.0 about 4. 1.0 is the default for that reason.

This makes the intervention a **runtime control parameter** rather than a permanent property of a checkpoint.

---

# Layer-selective ablation

The projection can also be restricted to specific layers.

For example:

```bash
python run.py \
  --pack /path/to/Ternary-Bonsai-2.5-27B-mlx-2bit \
  --layers 20,21,22,23 \
  "your prompt"
```

Layer selection provides another dimension for exploring the trade-off between behavioral intervention and capability retention.

Instead of producing many different checkpoints, experiments can vary:

```text
direction
×
alpha
×
layers
```

against the same immutable model pack.

---

# Direction files

The `directions/` directory contains:

| File                      | Purpose                                  |
| ------------------------- | ---------------------------------------- |
| `refusal_dir.safetensors` | MLX / Python                             |
| `refusal_dir_fp32.bin`    | Swift / C                                |
| `direction.json`          | Dimensions, application rule and caveats |

The Safetensors file stores the vector under:

```text
direction
```

The raw binary representation contains:

```text
5120 × little-endian Float32
```

with no header.

---

# Direction transfer

The refusal direction is the one estimated for the Bonsai 2 release, on the **BF16 base
model** both Bonsai 2 and 2.5 descend from. Bonsai 2.5 is a later release with different
weights, focused on agentic and tool use, so whether that direction still captures
refusal in 2.5 was an open question rather than an assumption.

It was measured, two ways:

* **Representation.** On a harmful prompt the direction carries up to 44% of the
  residual norm in the middle layers of 2.5 (layer 40, table under
  [Verify the ablation](#verify-the-ablation)) — the feature is still there, and still
  large.
* **Behaviour.** Projecting it out takes AdvBench refusals from 99% to the low single
  digits at `alpha = 1`, with no change in capability beyond noise
  ([Evaluation](#evaluation)).

So the direction is reused unchanged. If you want to re-derive one on 2.5 itself, the
fp16 expansion (`scripts/expand_pack.py`) gives you ordinary hidden states under
PyTorch to do it with.

---

# GGUF / llama.cpp

The same edit ships as a rank-1 LoRA adapter, so the published ternary GGUF stays
byte-identical. The adapter is in this repo — nothing to build:

```text
gguf/bonsai2.5-abliterate-lora.gguf     8,668,576 bytes
sha256  064338febbfb3a8d278345aaf6563ca8496daf5331b811f6a3b53a3178d6a3e2
```

```bash
llama-cli -m Ternary-Bonsai-2.5-27B-PTQ1_0.gguf --lora gguf/bonsai2.5-abliterate-lora.gguf
```

It holds 128 `lora_a`/`lora_b` pairs — one per projection writer — as F32, against
`general.architecture = qwen35` and `adapter.lora_alpha = 1.0`.

`W' = W - r (r^T W)` is rank 1, so the whole edit is `A = r^T W`, `B = -r`. llama.cpp
builds a LoRA into the graph as two extra matmuls on the activation and never merges it
into the base weights — which is what makes this work at 1.76 bits, where a baked edit
would simply be rounded away.

## An adapter belongs to one set of weights

`A = rᵀW` contains `W`. **Do not load the Bonsai 2 adapter on Bonsai 2.5.** Both packs
use the same 851 tensor names, so it loads without a warning — and then adds
`-r (rᵀW₂)` to `W₂.₅`, a rank-1 perturbation computed from a different model's matrix,
instead of projecting anything.

## W comes from the GGUF itself

There is no unfolded fp16 checkpoint for Bonsai 2.5, and none is needed. The weights are
exactly ternary, so the matrix the fork multiplies with is fully determined by the GGUF:
decode the codes to `{-s, 0, +s}`, undo the Hadamard fold with the sign vectors the GGUF
declares in its `prism.hadamard.*` metadata, and you have `W` in the plain hidden basis
(`bonsai_abliterate/unfold.py`, numpy only).

```bash
python scripts/export_gguf_lora.py \
  --base-gguf Ternary-Bonsai-2.5-27B-PTQ1_0.gguf \
  --out       bonsai2.5-abliterate-lora.gguf
```

Checked two ways. The decoder agrees with the pack's own `runtime/codec.py` to a relative
error of `0.000e+00` on `ffn_down`, `ssm_out` and `attn_output` (`--check`). And run on
the Bonsai 2 GGUF, it reproduces the adapter that was exported for Bonsai 2 from its
unfolded fp16 checkpoint: median relative error 8.9e-05 across all 256 tensors, cosine
1.000000 on every `lora_a` — the fp16 rounding of that checkpoint, nothing more.

## The `ssm_out` column order

This is the detail that decides whether the adapter works at all.

On a `prism.hadamard.gdn_v_grouped` pack, the fork stores `ssm_out`'s columns in
**grouped** value-head order `[head_dim, rep, n_k]`, while the linear-attention block
produces its output in **tiled** order `[head_dim, n_k, rep]`. The base matmul permutes
the activation to match before multiplying. The LoRA branch is handed the activation
*as produced* — tiled. So `lora_a` for `ssm_out` must be written in tiled order:

```text
A_tiled[h + hd·(k + n_k·r)] = A_grouped[h + hd·(r + rep·k)]
hd = 128, n_k = 16, rep = 3
```

Written in the file's own order, the 48 linear-attention writers load, run, and do
almost nothing: scaled to 100, an `ssm_out`-only adapter barely changes a reply, where
`ffn_down`-only or `attn_output`-only adapters destroy it. Measured on AdvBench at scale
1, that one permutation is the difference between **87%** and **4%** refusals.

The same applies to the Bonsai 2 GGUF under the current fork: there, the Bonsai 2
adapter with `ssm_out` in file order (and `token_embd` removed so that it loads) still
refused 84% of AdvBench at scale 1.

`export_gguf_lora.py` writes the tiled order (`unfold.lora_input_order`).

## The embedding is not on the adapter

The fork refuses a LoRA on a Hadamard latent lookup table, and refuses the whole adapter
with it:

```text
LoRA tensor 'token_embd.weight' targets a prism.hadamard latent lookup table, which is not supported
```

So the adapter carries the 128 projection writers and leaves `token_embd` out. The MLX
and PyTorch runtimes still project the embedding, which is why they have 129 sites. The
cost is small: the embedding's component along `r` is the same for a token in any
context, so it carries no harmful-versus-harmless signal of its own — and the measured
refusal rates of the two paths agree ([The exact projection agrees](#the-exact-projection-agrees)).
`--embedding` writes it anyway, for a runtime that accepts it.

## Strength

`--lora` applies it at scale 1.0, and that is the projection exactly, not an
approximation: llama.cpp computes `scale = adapter_scale * alpha / rank` with
`rank = lora_b->ne[0]`, and the exporter writes `alpha = 1` against rank 1.

Measured on the `PTQ1_0` pack with the fork, greedy, thinking off:

```text
scale 0       the published model: 99% of AdvBench refused
scale 1       exact projection: 4% refused; MMLU unchanged within noise
scale 1.5-2   the last refusals go; MMLU −2 to −4 points
scale 100     the model is destroyed (proof the adapter is in the graph)
```

See [Tuning `alpha`](#tuning-alpha) for the full sweep. Read a single prompt as a sample
of one: at full strength a few percent of every safety set still refuses.

## You need PrismML's fork

Stock llama.cpp cannot open these packs at all. `PTQ1_0` and `PQ2_0` are private ggml
type ids — 143 and 142, outside upstream's range — so upstream refuses them at header
parse. Even `gguf-py` raises:

```text
ValueError: np.uint32(143) is not a valid GGMLQuantizationType
```

which is why `bonsai_abliterate/gguf_min.py` reads the base header directly. Writing the
adapter still uses `gguf-py`, since an adapter holds only standard F32 tensors.

The 2.5 card also publishes a `Q2_0` file that stock llama.cpp loads. The adapter was not
tested against it, and the LoRA-basis behaviour above is the fork's; on any other build,
check that scale 0 reproduces the published model and that a large scale visibly breaks
it before trusting the adapter.

If the fork's build stops at "Provisioning UI assets", configure with
`-DLLAMA_USE_PREBUILT_UI=OFF`.

## What is verified

* The adapter loads on the `PTQ1_0` pack under the fork; scale 0 reproduces the
  published refusals and scale 100 destroys the model, so it is in the compute graph.
* Each writer family is live on its own: `ffn_down`, `attn_output` and (with the tiled
  order) `ssm_out`.
* The decoder matches the pack's codec exactly, and reproduces the Bonsai 2 adapter.
* `B @ A` removes the direction from every written output: the worst remaining leakage
  along `r` across the 128 sites is 1.4e-06 of what it was (`--verify`).
* Rerunning the exporter reproduces `gguf/bonsai2.5-abliterate-lora.gguf` byte for byte
  (same sha256).

Not verified: `PQ2_0` and `Q2_0`. Only `PTQ1_0` was run.

---

# Running on NVIDIA GPUs

The pack's quantized matrix multiplication currently has kernels for:

```text
Metal
CPU
```

but not CUDA through `mlx-cuda`:

```text
QuantizedMatmul has no CUDA implementation
```

So on an NVIDIA machine there are two GPU paths, and neither is the MLX pack itself:

* **llama.cpp** — the published GGUF and the adapter in `gguf/`. This is the serving
  path: 6 GB of weights, the ternary kernels, per-request strength.
* **PyTorch** — `scripts/expand_pack.py` writes the pack's exact ternary values as an
  ordinary fp16 `transformers` checkpoint (about 54 GB; a container change, not a
  requantization), and `bonsai_abliterate.torch_ablation.install` hooks the 129
  writers. This is the research path: hidden states, other directions, other layers.

The MLX pack still runs on an x86 CPU, which is how `selfcheck.py` was run for this
release — minutes per forward pass, useful for checking behaviour, not for serving:

```bash
pip install -r requirements.txt "mlx-cpu==0.32.0"
python scripts/selfcheck.py --pack /path/to/Ternary-Bonsai-2.5-27B-mlx-2bit
```

Keep `mlx`, `mlx-cpu` and `transformers` at the versions in `requirements.txt`: installing
`mlx-vlm` on its own can upgrade `mlx` past the `mlx-cpu` wheel and `transformers` past
the version the pack's runtime pins.

Or in Docker:

```bash
docker build \
  -t bonsai2.5-abliterate \
  -f docker/Dockerfile .

PACK=/path/to/Ternary-Bonsai-2.5-27B-mlx-2bit \
docker/run.sh \
  python run.py \
    --pack /pack \
    --max-new 32 \
    "your prompt"
```

---

# iOS / mlx-swift

A Swift reference implementation is included at:

```text
swift/RefusalAblation.swift
```

Load:

```text
directions/refusal_dir_fp32.bin
```

as:

```text
5120 little-endian Float32
```

and wrap each residual writer with:

```text
AblatedLinear
```

Once the model integration exists, the refusal intervention itself is small.

Per residual write it adds approximately:

```text
1 × dot product
1 × AXPY
```

No alternative model weights need to be stored.

`scripts/prepare_ios_pack.py` drops the vision tower and the redundant group biases for a
text-only app, as for Bonsai 2. The 2.5 pack has the same format and tensor shapes, but
the memory budget was not re-measured on it.

---

# Architecture

The key distinction of this release is that **uncensoring is a runtime property rather than a checkpoint property**.

```text
┌─────────────────────────────────────────────┐
│       Ternary Bonsai 2.5 · 27B              │
│                                             │
│       ~1.76 bits / weight                   │
│       original packed weights              │
└──────────────────┬──────────────────────────┘
                   │
                   │ weights remain
                   │ bit-identical
                   ▼
┌─────────────────────────────────────────────┐
│             Runtime Inference               │
│                                             │
│  residual writer                            │
│        │                                    │
│        ▼                                    │
│  y = writer(x)                              │
│        │                                    │
│        ▼                                    │
│  y' = y - α · dot(y,r) · r                  │
│        │                                    │
│        ▼                                    │
│  residual stream                            │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
                 output
```

### Properties

```text
Model parameters           27B
Effective weight size      ~1.76 bits/weight
Hidden dimension           5120

Residual intervention
sites                       129 (MLX, PyTorch)
                            128 (llama.cpp)

Weight modification        None
Weight re-quantization     None
Additional weight
quantization error         None

Intervention strength      Runtime adjustable
Layer selection            Runtime adjustable
Original behavior          alpha=0
Full projection            alpha=1
```

---

# Why runtime intervention is useful

Permanent weight editing couples a behavioral modification to a particular checkpoint.

Runtime intervention separates the two.

```text
                 MODEL
                   │
            immutable weights
                   │
                   ▼
              INFERENCE
                   │
          ┌────────┴────────┐
          │                 │
      alpha = 0         alpha = 1
          │                 │
          ▼                 ▼
      original          ablated
      behavior          behavior
```

The same architecture can potentially support more than one learned direction without generating another copy of the underlying model.

Conceptually:

```text
Immutable Model
      │
      ▼
Runtime Intervention
      │
      ├── direction
      ├── strength
      └── layer scope
      │
      ▼
Inference
```

This repository currently implements **refusal-direction ablation**.

---

# What this release is — and isn't

This is **not a newly trained 27B model**.

It is a runtime behavioral intervention for the existing Ternary Bonsai 2.5 27B packs.

The underlying model architecture, QAT training, ternary representation and original packed weights come from the upstream Bonsai release.

The contribution here is the runtime refusal-direction implementation, direction artifacts, residual-writer instrumentation and verification tooling.

That distinction is intentional:

> **The model stays immutable. Behavioral intervention happens at inference.**

---

# Responsible use

Removing a learned refusal direction can cause the model to respond to requests that the original model would decline.

The technique should therefore be treated as a research and inference-control mechanism, not as evidence that every resulting output is safe, correct or appropriate.

Deployments should apply their own access controls, policy enforcement and security boundaries appropriate to their use case.

---

# Credits

Built on:

**Ternary Bonsai 2.5 27B** by `prism-ml`

Runtime refusal-direction implementation and tooling by **[`OrcaRouter`](https://www.orcarouter.ai)**.

The original model pack is not redistributed or modified by this repository.

---

# License

Apache-2.0, matching the underlying model.
