---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - UNETR
  - UNet Transformer
family: transformer
registry_key: unetr
summary: Pure ViT encoder with a CNN decoder receiving skips from four evenly-spaced ViT layer outputs.
---

# UNETR

`UNETR` (`models/backbone/unetr.py`) uses a pure Vision Transformer (ViT) encoder with a CNN decoder that receives skip connections from intermediate ViT layer outputs ([[UNETR_Hatamizadeh2021_2103.10504.pdf|Hatamizadeh et al., 2022]]).

---

## Summary

UNETR tokenises the input patch directly into non-overlapping patches and processes the token sequence through $L$ = `transformer_layers` ViT layers without any CNN. The decoder extracts features from four evenly spaced ViT layers and uses CNN blocks to progressively recover spatial resolution. The skip-layer indices (0-based) are computed as $\{L/4 - 1,\; L/2 - 1,\; 3L/4 - 1,\; L - 1\}$; for the default $L = 8$ this gives layers $\{1, 3, 5, 7\}$. The constructor requires $L \ge 4$ and exactly four `decoder_features`.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | Input patch, $\mathbb{R}^{B \times C \times H \times W}$ |
| $\mathbf{z}_0$ | Patch embedding sequence |
| $\mathbf{z}_{l-1}$ | Input token sequence to layer $l$, $\mathbb{R}^{N \times d}$ |
| $\mathbf{z}'_l$ | Intermediate sequence after attention + residual |
| $\mathbf{z}_l$ | Output sequence of layer $l$ |
| $\mathbf{E}_{\text{pos}}$ | Learnable position embeddings, $\mathbb{R}^{N \times d}$ |
| $\mathbf{g}$ | Tap reshaped to the $(B, d, H/p, W/p)$ grid |
| $\mathbf{h}_0$ | Front projection output ($3\times3$ conv + norm + act) |
| $\mathbf{u}_s$ | $s$-th transposed-convolution (deconv) output |
| $\mathbf{c}_s$ | $s$-th $3\times3$ convolution output |
| $\mathbf{h}_s$ | $s$-th blue-block output after norm + act |
| $\mathbf{p}$ | Final projected skip feature map, width `decoder_features[...]` |
| $E$ | Linear patch embedding, $\mathbb{R}^{(p^2 C) \times d}$ |
| $B$ | Batch size |
| $C$ | Input channel count |
| $H, W$ | Input spatial dimensions (default `image_size = 256`, giving a $16 \times 16$ token grid) |
| $N$ | Number of patch tokens, $(H/p)(W/p)$ |
| $d$ | Hidden dimension, `embedding_dim` (default `544`) |
| $p$ | `patch_size` (default `16`) |
| $L$ | Number of ViT layers, `transformer_layers` |
| $S$ | Number of blue blocks (upsample steps): $3, 2, 1$ for $\mathbf{z}_3, \mathbf{z}_6, \mathbf{z}_9$ and $0$ for the $\mathbf{z}_{12}$ bottleneck |
| $\text{MSA}$ | Full multi-head self-attention (global, no window) |
| $\text{MLP}$ | Feed-forward network |
| $\text{LN}$ | Layer normalisation |
| $\text{Norm}$ | Normalisation (`normalization`, default batch) |
| $\text{Act}$ | Activation (`activation`, default ReLU) |

---

## Architecture

### ViT Patch Embedding

The input $\mathbf{x} \in \mathbb{R}^{B \times C \times H \times W}$ is split into $N = (H/p) \cdot (W/p)$ patches of size $p \times p$:

$$
\mathbf{z}_0 = [\mathbf{x}_{p,1}E; \, \mathbf{x}_{p,2}E; \, \dots; \, \mathbf{x}_{p,N}E] + \mathbf{E}_{\text{pos}}
$$

### Transformer Block

$$
\mathbf{z}'_l = \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}
$$

$$
\mathbf{z}_l = \text{MLP}(\text{LN}(\mathbf{z}'_l)) + \mathbf{z}'_l
$$

### Decoder Skip Connections

ViT outputs at the four selected layers (default $\{1, 3, 5, 7\}$, see Summary) are reshaped from $(B, N, d)$ to $(B, d, H/p, W/p)$ and processed by `ProgressiveProjectionHead` modules. Each head opens with a $3\times3$ conv + norm + act front projection (the paper's "consecutive $3\times3$ convolutional layers"), then applies a sequence of blue blocks, each composed of a $2\times2$ transposed convolution followed by a $3\times3$ conv + norm + act (Fig. 2 legend, "Deconv $2\times2\times2$, Conv $3\times3\times3$, BN, ReLU"). The three shallower skip maps use 3, 2, and 1 such blue blocks respectively, reaching the decoder feature widths `decoder_features[-1]`, `decoder_features[-2]`, `decoder_features[-3]`; the deepest layer output is projected (0 upsample steps, i.e. front projection only) to `decoder_features[0]` and serves as the bottleneck input. The original input image is additionally processed by a separate `ConvBlock` (`input_skip_conv`) to form the finest-resolution skip. The four CNN decoder blocks then upsample and concatenate with these skips in turn.

For a tap reshaped to the grid feature map $\mathbf{g} \in \mathbb{R}^{B \times d \times H/p \times W/p}$ and processed with $S$ blue blocks, the head is the recurrence

$$
\begin{aligned}
\mathbf{h}_0 &= \text{Conv}_{3\times3}(\mathbf{g}) \\
\mathbf{h}_0 &= \text{Act}(\text{Norm}(\mathbf{h}_0)) \\
\mathbf{u}_s &= \text{Deconv}_{2\times2}(\mathbf{h}_{s-1}) \\
\mathbf{c}_s &= \text{Conv}_{3\times3}(\mathbf{u}_s) \\
\mathbf{h}_s &= \text{Act}(\text{Norm}(\mathbf{c}_s)) \\
\mathbf{p} &= \mathbf{h}_S
\end{aligned}
$$

where lines for $\mathbf{u}_s, \mathbf{c}_s, \mathbf{h}_s$ repeat for $s = 1, \dots, S$, and the final projection $\mathbf{p}$ has spatial size $2^S$ times the token grid. The bottleneck tap uses $S = 0$, so the head reduces to the front projection $\mathbf{p} = \mathbf{h}_0$.

---

## Design Rationale

**Full global attention from the first layer.** Unlike [[SwinUNet]] (window attention) and [[TransUNet]] (attention only at bottleneck), UNETR applies full self-attention at every layer, at the cost of $O(N^2)$ attention complexity in the token count $N$. For the default configuration (`image_size = 256`, `patch_size = 16`), the token grid is $16 \times 16$, i.e. $N = 256$ tokens, which keeps the attention cost tractable.

> **Pure transformer encoder**
> UNETR uses no CNN encoder, so it carries no built-in inductive bias for local spatial structure — every layer sees the full token sequence through global self-attention.

**No CNN encoder.** UNETR has no inductive bias for local spatial structure. This may be a limitation for SAR data where fine-scale texture is informative. This is an open empirical question in the project context.

---

## Parameter Reference

See [[Configuration Layer]] → `UNETRConfig`.

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| `image_size` | $H, W$ | `256` | Input patch side length (must be divisible by `patch_size`) |
| `patch_size` | $p$ | `16` | Patch embedding kernel/stride |
| `embedding_dim` | $d$ | `544` | ViT token embedding dimension |
| `transformer_layers` | $L$ | `8` | Number of ViT layers (must be $\ge 4$) |
| `transformer_heads` | — | `8` | Attention heads per layer |
| `transformer_mlp_ratio` | — | `4.0` | FFN hidden expansion ratio |
| `decoder_features` | — | `[360, 184, 88, 48]` | CNN decoder channel widths (must have length 4) |
| `attention_dropout` | — | `0.0` | Dropout on attention weights |
| `stochastic_depth_rate` | — | `0.0` | Maximum DropPath rate across layers |
| `dropout` | — | `0.15` | ConvBlock and FFN dropout |
| `activation` | — | `"relu"` | CNN activation |
| `ffn_activation` | — | `"gelu"` | Transformer FFN activation |
| `normalization` | — | `"batch"` | CNN normalisation layer |
| `in_channels` | — | `1` | Input channel count |
| `out_channels` | — | `6` | Output channel count ($3K$ for $K=2$ Gaussians) |

The default `embedding_dim = 544` is divisible by `transformer_heads = 8`. Every field above (except `shape_logger_types`) is persisted per run by `ModelConfigIO` and reloaded verbatim at inference; see [[Model Zoo]] → Configuration Persistence.

---

## Paper fidelity

*Review date: 2026-06-04. Ground truth: Hatamizadeh et al., "UNETR: Transformers for 3D Medical Image Segmentation", arXiv:2103.10504 (Sec. 3.1 Architecture, Fig. 2, Eq. 1-6; [[UNETR_Hatamizadeh2021_2103.10504.pdf|PDF]]). Code under review: `models/backbone/unetr.py`, `configuration/architectures/backbone.py` (`UNETRConfig`). The original is 3D ($H\times W\times D$, $P^3$ patches, $3\times3\times3$ convs, Deconv $2\times2\times2$); our implementation is the 2D analog ($H\times W$, $p^2$ patches, $3\times3$ convs, Deconv $2\times2$) and is judged on the 2D analog of each structure.*

**Overall verdict: faithful 2D adaptation.** Patch-direct embedding (no CNN encoder), the pre-norm ViT encoder (Eq. 1-3), the $\{z_3,z_6,z_9,z_{12}\}$ even-spacing skip rule, the per-skip deconv-step counts (3/2/1), the bottleneck-deconv-then-merge wiring, the multi-scale concat-then-conv decoder, the full-resolution input-stem skip, and the $1\times1$ output head all reproduce Fig. 2. Each deconv is followed by a $3\times3$ conv + norm + act and the front projection is a $3\times3$ conv, matching the Fig. 2 legend. The patch embedding, transformer block, and attention are the shared `PatchEmbedding`, `TransformerBlock`, and `MultiHeadSelfAttention` from `models/blocks.py`, the same modules used by [[TransUNet]]; only `ProgressiveProjectionHead` and the `UNETR` shell are local to `models/backbone/unetr.py`. One minor deviation: the patch embedding adds a `LayerNorm` not present in Eq. 1.

| # | Component (paper ref) | Code site | Verdict |
|---|---|---|---|
| 1 | Direct patch embedding, no CNN encoder; linear projection of non-overlapping $p\times p$ patches (Eq. 1) | `PatchEmbedding` `blocks.py:419-441`, `UNETR.py:72-76,199` | Match (2D analog) |
| 1b | Learned 1D positional embedding, no class token (Eq. 1) | `UNETR.py:78-79,200` | Match |
| 1c | Extra `LayerNorm` on patch embedding (not in Eq. 1) | `PatchEmbedding` `blocks.py:434,440` | Deviation (minor) |
| 2 | ViT block: pre-norm, $z'=\mathrm{MSA}(\mathrm{LN}(z))+z$, $z=\mathrm{MLP}(\mathrm{LN}(z'))+z'$, GELU MLP (Eq. 2-3) | `TransformerBlock` `blocks.py:380-416` | Match |
| 2b | Scaled dot-product MSA, $K_h=K/n$ scaling (Eq. 4-6) | `MultiHeadSelfAttention` `blocks.py:343-377` | Match |
| 3 | Even-spaced skip taps $z_3,z_6,z_9,z_{12}$ via quartile rule; final LN on deepest tap | `UNETR.py:96-102,205-207` | Match (2D analog) |
| 4 | Reshape each tapped sequence to a 2D grid $(B,d,H/p,W/p)$ | `tokens_to_feature_map` `blocks.py:444-446`, `UNETR.py:207` | Match (2D analog) |
| 5 | Per-skip deconv counts: $z_9$=1, $z_6$=2, $z_3$=3 | `UNETR.py:105-130` | Match (count) |
| 5b | Blue-block composition: each deconv followed by Conv $3\times3$+BN+ReLU (Fig. 2 legend) | `ProgressiveProjectionHead` `UNETR.py:10-54` | Match (2D analog) |
| 6 | Bottleneck $z_{12}$ projected then deconv by 2, merged with $z_9$ (Fig. 2) | `UNETR.py:131-138,215,218-227` | Match (2D analog) |
| 7 | Decoder per scale: concat then conv blocks then deconv | `UNETR.py:149-176,217-227` | Accepted adaptation |
| 8 | Full-resolution input stem: raw input through conv blocks, concat at last stage (Fig. 2 bottom) | `UNETR.py:140-147,223,226` | Match (2D analog) |
| 9 | Output head $1\times1$ conv | `UNETR.py:188-192,231` | Match (2D analog) |
| 10 | Resolution / channel bookkeeping of every merge | `UNETR.py:149-176,217-230` | Match |

**Encoder (items 1-2).** The patch embedding is a strided $\mathrm{Conv2d}$ with kernel = stride = $p$ (shared `PatchEmbedding`, `blocks.py:419-441`), the exact equivalent of the per-patch linear projection $E$ in Eq. 1; the learned positional table (`UNETR.py:78-79`) is added with no class token, matching the paper's explicit note that the `[class]` token is dropped for segmentation. The transformer block (shared `TransformerBlock`, `blocks.py:411-416`) implements the pre-norm form of Eq. 2-3 verbatim — $\mathrm{LN}\to\mathrm{MSA}\to$ residual, then $\mathrm{LN}\to\mathrm{MLP}\to$ residual — with a GELU two-layer MLP (`ffn_activation = "gelu"`). The attention (shared `MultiHeadSelfAttention`, `blocks.py:363-377`) is standard scaled dot-product with the $1/\sqrt{K_h}$ scaling of Eq. 4. The paper fixes $L=12$, $K=768$; here $L$ and $d$ are hyperparameters (default $L=8$, $d=544$) and out of scope.

**Skip rule (item 3).** The paper taps the fixed layers $\{3,6,9,12\}$, i.e. the quartiles of $L=12$. The code generalises this to $\{L/4,\,L/2,\,3L/4,\,L\}$ (0-based: `total_layers//4 - 1`, etc., `UNETR.py:96-102`), which reproduces $\{2,5,8,11\}$ (0-based for $\{3,6,9,12\}$) at $L=12$ and stays evenly spaced for any $L$. The final transformer `LayerNorm` is applied only to the deepest tap (`UNETR.py:206`), consistent with the paper applying Norm at the encoder output / bottleneck; the three shallower taps are forwarded un-normed, which is the standard reading of Fig. 2. Match.

**Projection / blue block (items 4-5).** Each tapped sequence is reshaped to the grid (shared `tokens_to_feature_map`, `blocks.py:444-446`, called at `UNETR.py:207`) and passed through a `ProgressiveProjectionHead`. The number of upsample steps (3 for $z_3$, 2 for $z_6$, 1 for $z_9$, 0 for the $z_{12}$ bottleneck) exactly matches the blue-block counts in Fig. 2. The head (`UNETR.py:21-50`) opens with a $3\times3$ conv + norm + act front projection (matching the paper's "consecutive $3\times3$ convolutional layers" projection) and repeats `[ConvTranspose2d, Conv $3\times3$, Norm, Act]` per upsample step, the 2D analog of the Fig. 2 legend "Deconv $2\times2\times2$, Conv $3\times3\times3$, BN, ReLU". The bottleneck head (`upsample_steps=0`) is the $3\times3$ conv + norm + act front projection alone. Each head emits its target `decoder_features[...]` width, feeding the decoder wiring directly. Match.

**Bottleneck and decoder (items 6-7).** The $z_{12}$ tap is projected with 0 upsample steps (`UNETR.py:131-138`), staying at the $H/p$ grid; decoder step `idx0` then deconvs it $\times2$ and concatenates the $z_9$ projection (`UNETR.py:215,218-227`), reproducing the Fig. 2 "deconv bottleneck by 2, then merge with $z_9$" path. Each decoder level (`UNETR.py:217-227`) does ConvTranspose-up $\to$ resize-to-skip $\to$ concat $\to$ double-$3\times3$-conv ConvBlock. The paper draws the level as concat $\to$ yellow convs $\to$ green deconv, i.e. the deconv that the code places at the *start* of level $n{+}1$ the paper draws at the *end* of level $n$. This is the standard equivalent regrouping of the U-Net decoder ladder; the connectivity graph and channel flow are identical. Accepted adaptation. The yellow conv blocks (Conv $3\times3$ + BN + ReLU, $\times2$) are faithfully realised by the shared `ConvBlock` (`blocks.py:113-149`).

**Stem and output (items 8-9).** The raw input is processed by `input_skip_conv` (a double-$3\times3$ ConvBlock, `UNETR.py:140-147`) and concatenated at the final, full-resolution decoder stage (`UNETR.py:223,226`), matching the Fig. 2 bottom path (Input $\to$ two yellow convs at $H\times W\times64 \to$ concat before the output head). The output head is a $1\times1$ conv (`UNETR.py:188-192`); the paper's softmax is a loss-side detail and out of scope.

**Resolution / channel bookkeeping (item 10).** With default $p=16$ the token grid is $H/16$. Projected skip resolutions are $z_9\to H/8$ (1 deconv), $z_6\to H/4$ (2), $z_3\to H/2$ (3); the bottleneck stays at $H/16$ and is deconv'd to $H/8$ before the $z_9$ merge. The decoder ladder ($H/16\to H/8\to H/4\to H/2\to H$, channels $360\to184\to88\to48\to48$) meets each skip at the correct scale, and `final_upsample` resolves to `Identity` because `patch_size // 2**4 = 1`. Every concat site is dimensionally consistent. Match.

**Deviation summary.**
- *Patch-embedding `LayerNorm`* — minor. Paper: Eq. 1 (linear projection + pos embed only). Code: shared `PatchEmbedding` `blocks.py:434,440`. The extra `self.norm` is a benign modern-ViT stabiliser with no structural impact; because `PatchEmbedding` is shared with [[TransUNet]], it applies to both.

---

## Related Notes

- [[Model Zoo]] — Architecture comparison
- [[SwinUNet]] — Window-attention alternative
- [[TransUNet]] — CNN + ViT hybrid
- [[Configuration Layer]] — UNETRConfig
- [[DLR-TomoSAR Index]] — Master index
