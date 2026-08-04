---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - SwinUNet
  - Swin Transformer U-Net
  - Swin-UNet
family: transformer
registry_key: swin_unet
summary: Hierarchical Swin Transformer encoder with symmetric Swin-based decoder for dense regression on SAR patches.
---

# SwinUNet

`SwinUNet` (`models/backbone/swin_unet.py`) uses a hierarchical Swin Transformer ([[SwinTransformer_Liu2021_2103.14030.pdf|Liu et al., 2021]]) as the encoder and a symmetric Swin-based decoder with patch merging/expanding, adapted for dense regression on SAR patches ([[SwinUNet_Cao2021_2105.05537.pdf|Cao et al., 2022]]).

---

## Summary

SwinUNet partitions the input patch into non-overlapping windows and applies self-attention within each window. Shifted-window attention across layers enables cross-window information flow. Patch merging hierarchically reduces spatial resolution while increasing channel depth (analogous to max-pooling in CNN encoders).

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{Z} \in \mathbb{R}^{N \times C}$ | Input token sequence within a window |
| $\hat{\mathbf{Z}}$ | Intermediate token sequence after attention and residual |
| $\mathbf{Z}'$ | Output token sequence after MLP and residual |
| $N = M^2$ | Number of tokens per window |
| $N/4$ | Token count after merging $2\times2$ spatial regions |
| $C$ | Embedding dimension at the current level |
| $2C$ | Doubled channel count after patch-merging linear projection |
| $M$ | Window side length |
| $h$ | Number of attention heads |
| $d_k = C / h$ | Head dimension |
| $Q_i, K_i, V_i \in \mathbb{R}^{N \times d_k}$ | Query, key, value projections for head $i$ |
| $W^Q_i, W^K_i, W^V_i \in \mathbb{R}^{C \times d_k}$ | Per-head query, key, value projection matrices (implemented jointly as a single `Linear(C, 3C)`) |
| $W^O \in \mathbb{R}^{C \times C}$ | Output projection matrix (`output_projection`) |
| $B \in \mathbb{R}^{M^2 \times M^2}$ | Per-head relative position bias, indexed from a learnable table of shape $((2M-1)^2, h)$ |
| $\text{LN}$ | Layer normalisation |

---

## Architecture

### Swin Transformer Block

For a sequence of tokens $\mathbf{Z} \in \mathbb{R}^{N \times C}$ within a window of size $M \times M$:

**Window Multi-Head Self-Attention (W-MSA):**

$$
\text{head}_i = \text{Softmax}\!\left(\frac{Q_i K_i^T}{\sqrt{d_k}} + B\right) V_i
$$

$$
\text{W-MSA}(\mathbf{Z}) = \text{cat}[\text{head}_1, \dots, \text{head}_h] W^O
$$

where $Q_i = \mathbf{Z} W^Q_i$, $K_i = \mathbf{Z} W^K_i$, $V_i = \mathbf{Z} W^V_i$.

**Shifted-Window MSA (SW-MSA):** alternate layers shift the window partitioning by $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ to enable cross-window attention.

**Swin block:**

$$
\hat{\mathbf{Z}} = \text{W-MSA}(\text{LN}(\mathbf{Z})) + \mathbf{Z}
$$

$$
\mathbf{Z}' = \text{MLP}(\text{LN}(\hat{\mathbf{Z}})) + \hat{\mathbf{Z}}
$$

### Hierarchical Encoding

Patch merging at level $l$ concatenates $2\times2$ neighbouring token groups and projects to $2C$ channels:

$$
\text{PatchMerge}: \mathbb{R}^{N \times C} \to \mathbb{R}^{N/4 \times 2C}
$$

This halves spatial resolution and doubles channels at each level (analogous to CNN encoder downsampling).

---

## Design Rationale

**Global context.** For TomoSAR, some scattering structures span tens of pixels (building facades, roof-to-ground layover). CNN architectures with limited receptive fields struggle to model these. Swin attention within windows of configurable size provides long-range context at a cost proportional to window size rather than image size.

> **Resolution handling.** SwinUNet uses only window-relative position bias (the `relative_position_bias_table`, sized by `window_size`), not absolute positional embeddings. The implementation never reads `image_size` in `SwinUNet.__init__`: the token grid is inferred dynamically from the patch-embedding convolution output, and arbitrary patch sizes are accommodated by per-block window padding and the `match_spatial_size` interpolation guard. The `image_size` config field is therefore documentary only and does not constrain the runtime input resolution.

---

## Parameter Reference

See [[Configuration Layer]] → `SwinUNetConfig`. Per-stage embedding dimensions are derived as $\text{dims}[i] = C \cdot 2^{i}$ from `embedding_dim`; the implementation asserts $\text{dims}[i] \bmod h_i = 0$ for every stage, where $h_i$ is the per-stage `num_heads`.

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| `image_size` | — | `256` | Documentary nominal patch side length; not read at construction (resolution is inferred dynamically) |
| `patch_size` | $p$ | `4` | Patch embedding conv kernel/stride |
| `embedding_dim` | $C$ | `80` | Stage-0 token embedding dimension |
| `depths` | — | `[2, 2, 6, 2]` | Number of Swin blocks per stage |
| `num_heads` | $h$ | `[2, 5, 10, 20]` | Attention heads per stage |
| `window_size` | $M$ | `7` | Side length of the local attention window |
| `mlp_ratio` | — | `4.0` | FFN hidden expansion ratio |
| `dropout` | — | `0.30` | Dropout in attention output and FFN |
| `attention_dropout` | — | `0.10` | Dropout on attention weights |
| `ffn_activation` | — | `"gelu"` | FFN activation |
| `stochastic_depth_rate` | — | `0.10` | Maximum DropPath rate (linearly scaled across blocks) |
| `in_channels` | — | `1` | Input channel count |
| `out_channels` | — | `6` | Output channel count ($3K$ for $K=2$ Gaussians) |

With `embedding_dim = 80` and four stages, the per-stage dimensions are $[80, 160, 320, 640]$. The number of stages equals `len(depths)`, which must equal `len(num_heads)`. Every field above (except `shape_logger_types`) is persisted per run by `ModelConfigIO` and reloaded verbatim at inference; see [[Model Zoo]] → Configuration Persistence.

---

## Paper fidelity

**Review date:** 2026-06-04

**Ground-truth sources:**
- Cao et al. (2021), *Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation*, arXiv:2105.05537 (Sections 3.1–3.6, Fig. 1–2). [[SwinUNet_Cao2021_2105.05537.pdf|PDF]]
- Liu et al. (2021), *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*, arXiv:2103.14030 (Sections 3.1–3.3, Eq. 1–4, Fig. 2–4). [[SwinTransformer_Liu2021_2103.14030.pdf|PDF]]

**Overall verdict:** Faithful implementation. The Swin block, shifted-window attention with masking, relative position bias, patch merging, patch expanding, and skip-connection fusion all reproduce the paper formulations exactly. The only structural choice differing from Swin-Unet's headline configuration is the final upsampling head, which uses `ConvTranspose2d` rather than the $4\times$ `PatchExpand` + linear projection (an author-sanctioned alternative: the paper itself ablates transposed convolution as a valid upsampling choice in Table 3, where patch expanding merely scores marginally higher; see Deviations note). Stage symmetry and bottleneck depth follow the host configuration rather than being hard-wired to the paper's exact `[2,2,6,2]` + `x2` bottleneck, but the mechanism is correct.

**Component verdict table:**

| # | Component | Paper ref | Code | Verdict |
|---|---|---|---|---|
| 1 | Patch partition + linear embedding | Swin-Unet Fig. 1, Sec. 3.1 ($4\times4$) | `SwinUNet.py:305-313` | ACCEPTED ADAPTATION |
| 2 | Swin block pair (W-MSA / SW-MSA, pre-norm LN, MLP-GELU, residuals) | Swin Eq. 1–4, Fig. 2 | `SwinUNet.py:81-203`, `267` | MATCH |
| 3 | Cyclic shift + 9-region mask + reverse roll | Swin Sec. 3.2, Fig. 4 | `SwinUNet.py:118-152, 164-197` | MATCH |
| 4 | Relative position bias $(2M-1)^2$ table, index, inside softmax | Swin Eq. 4 | `SwinUNet.py:33-47, 58-62` | MATCH |
| 5 | Patch merging: concat $2\times2$ → LN($4C$) → Linear($4C{\to}2C$) | Swin Sec. 3.1; Swin-Unet Sec. 3 (patch merging) | `SwinUNet.py:205-229` | MATCH |
| 6 | Patch expanding: Linear($C{\to}2C$) → rearrange to $2\times$ res, $C/2$ | Swin-Unet Sec. 3.5 | `SwinUNet.py:232-248` | MATCH |
| 7 | Skip connection: concat encoder+decoder tokens → Linear halving | Swin-Unet Sec. 3.6 | `SwinUNet.py:367, 428-429` | MATCH |
| 8 | Bottleneck Swin blocks | Swin-Unet Sec. 3.4 (two blocks) | `SwinUNet.py:355, 413` | ACCEPTED ADAPTATION |
| 9 | Final $4\times$ patch expanding + linear projection head | Swin-Unet Fig. 1, Sec. 3.1 | `SwinUNet.py:382-392, 433-435` | ACCEPTED ADAPTATION |
| 10 | Encoder/decoder symmetry; $d^{-1/2}$ scaling; qkv bias | Swin Eq. 4–5 | `SwinUNet.py:26, 28, 361-380` | MATCH |

**Prose.**

*Swin block and attention (items 2–4).* The block follows Swin Eq. (1)–(4) precisely: pre-norm `LN` before both the attention and MLP sub-blocks, residual additions after each, and a 2-layer MLP with GELU at `mlp_ratio = 4` (`SwinUNet.py:110-116`, `180`, `200-202`). A single shared `SwinStage` class (`SwinUNet.py:251`) builds both the encoder and decoder stages; within it the shift alternates `0` and `window_size // 2` across consecutive blocks (`SwinUNet.py:267`), realising the W-MSA then SW-MSA pairing of Fig. 2 and matching $\lfloor M/2 \rfloor$ from Swin Sec. 3.2. Attention uses scaling $d_k^{-1/2}$ (`scale = head_dim ** -0.5`, line 26) and a fused `Linear(C, 3C)` with bias for $Q,K,V$ (line 28, qkv bias present), consistent with Eq. (5). The relative position bias is added inside the softmax (line 62), exactly as in Swin Eq. (4); the table has shape $((2M-1)^2, h)$ (line 34) and the index construction (lines 38–46) reproduces the reference algorithm: per-axis offset by $M-1$, row index scaled by $2M-1$, then summed. Note the paper's text contains a typographical inconsistency, writing $\tilde{B}\in\mathbb{R}^{(2M-1)\times(2M+1)}$ in Swin-Unet but $\mathbb{R}^{(2M-1)\times(2M-1)}$ in Swin Eq. (4); the code follows the correct $(2M-1)\times(2M-1)$ form.

*Shifted-window masking (item 3).* The cyclic shift uses `torch.roll` by $-\text{shift}$ on both spatial axes (line 165) with the inverse roll after attention (line 192), per Swin Sec. 3.2 "efficient batch computation" (Fig. 4). The mask builds the nine sub-regions from the three-way height/width slice product (lines 123–137), partitions into windows, and forms the pairwise difference mask with $-100$ for cross-region pairs and $0$ otherwise (lines 149–151). This is the standard reference construction; the $-100$ additive penalty is the conventional finite stand-in for $-\infty$ and is an accepted choice.

*Patch merging and expanding (items 5–6).* Patch merging concatenates the four $2\times2$ strided sub-grids, applies `LN` over the $4C$ concatenation, then a bias-free `Linear(4C, 2C)` (lines 220–228) — exactly Swin Sec. 3.1 and the Swin-Unet patch-merging description. Patch expanding (lines 232–248) is the subtle case demanded by the protocol: the paper's first-layer worked example takes $8C$ input, applies a linear to $2\times$ the dimension ($16C$), then rearranges to $2\times$ resolution and a quarter of that dimension ($4C$). In code, `output_dim = input_dim/2`, so `expand = Linear(input_dim, 4*output_dim) = Linear(8C, 16C)` (line 235) matches the $2\times$-dimension linear; the subsequent `view`/`permute`/`view` (lines 242–245) rearranges $16C$ into a $2\times2$ spatial block of `output_dim = 4C` channels, doubling $H$ and $W$ and reducing channels to a quarter of $16C$. This reproduces the paper's expand-then-rearrange exactly (LN applied post-rearrange, line 246). Verdict MATCH.

*Skip connections and symmetry (items 7, 10).* Decoder stages concatenate upsampled tokens with the corresponding encoder tokens along the channel axis and apply `Linear(2C, C)` (the `skip_projections`) to restore the dimension (lines 367, 428–429), as in Swin-Unet Sec. 3.6. Encoder and decoder stage depths are mirrored through `decoder_index` indexing into the shared `depths`/`num_heads` lists (lines 361–380), giving the symmetric U-shape of Fig. 1.

*Accepted adaptations.* (1) Patch embedding (item 1) uses a single `Conv2d(stride=patch_size)` rather than an explicit unfold + `Linear` on the flattened $p^2\cdot\text{in\_channels}$ vector; these are mathematically equivalent for non-overlapping patches and the convolutional form is the universal reference implementation. (2) Bottleneck (item 8): the paper fixes two successive Swin blocks; the code runs `depths[-1]` blocks in the deepest encoder stage (default `2`, so equal by default) followed by a single `bottleneck_norm` `LayerNorm` (line 355), with no separate bottleneck stage. The mechanism — deepest-resolution Swin blocks at unchanged dimension/resolution — is preserved; depth is configuration-driven. (3) Padding for non-divisible $H,W$ in the Swin block (lines 158–162) and patch merging (lines 214–219), plus the `match_spatial_size` interpolation guard (imported from `models/blocks.py`), are robustness additions for arbitrary SAR patch sizes and are explicitly in scope as accepted.

**Deviations.**

- *None of structural or minor severity.* The only item worth flagging is the final $4\times$ upsampling head (item 9). Swin-Unet Fig. 1 and Sec. 3.1 illustrate a final $4\times$ `PatchExpand` (expand-then-rearrange) followed by a linear projection; the code instead uses `ConvTranspose2d(dims[0], dims[0], kernel_size=patch_size, stride=patch_size)` for the $4\times$ upsample (`SwinUNet.py:382-387`), then a $1\times1$ `Conv2d` head (lines 388–392). The paper's Table 3 ablates transposed convolution as an author-sanctioned upsampling option (patch expanding scores marginally higher), so the convolutional head is a paper-endorsed variant rather than a fidelity defect; the output spatial geometry is identical either way. A strictly Fig. 1-matching variant would replace the final `ConvTranspose2d` + `Conv2d` with a $4\times$ `PatchExpanding` (expand factor $4^2$, rearrange to $4\times$ resolution) followed by a `Linear(dims[0], out_channels)` projection on the token sequence, then reshape to the image grid.

---

## Related Notes

- [[Model Zoo]] — Architecture comparison
- [[TransUNet]] — Alternative transformer-based architecture
- [[UNETR]] — Pure transformer encoder
- [[Configuration Layer]] — SwinUNetConfig
- [[DLR-TomoSAR Index]] — Master index
