---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - SegFormer
  - SegFormerLite
family: segformer
registry_key: segformer
summary: Hierarchical transformer encoder with efficient attention and an all-MLP multi-scale decoder.
---

# SegFormer

`SegFormerLite` (`models/backbone/segformer_lite.py`, registry name `"segformer"`) is a hierarchical transformer encoder with an all-MLP decoder ([[SegFormer_Xie2021_2105.15203.pdf|Xie et al., 2021]]): four stages of overlapping patch embeddings and efficient self-attention produce a feature pyramid, and a lightweight decoder projects, upsamples, and fuses all four scales before the output head.

The model is a dense per-pixel regressor for the TomoSAR Gaussian-mixture target. It ingests `in_channels` (default $1$) co-registered SLC channels and the `output_head` 1×1 convolution emits `out_channels` (default $6$) maps, which decode as $3K$ Gaussian-mixture parameters with $K = $ `out_channels` $/3$ Gaussians at $3$ parameters each (`params_per_gaussian = 3`); the default $6$ channels are $K = 2$ Gaussians.

---

## Summary

Stage $i$ embeds the previous resolution with an overlapping strided convolution (strides $4, 2, 2, 2$ → resolutions $P/4 \dots P/32$), then applies `depths[i]` transformer blocks. Each block is pre-norm: efficient self-attention with spatial reduction, then a MixFFN whose hidden layer contains a 3×3 depthwise convolution — the only positional signal in the network (no positional embeddings). The decoder maps every stage to `decoder_channels` with a 1×1 convolution, bilinearly upsamples all to the stage-1 resolution, concatenates, fuses, and upsamples ×4 to full resolution.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | Input token sequence to a block or sublayer |
| $N = hw$ | Number of tokens (spatial positions) at the current stage |
| $C$ | Channel (embedding) dimension at the current stage |
| $h, w$ | Token-grid height and width at the current stage |
| $R$ | Spatial-reduction ratio (`sr_ratios`, default $(4, 2, 2, 1)$) |
| $\text{SR}_R$ | Spatial reduction by ratio $R$, $\mathbb{R}^{N \times C} \to \mathbb{R}^{N/R^2 \times C}$ |
| $\text{MHA}$ | Multi-head attention |
| $\text{FC}_{C \to 4C}, \text{FC}_{4C \to C}$ | Pointwise (1×1) linear layers of the MixFFN |
| $\text{DWConv}_{3\times3}$ | 3×3 depthwise convolution inside the MixFFN |
| $\mathbf{u}_1$ | Token features after the expanding pointwise layer |
| $\mathbf{u}_2$ | Features after the 3×3 depthwise convolution |
| $\mathbf{u}_3$ | Features after the GELU activation |
| $\mathbf{u}_4$ | Features after the reducing pointwise layer (the MixFFN output) |

---

## Architecture

### Efficient Self-Attention

With $N = hw$ tokens, full attention costs $O(N^2)$. The keys and values are spatially reduced by a strided convolution with ratio $R$ (`sr_ratios`, default $(4, 2, 2, 1)$):

$$
\text{Attn}(\mathbf{x}) = \text{MHA}\big(\mathbf{x},\; \text{SR}_R(\mathbf{x}),\; \text{SR}_R(\mathbf{x})\big), \qquad \text{SR}_R: \mathbb{R}^{N \times C} \to \mathbb{R}^{N/R^2 \times C}
$$

reducing the cost to $O(N^2 / R^2)$. At a 64 px patch the stage resolutions are $16, 8, 4, 2$, so even the largest attention map is $256 \times 16$.

### MixFFN

$$
\begin{aligned}
\mathbf{u}_1 &= \text{FC}_{C \to 4C}(\mathbf{x}) \\
\mathbf{u}_2 &= \text{DWConv}_{3\times3}(\mathbf{u}_1) \\
\mathbf{u}_3 &= \text{GELU}(\mathbf{u}_2) \\
\mathbf{u}_4 &= \text{FC}_{4C \to C}(\mathbf{u}_3) \\
\text{MixFFN}(\mathbf{x}) &= \mathbf{u}_4
\end{aligned}
$$

The depthwise convolution leaks local positional information into the token stream, replacing explicit positional encodings — which also removes any resolution-dependence of the encoder.

---

## Design Rationale

**A second transformer family.** [[SwinUNet]], [[TransUNet]], and [[UNETR]] all pair attention with heavy convolutional decoders. SegFormer tests the opposite hypothesis: that with a strong multi-scale encoder, a near-trivial MLP decoder suffices for dense prediction. If the transformer models underperform on this problem, SegFormer disambiguates whether the encoder or the decoder side was the bottleneck.

---

## Parameter Reference

See [[Configuration Layer]] → `SegFormerLiteConfig` (`in_channels`, `out_channels`, `params_per_gaussian`, `embedding_dims`, `depths`, `num_heads`, `sr_ratios`, `mlp_ratio`, `decoder_channels`, `dropout`, `attention_dropout`, `ffn_activation`, `stochastic_depth_rate`, `init_mode`). The loader sets `in_channels` from the representation channel count and `out_channels = 3K` from the requested Gaussian count.

---

## Paper fidelity

**Review date:** 2026-06-04
**Citation:** Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* arXiv:2105.15203. (Ground truth: Section 3, Fig. 2, Eq. 1–3 p. 4, Eq. 4 p. 5, Table 6 p. 11.) [[SegFormer_Xie2021_2105.15203.pdf|PDF]]

**Overall verdict:** MATCH (faithful Lite reimplementation). The mechanism and topology of the MiT encoder and All-MLP decoder are reproduced exactly. Every divergence from the reference checkpoints is either justified Lite scaling or a hyperparameter choice (out of scope). Two minor mechanism notes are recorded: the $\mathrm{SR}$ reduction is realised as a strided convolution rather than the literal Reshape+Linear of Eq. 2 (functionally equivalent, and what the official MiT code does), and the per-block residual sublayer outputs are not independently dropout-regularised inside attention (the paper does not specify this either way).

### Verdict table

| # | Component | Paper ref | Code | Verdict |
|---|-----------|-----------|------|---------|
| 1 | Overlapped patch embedding $K{=}7,S{=}4,P{=}3$ then $K{=}3,S{=}2,P{=}1$ | §3.1 "Overlapped Patch Merging"; Table 6 | `SegFormerLite.py:116-117`, `OverlapPatchEmbedding:11-22` | MATCH |
| 1b | LayerNorm after projection conv (conv → flatten → LN) | §3.1; official MiT | `SegFormerLite.py:18-21` | MATCH |
| 2 | Efficient self-attention: $\mathrm{SR}_R$ on K,V; Q full length; norm after reduction | Eq. 1–2 (p. 4) | `EfficientSelfAttention:25-46` | MATCH (SR as strided conv — see note) |
| 3 | Mix-FFN $\mathbf{x}\to\mathrm{MLP}\to\mathrm{DWConv}_{3\times3}\to\mathrm{GELU}\to\mathrm{MLP}$, residual added (see Architecture MixFFN block) | Eq. 3 (p. 4) | `MixFFN:49-71`, residual `SegFormerBlock:85` | MATCH |
| 4 | No positional encoding anywhere | §3.1 "Mix-FFN"; Abstract | entire file — none present | MATCH |
| 5 | Pre-norm block + residuals + stochastic depth | §3.1; Fig. 2 (×N block) | `SegFormerBlock:83-86`, droppath rates `:119-120` | MATCH |
| 6 | All-MLP decoder: unify → upsample 1/4 → fuse concat → predict, all channel-wise linear | Eq. 4 (p. 5) | `decode_projections:146`, `fuse:148-153`, `output_head:155`, fwd `:159-180` | MATCH (ACCEPTED: BN+act+dropout added to fuse) |
| 7 | Final upsample to full resolution | §3.2 (mask at $H/4$, pipeline upsamples) | `interpolate` `SegFormerLite.py:178` | ACCEPTED ADAPTATION |
| 8 | 4 stages, resolutions 1/4, 1/8, 1/16, 1/32 | Fig. 2; §3.1 | `n_stages=len(dims)=4`, strides `[4,2,2,2]` `:117` | MATCH |
| 9 | MHSA head count + scaling fidelity | Eq. 1 | `nn.MultiheadAttention` `:28` | MATCH |
| 10 | Channel-increase / resolution-shrink pyramid; stage-3 heavy | Fig. 2; Table 6 caption | `embedding_dims=[40,80,192,320]` | MATCH (ACCEPTED: Lite dims/depths) |

### Prose

**Overlapped patch embedding (point 1).** `kernel_sizes = [7] + [3]*(n_stages-1)` and `strides = [4] + [2]*(n_stages-1)` (`SegFormerLite.py:116-117`) with padding `kernel_size // 2` (i.e. $3$ and $1$) reproduce the paper's $K_1{=}7,S_1{=}4,P_1{=}3$ and $K_i{=}3,S_i{=}2,P_i{=}1$ exactly (§3.1, Table 6). The [[LayerNorm]] sits *after* the projection convolution on the flattened token sequence (`OverlapPatchEmbedding:18-21`: conv → `flatten(2).transpose` → `norm`), matching the official MiT ordering. MATCH.

**Efficient self-attention (point 2).** Eq. 2 specifies reducing the sequence-to-be-reduced ($K$ and $V$) by ratio $R$ via Reshape$(\tfrac{N}{R},C{\cdot}R)$ then Linear$(C{\cdot}R,C)$. The code (`EfficientSelfAttention:36-45`) reshapes tokens back to $B{\times}C{\times}h{\times}w$, applies a stride-$R$ convolution `spatial_reduction` (kernel $=$ stride $= R$), flattens, and applies `sr_norm` ([[LayerNorm]]) — this is the official MiT realisation of $\mathrm{SR}_R$ and is mathematically equivalent to the Reshape+Linear formulation (a non-overlapping strided conv is a reshape followed by a linear map). The query `x` is passed at full length and only `kv` is reduced; `out, _ = self.attention(x, kv, kv, ...)` confirms reduction applied to *both* K and V (`:45`). Norm placement is post-reduction as in Eq. 2. MATCH; the conv-vs-Linear realisation is a faithful, standard adaptation, not a deviation.

**Mix-FFN (point 3).** `MixFFN.forward` (`:60-71`) runs `fc1` (1×1 conv = MLP) → `dwconv` (3×3 depthwise, `groups=hidden_dim`, `:55`) → GELU `activation` → `fc2`. The depthwise 3×3 therefore sits *between* fc1 and the GELU, exactly as the $\mathrm{MLP}\to\mathrm{Conv}_{3\times3}\to\mathrm{GELU}$ ordering of Eq. 3. The outer residual $\mathbf{x}+\cdots$ is applied in the block (`SegFormerBlock:85`). MATCH.

**No positional encoding (point 4).** No `nn.Embedding`, no learned positional table, no `pos_embed` parameter, no sinusoidal addition anywhere in the file. Location information is carried solely by the Mix-FFN depthwise convolution and the overlapping patch convs, as the paper claims (§3.1 "Mix-FFN"). MATCH.

**Transformer block (point 5).** Pre-norm is explicit: `x + drop_path(attention(norm1(x)))` then `x + drop_path(ffn(norm2(x)))` (`SegFormerBlock:83-86`), i.e. [[LayerNorm]] precedes both attention and FFN, both with residuals. Stochastic depth is a linearly-spaced schedule over all blocks (`drop_path_rates`, `:119-120`) via the shared `DropPath` module (`models/blocks.py:9-21`), matching the standard SegFormer training recipe and Fig. 2's repeated block. MATCH.

**All-MLP decoder (point 6).** Eq. 4's four steps map cleanly: (1) $\hat F_i=\mathrm{Linear}(C_i,C)$ → `decode_projections` are 1×1 convs (`:146`), channel-wise linear; (2) $\mathrm{Upsample}$ to stage-1 resolution → `functional.interpolate(..., target_size, mode="bilinear")` where `target_size = stage_outputs[0].shape[2:]` (the 1/4 map) (`:168,173-174`); (3) $F=\mathrm{Linear}(4C,C)(\mathrm{Concat})$ → `fuse` first layer is a 1×1 conv $n\_stages{\cdot}C\to C$ (`:149`); (4) $M=\mathrm{Linear}(C,N_{cls})$ → `output_head` 1×1 conv $C\to$ `out_channels` $=3K$ (`:155`). Every operator is a 1×1 conv (channel-wise linear), never a 3×3 — the defining property of the All-MLP decoder is preserved. ACCEPTED ADAPTATION: `fuse` appends BatchNorm + activation + Dropout2d after the fusion linear (`:150-152`); the paper's Eq. 4 fuse is a bare Linear, but the official mmsegmentation `SegformerHead` likewise wraps fusion in a Conv-BN-ReLU `ConvModule`, so this matches the reference implementation rather than the bare equation.

**Final upsampling (point 7).** The paper's decoder emits the mask at $H/4\times W/4$ and the training/eval pipeline upsamples to full resolution. Here the model upsamples internally: after `fuse`, `functional.interpolate(x, size=input_size, ...)` restores full input resolution before `output_head` (`:177-178`). ACCEPTED ADAPTATION — same prediction, the ×4 restoration is simply folded into the module instead of the loss pipeline (sensible for the dense TomoSAR regression head). Note the head is applied *after* the upsample, whereas Eq. 4 applies the class linear at $1/4$ then upsamples; both produce a full-resolution per-pixel linear map, so this is functionally equivalent.

**Stage count and resolutions (point 8).** Four stages (`len(embedding_dims)==4`), strides $4,2,2,2$ give cumulative downsampling $1/4,1/8,1/16,1/32$ exactly as Fig. 2's $\tfrac{H}{4}{\times}\tfrac{W}{4}{\times}C_1 \dots \tfrac{H}{32}{\times}\tfrac{W}{32}{\times}C_4$. MATCH.

**MultiheadAttention substitution (point 9).** `nn.MultiheadAttention(embedding_dim, num_heads, dropout=attention_dropout, batch_first=True)` (`:28`) implements standard scaled-dot-product MHSA with per-head scaling $1/\sqrt{d_{head}}$ (Eq. 1) and `num_heads` $=(1,2,4,8)$ across stages — the same head progression as MiT (Table 6: $N_i$). The substitution is faithful: queries, keys, values are linearly projected per head and combined identically to the paper's formulation. MATCH.

**Other Fig. 2 specifics (point 10).** The channel-increase / resolution-shrink pyramid and the assignment of most depth to stage 3 (Table 6 caption) are configurable; the Lite default `embedding_dims=[40,80,192,320]`, `depths=[2,2,2,2]`, `sr_ratios=(4,2,2,1)` is a scaled-down MiT-B0 variant — ACCEPTED Lite scaling (dims, depths, and decoder channel $C$ are all hyperparameters). A per-stage trailing [[LayerNorm]] (`SegFormerStage:93,99`) before reshaping back to a feature map matches the official MiT `norm` applied at each stage output. MATCH.

**Deviations:** none of structural or minor severity. The only items worth flagging are the two ACCEPTED adaptations above (internal final upsample; Conv-BN-act fuse), both of which align the code with the official mmsegmentation implementation rather than departing from it.

---

## Related Notes

- [[SwinUNet]] / [[TransUNet]] / [[UNETR]] — Other transformer architectures in the zoo
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — SegFormerLiteConfig
- [[DLR-TomoSAR Index]] — Master index
