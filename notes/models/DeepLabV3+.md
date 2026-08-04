---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - DeepLabV3+
  - DeepLabV3Plus
  - ASPP encoder-decoder
family: deeplab
registry_key: deeplabv3plus
summary: Residual encoder at stride 8 with Atrous Spatial Pyramid Pooling and a light decoder fusing low-level detail.
---

# DeepLabV3+

`DeepLabV3Plus` (`models/backbone/deeplab_v3_plus.py`) is an encoder-decoder with Atrous Spatial Pyramid Pooling ([[DeepLabV3Plus_Chen2018_1802.02611.pdf|Chen et al., 2018]]): a residual encoder at output stride 8, an ASPP module that aggregates multi-rate dilated context, and a light decoder that fuses low-level detail before upsampling to full resolution.

---

## Summary

A stride-2 stem halves the input, residual stages (reusing the [[ResUNet]] `ResidualConvBlock`) downsample twice more to $P/8$, ASPP mixes parallel dilated branches plus a global-pooling branch, and the decoder concatenates the upsampled ASPP output with projected stride-2 low-level features before two refinement convolutions and bilinear upsampling to full resolution.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{f}$ | Encoder output feature map, $\mathbf{f} \in \mathbb{R}^{B \times F_4 \times \frac{P}{8} \times \frac{P}{8}}$ |
| $B$ | Batch size |
| $P$ | Input patch side length (pixels) |
| $F_1, F_4$ | Channel counts of the first and fourth encoder stages |
| $\mathbf{b}_0$ | $1\times1$ ASPP branch output |
| $\mathbf{b}_r$ | Atrous ASPP branch output at dilation rate $r$ |
| $\mathbf{g}$ | Global-pooled vector after $1\times1$ conv $\to$ norm $\to$ activation |
| $\mathbf{b}_\text{img}$ | Image-pooling branch output (upsampled $\mathbf{g}$) |
| $\mathbf{h}$ | Concatenation of all ASPP branch outputs |
| $\text{Conv}_{1\times1}$ | $1\times1$ convolution (conv $\to$ norm $\to$ activation unit) |
| $\text{Conv}_{3\times3}^{r}$ | $3\times3$ convolution with dilation $r$ (receptive field $1 + 2r$ per side) |
| $\text{GAP}$ | Global average pooling to $1 \times 1$ |
| $\text{up}$ | Bilinear upsampling |
| $\text{concat}$ | Channel-wise concatenation |
| $\mathcal{R}$ | Set of atrous rates (default $(1, 2, 4)$) |
| $r$ | A single atrous dilation rate, $r \in \mathcal{R}$ |
| $\mathrm{OS}$ | Output stride of the encoder |

---

## Architecture

### ASPP

For encoder output $\mathbf{f} \in \mathbb{R}^{B \times F_4 \times \frac{P}{8} \times \frac{P}{8}}$ and atrous rates $\mathcal{R}$ (default $(1, 2, 4)$, scaled to the $8 \times 8$ feature map of a 64 px patch):

$$
\begin{aligned}
\mathbf{b}_0 &= \text{Conv}_{1\times1}(\mathbf{f}) \\
\mathbf{b}_r &= \text{Conv}_{3\times3}^{r}(\mathbf{f}), \qquad r \in \mathcal{R} \\
\mathbf{g} &= \text{Conv}_{1\times1}(\text{GAP}(\mathbf{f})) \\
\mathbf{b}_\text{img} &= \text{up}(\mathbf{g}) \\
\mathbf{h} &= \text{concat}\big[\mathbf{b}_0,\; \{\mathbf{b}_r\}_{r \in \mathcal{R}},\; \mathbf{b}_\text{img}\big] \\
\text{ASPP}(\mathbf{f}) &= \text{Conv}_{1\times1}(\mathbf{h})
\end{aligned}
$$

The pooled map $\mathbf{g}$ passes through a $1\times1$ conv → norm → activation (GroupNorm here rather than the configured norm, see Paper fidelity Deviation 3) and is broadcast back by bilinear upsampling. Every branch, including the image-pooling branch, is a conv → norm → activation unit; all branches output $F_4 / 2$ channels and the projection returns $F_4 / 2$ channels with dropout.

### Decoder

The ASPP output is upsampled ×4 to the stride-2 resolution, concatenated with the 1×1-projected low-level features (first encoder stage, $\max(F_1/2, 16)$ channels), refined by two 3×3 conv-norm-act blocks, and upsampled ×2 to full resolution before the 1×1 output head.

---

## Design Rationale

**Context without resolution loss.** Each tomographic pixel's Gaussian parameters depend on neighbourhood context (building facades and layover patterns extend over many pixels), but the output must stay dense. Dilated convolutions grow the receptive field without further downsampling, which is the regime where U-Net decoders spend most of their capacity recovering resolution.

> ASPP is the only architecture in the zoo that aggregates context at multiple explicit dilation rates simultaneously; if scatterer structure has a characteristic spatial scale, one of the parallel branches can lock onto it directly.

---

## Parameter Reference

See [[Configuration Layer]] → `DeepLabV3PlusConfig`. The param groups expose a dedicated `aspp_lr`/`aspp_wd` for the context module.

---

## Paper fidelity

**Review date.** 2026-06-04

**Citation.** Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*. arXiv:1802.02611v3. Architecture in Sections 3.1–3.2, Figs. 1–2 and 4; decoder ablations in Section 4.1 (Tables 1–2). [[DeepLabV3Plus_Chen2018_1802.02611.pdf|PDF]]

This implementation adapts the macro-structure of DeepLabV3+ (encoder + ASPP + low-level-fusion decoder) to the dense TomoSAR regression task. The skeleton — parallel ASPP, reduce-then-concat decoder, bilinear recovery — is faithfully reproduced. Two choices distinguish the backbone from the paper's headline configuration, but on review both are **justified adaptations rather than open deviations**. First, the code reaches $\mathrm{OS}=8$ with a shallow residual stack and `MaxPool` placement, with no dilation anywhere in the backbone; this is conformant, because the paper itself trains its encoder at $\mathrm{OS}=32$ *without* any atrous convolution and only introduces backbone dilation as an optional inference-time densification (Sec. 4.1), so a non-atrous striding backbone is an in-paper configuration, not a departure. Second, the code uses regular (non-separable) convolutions throughout, whereas the paper's $\mathrm{SC}$ rows (Table 5) apply depthwise-separable convolution; the paper presents separable convolution as an *efficiency option* (33–41% Multiply-Add reduction at similar accuracy), not as a property the architecture requires, so dense convolutions are a justified parameterisation choice for the small SAR patches. The ASPP, where the paper does mandate dilation for multi-scale context, retains its atrous branches faithfully.

### Verdict table

| # | Component | Paper ref | Code ref | Verdict |
|---|-----------|-----------|----------|---------|
| 1 | ASPP: 1×1 + three 3×3 atrous + image pooling, concat → 1×1 project → dropout | Fig. 2; Sec. 3.1 | `deeplab_v3_plus.py:34-72` | MATCH (branch structure; rates are hyperparameters) |
| 1b | Image-pooling branch: GAP → 1×1 → norm → upsample | Fig. 2; Sec. 3.1 | `deeplab_v3_plus.py:52-57,68-70` | MATCH — GroupNorm used in lieu of BN, see Deviation 3 |
| 2 | Encoder output stride via dilated/atrous backbone | Sec. 3.1 ("DeepLabv3 as encoder"); Fig. 1(c) | `deeplab_v3_plus.py:95-110`; `configuration/architectures/backbone.py` | JUSTIFIED ADAPTATION — backbone has no atrous conv; OS=8 via MaxPool (paper trains OS=32 without atrous) |
| 3 | ASPP output bilinearly upsampled ×4, then concat low-level | Sec. 3.1 ("Proposed decoder") | `deeplab_v3_plus.py:144-149` | ACCEPTED ADAPTATION — ×4 to the OS=2 tap (consistent with deviation 4) |
| 4 | Low-level tap at Conv2 (OS=4), 1×1 reduce to 48 ch | Sec. 3.1; Sec. 4.1 / Table 1 | `deeplab_v3_plus.py:122,141-142,147` | ACCEPTED ADAPTATION — tap at stage 0 (OS=2); reduce-then-concat preserved (ratio is hyperparameter) |
| 5 | Post-concat refinement: two 3×3, 256 ch, then upsample ×4 | Sec. 4.1 / Table 2; Fig. 2 | `deeplab_v3_plus.py:124-127,149-151` | MATCH (two 3×3 convs; channel count is hyperparameter) |
| 6 | Depthwise-separable convs in ASPP and decoder | Sec. 3.3; Table 5 ($\mathrm{SC}$) | `deeplab_v3_plus.py:24-28,46-49,125-126` | JUSTIFIED ADAPTATION — regular convolutions (separable convs are a paper efficiency option) |
| 7 | Output head: 1×1 conv to class logits | Sec. 4.1 ("DeepLabv3 feature map") | `deeplab_v3_plus.py:129` | MATCH |
| 8 | Backbone block composition (Xception/ResNet residual blocks) | Fig. 4; Sec. 3.2 | `deeplab_v3_plus.py:98-110`; `blocks.py:152-207` | ACCEPTED ADAPTATION — generic pre-activation residual blocks |
| 9 | Final upsampling path (decoder output → full resolution) | Fig. 2 ("Upsample by 4") | `deeplab_v3_plus.py:151` | ACCEPTED ADAPTATION — ×2 to full res (consequence of OS=2 tap) |
| 10 | Encoder feature map = last map before logits, 256 ch | Sec. 3.1 | `deeplab_v3_plus.py:86,112-120` | MATCH (ASPP channels = $F_4/2$; count is hyperparameter) |

### Prose

**Overall verdict: ACCEPTED ADAPTATION; the two backbone differences (no atrous, non-separable) are justified adaptations, not open deviations.** The encoder–ASPP–decoder topology of Fig. 2 is reproduced with high fidelity at the level the paper actually specifies as load-bearing: the ASPP module (one $1\times1$ branch, three $3\times3$ atrous branches whose count and structure match Fig. 2, an image-level pooling branch, concatenation, a $1\times1$ projection, and dropout) is present at `deeplab_v3_plus.py:34-72`; the decoder reduces low-level features with a $1\times1$ conv, concatenates with the bilinearly-upsampled ASPP output, refines with **two** $3\times3$ conv–norm–act blocks (matching the Table 2 best entry $[3\times3,256]\times2$), and recovers full resolution by bilinear upsampling (`deeplab_v3_plus.py:122-151`); the head is a single $1\times1$ conv (`:129`). Rate values $(1,2,4)$, channel counts ($F_4/2$ for ASPP, $\max(F_1/2,16)$ for the low-level reduction), and tap depth are hyperparameters and out of scope.

**Adaptation 1 (justified) — no atrous convolution in the backbone.** The paper's *headline* dense-prediction setting dilates the last residual blocks of a deep ImageNet backbone so features stay at $\mathrm{OS}=16$ or $8$ without further striding (Sec. 3.1; Fig. 1(c)). The code instead reaches $\mathrm{OS}=8$ purely by `MaxPool2d` placement (`deeplab_v3_plus.py:109`: pooling only at stages $1$ and $2$) over a four-stage `ResidualConvBlock` stack that carries no dilation argument at all (`blocks.py:152-207`). **This is a justified adaptation, not an open deviation:** the paper itself trains the encoder at $\mathrm{OS}=32$ *without* atrous convolution and treats backbone dilation as an optional resolution-recovery step, so a non-atrous striding backbone is an in-paper configuration. The **ASPP still uses dilation** (`deeplab_v3_plus.py:49`), so the multi-scale-context mechanism the paper does mandate is intact; only the optional backbone densification is omitted. Should higher backbone resolution prove useful, a `dilation` argument on `ResidualConvBlock` could replace the deepest `MaxPool` with dilated convolutions, but this is an enhancement rather than a fidelity fix.

**Adaptation 2 (justified) — regular convolutions instead of depthwise-separable.** Section 3.3 and the $\mathrm{SC}$ rows of Table 5 apply depthwise-separable convolution in both ASPP and decoder for a 33–41% Multiply-Add reduction at *similar* accuracy. The code uses dense `nn.Conv2d` everywhere (`deeplab_v3_plus.py:25,125-126`). The paper frames separable convolution as an **efficiency option** rather than an accuracy-critical component, so dense convolutions are a justified parameterisation choice and not an open deviation; the information flow is unchanged. A depthwise-separable `ConvNormAct` variant (depthwise $3\times3$ with `groups=in_channels` + norm + act, then pointwise $1\times1$) remains available as an efficiency lever for the ASPP $3\times3$ branches and the decoder refinement convs if Multiply-Add budget becomes binding.

**Deviation 3 (minor) — image-pooling branch normalization uses GroupNorm.** Fig. 2 and the inherited DeepLabV3 image-level branch are GAP → $1\times1$ conv → BN → ReLU → upsample. The code's `pool_branch` is `AdaptiveAvgPool2d(1)` → $1\times1$ conv (`bias=config.conv_bias`, i.e. `False`) → norm → activation (`deeplab_v3_plus.py:52-57`), compositionally consistent with the other branches, except that the normalization is hard-wired to GroupNorm rather than the configured `normalization`.

The normalization is built as `build_norm2d("group", ...)` rather than inheriting the configured `normalization` (default `"batch"`). The reason is a degeneracy specific to this branch: the norm here acts on the post-GAP feature map of shape $[N, C, 1, 1]$, so `nn.BatchNorm2d` would estimate its batch statistics over only $N \cdot H \cdot W = N$ elements per channel; at batch size $1$ the per-channel variance is identically zero and the layer is ill-defined in training mode. `GroupNorm` normalizes per sample over channel groups and is therefore well-defined for any $N$, including $N = 1$. This matches the stabilizing *intent* of the paper's BN in the image-level branch while remaining robust to the small-batch regime; it is the standard batch-size-independent substitute. The other ASPP branches, which act on full-resolution feature maps, retain the configured normalization and are unaffected. Verified by a train-mode forward/backward at batch size $1$ (finite output, no crash) in addition to the batch-size-$2$ tests at $64\times64$ and $70\times70$.

**Accepted adaptations.** The low-level tap at stage 0 ($\mathrm{OS}=2$) rather than Conv2 ($\mathrm{OS}=4$), the resulting ASPP-upsample factor (×4 to $\mathrm{OS}=2$) and final upsample (×2 to full resolution) form one self-consistent choice: the reduce-then-concat-then-refine structure the paper specifies is preserved; only the absolute stride of the fusion point shifts, which is reasonable for the smaller SAR input patches. The generic pre-activation residual blocks standing in for ResNet/Xception blocks are likewise an accepted backbone substitution.

---

## Related Notes

- [[ResUNet]] — Source of the residual block
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — DeepLabV3PlusConfig
- [[DLR-TomoSAR Index]] — Master index
