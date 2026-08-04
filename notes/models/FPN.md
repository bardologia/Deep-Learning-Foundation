---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - FPN
  - FPNNet
  - Feature Pyramid Network
family: fpn
registry_key: fpn
summary: Residual bottom-up encoder with top-down lateral pyramid and a Panoptic-FPN semantic branch merged before the head.
---

# FPN

`FPNNet` (`models/backbone/fpn_net.py`) is a Feature Pyramid Network ([[FPN_Lin2017_1612.03144.pdf|Lin et al., 2017]]) adapted for dense regression: a residual bottom-up encoder, a top-down pathway with lateral connections that gives every pyramid level the same channel width, and a Panoptic-FPN semantic branch ([[PanopticFPN_Kirillov2019_1901.02446.pdf|Kirillov et al., 2019]]) that brings each pyramid level to the target scale through depth-scaled interleaved upsampling stages whose outputs are summed before the head.

---

## Summary

The bottom-up path is a stack of residual stages (reusing the [[ResUNet]] block) at resolutions $P, P/2, P/4, P/8$. Lateral 1×1 convolutions project every stage to `pyramid_channels`; the top-down path adds each upsampled coarser level to the lateral projection and smooths with a 3×3 convolution. Each pyramid level then passes through a semantic segmentation block that, following Kirillov et al. (2019), reaches the target (full) resolution through $N_i = \log_2(\text{stride}_i/\text{stride}_\text{target})$ interleaved stages of (3×3 conv → norm → act → 2× bilinear upsample), with the shallowest level ($N=0$) reduced to `segmentation_convs` 3×3 conv → norm → act units and no upsample; the equal-scale outputs are merged by summation, fused (`fuse_block`), and mapped by a single $1\times1$ `output_head` to `out_channels`. The network takes `in_channels = 1` and emits a flat `out_channels = 6` channel stack — `params_per_gaussian` $= 3$ times $K = 2$ Gaussians; with the default [[Head Conv]], `FPNNet` produces all $3K$ mixture parameters from one convolution; [[Head Per-Gaussian]] and the other output heads remain selectable via the `head` config field.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{c}_i$ | Bottom-up feature at level $i$ (channels `features[i]`) |
| $\mathbf{p}_i$ | Pyramid feature at level $i$ (channels `pyramid_channels`, identical at every level) |
| $\mathbf{s}_{i,k}$ | Semantic-branch feature at level $i$ after stage $k$ |
| $\mathbf{s}_i$ | Final semantic output of level $i$ at the common target scale |
| $\mathbf{s}$ | Summed semantic map over all levels |
| $i$ | Pyramid level index |
| $k$ | Semantic-branch stage index, $k = 1, \dots, N_i$ |
| $N_i$ | Number of upsampling stages for level $i$, $N_i = \log_2(\text{stride}_i / \text{stride}_\text{target})$ |
| $P$ | Input patch side length (pixels) |
| $\text{Conv}_{1\times1}$ | $1\times1$ lateral convolution |
| $\text{Conv}_{3\times3}$ | $3\times3$ convolution |
| $\text{up}$ | Bilinear upsampling by the resolution ratio |
| $\text{up}_{\times2}$ | $2\times$ bilinear upsampling |
| $\text{norm}$ | Normalisation |
| $\text{act}$ | Activation |
| $\text{stride}_i$ | Encoder stride at level $i$ |
| $\text{stride}_\text{target}$ | Stride of the target (full) resolution |
| `pyramid_channels` | Uniform pyramid channel width |
| `segmentation_convs` | Number of $3\times3$ conv → norm → act units within each semantic-branch stage |

---

## Architecture

### Top-Down Pathway

$$
\begin{aligned}
\mathbf{l}_i &= \text{Conv}_{1\times1}(\mathbf{c}_i) \\
\mathbf{u}_i &= \text{up}(\mathbf{p}_{i+1}) \\
\mathbf{p}_i &= \text{Conv}_{3\times3}(\mathbf{l}_i + \mathbf{u}_i)
\end{aligned}
$$

The equal-width pyramid means semantic strength is uniform across scales — coarse levels contribute context, fine levels contribute localisation, and the sum weights them equally.

### Semantic Branch

$$
\begin{aligned}
N_i &= \log_2\!\frac{\text{stride}_i}{\text{stride}_\text{target}} \\
\mathbf{s}_{i,0} &= \mathbf{p}_i \\
\mathbf{t}_{i,k} &= \text{Conv}_{3\times3}(\mathbf{s}_{i,k-1}) \\
\mathbf{a}_{i,k} &= \text{act}(\text{norm}(\mathbf{t}_{i,k})) \\
\mathbf{s}_{i,k} &= \text{up}_{\times2}(\mathbf{a}_{i,k}), \qquad k = 1, \dots, N_i \\
\mathbf{s}_i &= \mathbf{s}_{i,N_i}
\end{aligned}
$$

Each pyramid level $\mathbf{p}_i$ is decoded to the common target scale by $N_i$ interleaved stages indexed by $k$, where one stage $\mathbf{s}_{i,k-1} \mapsto \mathbf{s}_{i,k}$ is a 3×3 conv, normalisation, activation, and a $2\times$ bilinear upsample (Kirillov et al. 2019, Sec. 3.1, Fig. 3). Deeper (coarser) levels receive more stages; with encoder strides $\{1,2,4,8\}$ and the target at level 0, the levels take $N = 0,1,2,3$ stages respectively. The level-0 case ($N=0$) degenerates to `segmentation_convs` 3×3 conv → norm → act units with no upsample (`fpn_net.py:18-24`). Repeated $2\times$ steps insert learned refinement at each scale rather than collapsing the decode into one large-factor interpolation; a residual `interpolate` only corrects the floor-rounding of odd input sizes (`fpn_net.py:110-111`). The equal-scale maps $\mathbf{s}_i$ are then summed:

$$
\mathbf{s} = \sum_i \mathbf{s}_i .
$$

The `segmentation_convs` hyperparameter sets the number of 3×3 conv → norm → act units composed within each stage.

---

## Design Rationale

**A decoder with almost no capacity.** The FPN decoder is deliberately minimal (one 1×1, one 3×3, and a small block per level) compared to the symmetric decoders of the U-Net family, concentrating parameters in the encoder. If the benchmark shows decoder capacity matters little for this task, FPN converts that observation into a cheaper architecture; if FPN lags, the decoder side is doing real work in the U-shaped models.

---

## Parameter Reference

See [[Configuration Layer]] → `FPNNetConfig` (`features`, `pyramid_channels`, `segmentation_convs` — convs per semantic-branch stage).

---

## Paper fidelity

*Review date: 2026-06-04.* Ground truth: Lin et al., *Feature Pyramid Networks for Object Detection*, arXiv:1612.03144 (Sec. 3, Fig. 3) [[FPN_Lin2017_1612.03144.pdf|PDF]] and Kirillov et al., *Panoptic Feature Pyramid Networks*, arXiv:1901.02446 (Sec. 3.1, Fig. 3) [[PanopticFPN_Kirillov2019_1901.02446.pdf|PDF]]. Comparison against `models/backbone/fpn_net.py` and `configuration/architectures/backbone.py` (`FPNNetConfig`, build helpers).

### Verdict table

| # | Dimension | Verdict | Evidence |
|---|-----------|---------|----------|
| 1 | Bottom-up feature hierarchy, stride-2 between stages | MATCH | Lin Sec. 3 "scaling step of 2"; `fpn_net.py:58` `MaxPool2d(kernel_size=2)` per stage (strides $\{1,2,4,8\}$) |
| 2 | Lateral $1\times1$ conv on each bottom-up map | MATCH | Lin Fig. 3, Sec. 3; `fpn_net.py:71` |
| 3 | Top-down upsample $\times2$ + element-wise ADD | MATCH (add) / DEVIATION-minor (bilinear vs nearest) | Lin Sec. 3 "nearest neighbor upsampling", merge by addition; `fpn_net.py:100-101` bilinear interpolate + `lateral + top_down` |
| 4 | Post-merge $3\times3$ anti-aliasing conv per level | MATCH | Lin Sec. 3 "append a $3\times3$ convolution on each merged map"; `fpn_net.py:72,103` |
| 5 | Uniform pyramid channel width | MATCH | Lin Sec. 3 ($d{=}256$, uniform); Kirillov Sec. 3.1 "same channel dimension"; `fpn_net.py:71-72`, `pyramid_channels` |
| 6 | Semantic branch: depth-scaled upsampling stages, SUM merge | MATCH | Kirillov Sec. 3.1 + Fig. 3 (progressive 3x3+norm+act+2x stages, more for deeper levels); `fpn_net.py:12-38,74-77` `SegmentationBlock(channels, n_stages, convs_per_stage, ...)` emits $N_i=\log_2(\text{stride}_i/\text{stride}_\text{target})$ interleaved stages ($N=0,1,2,3$), SUM merge |
| 7 | Fused map predicted ($1\times1$) and upsampled | MATCH (predict) / DEVIATION-minor (no final upsample) | Kirillov Sec. 3.1 final $1\times1$ + $4\times$ upsample; `fpn_net.py:86,116` |
| 8 | Index-0 full-resolution ($1/1$) level | ACCEPTED ADAPTATION | Papers start pyramid at $1/4$; code uses an identity-stride level 0 (`fpn_net.py:58` `nn.Identity()` when `index == 0`) |
| 9 | Encoder block composition | ACCEPTED ADAPTATION | Lin Sec. 3 backbone-agnostic; code uses pre-activation `ResidualConvBlock` ([[ResUNet]]), constructed stride-1 with separate maxpool |
| 10 | Smooth/lateral layers carry no non-linearity | MATCH | Lin Sec. 4.1 "no non-linearities in these extra layers"; `fpn_net.py:71-72` bare convs |

### Prose

The **bottom-up pathway, lateral connections, additive merge, post-merge smoothing, and uniform channel width** are all faithful to Lin Sec. 3 and Fig. 3. The encoder reuses the pre-activation [[ResUNet]] `ResidualConvBlock`, constructed without a `stride` argument so every block stays stride-$1$ and spatial reduction is delegated entirely to the separate `MaxPool2d(kernel_size=2)` layers (level 0 uses `nn.Identity()`) (`fpn_net.py:54-69`). This is consistent with Lin's explicitly *generic-backbone* assumption (Sec. 3, "independent of the backbone convolutional architecture"), so the construction is paper-conformant.

Two **minor deviations** concern the top-down resampling: the paper specifies nearest-neighbour upsampling "for simplicity", whereas the code uses `mode="bilinear"` (`fpn_net.py:100`). Bilinear is a strictly richer interpolation and is the choice Panoptic FPN itself makes in the semantic branch, so this is a benign, well-motivated minor deviation. Relatedly, the final output is produced by the $1\times1$ head at the merge resolution with no subsequent upsample (`fpn_net.py:116`); because the merge resolution is already full input resolution (level 0 is $1/1$), this is correct for the adapted geometry and only a nominal difference from Kirillov's "$1\times1$ then $4\times$ upsample".

The semantic segmentation branch follows the *depth-scaled* upsampling decoder of Kirillov Sec. 3.1 / Fig. 3: each FPN level is brought to the target scale by a sequence of upsampling stages, where a stage is $3\times3$ conv + norm + activation + $2\times$ bilinear, and **deeper (coarser) levels receive more stages** (in the paper the $1/32$ level gets three stages, $1/16$ two, $1/8$ one, $1/4$ a single conv with no upsample), after which the equal-scale maps are summed.

`SegmentationBlock(channels, n_stages, convs_per_stage, activation, normalization, bias)` emits $N_i = \log_2(\text{stride}_i/\text{stride}_\text{target})$ interleaved stages of ($3\times3$ conv → norm → act → $2\times$ bilinear upsample); the shallowest level ($N_i=0$) degenerates to `convs_per_stage` $3\times3$ conv → norm → act units with no upsample (`fpn_net.py:18-24`), generalising the single-conv $1/4$ level of Kirillov Fig. 3 to the configured per-stage conv count. The blocks are built with `n_stages = index` over `features` (`fpn_net.py:74-77`); with the encoder strides $\{1,2,4,8\}$ and the target at level 0, the four levels receive $N = 0,1,2,3$ stages. The per-level loop reaches the common scale through these repeated $2\times$ stages; the residual `interpolate` absorbs only the floor-rounding mismatch produced by odd input sizes (`fpn_net.py:110-111`). The element-wise SUM merge matches the paper (Kirillov ablation Table 1d confirms sum over concat). The `segmentation_convs` hyperparameter sets the number of $3\times3$ conv → norm → act units *within each stage*, with tunable choices $\{1,2,3\}$ (`configuration/architectures/backbone.py`). The bottom-up / lateral / top-down core, pyramid width, and per-stage conv count are the architecture's hyperparameters.

---

## Related Notes

- [[ResUNet]] — Source of the residual block
- [[LinkNet]] — Another light-decoder design (additive skips)
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — FPNNetConfig
- [[DLR-TomoSAR Index]] — Master index
