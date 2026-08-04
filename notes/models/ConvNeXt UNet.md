---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - ConvNeXtUNet
  - ConvNeXt U-Net
family: convnext
registry_key: convnext_unet
summary: U-Net topology with ConvNeXt inverted-bottleneck blocks (depthwise-7x7, LayerNorm, GELU) at every stage.
---

# ConvNeXt UNet

`ConvNeXtUNet` (`models/backbone/convnext_unet.py`) places ConvNeXt blocks ([[ConvNeXt_Liu2022_2201.03545.pdf|Liu et al., 2022]]) inside the U-Net topology: each encoder/decoder stage is a stack of depthwise-7×7 inverted-bottleneck blocks with LayerNorm and GELU, connected by strided-conv downsampling, transposed-conv upsampling, and skip concatenation.

---

## Summary

A 3×3 stem projects the input to `features[0]` at full resolution. Each stage applies a 1×1 channel projection followed by `blocks_per_stage` ConvNeXt blocks; downsampling between stages is LayerNorm + 2×2 stride-2 convolution (the ConvNeXt convention), and the decoder mirrors the encoder with skip concatenation as in [[UNet]]. Stochastic depth increases linearly across the block sequence via `DropPath`.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | Block input feature map |
| $\mathbf{y}$ | Block output feature map |
| $C$ | Channel dimension at the current stage |
| $\mathbf{u}_1$ | Features after depthwise spatial mixing |
| $\mathbf{u}_2$ | Features after LayerNorm |
| $\mathbf{u}_3$ | Features after the $C \to 4C$ pointwise expansion |
| $\mathbf{u}_4$ | Features after the GELU activation |
| $\mathbf{u}_5$ | Features after the $4C \to C$ pointwise reduction (the residual branch) |
| $\text{DWConv}_{7\times7}$ | Depthwise convolution, one filter per channel, receptive field 7×7 |
| $\text{LN}$ | LayerNorm over the channel dimension (channels-last permutation) |
| $\text{FC}$ | Pointwise linear layers forming a 4× inverted bottleneck (`ffn_ratio`) |
| $\text{GELU}$ | Gaussian Error Linear Unit activation |
| $\text{DropPath}$ | Stochastic depth applied to the residual branch |
| $\gamma$ | Layer Scale, a learnable per-channel vector initialised to $10^{-6}$ (`layer_scale_init`) that scales the residual branch before drop path; disabled ($\gamma\equiv1$, no parameter) when `layer_scale_init` $\le 0$ |

---

## Architecture

### ConvNeXt Block

$$
\begin{aligned}
\mathbf{u}_1 &= \text{DWConv}_{7\times7}(\mathbf{x}) \\
\mathbf{u}_2 &= \text{LN}(\mathbf{u}_1) \\
\mathbf{u}_3 &= \text{FC}_{C \to 4C}(\mathbf{u}_2) \\
\mathbf{u}_4 &= \text{GELU}(\mathbf{u}_3) \\
\mathbf{u}_5 &= \text{FC}_{4C \to C}(\mathbf{u}_4) \\
\mathbf{y} &= \mathbf{x} + \text{DropPath}(\gamma \odot \mathbf{u}_5)
\end{aligned}
$$

The block separates spatial mixing (depthwise, cheap, large kernel) from channel mixing (pointwise, where the parameters live) — the convolutional analogue of a transformer block.

---

## Design Rationale

**Modernised convolutions under the same topology.** All convolutional models in the zoo use the classic conv-norm-act recipe with 3×3 kernels. ConvNeXt brings the design choices that let CNNs match transformers — large depthwise kernels, inverted bottlenecks, LayerNorm, GELU, stochastic depth — while keeping the exact U-Net macro-structure, so any gain is attributable to the block design rather than the topology.

---

## Parameter Reference

See [[Configuration Layer]] → `ConvNeXtUNetConfig` (`features`, `blocks_per_stage`, `ffn_ratio`, `stochastic_depth_rate`, `layer_scale_init`, `bottleneck_factor`).

---

## Paper fidelity

*Review date:* 2026-06-04. *Citation:* Liu, Mao, Wu, Feichtenhofer, Darrell, Xie, *A ConvNet for the 2020s*, CVPR 2022 (arXiv:2201.03545). *Ground truth:* [[ConvNeXt_Liu2022_2201.03545.pdf]].

This model is a deliberate hybrid: the **encoder** is judged strictly against the ConvNeXt paper, while the **decoder** is only required to be a coherent UNet-style decoder. Stage compute ratio and other hyperparameters are out of scope.

**Verdict:** the encoder block is a faithful reproduction of the ConvNeXt block (Fig. 4) and the separate-downsampling convention (Sec. 2.6 / Sec. 2 page-6 paragraph), including Layer Scale ($\gamma$, Sec. 3.1): the block applies a learnable per-channel $\gamma$ initialised to $10^{-6}$ to the residual branch before drop path. The patchify stem, the per-stage $1\times1$ projection, and the $1\times1$ output head are accepted adaptations for dense prediction under a U-Net topology. The decoder is correctly wired.

| Dimension | Paper ref | Code | Classification |
|---|---|---|---|
| Block op order (DWConv$_{7\times7}\to$LN$\to$FC$\to$GELU$\to$FC$\to\gamma\to$residual) | Fig. 4 | `convnext_unet.py:36-49` | MATCH |
| Pointwise layers channels-last as Linears | Fig. 4 (note "$1\times1$ convs $\equiv$ Linear") | `convnext_unet.py:29-31,42-44` | MATCH |
| Inverted-bottleneck ratio 4 | Sec. 2.4 | `configuration/architectures/backbone.py` (`ffn_ratio=4.0`) | MATCH |
| LayerNorm over $C$ only (channels-last) | Sec. 2.6 "BN$\to$LN" | `convnext_unet.py:11-19,28,41` | MATCH |
| Layer Scale $\gamma$ (init $10^{-6}$, per-channel, before drop path) | Sec. 3.1 | `convnext_unet.py:34,45-46`; `configuration/architectures/backbone.py` (`layer_scale_init=1e-6`) | MATCH |
| Stochastic depth linearly increasing across blocks | Sec. 2.1; Sec. 3.1 | `convnext_unet.py:79-86` | MATCH |
| Patchify stem ($4\times4$ conv stride 4 + LN) | Sec. 2.2 | `convnext_unet.py:90-93` ($3\times3$ stride 1) | ACCEPTED ADAPTATION |
| Separate downsampling (LN + $2\times2$ conv stride 2) | Sec. 2.6 page-6 | `convnext_unet.py:100-103` | MATCH |
| Channel change location (official folds into downsample conv) | Sec. 2.6 / official impl | `convnext_unet.py:55` (per-stage $1\times1$ projection) | ACCEPTED ADAPTATION |
| Stage compute ratio $(3,3,9,3)$ | Sec. 2.2 | uniform `blocks_per_stage` | out of scope (hyperparameter) |
| Decoder: transposed conv $\to$ concat skip $\to$ ConvNeXt stage | — (UNet) | `convnext_unet.py:108-136` | MATCH (coherent) |
| Output head | — (dense prediction) | `convnext_unet.py:115` ($1\times1$ conv) | ACCEPTED ADAPTATION |

**Encoder fidelity (strict).** The block in `convnext_unet.py:36-49` reproduces Fig. 4's ConvNeXt column exactly: a $7\times7$ depthwise convolution (`dwconv`, `groups=channels`, $\text{pad}=3$) for spatial mixing, a single LayerNorm over the channel dimension via a channels-last permutation, a $C\to4C$ pointwise expansion (`fc1`), one GELU, a $4C\to C$ pointwise reduction (`fc2`), Layer Scale ($\gamma$), and an additive residual. Realising the pointwise layers as `nn.Linear` in channels-last layout is exactly the official implementation's choice and is mathematically identical to $1\times1$ convolutions, so the *fewer activations / fewer norms* micro-design (Sec. 2.6 — one LN, one GELU per block) is honoured. Inter-stage downsampling (`convnext_unet.py:100-103`) is a `ChannelLayerNorm` followed by a $2\times2$ stride-2 convolution, matching the "separate downsampling layers" with a normalisation before each spatial-resolution change (Sec. 2, page 6). Stochastic depth (`convnext_unet.py:79-86`) is a single linearly increasing schedule consumed monotonically across all $2N+1$ stages (encoder, bottleneck, decoder), consistent with the linear drop-path rule.

**Layer Scale.** The paper (Sec. 3.1) states "Layer Scale of initial value $10^{-6}$ is applied", i.e. a learnable per-channel vector $\gamma$ initialised to $10^{-6}$ multiplies the block's residual branch immediately before drop path: $\mathbf{y}=\mathbf{x}+\text{DropPath}(\gamma\odot f(\mathbf{x}))$. `ConvNeXtBlock` registers `self.gamma = nn.Parameter(layer_scale_init * torch.ones(channels))` (`convnext_unet.py:34`) and applies `x = self.gamma * x` after `fc2` while the tensor is still channels-last, before the permute-back and `self.drop_path` (`convnext_unet.py:45-47`), matching the official `facebookresearch/ConvNeXt` block. The initial value is threaded as `layer_scale_init` ($10^{-6}$ by default) through `ConvNeXtUNetConfig` (`configuration/architectures/backbone.py`) $\to$ `ConvNeXtUNet` $\to$ `ConvNeXtStage` $\to$ `ConvNeXtBlock`, alongside the `ffn_ratio`/`drop_path` threading. When `layer_scale_init` $\le 0$ the parameter is `None` and the branch is added unscaled. The $\gamma$ parameters live inside the encoder/bottleneck/decoder submodules, so `get_param_groups` covers them automatically.

**Accepted adaptations.** (i) *Stem* — the paper's patchify stem is a $4\times4$ stride-4 non-overlapping convolution that downsamples $4\times$ at entry (Sec. 2.2); the code uses a $3\times3$ stride-1 convolution plus LN (`convnext_unet.py:90-93`), preserving full input resolution. This is the standard dense-prediction adaptation: a U-Net needs a full-resolution feature map for the first skip connection, and all spatial reduction is deferred to the separate downsampling layers. Justified, non-structural. (ii) *Per-stage $1\times1$ projection* (`convnext_unet.py:55`) — the official implementation folds the channel change into the $2\times2$ downsample convolution, whereas here the downsample keeps channels constant ($C\to C$) and a $1\times1$ convolution at the head of each stage performs the channel change. The two placements are functionally equivalent (a linear channel remap adjacent to the resolution change); the code's choice is what lets the decoder feed concatenated $2C$ skips into a stage cleanly. Justified. (iii) *Output head* — a $1\times1$ convolution to `out_channels` (`convnext_unet.py:115`) replaces the classification head (final LN + global average pool + linear); this is the canonical dense-prediction head and the correct choice for a U-Net.

**Decoder (coherence check only).** Wiring is correct (`convnext_unet.py:108-136`): `reversed_features = [bottleneck_channels] + features[::-1]`; each step upsamples with a $2\times2$ stride-2 transposed convolution from `reversed_features[i]` to `reversed_features[i+1]`, aligns to the skip via `match_spatial_size`, concatenates the encoder skip (channels `reversed_features[i+1]`) to give $2\times$ channels, and runs a `ConvNeXtStage` whose $1\times1$ projection maps $2C\to C$. Skips are captured pre-downsample (`convnext_unet.py:125`), so spatial sizes and channel counts match at each concat. Bottleneck channel count is `features[-1] * bottleneck_factor` (`convnext_unet.py:77`). No channel-bookkeeping errors found.

---

## Related Notes

- [[UNet]] — Shared topology
- [[ResUNet]] — Classic residual counterpart
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — ConvNeXtUNetConfig
- [[DLR-TomoSAR Index]] — Master index
