---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - UNetSkip
family: resunet
registry_key: unet_skip
summary: Residual U-Net variant using MaxPool2d downsampling instead of ResUNet's stride-2 convolutions.
---

# UNet Skip

`UNetSkip` (`models/backbone/resunet.py`, registry key `"unet_skip"`) is a U-Net with residual conv blocks and `MaxPool2d` downsampling. It subclasses `ResUNetBackbone` and is instantiated with `downsample="maxpool"`, so every residual unit runs at stride 1 and all spatial reduction is performed by separate `MaxPool2d(2)` layers.

---

## Summary

Each encoder stage applies a residual block (the shared `ResidualConvBlock` from `models/blocks.py`, the same block used by [[ResUNet]], here at its default stride 1 with `first_unit = False`) followed by a `MaxPool2d(2)` downsampling step; the deepest stage is a single residual bottleneck (stride 1); each decoder stage upsamples, concatenates the corresponding encoder skip, and applies a residual block. A final $1\times1$ convolution projects to the output channels. The topology, concatenative skip connections, and upsampling are identical to [[UNet]]; the only difference from [[UNet]] is that the plain double-convolution block is replaced by a residual block.

The parameterised submodules are `encoder_blocks`, `bottleneck`, `upsample_layers`, `decoder_blocks`, and `output_head` (inherited from `ResUNetBackbone`); `downsample_layers` holds the `MaxPool2d` layers. Because `MaxPool2d` carries no learnable parameters, `downsample_layers` contributes no keys to the state dict, so the saved tensors are exactly `encoder_blocks`, `bottleneck`, `upsample_layers`, `decoder_blocks`, and `output_head`.

---

## Symbols

| $symbol$ | meaning |
|---|---|
| $\mathbf{x}$ | input feature map, $\mathbf{x} \in \mathbb{R}^{B \times C_{\text{in}} \times H \times W}$ |
| $\mathbf{u}_i$ | intermediate activations of the main branch |
| $B$ | batch size |
| $H, W$ | feature-map height and width |
| $C_{\text{in}}, C_{\text{out}}$ | input and output channel counts |
| $\text{Conv}_{3\times3}$ | $3\times3$ convolution with padding 1 and stride 1 (no spatial size change); the first maps $C_{\text{in}} \to C_{\text{out}}$, the second $C_{\text{out}} \to C_{\text{out}}$ |
| $\text{Conv}_{1\times1}$ | $1\times1$ convolution used as the learned projection shortcut |
| $\text{Norm}$ | normalisation layer (`normalization`; default `"batch"`) |
| $\text{Act}$ | activation function (`activation`; default `"relu"`) |
| $\text{Proj}$ | identity or learned $1\times1$ projection (defined below) |
| $\text{ResBlock}$ | residual block (stride 1, `first_unit = False`) |
| $K$ | number of Gaussian components |

---

## Architecture

The block uses pre-activation ordering (`ResidualConvBlock` from `models/blocks.py`, the same block as [[ResUNet]], here at stride 1): each convolution is preceded by normalisation and activation. For input $\mathbf{x}$ with $C_{\text{in}}$ channels and output $C_{\text{out}}$:

$$
\begin{aligned}
\mathbf{u}_1 &= \text{Act}(\text{Norm}(\mathbf{x})) \\
\mathbf{u}_2 &= \text{Conv}_{3\times3}(\mathbf{u}_1) \\
\mathbf{u}_3 &= \text{Act}(\text{Norm}(\mathbf{u}_2)) \\
\mathbf{u}_4 &= \text{Conv}_{3\times3}(\mathbf{u}_3) \\
\text{ResBlock}(\mathbf{x}) &= \mathbf{u}_4 + \text{Proj}(\mathbf{x})
\end{aligned}
$$

Optional `Dropout2d` is appended after the final convolution of the main branch when `dropout > 0`.

$$
\text{Proj}(\mathbf{x}) = \begin{cases}
\mathbf{x} & C_{\text{in}} = C_{\text{out}} \\
\text{Conv}_{1\times1, C_{\text{in}} \to C_{\text{out}}}(\mathbf{x}) & C_{\text{in}} \neq C_{\text{out}}
\end{cases}
$$

The projection is applied when input and output channels differ. Because every block runs at stride 1, the shortcut (`self.shortcut`) is a $1\times1$ convolution only when channels change and `nn.Identity` otherwise; the residual shortcut is summed (not concatenated) with the main branch output. No stride is ever applied inside the block — all spatial reduction is delegated to the separate `MaxPool2d(2)` layers in the encoder.

---

## Relationship to ResUNet

`UNetSkip` and [[ResUNet]] are sibling subclasses of `ResUNetBackbone`, differing only in the `downsample` mode passed to the backbone constructor: `UNetSkip` uses `downsample="maxpool"`, [[ResUNet]] uses the backbone default `downsample="stride"`. This single switch sets the encoder stride, first-unit handling, bottleneck stride, and downsampling operator:

| Aspect | `UNetSkip` (`downsample="maxpool"`) | [[ResUNet]] (`downsample="stride"`) |
|---|---|---|
| Encoder downsampling | `MaxPool2d(2)` after each residual block | stride-2 first conv inside residual blocks 1+ |
| Bottleneck stride | 1 | 2 |
| Residual block stride | 1 everywhere | 1 (block 0, decoder) / 2 (encoder blocks 1+) |
| First-unit handling | `first_unit = False` everywhere (full pre-activation) | encoder block 0 passes `first_unit = True`, dropping its leading `Norm → Act` |
| Residual block class | `ResidualConvBlock` (from `models/blocks.py`) | `ResidualConvBlock` (same class) |
| Skip / decoder mechanics | identical | identical |

Both share the same `ResidualConvBlock` definition; in `UNetSkip` the backbone passes `stride = 1` and `first_unit = False` for every block, so every block — including encoder block 0 — keeps the full pre-activation `Norm → Act → Conv → Norm → Act → Conv` ordering and stride 1. Consequently `encoder_blocks.0.layers.0` is a `BatchNorm` (see State Dict Layout).

---

## State Dict Layout

The state-dict keys and shapes follow directly from the submodule structure:

- Top-level state-dict module prefixes: `encoder_blocks`, `bottleneck`, `upsample_layers`, `decoder_blocks`, `output_head` — i.e. every parameterised submodule. `downsample_layers` (`MaxPool2d`) holds no parameters and therefore appears with zero keys.
- Encoder block indices `0–3`, decoder block indices `0–3`, upsample indices `0–3`, plus a single `bottleneck`, consistent with the default four-level `features = [64, 128, 256, 512]`.
- The first encoder convolution weight is `encoder_blocks.0.layers.2.weight` of shape $(64, C_{\text{in}}, 3, 3)$ and the output head is `output_head.weight` of shape $(C_{\text{out}}, 64, 1, 1)$, reflecting the leading $\text{Norm}\to\text{Act}$ pre-activation order (the `BatchNorm` at `layers.0` precedes the first conv at `layers.2`) and the $1\times1$ output projection.
- With the default `in_channels = 1` and `out_channels = 6` these resolve to $C_{\text{in}} = 1$ and $C_{\text{out}} = 6$ (two Gaussians at three parameters each, `params_per_gaussian = 3`); both are overridable via config so that, for example, a three-pass `mag_real_imag` input gives $C_{\text{in}} = 9$ and five Gaussians give $C_{\text{out}} = 15$.

---

## Parameter Reference

See [[Configuration Layer]] → `UNetSkipConfig`. The configurable fields and their defaults mirror [[UNet]] and [[ResUNet]].

| Parameter | Default | Description |
|---|---|---|
| `features` | `[64, 128, 256, 512]` | Encoder channel widths |
| `bottleneck_factor` | `2` | Bottleneck channel multiplier |
| `dropout` | `0.15` | Per-block dropout |
| `activation` | `"relu"` | Activation function |
| `normalization` | `"batch"` | Normalisation layer |
| `upsample_mode` | `"convtranspose"` | Upsampling mode |
| `in_channels` | `1` | Input channel count |
| `out_channels` | `6` | Output channel count ($3K$ for $K=2$ Gaussians) |

`UNetSkipConfig.get_param_groups` partitions parameters into `encoder` (encoder blocks **and** the `MaxPool` `downsample_layers`), `bottleneck`, `decoder` (upsample layers and decoder blocks), and `output_head`; empty groups are dropped. The `downsample_layers` are included for completeness even though `MaxPool2d` exposes no trainable parameters.

---

## Paper fidelity

**Review date:** 2026-06-04

`UNetSkip` is a **project-specific hybrid with no single publication**. Its three structural ingredients each trace to a different source:

- **Topology** — the encoder/bottleneck/decoder layout with concatenative skip connections follows [[UNet]] (O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," *MICCAI*, 2015; arXiv:1505.04597; [[UNet_Ronneberger2015_1505.04597.pdf|PDF]]).
- **Blocks** — the residual conv blocks follow the residual-unit (pre-activation BN-ReLU-Conv with summed shortcut) formulation of Z. Zhang, Q. Liu, Y. Wang, "Road Extraction by Deep Residual U-Net," *IEEE Geoscience and Remote Sensing Letters*, 2018 (arXiv:1711.10684), Fig. 1(b), Sec. II-A2, Eq. 1–3. [[ResUNet_Zhang2018_1711.10684.pdf|PDF]]
- **Downsampling** — `MaxPool2d(2)` **deviates from Zhang et al.**, who explicitly replace pooling with stride-2 convolutions (Sec. II-A; Table I). The pooling choice here is the classic [[UNet]] decimation operator.

Consequently this architecture cannot be checked against one paper. The residual-block internals are faithful to Zhang et al. (verified in the [[ResUNet]] note); the spatial-reduction operator is deliberately the [[UNet]] `MaxPool` rather than Zhang's stride-2 convolution. The combination is a residual U-Net with pooled downsampling.

---

## Related Notes

- [[ResUNet]] — Sibling `ResUNetBackbone` subclass with stride-2 downsampling
- [[UNet]] — Base topology and the source of the `MaxPool` downsampling
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — UNetSkipConfig
- [[DLR-TomoSAR Index]] — Master index
