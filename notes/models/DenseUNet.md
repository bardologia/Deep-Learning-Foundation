---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - DenseUNet
  - FC-DenseNet
  - Tiramisu
family: densenet
registry_key: dense_unet
summary: Fully-convolutional DenseNet (Tiramisu) with dense blocks in a U-shaped down/up path and skip concatenation.
---

# DenseUNet

`DenseUNet` (`models/backbone/dense_unet.py`) is a fully-convolutional DenseNet (FC-DenseNet / "Tiramisu", [[FCDenseNet_Tiramisu_Jegou2017_1611.09326.pdf|Jégou et al., 2017]]): dense blocks in which every layer receives the concatenation of all preceding feature maps, arranged in a U-shaped down/up path with skip concatenation.

---

## Summary

The model is registered as `dense_unet` in `models/__init__.py`. A 3×3 stem maps `in_channels` (default $1$) to $3g$ channels (growth rate $g$ = `growth_rate`). Each down block is a `DenseBlock` of `block_layers[i]` layers; the block's new features are concatenated onto its input, stored as the skip, and passed through a `TransitionDown` (1×1 conv + MaxPool). The bottleneck dense block emits only its new features ($g \cdot L_b$ channels), which keeps the upsampling path narrow. Each up step is a transposed convolution, concatenation with the skip, and another dense block whose new features feed the next step; the final block's new features go to the 1×1 `output_head`, which emits `out_channels` (default $6$) per-pixel maps. With `params_per_gaussian = 3` and `out_channels = 6` the head produces the $3K$ parameters of a $K=2$ Gaussian-mixture range profile per pixel.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $H, W$ | Spatial height and width |
| $C_{in}$ | Input channel count of a block |
| $\mathbf{x}_\ell$ | New feature map produced by layer $\ell$ ($g$ channels) |
| $\mathbf{h}_\ell$ | Concatenation $[\mathbf{x}_0, \dots, \mathbf{x}_{\ell-1}]$ of all preceding feature maps |
| $\mathbf{a}_\ell$ | Normalized, activated input to layer $\ell$'s convolution |
| $\text{Norm}$ | Normalization (batch norm) |
| $\text{Act}$ | Activation (ReLU) |
| $\text{Conv}_{k\times k}$ | $k\times k$ convolution |
| $H_\ell$ | Dense-layer composite function (BN–ReLU–Conv$3\times3$–Dropout) |
| $\ell$ | Layer index inside a dense block |
| $L$ | Number of layers in a dense block |
| $L_b$ | Number of layers in the bottleneck dense block |
| $g$ | Growth rate (`growth_rate`); channels added per layer |
| $\theta$ | Transition compression factor ($\theta=1$, no compression) |

---

## Architecture

### Dense Layer and Block

Layer $\ell$ inside a block computes $g$ new channels from everything before it:

$$
\begin{aligned}
\mathbf{h}_{\ell} &= [\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{\ell-1}] \\
\mathbf{a}_{\ell} &= \text{Act}(\text{Norm}(\mathbf{h}_{\ell})) \\
\mathbf{x}_{\ell} &= \text{Conv}_{3\times3}(\mathbf{a}_{\ell}) \in \mathbb{R}^{B \times g \times H \times W}
\end{aligned}
$$

A block of $L$ layers adds $L \cdot g$ channels. Parameters grow as $O(L^2 g^2)$ per block, so the network reaches a given depth with far fewer parameters per layer than plain CNNs.

---

## Design Rationale

**Feature reuse instead of feature recomputation.** Dense connectivity gives every layer direct access to all earlier features and a direct gradient path to the loss — an extreme version of the skip philosophy that [[ResUNet]] applies per block and [[UNet++]] applies across the decoder grid. Per parameter, dense blocks are the most feature-efficient design in the zoo; capacity matching scales $g$ (reaching $g \approx 90$ at the UNet budget), making this a deep-and-thin counterpoint to the wide-and-shallow models.

---

## Parameter Reference

See [[Configuration Layer]] → `DenseUNetConfig` (`growth_rate`, `block_layers`, `bottleneck_layers`).

---

## Paper fidelity

*Review date: 2026-06-04.* Ground truth: Huang et al., *Densely Connected Convolutional Networks*, CVPR 2017 (arXiv:1608.06993) [[DenseNet_Huang2017_1608.06993.pdf|PDF]] — hereafter [[DenseNet_Huang2017_1608.06993.pdf|DenseNet]]; and Jégou et al., *The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation*, CVPRW 2017 (arXiv:1611.09326) [[FCDenseNet_Tiramisu_Jegou2017_1611.09326.pdf|PDF]] — hereafter [[FCDenseNet_Tiramisu_Jegou2017_1611.09326.pdf|FC-DenseNet]]. Code reviewed: `models/backbone/dense_unet.py` and `configuration/architectures/backbone.py` (`DenseUNetConfig`; shared builders).

The implementation is a faithful **FC-DenseNet (Tiramisu)** with one deliberate, paper-conformant choice (no bottleneck $1\times1$ inside dense layers) and one minor numerical robustness addition (spatial-size matching on the skip). No structural deviations were found. The single subtle point on which Tiramisu implementations usually fail — propagating *only the last dense block's new features* through the upsampling path rather than the full concatenation — is handled correctly.

### Verdict table

| # | Dimension | Paper reference | Code | Verdict |
|---|-----------|-----------------|------|---------|
| 1 | Dense-layer composite $H_\ell$ | [[FCDenseNet_Tiramisu_Jegou2017_1611.09326.pdf|FC-DenseNet]] Table 1 (BN–ReLU–Conv$3\times3$–Dropout); [[DenseNet_Huang2017_1608.06993.pdf|DenseNet]] §3 | `DenseLayer` | MATCH |
| 1b | No bottleneck $1\times1$ (DenseNet-B *not* used) | [[FCDenseNet_Tiramisu_Jegou2017_1611.09326.pdf|FC-DenseNet]] Table 1 uses plain $3\times3$; [[DenseNet_Huang2017_1608.06993.pdf|DenseNet]] §3 lists DenseNet-B as optional | `DenseLayer` ($3\times3$ conv only) | MATCH (Tiramisu convention; DenseNet-B is optional) |
| 2 | Dense connectivity $\mathbf{x}_\ell = H_\ell([\mathbf{x}_0,\dots,\mathbf{x}_{\ell-1}])$, input $= C_{in} + \ell g$ | [[DenseNet_Huang2017_1608.06993.pdf|DenseNet]] Eq. 2; [[FCDenseNet_Tiramisu_Jegou2017_1611.09326.pdf|FC-DenseNet]] Eq. 3 | `DenseBlock.__init__` (`input_channels + index * growth_rate`), `DenseBlock.forward` | MATCH |
| 3 | Block returns *new* features only; encoder re-concatenates input+new for skip, decoder does **not** | [[FCDenseNet_Tiramisu_Jegou2017_1611.09326.pdf|FC-DenseNet]] §3.2, Fig. 1–2 | `DenseBlock.forward` returns `cat(new_features)`; encoder `forward` re-concats; decoder `forward` does not | MATCH |
| 4 | Transition Down: BN–ReLU–Conv$1\times1$–Dropout–MaxPool$2\times2$ | [[FCDenseNet_Tiramisu_Jegou2017_1611.09326.pdf|FC-DenseNet]] Table 1 | `TransitionDown` | MATCH |
| 5 | Transition Up: $3\times3$ transposed conv, stride 2, transporting only last block's new features | [[FCDenseNet_Tiramisu_Jegou2017_1611.09326.pdf|FC-DenseNet]] Table 1, §3.2 | `trans_up` (`nn.ConvTranspose2d`); `up_channels = block.new_channels` | ACCEPTED ADAPTATION (kernel $2\times2$ vs paper $3\times3$ — both stride-2 transposed conv; kernel size is a hyperparameter, structurally identical) |
| 6 | Skip contents = full input+new concat of matching encoder block | [[FCDenseNet_Tiramisu_Jegou2017_1611.09326.pdf|FC-DenseNet]] Fig. 1 | `skip_connections.append(x)` after `cat([x, new_features])` | MATCH |
| 7 | Bottleneck dense block emits new features only, keeping up-path narrow | [[FCDenseNet_Tiramisu_Jegou2017_1611.09326.pdf|FC-DenseNet]] §3.2–3.3 | `self.bottleneck`; `bottleneck_new_channels = self.bottleneck.new_channels` | MATCH |
| 8 | Output head: $1\times1$ conv to `out_channels` maps | [[FCDenseNet_Tiramisu_Jegou2017_1611.09326.pdf|FC-DenseNet]] §3.3 (final $1\times1$) | `output_head` `nn.Conv2d(up_channels, out_channels, 1)` | MATCH (here a $3K$ Gaussian-mixture regression head, not segmentation logits; activation/loss external — out of scope) |
| 9 | Growth rate $g$; stem $= 3g$; no transition compression $\theta$ | [[FCDenseNet_Tiramisu_Jegou2017_1611.09326.pdf|FC-DenseNet]] Table 2 ($m=48=3g$ at $g{=}16$); [[DenseNet_Huang2017_1608.06993.pdf|DenseNet]] §3 compression is the optional DenseNet-C variant | stem `3 * growth`; `TransitionDown` conv channels→channels ($\theta=1$) | MATCH (no compression; Tiramisu uses $\theta=1$) |
| 10 | Spatial-size matching before skip concat | not in paper (same-conv blocks preserve size) | `match_spatial_size` | ACCEPTED ADAPTATION (defensive interpolation; no-op when sizes agree, which they do for power-of-two inputs) |

### Channel arithmetic check

Reproducing [[FCDenseNet_Tiramisu_Jegou2017_1611.09326.pdf|FC-DenseNet]] Table 2 (FC-DenseNet103, $g{=}16$, down blocks $4,5,7,10,12$, bottleneck $15$) with the code's conventions yields the exact published down-path totals $112,192,304,464,656$ and TU-concatenation inputs $896,656,464,304,192$. Table 2's reported `TU + DB` $m$ values (e.g. $1088 = 896+192$) are the *total* feature count at that resolution; the value *propagated to the next Transition Up* is the new-features-only count ($192$), which is exactly what `up_channels = block.new_channels` carries. This confirms the decoder does not re-concatenate its input into the upsampled stream, per [[FCDenseNet_Tiramisu_Jegou2017_1611.09326.pdf|FC-DenseNet]] §3.2 ("the input of a dense block is not concatenated with its output" in the upsampling path).

### Prose

Every load-bearing dimension of the Tiramisu specification is reproduced. The composite function is BN–ReLU–Conv$3\times3$ with optional dropout ([[FCDenseNet_Tiramisu_Jegou2017_1611.09326.pdf|FC-DenseNet]] Table 1), and the DenseNet-B bottleneck $1\times1$ is intentionally omitted — a legitimate choice since the Tiramisu itself omits it. Dense connectivity ([[DenseNet_Huang2017_1608.06993.pdf|DenseNet]] Eq. 2 / [[FCDenseNet_Tiramisu_Jegou2017_1611.09326.pdf|FC-DenseNet]] Eq. 3) is implemented with the correct growing input width $C_{in}+\ell g$. The asymmetry between encoder and decoder block-output handling — the encoder re-concatenates input with new features to form the skip and the down-stream, while the decoder propagates only new features to avoid the feature-map explosion — is the defining detail of the Tiramisu, and it is correct here.

Two non-structural adaptations: (i) the transposed convolution uses a $2\times2$ kernel rather than the paper's $3\times3$, which changes only the learned upsampling filter footprint, not the architecture (ACCEPTED ADAPTATION, minor); (ii) `match_spatial_size` interpolates the upsampled tensor to the skip's spatial size before concatenation (ACCEPTED ADAPTATION) — a no-op for power-of-two input dimensions and harmless robustness otherwise. No compression factor $\theta$ is applied at transitions, matching the Tiramisu ($\theta=1$); DenseNet-C compression is an optional DenseNet variant, not part of FC-DenseNet. Overall verdict: **MATCH** (faithful FC-DenseNet/Tiramisu; no deviations).

---

## Related Notes

- [[UNet++]] — Dense connectivity across the decoder grid
- [[ResUNet]] — Per-block residual alternative
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — DenseUNetConfig
- [[DLR-TomoSAR Index]] — Master index
