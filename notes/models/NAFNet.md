---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - Nonlinear Activation Free Network
family: nafnet
registry_key: nafnet
summary: Nonlinearity-free restoration backbone of gated blocks with simplified channel attention, additive skips, and PixelShuffle upsampling (Chen et al., 2022).
---

# NAFNet

`NAFNet` (`models/backbone/nafnet.py`) is a Nonlinear Activation Free Network (Chen et al., *Simple Baselines for Image Restoration*, ECCV 2022, arXiv:2204.04676) adapted for dense regression. It is the zoo's representative of the image-restoration family: architectures built for high-fidelity per-pixel output rather than semantic abstraction, which is the closer match to Gaussian-parameter regression than the segmentation-derived designs.

---

## Summary

The network is a U-shaped encoder-decoder of `NAFBlock`s with additive skip connections. Each `NAFBlock` carries two residual branches: a spatial branch (LayerNorm → $1\times1$ expansion → $3\times3$ depthwise conv → SimpleGate → simplified channel attention → $1\times1$ projection) and an FFN branch (LayerNorm → $1\times1$ expansion → SimpleGate → $1\times1$ projection), each scaled by a zero-initialised learnable per-channel factor ($\beta$, $\gamma$). No ReLU, GELU, or any conventional activation appears anywhere: SimpleGate — an elementwise product of the two channel halves — is the only nonlinearity. Downsampling is a stride-2 $2\times2$ convolution that doubles channels; upsampling is a $1\times1$ convolution + `PixelShuffle` that halves them. The default `width = 32`, `enc_blocks = [2, 2, 4, 8]`, `middle_blocks = 12`, `dec_blocks = [2, 2, 2, 2]` reproduces the published SIDD configuration at 29.2M parameters, inside the zoo's reference band.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | Block input feature map (channels $c$) |
| $\mathbf{X}_1, \mathbf{X}_2$ | The two channel halves entering SimpleGate |
| $\beta, \gamma$ | Zero-initialised learnable residual scales of the two branches |
| $c$ | Block channel width (`width` $\times\, 2^i$ at stage $i$) |
| $\text{DW}_{3\times3}$ | Depthwise $3\times3$ convolution |
| $\text{SCA}$ | Simplified channel attention: global average pool → $1\times1$ conv → channelwise multiply |
| $\text{LN}$ | LayerNorm over the channel dimension |

---

## Architecture

### SimpleGate

$$
\text{SimpleGate}(\mathbf{X}) = \mathbf{X}_1 \odot \mathbf{X}_2
$$

The feature map is split into two halves along channels and multiplied elementwise — a degenerate gated linear unit that replaces both the activation function and, combined with SCA, the full channel-attention module of prior restoration networks.

### NAFBlock

$$
\begin{aligned}
\mathbf{u} &= \mathbf{x} + \beta  \cdot \text{Conv}_{1\times1}\!\Big(\text{SCA}\big(\text{SimpleGate}(\text{DW}_{3\times3}(\text{Conv}_{1\times1}(\text{LN}(\mathbf{x}))))\big)\Big) \\
\mathbf{y} &= \mathbf{u} + \gamma \cdot \text{Conv}_{1\times1}\!\Big(\text{SimpleGate}\big(\text{Conv}_{1\times1}(\text{LN}(\mathbf{u}))\big)\Big)
\end{aligned}
$$

The $1\times1$ expansions widen channels by `dw_expand` and `ffn_expand` respectively before SimpleGate halves them again. $\beta$ and $\gamma$ start at zero, so every block begins as the identity and training grows the residual contributions.

### Encoder-Decoder

Stage $i$ of the encoder stacks `enc_blocks[i]` NAFBlocks at width $c_i = \texttt{width}\cdot 2^i$, then downsamples with a stride-2 $2\times2$ convolution to $2c_i$ channels. The bottleneck runs `middle_blocks` blocks at the coarsest width. Each decoder stage upsamples with $1\times1$ conv + `PixelShuffle(2)` (halving channels), **adds** the matching encoder skip, and stacks `dec_blocks[i]` blocks. A $3\times3$ `output_head` maps the full-resolution feature map to the $3K$ parameter channels; the global input-to-output residual of the restoration setting is omitted because input and output live in different spaces here.

---

## Design Rationale

**The restoration hypothesis.** Every other encode-decode model in the [[Model Zoo]] descends from semantic segmentation, which trades spatial fidelity for abstraction. Restoration architectures make the opposite trade — shallow semantic depth, high per-pixel fidelity, cheap global mixing through channel statistics — and NAFNet is the strongest simple representative of that family. If the segmentation prior is wrong for Gaussian-parameter regression, this is the model that shows it.

**Simplicity as control.** NAFNet reached state of the art on GoPro and SIDD while removing activations, self-attention, and multi-stage refinement; its performance is attributable to block topology and normalisation, not to any exotic component. That makes it a clean single hypothesis to test.

---

## Parameter Reference

See [[Configuration Layer]] → `NAFNetConfig` (`width` — the capacity-matching knob; `enc_blocks`, `middle_blocks`, `dec_blocks`; `dw_expand`, `ffn_expand`).

---

## Paper fidelity

Ground truth: Chen, Chu, Zhang, Sun, *Simple Baselines for Image Restoration*, ECCV 2022, arXiv:2204.04676, and the official `megvii-research/NAFNet` implementation. Comparison against `models/backbone/nafnet.py`.

| # | Dimension | Verdict | Evidence |
|---|-----------|---------|----------|
| 1 | NAFBlock spatial branch (LN → 1×1 expand → 3×3 DW → SimpleGate → SCA → 1×1) | MATCH | Paper Sec. 3, Fig. 4; `nafnet.py` `NAFBlock.forward` first branch |
| 2 | NAFBlock FFN branch (LN → 1×1 expand → SimpleGate → 1×1) | MATCH | Paper Sec. 3; second branch |
| 3 | Zero-initialised learnable residual scales $\beta, \gamma$ | MATCH | Official implementation; `nn.Parameter(torch.zeros(...))` |
| 4 | SimpleGate as the only nonlinearity | MATCH | Paper Sec. 3.2; no `build_activation` anywhere in the module |
| 5 | Stride-2 2×2 conv down, 1×1 conv + PixelShuffle up, additive skips | MATCH | Official implementation; `downsample_layers`, `upsample_layers`, `x = x + skip` |
| 6 | LayerNorm over channels | MATCH | Paper Sec. 3.1; shared `ChannelLayerNorm` (`models/blocks.py`) |
| 7 | Global input residual and image-space 3×3 ending conv | ACCEPTED ADAPTATION | Restoration maps image→image; here the head maps features → $3K$ parameter channels, so the global residual is dropped |
| 8 | Default size (width 32, [2,2,4,8]/12/[2,2,2,2], 29.2M) | MATCH | Official SIDD configuration; verified 29,160,006 parameters at $C_{\text{in}}=1$, $C_{\text{out}}=6$ |
| 9 | `match_spatial_size` before skip addition | ACCEPTED ADAPTATION | Guards odd input sizes; identity for the power-of-two patches used in training |

---

## Related Notes

- [[ConvNeXt UNet]] — Modern conv design under the segmentation topology; NAFNet is the restoration counterpart
- [[HRNet]] — The other fidelity-first design (no resolution bottleneck)
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — NAFNetConfig
- [[DLR-TomoSAR Index]] — Master index
