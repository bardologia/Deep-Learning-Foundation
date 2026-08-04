---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - ConvNeXt2dImageAutoencoder
  - convnext2d_ae
family: image-autoencoder
registry_key: convnext2d_ae
summary: ConvNeXt-block image autoencoder for the SAR stack, sharing design with the ConvNeXt UNet backbone.
---

# ConvNeXt2D Image Autoencoder

`ConvNeXt2dImageAutoencoder` (`models/image_autoencoder/convnext2d.py`) is a [[Model Zoo|image autoencoder]] built from ConvNeXt blocks — the modern convolution design also used by the [[ConvNeXt UNet]] backbone. It encodes the SAR image stack into a 2D spatial embedding and reconstructs it; the trained encoder is reused as the pretrained image front-end of the [[JEPA Profile-Embedding|JEPA]] predictor.

---

## Summary

A ConvNeXt encoder maps the input stack of $C_{\text{in}}$ channels to a spatial embedding of $d$ channels, downsampling by a power-of-two factor; a mirrored decoder reconstructs the stack. A $3\times3$ stem lifts the input to the base width, then $D$ `ConvNeXtBlock`s (`models/blocks.py`) run at that resolution. Each of the $n$ downsampling stages applies a `ChannelLayerNorm` and a stride-2 $2\times2$ convolution that doubles the channels, followed by $D$ ConvNeXt blocks. A $1\times1$ convolution maps to the embedding. The decoder reverses this with `build_upsample` stages and ConvNeXt refinement before a $1\times1$ head back to $C_{\text{in}}$. The embedding is passed through `embedding_norm` (default `none`) via the shared `EmbeddingNorm` mixin (`models/image_autoencoder/base.py`). The number of stages is $n = \log_2(\texttt{downsample\_factor})$.

---

## Symbols

| $symbol$ | meaning |
|---|---|
| $\mathbf{X}$ | input image stack, $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ |
| $C_{\text{in}}$ | input channels (`in_channels`, default 1) |
| $b$ | base channels (`base_channels`, default 32) |
| $D$ | blocks per stage (`depth`, default 2) |
| $n$ | downsampling stages, $n = \log_2(\texttt{downsample\_factor})$ (default factor 2, so $n=1$) |
| $r$ | inverted-bottleneck ratio ($\texttt{FFN\_RATIO} = 4$) |
| $d$ | embedding dimension (`embedding_dim`, default 24) |
| $\text{Act}$ | activation (`activation`; default `gelu`) |
| $\mathbf{Z}$ | spatial embedding, $\mathbf{Z} \in \mathbb{R}^{d \times H' \times W'}$ |
| $\hat{\mathbf{X}}$ | reconstructed stack |

---

## Architecture

A ConvNeXt block at width $C$: a depthwise $7\times7$ convolution, channel `LayerNorm`, an inverted-bottleneck MLP ($C \to rC \to C$ via $1\times1$ projections with activation), a learnable layer-scale $\gamma$, and a residual add:

$$
\text{CNX}(\mathbf{x}) = \mathbf{x} + \gamma \odot \Big(W_2\,\text{Act}\big(W_1\,\text{LN}(\text{DWConv}_{7}(\mathbf{x}))\big)\Big), \quad W_1 \in \mathbb{R}^{rC \times C}, \; W_2 \in \mathbb{R}^{C \times rC}
$$

Encoder: a $3\times3$ stem to width $b$, $D$ ConvNeXt blocks, then $n$ downsampling stages (`ChannelLayerNorm` + stride-2 $2\times2$ conv doubling channels, then $D$ blocks), and a $1\times1$ projection:

$$
\begin{aligned}
\mathbf{U} &= \text{CNX}^{\,\circ D}(\text{Conv}_{3}:\, C_{\text{in}} \to b\,(\mathbf{X})) \\
\mathbf{V} &= \text{Down}^{(n)}(\mathbf{U}) \in \mathbb{R}^{b \cdot 2^{n} \times H' \times W'}, \quad H' = H/2^{n} \\
\mathbf{Z} &= \text{EmbeddingNorm}\big(\text{Conv}_{1\times1}:\, b\cdot 2^{n} \to d\,(\mathbf{V})\big)
\end{aligned}
$$

Decoder: a $1\times1$ lift from $d$ to the bottleneck width, $n$ upsampling stages (`upsample_mode`, default `convtranspose`) halving channels with $D$ ConvNeXt blocks each, $D{-}1$ refinement blocks, and a $1\times1$ head to $C_{\text{in}}$:

$$
\hat{\mathbf{X}} = \text{Conv}_{1\times1}\big(\text{Refine}(\text{Up}^{(n)}(\text{Conv}_{1\times1}:\, d \to b\cdot 2^{n}\,(\mathbf{Z})))\big)
$$

`encode_features` bilinearly resizes $\mathbf{Z}$ to a requested spatial size for JEPA. Training minimizes a reconstruction loss between $\mathbf{X}$ and $\hat{\mathbf{X}}$ (default MSE).

---

## When to use

When evaluating modern convolution design — large depthwise kernels, inverted bottlenecks, LayerNorm and layer-scale — for the image front-end. ConvNeXt blocks see wider spatial context than the plain $3\times3$ stacks of the [[Conv2D Image Autoencoder]] while staying parameter-efficient, and reuse the same blocks as the [[ConvNeXt UNet]] backbone.
