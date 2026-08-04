---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - ResNet2dImageAutoencoder
  - resnet2d_ae
family: image-autoencoder
registry_key: resnet2d_ae
summary: Residual (pre-activation) image autoencoder; deeper counterpart to the default Conv2D Image Autoencoder.
---

# ResNet2D Image Autoencoder

`ResNet2dImageAutoencoder` (`models/image_autoencoder/resnet2d.py`) is a [[Model Zoo|image autoencoder]]: a deeper, residual counterpart to the default [[Conv2D Image Autoencoder]]. It encodes the SAR image stack into a 2D spatial embedding and reconstructs it; the trained encoder is reused as the pretrained image front-end of the [[JEPA Profile-Embedding|JEPA]] predictor.

---

## Summary

A residual convolutional encoder maps the input stack of $C_{\text{in}}$ channels to a spatial embedding of $d$ channels, downsampling by a power-of-two factor; a mirrored decoder reconstructs the stack. Every unit is a pre-activation `ResidualConvBlock` (`models/blocks.py`): the stem runs at the base width (with `first_unit=True` to skip the leading norm/activation), each of the $n$ downsampling stages uses a stride-2 residual block that doubles the channel count, and a $1\times1$ convolution maps to the embedding. The decoder reverses this with `build_upsample` stages and residual refinement before a $1\times1$ head back to $C_{\text{in}}$. The embedding is passed through `embedding_norm` (default `none`) via the shared `EmbeddingNorm` mixin (`models/image_autoencoder/base.py`). The number of stages is $n = \log_2(\texttt{downsample\_factor})$.

---

## Symbols

| $symbol$ | meaning |
|---|---|
| $\mathbf{X}$ | input image stack, $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ |
| $C_{\text{in}}$ | input channels (`in_channels`, default 1) |
| $b$ | base channels (`base_channels`, default 32) |
| $D$ | stem/refine depth (`depth`, default 2) |
| $n$ | downsampling stages, $n = \log_2(\texttt{downsample\_factor})$ (default factor 2, so $n=1$) |
| $d$ | embedding dimension (`embedding_dim`, default 24) |
| $\text{Act}$ | activation (`activation`; default `gelu`) |
| $\text{Norm}$ | normalisation (`normalization`; default `batch`) |
| $\mathbf{Z}$ | spatial embedding, $\mathbf{Z} \in \mathbb{R}^{d \times H' \times W'}$ |
| $\hat{\mathbf{X}}$ | reconstructed stack |

---

## Architecture

A pre-activation residual unit, with $f$ the two-convolution body and a $1\times1$ projection shortcut when channels or stride change:

$$
\text{Res}(\mathbf{x}) = f(\mathbf{x}) + \text{shortcut}(\mathbf{x}), \qquad f = \text{Conv}_{3}\!\circ\text{Act}\circ\text{Norm}\circ\text{Conv}_{3}\!\circ\text{Act}\circ\text{Norm}
$$

Encoder: a stem of $D$ residual blocks at width $b$, then $n$ stride-2 residual stages each doubling channels (with $D{-}1$ residual blocks at the new width), and a $1\times1$ projection to the embedding:

$$
\begin{aligned}
\mathbf{U} &= \text{Stem}(\mathbf{X}) \in \mathbb{R}^{b \times H \times W} \\
\mathbf{V} &= \text{Down}^{(n)}(\mathbf{U}) \in \mathbb{R}^{b \cdot 2^{n} \times H' \times W'}, \quad H' = H/2^{n} \\
\mathbf{Z} &= \text{EmbeddingNorm}\big(\text{Conv}_{1\times1}:\, b\cdot 2^{n} \to d\,(\mathbf{V})\big)
\end{aligned}
$$

Decoder: a $1\times1$ lift from $d$ to the bottleneck width, $n$ upsampling stages (`upsample_mode`, default `convtranspose`) each halving channels with a residual block, $D{-}1$ residual refinement blocks, and a $1\times1$ head to $C_{\text{in}}$:

$$
\hat{\mathbf{X}} = \text{Conv}_{1\times1}\big(\text{Refine}(\text{Up}^{(n)}(\text{Conv}_{1\times1}:\, d \to b\cdot 2^{n}\,(\mathbf{Z})))\big)
$$

`encode_features` bilinearly resizes $\mathbf{Z}$ to a requested spatial size for JEPA. Training minimizes a reconstruction loss between $\mathbf{X}$ and $\hat{\mathbf{X}}$ (default MSE).

---

## When to use

When the default [[Conv2D Image Autoencoder]] underfits the image domain and a deeper encoder is wanted without training instability. The pre-activation residual path supports more downsampling stages and greater depth at higher parameter cost, while keeping the same encode-to-spatial-embedding contract JEPA depends on.
