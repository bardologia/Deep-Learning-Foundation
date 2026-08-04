---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - DilatedConv2dImageAutoencoder
  - dilated2d_ae
family: image-autoencoder
registry_key: dilated2d_ae
summary: Atrous-convolution image autoencoder keeping full input resolution in the embedding (no downsampling).
---

# Dilated2D Image Autoencoder

`DilatedConv2dImageAutoencoder` (`models/image_autoencoder/dilated2d.py`) is a [[Model Zoo|image autoencoder]] that grows its receptive field through atrous (dilated) convolutions instead of spatial downsampling, so the embedding keeps the full input grid size. It encodes the SAR image stack into a 2D spatial embedding and reconstructs it; the trained encoder is reused as the pretrained image front-end of the [[JEPA Profile-Embedding|JEPA]] predictor.

---

## Summary

A stem convolution lifts the input stack of $C_{\text{in}}$ channels to the base width $b$, then $m$ dilated residual blocks are applied with dilation $2^i$ at block $i$, doubling the receptive field each block while preserving the spatial size (padding equal to the dilation). A $1\times1$ convolution maps to the embedding $d$ at the original resolution. The decoder is symmetric: a $1\times1$ lift, the same dilated residual stack, and a $1\times1$ head back to $C_{\text{in}}$. The embedding is passed through `embedding_norm` (default `none`) via the shared `EmbeddingNorm` mixin (`models/image_autoencoder/base.py`). There is no spatial downsampling — `downsample_factor` is fixed at 1 — so $\mathbf{Z}$ has the same height and width as $\mathbf{X}$.

---

## Symbols

| $symbol$ | meaning |
|---|---|
| $\mathbf{X}$ | input image stack, $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ |
| $C_{\text{in}}$ | input channels (`in_channels`, default 1) |
| $b$ | base channels (`base_channels`, default 32) |
| $m$ | dilated residual blocks per side (`dilation_depth`, default 3) |
| $2^i$ | dilation of block $i$, $i = 0, \dots, m-1$ |
| $d$ | embedding dimension (`embedding_dim`, default 24) |
| $\text{Act}$ | activation (`activation`; default `gelu`) |
| $\text{Norm}$ | normalisation (`normalization`; default `batch`) |
| $\mathbf{Z}$ | spatial embedding, $\mathbf{Z} \in \mathbb{R}^{d \times H \times W}$ |
| $\hat{\mathbf{X}}$ | reconstructed stack |

---

## Architecture

A dilated residual block at dilation $\delta$, with $\text{Conv}^{2\text{D}}_{3,\delta}$ a size-preserving $3\times3$ convolution of dilation $\delta$:

$$
\text{D}_\delta(\mathbf{x}) = \text{Act}\!\Big(\mathbf{x} + \text{Norm}\,\text{Conv}^{2\text{D}}_{3,\delta}\big(\text{Act}\,\text{Norm}\,\text{Conv}^{2\text{D}}_{3,\delta}(\mathbf{x})\big)\Big)
$$

Encoder: a $3\times3$ stem to width $b$, $m$ dilated residual blocks with geometrically increasing dilation, and a $1\times1$ projection at unchanged resolution:

$$
\begin{aligned}
\mathbf{U} &= \text{Act}\,\text{Norm}\,\text{Conv}_{3}:\, C_{\text{in}} \to b\,(\mathbf{X}) \\
\mathbf{V} &= \big(\text{D}_{2^{m-1}} \circ \cdots \circ \text{D}_{2^0}\big)(\mathbf{U}) \in \mathbb{R}^{b \times H \times W} \\
\mathbf{Z} &= \text{EmbeddingNorm}\big(\text{Conv}_{1\times1}:\, b \to d\,(\mathbf{V})\big)
\end{aligned}
$$

Decoder: a $1\times1$ lift from $d$ to $b$, the same dilated stack, and a $1\times1$ head to $C_{\text{in}}$:

$$
\hat{\mathbf{X}} = \text{Conv}_{1\times1}:\, b \to C_{\text{in}}\,\Big(\big(\text{D}_{2^{m-1}} \circ \cdots \circ \text{D}_{2^0}\big)\big(\text{Conv}_{1\times1}:\, d \to b\,(\mathbf{Z})\big)\Big)
$$

`encode_features` bilinearly resizes $\mathbf{Z}$ to a requested spatial size for JEPA. Training minimizes a reconstruction loss between $\mathbf{X}$ and $\hat{\mathbf{X}}$ (default MSE).

---

## When to use

When fine spatial detail must survive the bottleneck — the embedding stays at full input resolution, unlike the downsampling [[Conv2D Image Autoencoder]] or [[ResNet2D Image Autoencoder]]. Dilation supplies wide context for the per-pixel embedding without an encode-decode resolution round trip; it is a pure channel bottleneck with a large receptive field.
