---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - Conv2dImageAutoencoder
  - conv2d_ae
family: image-autoencoder
registry_key: conv2d_ae
summary: Default plain-convolutional image autoencoder for the SAR stack, pretraining the JEPA image front-end.
---

# Conv2D Image Autoencoder

`Conv2dImageAutoencoder` (`models/image_autoencoder/conv2d.py`) is the [[Model Zoo|image autoencoder]]. Unlike the profile autoencoders, it learns the latent space of the *input* domain: it encodes the SAR image stack into a 2D spatial embedding and reconstructs it. The trained encoder is reused as the pretrained image front-end of the [[JEPA Profile-Embedding|JEPA]] predictor.

---

## Summary

A convolutional encoder maps the input stack of $C_{\text{in}}$ channels to a spatial embedding of $d$ channels, optionally downsampling by a power-of-two factor; a mirrored decoder reconstructs the stack. The encoder is a stem of `ConvBlock`s at the base width, then $n$ stride-2 downsampling stages that double the channel count, then a $1\times1$ convolution to the embedding dimension. The decoder reverses this with `build_upsample` stages and refinement `ConvBlock`s before a $1\times1$ head back to $C_{\text{in}}$. The embedding is passed through `embedding_norm` (default `none`) via the shared `EmbeddingNorm` mixin (`models/image_autoencoder/base.py`). The number of downsampling stages is $n = \log_2(\texttt{downsample\_factor})$, which must be a power of two.

---

## Symbols

| $symbol$ | meaning |
|---|---|
| $\mathbf{X}$ | input image stack, $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ |
| $C_{\text{in}}$ | input channels (`in_channels`, default 1) |
| $b$ | base channels (`base_channels`, default 32) |
| $D$ | stem/refine depth (`depth`, default 2) |
| $n$ | downsampling stages, $n = \log_2(\texttt{downsample\_factor})$ (default factor 1, so $n=0$) |
| $d$ | embedding dimension (`embedding_dim`, default 24) |
| $\text{Act}$ | activation (`activation`; default `gelu`) |
| $\text{Norm}$ | normalisation (`normalization`; default `batch`) |
| $\mathbf{Z}$ | spatial embedding, $\mathbf{Z} \in \mathbb{R}^{d \times H' \times W'}$ |
| $\hat{\mathbf{X}}$ | reconstructed stack |

---

## Architecture

Encoder: a stem of $D$ `ConvBlock`s ($C_{\text{in}} \to b$, then $b \to b$), followed by $n$ downsampling stages each a stride-2 convolution with $\text{Norm}$ and $\text{Act}$ that doubles channels, plus a `ConvBlock`, and a final $1\times1$ projection to the embedding:

$$
\begin{aligned}
\mathbf{U} &= \text{Stem}(\mathbf{X}) \in \mathbb{R}^{b \times H \times W} \\
\mathbf{V} &= \text{Down}^{(n)}(\mathbf{U}) \in \mathbb{R}^{b \cdot 2^{n} \times H' \times W'}, \quad H' = H/2^{n} \\
\mathbf{Z} &= \text{EmbeddingNorm}\big(\text{Conv}_{1\times1}:\, b\cdot 2^{n} \to d\,(\mathbf{V})\big)
\end{aligned}
$$

Decoder: a $1\times1$ lift from $d$ back to the bottleneck width, $n$ upsampling stages (`upsample_mode`, default `convtranspose`) halving channels, $D{-}1$ refinement `ConvBlock`s, and a $1\times1$ head to $C_{\text{in}}$:

$$
\hat{\mathbf{X}} = \text{Conv}_{1\times1}\big(\text{Refine}(\text{Up}^{(n)}(\text{Conv}_{1\times1}:\, d \to b\cdot 2^{n}\,(\mathbf{Z})))\big)
$$

`encode_features` additionally bilinearly resizes $\mathbf{Z}$ to a requested spatial size, which is how JEPA consumes the embedding as a feature map. Training minimizes a reconstruction loss between $\mathbf{X}$ and $\hat{\mathbf{X}}$ (`ImageAeLossConfig`, default MSE).

---

## When to use

The single image autoencoder, used whenever JEPA operates on a learned image representation rather than the raw stack. With `downsample_factor = 1` it is a pure channel bottleneck at full resolution; raise the factor to trade spatial resolution for a more compressed embedding.
