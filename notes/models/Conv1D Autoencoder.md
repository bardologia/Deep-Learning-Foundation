---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - Conv1dAutoencoder
  - conv1d_ae
family: profile-autoencoder
registry_key: conv1d_ae
summary: 1D convolutional profile autoencoder exploiting local smoothness along the range axis.
---

# Conv1D Autoencoder

`Conv1dAutoencoder` (`models/autoencoder/conv1d.py`) is a [[Model Zoo|profile autoencoder]] that treats the elevation profile as a 1D signal and applies convolutions along the range axis, exploiting the local smoothness of the fitted profile. Like the other profile autoencoders it defines the output latent space later targeted by [[JEPA Profile-Embedding|JEPA]].

---

## Summary

Each pixel's profile $\mathbf{p} \in \mathbb{R}^{L}$ is reshaped to a single-channel 1D sequence via `AutoencoderBlocks.to_sequence`, processed independently, and reshaped back with `from_sequence`. The encoder is two $1\text{D}$ convolutions (kernel $k$, padding $k//2$, channel width $c$) with activations, followed by adaptive average pooling to a single position and a linear projection to the embedding $d$. The embedding is L2-normalized by the shared `EmbeddingNorm` mixin (`models/autoencoder/base.py`). The decoder projects $d$ back to $c \times L$, then two convolutions map it down to the reconstructed single-channel profile.

---

## Symbols

| $symbol$ | meaning |
|---|---|
| $\mathbf{p}$ | input profile, $\mathbf{p} \in \mathbb{R}^{L}$ per pixel |
| $L$ | profile length (`profile_length`, default 256) |
| $c$ | convolution channels (`seq_channels`, default 32) |
| $k$ | kernel size (`seq_kernel_size`, default 5) |
| $d$ | embedding dimension (`embedding_dim`, default 24) |
| $\text{Act}$ | activation function (`activation`; default `gelu`) |
| $\mathbf{z}$ | latent embedding, $\mathbf{z} \in \mathbb{R}^{d}$ |
| $\hat{\mathbf{p}}$ | reconstructed profile |

---

## Architecture

Encoder, with $\text{Conv}^{1\text{D}}_{k}$ a padded kernel-$k$ convolution and $\text{GAP}$ global average pooling over the length axis:

$$
\begin{aligned}
\mathbf{h}_1 &= \text{Act}(\text{Conv}^{1\text{D}}_{k}:\, 1 \to c\,(\mathbf{p})) \\
\mathbf{h}_2 &= \text{Act}(\text{Conv}^{1\text{D}}_{k}:\, c \to c\,(\mathbf{h}_1)) \\
\mathbf{g}   &= \text{GAP}(\mathbf{h}_2) \in \mathbb{R}^{c} \\
\mathbf{z}   &= \text{EmbeddingNorm}(W_{\text{head}}\,\mathbf{g})
\end{aligned}
$$

Decoder, with a linear lift to $c \times L$ followed by two convolutions back to one channel:

$$
\begin{aligned}
\mathbf{f}     &= \text{reshape}\!\left(W_{\text{proj}}\,\mathbf{z}\right) \in \mathbb{R}^{c \times L} \\
\hat{\mathbf{p}} &= \text{Conv}^{1\text{D}}_{k}:\, c \to 1\,\big(\text{Act}(\text{Conv}^{1\text{D}}_{k}:\, c \to c\,(\mathbf{f}))\big)
\end{aligned}
$$

Training minimizes a reconstruction loss between $\mathbf{p}$ and $\hat{\mathbf{p}}$ (`AutoencoderLossConfig`, default MSE).

---

## When to use

When the local structure of the elevation profile carries the signal — adjacent range bins are correlated, and convolution shares parameters across positions while a pure [[MLP Autoencoder]] does not. Heavier than the MLP but lighter than the attention-based [[Transformer1D Autoencoder]].
