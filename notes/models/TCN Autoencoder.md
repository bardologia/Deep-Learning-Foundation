---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - TcnAutoencoder
  - tcn_ae
family: profile-autoencoder
registry_key: tcn_ae
summary: Dilated-residual temporal convolutional network profile autoencoder with exponentially growing receptive field.
---

# TCN Autoencoder

`TcnAutoencoder` (`models/profile_autoencoder/tcn.py`) is a [[Model Zoo|profile autoencoder]] built from a temporal convolutional network: stacked dilated residual convolutions along the range axis whose receptive field grows exponentially with depth. It captures both narrow and broad elevation features without the cost of attention. Like the other profile autoencoders it defines the output latent space later targeted by [[JEPA Profile-Embedding|JEPA]].

---

## Summary

Each pixel's profile $\mathbf{p} \in \mathbb{R}^{L}$ is reshaped to a single-channel 1D sequence via `ProfileAutoencoderBlocks.to_sequence`. A stem convolution lifts it to $c$ channels, then $n$ dilated residual blocks are applied with dilation $2^i$ at block $i$, so the effective receptive field doubles each block while the length $L$ is preserved (padding $d\cdot(k-1)/2$). Adaptive average pooling collapses the length axis and a linear head projects to the embedding $d$, L2-normalized by the shared `EmbeddingNorm` mixin (`models/profile_autoencoder/base.py`). The decoder projects $d$ back to $c \times L$, applies the same dilated residual stack, and a final convolution maps to the reconstructed single-channel profile.

---

## Symbols

| $symbol$ | meaning |
|---|---|
| $\mathbf{p}$ | input profile, $\mathbf{p} \in \mathbb{R}^{L}$ per pixel |
| $L$ | profile length (`profile_length`, default 256) |
| $c$ | convolution channels (`seq_channels`, default 32) |
| $k$ | kernel size (`seq_kernel_size`, default 3) |
| $n$ | number of dilated residual blocks per side (`depth`, default 3) |
| $2^i$ | dilation of block $i$, $i = 0, \dots, n-1$ |
| $d$ | embedding dimension (`embedding_dim`, default 24) |
| $\text{Act}$ | activation function (`activation`; default `gelu`) |
| $\mathbf{z}$ | latent embedding, $\mathbf{z} \in \mathbb{R}^{d}$ |
| $\hat{\mathbf{p}}$ | reconstructed profile |

---

## Architecture

A dilated residual block at dilation $\delta$, with $\text{Conv}^{1\text{D}}_{k,\delta}$ a length-preserving kernel-$k$ convolution of dilation $\delta$:

$$
\text{TCN}_\delta(\mathbf{x}) = \text{Act}\!\Big(\mathbf{x} + \text{Conv}^{1\text{D}}_{k,\delta}\big(\text{Act}(\text{Conv}^{1\text{D}}_{k,\delta}(\mathbf{x}))\big)\Big)
$$

Encoder, with $\text{GAP}$ global average pooling over the length axis:

$$
\begin{aligned}
\mathbf{h}_0 &= \text{Conv}^{1\text{D}}_{k}:\, 1 \to c\,(\mathbf{p}) \\
\mathbf{h}_n &= \big(\text{TCN}_{2^{n-1}} \circ \cdots \circ \text{TCN}_{2^0}\big)(\mathbf{h}_0) \\
\mathbf{z}   &= \text{EmbeddingNorm}\big(W_{\text{head}}\,\text{GAP}(\mathbf{h}_n)\big)
\end{aligned}
$$

Decoder, lifting to $c \times L$ before the dilated stack and a final $c \to 1$ convolution:

$$
\hat{\mathbf{p}} = \text{Conv}^{1\text{D}}_{k}:\, c \to 1\,\Big(\big(\text{TCN}_{2^{n-1}} \circ \cdots \circ \text{TCN}_{2^0}\big)\big(\text{reshape}(W_{\text{proj}}\,\mathbf{z})\big)\Big)
$$

Training minimizes a reconstruction loss between $\mathbf{p}$ and $\hat{\mathbf{p}}$ (default MSE).

---

## When to use

When the elevation profile has structure at several scales at once — sharp single-bin peaks alongside broad lobes. The exponentially growing dilation reaches long range with few parameters, sitting between the purely local [[Conv1D Autoencoder]] and the global but heavier [[Transformer1D Autoencoder]].
