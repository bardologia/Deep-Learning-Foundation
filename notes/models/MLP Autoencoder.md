---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - MlpAutoencoder
  - mlp_ae
family: profile-autoencoder
registry_key: mlp_ae
summary: Default, cheapest profile autoencoder; fully-convolutional symmetric MLP baseline supplying the JEPA target latent.
---

# MLP Autoencoder

`MlpAutoencoder` (`models/autoencoder/mlp.py`) is the default [[Model Zoo|profile autoencoder]]. It learns the latent space of the *output* domain: it encodes a fitted, normalized elevation profile into a low-dimensional embedding and reconstructs it. The encoder later supplies the target latent that the [[JEPA Profile-Embedding|JEPA]] predictor regresses towards.

---

## Summary

The profile of length $L$ is treated as a flat feature vector and processed per pixel. Both encoder and decoder are symmetric multilayer perceptrons built from `AutoencoderBlocks.mlp_stack`, which is implemented with $1\times1$ convolutions so that every spatial pixel of a $B \times L \times H \times W$ tensor is mapped independently and the network stays fully convolutional. The encoder compresses $L \to d$ through a hidden width $h$; the decoder mirrors it $d \to L$. The latent embedding is passed through the configured `embedding_norm` (default L2) before decoding, via the shared `EmbeddingNorm` mixin in `AutoencoderBase` (`models/autoencoder/base.py`). It is the cheapest member of the zoo and the strongest baseline.

---

## Symbols

| $symbol$ | meaning |
|---|---|
| $\mathbf{p}$ | input profile, $\mathbf{p} \in \mathbb{R}^{L}$ per pixel |
| $L$ | profile length (`profile_length`, default 256) |
| $h$ | hidden width (`hidden_dim`, default 128) |
| $d$ | embedding dimension (`embedding_dim`, default 24) |
| $D$ | network depth (`depth`, default 2 hidden blocks) |
| $\text{Act}$ | activation function (`activation`; default `gelu`) |
| $\mathbf{z}$ | latent embedding, $\mathbf{z} \in \mathbb{R}^{d}$ |
| $\hat{\mathbf{p}}$ | reconstructed profile |

---

## Architecture

Each $1\times1$ block applies a pointwise linear map, an activation, and optional dropout. For the encoder with $D$ hidden blocks:

$$
\begin{aligned}
\mathbf{u}_0 &= \mathbf{p} \\
\mathbf{u}_k &= \text{Act}(W_k \mathbf{u}_{k-1} + \mathbf{b}_k), \quad k = 1, \dots, D \\
\mathbf{z}_{\text{raw}} &= W_{\text{out}} \mathbf{u}_D + \mathbf{b}_{\text{out}} \\
\mathbf{z} &= \text{EmbeddingNorm}(\mathbf{z}_{\text{raw}})
\end{aligned}
$$

with $W_1 \in \mathbb{R}^{h \times L}$, $W_{2 \dots D} \in \mathbb{R}^{h \times h}$, and $W_{\text{out}} \in \mathbb{R}^{d \times h}$. The decoder is the mirror map $d \to h \to \dots \to L$, producing $\hat{\mathbf{p}}$ with no terminal normalization. Training minimizes a reconstruction loss between $\mathbf{p}$ and $\hat{\mathbf{p}}$ (`AutoencoderLossConfig`, default MSE).

---

## When to use

The default starting point. With no spatial or sequential inductive bias it is the smallest and fastest profile autoencoder, and in practice a very strong reconstruction baseline. Prefer [[Conv1D Autoencoder]] when local smoothness of the profile should be exploited, or [[Transformer1D Autoencoder]] when long-range dependencies along the elevation axis matter.
