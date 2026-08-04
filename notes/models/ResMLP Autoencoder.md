---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - ResMlpAutoencoder
  - resmlp_ae
family: profile-autoencoder
registry_key: resmlp_ae
summary: Pre-norm residual-MLP profile autoencoder; deeper variant of the MLP Autoencoder for better-conditioned training.
---

# ResMLP Autoencoder

`ResMlpAutoencoder` (`models/profile_autoencoder/resmlp.py`) is a [[Model Zoo|profile autoencoder]] that treats the elevation profile as a flat vector, like the [[MLP Autoencoder]], but replaces the plain dense stack with pre-norm residual blocks so the encoder and decoder can go deeper without optimisation trouble. Like the other profile autoencoders it defines the output latent space later targeted by [[JEPA Profile-Embedding|JEPA]].

---

## Summary

Each pixel's profile $\mathbf{p} \in \mathbb{R}^{L}$ is reshaped to a vector via `ProfileAutoencoderBlocks.to_sequence`, processed independently, and reshaped back with `from_sequence`. The encoder lifts the profile to width $h$ with a linear layer, applies $n$ residual MLP blocks, and projects to the embedding $d$. Each residual block is pre-norm: $\mathbf{x} \mapsto \mathbf{x} + W_2\,\text{Act}(W_1\,\text{LN}(\mathbf{x}))$, so the identity path keeps gradients well-conditioned at depth. The embedding is L2-normalized by the shared `EmbeddingNorm` mixin (`models/profile_autoencoder/base.py`). The decoder mirrors the encoder, lifting $d$ back to $h$, applying $n$ residual blocks, and projecting to the reconstructed profile of length $L$.

---

## Symbols

| $symbol$ | meaning |
|---|---|
| $\mathbf{p}$ | input profile, $\mathbf{p} \in \mathbb{R}^{L}$ per pixel |
| $L$ | profile length (`profile_length`, default 256) |
| $h$ | hidden width (`hidden_dim`, default 128) |
| $n$ | number of residual blocks per side (`depth`, default 3) |
| $d$ | embedding dimension (`embedding_dim`, default 24) |
| $\text{Act}$ | activation function (`activation`; default `gelu`) |
| $\text{LN}$ | `LayerNorm` over the width axis |
| $\mathbf{z}$ | latent embedding, $\mathbf{z} \in \mathbb{R}^{d}$ |
| $\hat{\mathbf{p}}$ | reconstructed profile |

---

## Architecture

A residual block at width $h$, with $\text{LN}$ applied before the dense pair (pre-norm):

$$
\text{ResMLP}(\mathbf{x}) = \mathbf{x} + W_2\,\text{Act}\!\big(W_1\,\text{LN}(\mathbf{x})\big), \qquad W_1, W_2 \in \mathbb{R}^{h \times h}
$$

Encoder, with $\circ$ the $n$-fold composition of residual blocks:

$$
\begin{aligned}
\mathbf{x}_0 &= W_{\text{embed}}\,\mathbf{p}, \quad W_{\text{embed}} \in \mathbb{R}^{h \times L} \\
\mathbf{x}_n &= \text{ResMLP}^{\,\circ n}(\mathbf{x}_0) \\
\mathbf{z}   &= \text{EmbeddingNorm}(W_{\text{head}}\,\mathbf{x}_n), \quad W_{\text{head}} \in \mathbb{R}^{d \times h}
\end{aligned}
$$

Decoder, symmetric with its own residual blocks:

$$
\hat{\mathbf{p}} = W_{\text{out}}\,\text{ResMLP}^{\,\circ n}\!\big(W_{\text{lift}}\,\mathbf{z}\big), \qquad W_{\text{lift}} \in \mathbb{R}^{h \times d}, \; W_{\text{out}} \in \mathbb{R}^{L \times h}
$$

Training minimizes a reconstruction loss between $\mathbf{p}$ and $\hat{\mathbf{p}}$ (default MSE).

---

## When to use

When the flat-vector treatment of the [[MLP Autoencoder]] is the right inductive bias but its shallow stack underfits. The residual path and pre-norm let `ResMLP` add depth (and therefore capacity) cheaply and stably, without the locality assumption of the [[Conv1D Autoencoder]] or the cost of the attention-based [[Transformer1D Autoencoder]].
