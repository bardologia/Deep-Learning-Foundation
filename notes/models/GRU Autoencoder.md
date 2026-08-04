---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - GruAutoencoder
  - gru_ae
family: profile-autoencoder
registry_key: gru_ae
summary: Recurrent (GRU) profile autoencoder modelling the elevation profile as an ordered sequence; most compact model.
---

# GRU Autoencoder

`GruAutoencoder` (`models/profile_autoencoder/gru.py`) is a [[Model Zoo|profile autoencoder]] that reads the elevation profile as a sequence with a gated recurrent unit, modelling ordering along the range axis explicitly. It is the most compact model in the profile zoo. Like the other profile autoencoders it defines the output latent space later targeted by [[JEPA Profile-Embedding|JEPA]].

---

## Summary

Each pixel's profile $\mathbf{p} \in \mathbb{R}^{L}$ is reshaped to a length-$L$ sequence of scalar features via `ProfileAutoencoderBlocks.to_sequence` (input feature dimension $1$). A multi-layer, optionally bidirectional GRU sweeps the sequence; the final hidden state of the last layer (concatenated across directions) is projected by a linear head to the embedding $d$, L2-normalized by the shared `EmbeddingNorm` mixin (`models/profile_autoencoder/base.py`). The decoder projects $d$ to the hidden width $h$, repeats it across all $L$ time steps as the input sequence, runs a unidirectional GRU, and a per-step linear layer emits the reconstructed scalar at each position.

---

## Symbols

| $symbol$ | meaning |
|---|---|
| $\mathbf{p}$ | input profile, $\mathbf{p} \in \mathbb{R}^{L}$ per pixel |
| $L$ | profile length (`profile_length`, default 256) |
| $h$ | GRU hidden width (`hidden_dim`, default 64) |
| $n$ | recurrent layers (`depth`, default 1) |
| $b$ | direction count, $2$ if `bidirectional` (default `True`) else $1$ |
| $d$ | embedding dimension (`embedding_dim`, default 24) |
| $\mathbf{z}$ | latent embedding, $\mathbf{z} \in \mathbb{R}^{d}$ |
| $\hat{\mathbf{p}}$ | reconstructed profile |

---

## Architecture

Encoder, with $\text{GRU}_{\text{enc}}$ an $n$-layer ($b$-directional) recurrent network over the $L$ scalar steps and $\mathbf{h}^{(L)}$ its final-layer hidden state(s):

$$
\begin{aligned}
\mathbf{h}^{(L)} &= \text{GRU}_{\text{enc}}(\mathbf{p}) \in \mathbb{R}^{b \cdot h} \quad \text{(directions concatenated)} \\
\mathbf{z}       &= \text{EmbeddingNorm}\big(W_{\text{head}}\,\mathbf{h}^{(L)}\big), \quad W_{\text{head}} \in \mathbb{R}^{d \times (b\,h)}
\end{aligned}
$$

Decoder, broadcasting the lifted embedding to all positions and decoding with a unidirectional GRU $\text{GRU}_{\text{dec}}$:

$$
\begin{aligned}
\mathbf{s}_t   &= W_{\text{lift}}\,\mathbf{z}, \quad t = 1, \dots, L, \;\; W_{\text{lift}} \in \mathbb{R}^{h \times d} \\
\mathbf{o}_{1:L} &= \text{GRU}_{\text{dec}}(\mathbf{s}_{1:L}) \\
\hat{p}_t      &= W_{\text{out}}\,\mathbf{o}_t, \quad W_{\text{out}} \in \mathbb{R}^{1 \times h}
\end{aligned}
$$

Training minimizes a reconstruction loss between $\mathbf{p}$ and $\hat{\mathbf{p}}$ (default MSE).

---

## When to use

When a compact model is wanted that still represents the profile as an ordered sequence rather than a flat vector. The GRU carries state along the range axis with a small parameter budget — the cheapest profile model here — though it is inherently sequential and less parallel than the [[Conv1D Autoencoder]] or [[Transformer1D Autoencoder]].
