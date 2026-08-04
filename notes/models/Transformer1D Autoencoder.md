---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - Transformer1dAutoencoder
  - transformer1d_ae
family: profile-autoencoder
registry_key: transformer1d_ae
summary: Self-attention profile autoencoder capturing long-range dependencies; highest-capacity profile model in the zoo.
---

# Transformer1D Autoencoder

`Transformer1dAutoencoder` (`models/autoencoder/transformer1d.py`) is a [[Model Zoo|profile autoencoder]] that models the elevation profile with self-attention, capturing long-range dependencies between distant range bins. As with the other profile autoencoders, its encoder defines the output latent space targeted by [[JEPA Profile-Embedding|JEPA]].

---

## Summary

Each pixel's profile $\mathbf{p} \in \mathbb{R}^{L}$ is reshaped to a per-pixel token via `AutoencoderBlocks.to_sequence`. The encoder linearly embeds the profile to a model width $h$, processes it through a stack of `nn.TransformerEncoderLayer` blocks (`batch_first`), and projects the attended representation to the embedding $d$; the result is L2-normalized by the shared `EmbeddingNorm` mixin (`models/autoencoder/base.py`). The decoder mirrors this: it lifts $d$ back to $h$, runs a second transformer stack, and projects to the reconstructed profile of length $L$. It is the highest-capacity profile autoencoder in the zoo.

---

## Symbols

| $symbol$ | meaning |
|---|---|
| $\mathbf{p}$ | input profile, $\mathbf{p} \in \mathbb{R}^{L}$ per pixel |
| $L$ | profile length (`profile_length`, default 256) |
| $h$ | model width (`hidden_dim`, default 128) |
| $D$ | number of transformer layers (`depth`, default 2) |
| $H$ | attention heads (`num_heads`, default 4) |
| $d$ | embedding dimension (`embedding_dim`, default 24) |
| $\text{Act}$ | feedforward activation (`activation`; default `gelu`, falls back to `gelu` if not in {relu, gelu}) |
| $\mathbf{z}$ | latent embedding, $\mathbf{z} \in \mathbb{R}^{d}$ |
| $\hat{\mathbf{p}}$ | reconstructed profile |

---

## Architecture

Each transformer layer applies multi-head self-attention and a feedforward block with $\dim_{\text{ff}} = 2h$. Encoder:

$$
\begin{aligned}
\mathbf{t}_0 &= W_{\text{embed}}\,\mathbf{p} \in \mathbb{R}^{h} \\
\mathbf{t}_D &= \text{TransformerEncoder}^{(D)}(\mathbf{t}_0) \\
\mathbf{z}   &= \text{EmbeddingNorm}(W_{\text{head}}\,\mathbf{t}_D)
\end{aligned}
$$

Decoder, a mirror with an independent transformer stack:

$$
\begin{aligned}
\mathbf{s}_0 &= W_{\text{embed}}'\,\mathbf{z} \in \mathbb{R}^{h} \\
\mathbf{s}_D &= \text{TransformerEncoder}'^{(D)}(\mathbf{s}_0) \\
\hat{\mathbf{p}} &= W_{\text{head}}'\,\mathbf{s}_D \in \mathbb{R}^{L}
\end{aligned}
$$

Each profile is a single token, so attention here mixes the embedded representation rather than a long token sequence; depth and width set the capacity. Training minimizes a reconstruction loss between $\mathbf{p}$ and $\hat{\mathbf{p}}$ (`AutoencoderLossConfig`, default MSE).

---

## When to use

When dependencies span the full elevation axis and the extra parameter budget over [[Conv1D Autoencoder]] and [[MLP Autoencoder]] is justified. The most expensive profile autoencoder; start with the MLP and escalate only if reconstruction quality demands it.
