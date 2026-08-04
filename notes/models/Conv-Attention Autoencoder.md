---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - CnnAttnAutoencoder
  - cnn_attn_ae
family: profile-autoencoder
registry_key: cnn_attn_ae
summary: Convolutional tokeniser plus multi-token self-attention; most expressive profile autoencoder.
---

# Conv-Attention Autoencoder

`CnnAttnAutoencoder` (`models/profile_autoencoder/cnn_attn.py`) is a [[Model Zoo|profile autoencoder]] that combines a convolutional tokeniser with genuine multi-token self-attention. Unlike the [[Transformer1D Autoencoder]], which embeds the whole profile as a single token, this model splits the profile into patches and lets attention relate the patches to one another — local convolution plus global attention. It is the most expressive profile model. Like the others it defines the output latent space later targeted by [[JEPA Profile-Embedding|JEPA]].

---

## Summary

Each pixel's profile $\mathbf{p} \in \mathbb{R}^{L}$ is reshaped to a single-channel 1D sequence via `ProfileAutoencoderBlocks.to_sequence`. A convolutional stem lifts it to $c$ channels, then a strided convolution of stride $s$ tokenises it into $T = \lfloor L/s \rfloor$ tokens of width $h$. A learnable positional embedding is added and $n$ shared `TransformerBlock` layers (`models/blocks.py`) apply pre-norm self-attention across the $T$ tokens. Mean pooling over tokens followed by a linear head yields the embedding $d$, L2-normalized by the shared `EmbeddingNorm` mixin (`models/profile_autoencoder/base.py`). The decoder lifts $d$ to $h$, broadcasts it to $T$ tokens (plus positional embedding), applies $n$ transformer layers, maps each token to an $s$-sample patch, and a final linear layer assembles the flattened patches into the reconstructed profile of length $L$.

---

## Symbols

| $symbol$ | meaning |
|---|---|
| $\mathbf{p}$ | input profile, $\mathbf{p} \in \mathbb{R}^{L}$ per pixel |
| $L$ | profile length (`profile_length`, default 256) |
| $c$ | stem convolution channels (`seq_channels`, default 32) |
| $k$ | stem kernel size (`seq_kernel_size`, default 5) |
| $s$ | patch size / tokeniser stride (`patch_size`, default 8) |
| $T$ | token count, $T = \lfloor L/s \rfloor$ (default 32) |
| $h$ | token / model width (`hidden_dim`, default 128) |
| $n$ | transformer layers per side (`depth`, default 2) |
| heads | attention heads (`num_heads`, default 4) |
| $d$ | embedding dimension (`embedding_dim`, default 24) |
| $\mathbf{z}$ | latent embedding, $\mathbf{z} \in \mathbb{R}^{d}$ |
| $\hat{\mathbf{p}}$ | reconstructed profile |

---

## Architecture

Encoder, with $\text{Conv}^{1\text{D}}_{s\downarrow s}$ a stride-$s$ tokeniser, $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{T \times h}$ the positional table, and $\text{TBlock}$ a pre-norm transformer layer:

$$
\begin{aligned}
\mathbf{f}      &= \text{Act}(\text{Conv}^{1\text{D}}_{k}:\, 1 \to c\,(\mathbf{p})) \\
\mathbf{T}_0    &= \text{Conv}^{1\text{D}}_{s \downarrow s}:\, c \to h\,(\mathbf{f})^{\top} + \mathbf{E}_{\text{pos}} \in \mathbb{R}^{T \times h} \\
\mathbf{T}_n    &= \text{TBlock}^{\,\circ n}(\mathbf{T}_0) \\
\mathbf{z}      &= \text{EmbeddingNorm}\Big(W_{\text{head}}\,\tfrac{1}{T}\textstyle\sum_{t} \mathbf{T}_n[t]\Big)
\end{aligned}
$$

Decoder, broadcasting the lifted embedding to $T$ tokens and reassembling $s$-sample patches:

$$
\begin{aligned}
\mathbf{U}_0     &= \mathbf{1}_T \otimes (W_{\text{lift}}\,\mathbf{z}) + \mathbf{E}'_{\text{pos}} \\
\mathbf{U}_n     &= \text{TBlock}^{\,\circ n}(\mathbf{U}_0) \\
\hat{\mathbf{p}} &= W_{\text{out}}\,\text{flatten}\big(W_{\text{patch}}\,\mathbf{U}_n\big), \quad W_{\text{patch}} \in \mathbb{R}^{s \times h}, \; W_{\text{out}} \in \mathbb{R}^{L \times (T s)}
\end{aligned}
$$

The final linear layer maps the $T \cdot s$ assembled samples to exactly $L$ outputs, so any $L$ is reconstructed regardless of divisibility by $s$. Training minimizes a reconstruction loss between $\mathbf{p}$ and $\hat{\mathbf{p}}$ (default MSE).

---

## When to use

When interactions between distant parts of the profile matter and a single-token embedding is too coarse. The convolutional tokeniser injects locality before attention models patch-to-patch relations, the most expressive — and most parameter-heavy — profile option, a step up from the degenerate single-token [[Transformer1D Autoencoder]].
