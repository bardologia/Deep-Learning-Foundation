---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - ViTImageAutoencoder
  - vit_ae
family: image-autoencoder
registry_key: vit_ae
summary: Patch-based masked-autoencoder-style ViT image autoencoder modelling global structure across the SAR stack.
---

# ViT Image Autoencoder

`ViTImageAutoencoder` (`models/image_autoencoder/vit.py`) is a [[Model Zoo|image autoencoder]] in the masked-autoencoder mould: it tokenises the SAR image stack into patches and relates them with a transformer encoder and decoder, modelling global structure across the stack. The trained encoder is reused as the pretrained image front-end of the [[JEPA Profile-Embedding|JEPA]] predictor.

---

## Summary

A `PatchEmbedding` (`models/blocks.py`) tokenises the input stack into a $g_h \times g_w$ grid of $h$-dimensional tokens with a stride-$s$ convolution. Position is injected by a depthwise $3\times3$ convolution applied to the token feature map and added back (a convolutional position encoding that works at any input size), then $L$ shared `TransformerBlock` layers apply pre-norm self-attention across all tokens. A final `LayerNorm` and a $1\times1$ convolution map the token grid to the spatial embedding of $d$ channels (size $g_h \times g_w$). The decoder lifts $d$ back to $h$, re-applies the convolutional position encoding and $L$ transformer layers, and a `ConvTranspose2d` of kernel/stride $s$ un-patchifies the tokens to the reconstructed stack at full resolution. The embedding is passed through `embedding_norm` (default `none`) via the shared `EmbeddingNorm` mixin (`models/image_autoencoder/base.py`). The decoder infers $g_h, g_w$ from the embedding, so the spatial size flows through automatically.

---

## Symbols

| $symbol$ | meaning |
|---|---|
| $\mathbf{X}$ | input image stack, $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ |
| $C_{\text{in}}$ | input channels (`in_channels`, default 1) |
| $s$ | patch size / tokeniser stride (`patch_size`, default 8) |
| $g_h, g_w$ | token grid, $g_h = H/s$, $g_w = W/s$ |
| $h$ | model width (`hidden_dim`, default 192) |
| $L$ | transformer layers per side (`depth`, default 4) |
| heads | attention heads (`num_heads`, default 6) |
| $r$ | FFN ratio (`mlp_ratio`, default 4) |
| $d$ | embedding dimension (`embedding_dim`, default 24) |
| $\mathbf{Z}$ | spatial embedding, $\mathbf{Z} \in \mathbb{R}^{d \times g_h \times g_w}$ |
| $\hat{\mathbf{X}}$ | reconstructed stack |

---

## Architecture

Encoder, with $\text{PE}$ the patch embedding, $\text{CPE}$ a depthwise $3\times3$ convolutional position encoding added to the tokens, and $\text{TBlock}$ a pre-norm transformer layer:

$$
\begin{aligned}
\mathbf{T}_0 &= \text{PE}_s(\mathbf{X}) \in \mathbb{R}^{(g_h g_w) \times h} \\
\mathbf{T}_0 &\mathrel{+}= \text{CPE}(\mathbf{T}_0) \\
\mathbf{T}_L &= \text{TBlock}^{\,\circ L}(\mathbf{T}_0) \\
\mathbf{Z}   &= \text{EmbeddingNorm}\big(\text{Conv}_{1\times1}:\, h \to d\,(\text{map}(\text{LN}(\mathbf{T}_L)))\big) \in \mathbb{R}^{d \times g_h \times g_w}
\end{aligned}
$$

Decoder, lifting the embedding to tokens, re-attending, and un-patchifying with a transposed convolution of stride $s$:

$$
\begin{aligned}
\mathbf{U}_0 &= \text{Conv}_{1\times1}:\, d \to h\,(\mathbf{Z}), \quad \mathbf{U}_0 \mathrel{+}= \text{CPE}(\mathbf{U}_0) \\
\mathbf{U}_L &= \text{TBlock}^{\,\circ L}(\mathbf{U}_0) \\
\hat{\mathbf{X}} &= \text{ConvT}_{s \uparrow s}:\, h \to C_{\text{in}}\,(\text{map}(\text{LN}(\mathbf{U}_L)))
\end{aligned}
$$

`encode_features` bilinearly resizes $\mathbf{Z}$ to a requested spatial size for JEPA. Training minimizes a reconstruction loss between $\mathbf{X}$ and $\hat{\mathbf{X}}$ (default MSE).

---

## When to use

When global, long-range relationships across the SAR stack carry the signal and a convolutional receptive field is too local. Patch tokenisation with full self-attention is the most expressive — and most parameter-heavy — image front-end here; the convolutional position encoding keeps it usable across varying patch sizes, unlike a fixed positional table. Prefer the [[Conv2D Image Autoencoder]] or [[ConvNeXt2D Image Autoencoder]] when the parameter budget is tight.
