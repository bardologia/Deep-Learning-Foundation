---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
  - tomosar/head
aliases:
  - ConvHead
family: output-head
head_key: conv
summary: Default output head; a single convolution maps the backbone embedding directly to the packed Gaussian parameter channels.
---

# Head Conv

The `conv` head (`OutputHeadsMixin._build_conv_head`, `models/blocks.py`) is the default output head on every backbone: a single convolution mapping the backbone embedding straight to the `out_channels` packed parameter channels. It is selected with `head = "conv"` on any backbone architecture config, or `backbone_head = "conv"` in the entry configs.

---

## Summary

Every backbone ends its encode-decode path in an embedding $\mathbf{e} \in \mathbb{R}^{B \times F \times P_H \times P_W}$ and calls `OutputHeadsMixin._build_output_head`, which for `conv` attaches

$$\hat{\boldsymbol{\theta}} = W * \mathbf{e} + b, \qquad \hat{\boldsymbol{\theta}} \in \mathbb{R}^{B \times 3K \times P_H \times P_W},$$

a single convolution with kernel size `conv_head_kernel_size` (a class attribute of the mixin; $1{\times}1$ everywhere except NAFNet, which keeps its native $3{\times}3$ restoration head). All $3K$ output channels share the same linear projection of the embedding; no Gaussian-layout resolution is performed and the interleaved $[a_1, \mu_1, \sigma_1, a_2, \mu_2, \sigma_2, \dots]$ layout is produced directly by channel order.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{e}$ | Backbone embedding, $\mathbf{e} \in \mathbb{R}^{B \times F \times P_H \times P_W}$ |
| $F$ | Embedding channel count (`embedding_channels`, set by each backbone) |
| $\hat{\boldsymbol{\theta}}$ | Packed parameter output, $\in \mathbb{R}^{B \times 3K \times P_H \times P_W}$ |
| $K$ | Number of Gaussian components, $K = \texttt{out\_channels} / \texttt{params\_per\_gaussian}$ |

---

## Interface

- Selection: `head = "conv"` (default) on every backbone config; `backbone_head` in `BackboneEntryConfig`, `JepaEntryConfig`, `CrossValidationConfig`.
- Module: `output_head` (`nn.Conv2d`), parameters returned by `OutputHeadsMixin.head_parameters` and optimized as the `output_head` param group with `output_head_lr` / `output_head_wd`.
- State dict: `output_head.weight`, `output_head.bias` — identical keys and shapes to the pre-head-axis single-head models, so their checkpoints load unchanged.

---

## Related Notes

- [[Head Multihead]], [[Head Per-Gaussian]], [[Head Set-Prediction]] — the alternative output heads on the same embedding.
- [[Model Zoo]] — backbone families the head attaches to.
