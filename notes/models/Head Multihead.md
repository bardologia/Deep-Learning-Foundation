---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
  - tomosar/head
aliases:
  - MultiheadHead
  - TripleHeads
family: output-head
head_key: multihead
summary: Three parameter-type-specific PixelMLP heads (amplitude, mean, sigma) replacing the single shared conv projection; selectable on every backbone.
---

# Head Multihead

The `multihead` head (`OutputHeadsMixin._build_triple_heads` / `_triple_head_forward`, `models/blocks.py`) attaches three independent pixel-wise MLP output heads to the backbone embedding — one for amplitude, one for mean elevation, and one for sigma — rather than a single shared convolution. It is selected with `head = "multihead"` on any backbone architecture config.

---

## Summary

The backbone (encoder, bottleneck, decoder) is untouched. The shared embedding is fed to three parameter-type-specific `PixelMLP` heads that independently predict all $K$ values of their respective parameter type. The factorization axis is the parameter type: each head sees the full embedding but specializes its two-layer projection to one physical quantity.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{e}$ | Shared backbone embedding, $\mathbf{e} \in \mathbb{R}^{B \times F \times P_H \times P_W}$ |
| $\mathbf{a}, \boldsymbol{\mu}, \boldsymbol{\sigma}$ | Amplitude, mean, and sigma head outputs, each $\in \mathbb{R}^{B \times K \times P_H \times P_W}$ |
| $\hat{\boldsymbol{\theta}}$ | Interleaved parameter output, $\in \mathbb{R}^{B \times 3K \times P_H \times P_W}$ |
| $F$ | Embedding channel count (`embedding_channels`, set by each backbone) |
| $H$ | Hidden channel count, $H = \max(\lfloor F / 2 \rfloor, 16)$ |
| $K$ | Number of Gaussian components, $K = \texttt{out\_channels} / \texttt{params\_per\_gaussian}$ |
| $\text{Conv}_{1\times1}$ | Pointwise (1×1) convolution with bias |

---

## Structure

Each of the three heads (`head_amp`, `head_mu`, `head_sigma`) is a `PixelMLP`:

$$\text{PixelMLP}(\mathbf{e}) = \text{Conv}_{1\times1}\big(\text{Act}(\text{Conv}_{1\times1}(\mathbf{e}))\big), \qquad F \to H \to K.$$

The activation is the backbone's `activation` field; transformer-family backbones without one (SwinUNet, SegFormer, ConvNeXt UNet) use their `ffn_activation`, and NAFNet uses GELU (`_head_activation` overrides).

`_triple_head_forward` stacks the three head outputs along a new parameter axis and reshapes to the interleaved layout:

$$\hat{\boldsymbol{\theta}} = \text{reshape}\big(\text{stack}(\mathbf{a}, \boldsymbol{\mu}, \boldsymbol{\sigma}; \text{dim}=2),\ B \times 3K \times P_H \times P_W\big),$$

with slot index outer and parameter type inner, giving $[a_1, \mu_1, \sigma_1, a_2, \mu_2, \sigma_2, \dots]$ — identical channel ordering to the [[Head Conv]] output, so all heads are interchangeable downstream.

---

## When to use

Systematic bias across parameter types: separate heads let amplitude, mean, and spread each learn their own projection statistics instead of sharing one linear map. The per-slot complement is [[Head Per-Gaussian]].

---

## Interface

- Selection: `head = "multihead"` on every backbone config; `backbone_head` in the entry configs; `unet-multihead` style compound keys in benchmark and tuning sweeps.
- Modules: `head_amp`, `head_mu`, `head_sigma`; parameters returned by `OutputHeadsMixin.head_parameters` and optimized as the `output_head` param group with `output_head_lr` / `output_head_wd`.
- Gaussian layout: `_resolve_gaussian_layout` requires `out_channels` divisible by `params_per_gaussian`.
- State dict: keys match the retired `unet_multihead` / `resunet_multihead` registry models, so those checkpoints load as `unet` / `resunet` with `head = "multihead"`.

---

## Related Notes

- [[Head Conv]], [[Head Per-Gaussian]], [[Head Set-Prediction]] — the other output heads.
- [[Model Zoo]] — backbone families the head attaches to.
