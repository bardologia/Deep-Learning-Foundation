---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
  - tomosar/head
aliases:
  - PerGaussianHead
family: output-head
head_key: per_gaussian
summary: K slot-specific PixelMLP heads, one per Gaussian component, each predicting its slot's full (amplitude, mean, sigma) triple; selectable on every backbone.
---

# Head Per-Gaussian

The `per_gaussian` head (`OutputHeadsMixin._build_per_gaussian_heads` / `_per_gaussian_forward`, `models/blocks.py`) attaches $K$ independent pixel-wise MLP heads to the backbone embedding, one per Gaussian slot, each predicting all `params_per_gaussian` parameters for its slot. It is selected with `head = "per_gaussian"` on any backbone architecture config.

---

## Summary

The backbone is untouched. The shared embedding feeds $K$ `PixelMLP` heads held in a `nn.ModuleList` (`gaussian_heads`); each head independently regresses its slot's $(a_k, \mu_k, \sigma_k)$ triple. The factorization axis is the slot: per-Gaussian shares one MLP per slot across parameter types, whereas [[Head Multihead]] shares one MLP per parameter type across slots.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{e}$ | Shared backbone embedding, $\mathbf{e} \in \mathbb{R}^{B \times F \times P_H \times P_W}$ |
| $\hat{\boldsymbol{\theta}}_k$ | Slot $k$ head output, $\in \mathbb{R}^{B \times 3 \times P_H \times P_W}$ |
| $\hat{\boldsymbol{\theta}}$ | Interleaved parameter output, $\in \mathbb{R}^{B \times 3K \times P_H \times P_W}$ |
| $F$ | Embedding channel count (`embedding_channels`, set by each backbone) |
| $H$ | Hidden channel count, $H = \max(\lfloor F / 2 \rfloor, 16)$ |
| $K$ | Number of Gaussian components, $K = \texttt{out\_channels} / \texttt{params\_per\_gaussian}$ |

---

## Structure

Each slot head is a `PixelMLP` mapping $F \to H \to \texttt{params\_per\_gaussian}$ through two $1{\times}1$ convolutions with one activation (the backbone's `activation`, or `ffn_activation` / GELU where the backbone defines no plain activation):

$$\hat{\boldsymbol{\theta}}_k = \text{Conv}_{1\times1}\big(\text{Act}(\text{Conv}_{1\times1}(\mathbf{e}))\big), \qquad k = 1, \dots, K.$$

`_per_gaussian_forward` stacks the slot outputs and reshapes to the interleaved $[a_1, \mu_1, \sigma_1, a_2, \mu_2, \sigma_2, \dots]$ layout, slot outer and parameter inner — the same channel ordering as every other head.

---

## When to use

Systematic bias across Gaussian slots: one head per component imposes slot independence, so a dominant slot cannot pull the projection statistics of the others. See [[Gaussian Slot Collapse]] context: slot-specialized heads are one of the levers against slots 3+ collapsing to zero.

---

## Interface

- Selection: `head = "per_gaussian"` on every backbone config; `backbone_head` in the entry configs; `resunet-per_gaussian` style compound keys in benchmark and tuning sweeps.
- Modules: `gaussian_heads` (`nn.ModuleList` of $K$ `PixelMLP`); parameters returned by `OutputHeadsMixin.head_parameters` and optimized as the `output_head` param group with `output_head_lr` / `output_head_wd`.
- Gaussian layout: `_resolve_gaussian_layout` requires `out_channels` divisible by `params_per_gaussian`.
- State dict: keys match the retired `unet_pergaussian` / `resunet_pergaussian` registry models, so those checkpoints load as `unet` / `resunet` with `head = "per_gaussian"`.

---

## Related Notes

- [[Head Conv]], [[Head Multihead]], [[Head Set-Prediction]] — the other output heads.
- [[Model Zoo]] — backbone families the head attaches to.
