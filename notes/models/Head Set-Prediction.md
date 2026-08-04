---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
  - tomosar/head
aliases:
  - SetPredictionHead
  - SetPredHead
family: output-head
head_key: set_pred
summary: Per-slot PixelMLP heads plus a sigmoid existence gate that blends each slot's amplitude toward a learned off level; selectable on every backbone, pairs with hungarian matching.
---

# Head Set-Prediction

The `set_pred` head (`OutputHeadsMixin._build_set_prediction_heads` / `_set_prediction_forward`, `models/blocks.py`) extends [[Head Per-Gaussian]] with an existence gate: an existence-logit `PixelMLP` emits one logit per Gaussian slot, and its sigmoid blends the regressed amplitude toward a learned per-slot off level. It is selected with `head = "set_pred"` on any backbone architecture config.

---

## Summary

The head makes slot existence an explicit, learnable decision instead of forcing the amplitude regressor to encode "off" as a regressed zero. Mean and sigma pass through ungated; only the amplitude channel is blended.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{e}$ | Shared backbone embedding, $\mathbf{e} \in \mathbb{R}^{B \times F \times P_H \times P_W}$ |
| $\ell_k$ | Existence logit of slot $k$ (per pixel) |
| $g_k$ | Sigmoid existence gate, $g_k = \sigma(\ell_k) \in [0, 1]$ |
| $a_k$ | Raw regressed amplitude of slot $k$ (normalized output space) |
| $o_k$ | Learned scalar off level of slot $k$ (`amp_off`, initialized at 0) |
| $\hat{a}_k$ | Gated amplitude emitted in the $3K$-channel output |
| $K$ | Number of Gaussian components, $K = \texttt{out\_channels} / \texttt{params\_per\_gaussian}$ |

---

## Structure

On top of the $K$ per-slot `PixelMLP` heads (`gaussian_heads`), the head builds an existence-logit `PixelMLP` (`existence_head`, $F \to H \to K$) and a learned off-level vector `amp_off` $\in \mathbb{R}^{K}$. The forward pass gates only the amplitude:

$$g_k = \sigma(\ell_k), \qquad \hat{a}_k = g_k\,a_k + (1 - g_k)\,o_k,$$

while $\mu_k$ and $\sigma_k$ pass through unchanged. The off level lives in normalized output space because normalized zero is not physical zero; training drives $o_k$ toward the normalized encoding of physical amplitude zero.

---

## When to use

Attacking [[Gaussian Slot Collapse]]: the gate decouples the slot on/off decision from amplitude regression, so switching a slot off does not require the amplitude head to regress an exact value. Pairs with hungarian param matching (`param_matching = hungarian` in the loss config), which lets slots claim ground-truth components in any order.

---

## Interface

- Selection: `head = "set_pred"` on every backbone config; `backbone_head` in the entry configs; `resunet-set_pred` style compound keys in benchmark and tuning sweeps.
- Modules: `gaussian_heads`, `existence_head`, `amp_off`; parameters returned by `OutputHeadsMixin.head_parameters` and optimized as the `output_head` param group with `output_head_lr` / `output_head_wd`.
- Gaussian layout: `_resolve_gaussian_layout` requires `out_channels` divisible by `params_per_gaussian`.
- State dict: keys match the retired `unet_setpred` / `resunet_setpred` registry models, so those checkpoints load as `unet` / `resunet` with `head = "set_pred"`.

---

## Related Notes

- [[Head Conv]], [[Head Multihead]], [[Head Per-Gaussian]] — the other output heads.
- [[Model Zoo]] — backbone families the head attaches to.
