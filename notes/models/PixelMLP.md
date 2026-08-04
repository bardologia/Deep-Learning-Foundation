---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - PixelMLPNet
  - Pixel-wise MLP
family: pixel_mlp
registry_key: pixel_mlp
summary: Pixel-wise MLP control baseline realised as a stack of 1×1 convolutions; zero spatial context by construction.
---

# PixelMLP

`PixelMLPNet` (`models/backbone/pixel_baselines.py`) is the no-spatial-context control of the zoo: a per-pixel multilayer perceptron realised as a stack of $1\times1$ convolutions. Every output pixel is a function of exactly one input pixel's channel vector — the receptive field is $1\times1$ regardless of depth — so the model can express arbitrary per-pixel channel mappings but no spatial reasoning whatsoever.

---

## Summary

The trunk applies $L$ layers of ($1\times1$ conv → norm → act → optional element-wise `Dropout`), each mapping the per-pixel channel vector to the next hidden width in `features`; a final $1\times1$ `output_head` emits the flat `out_channels` $= 3K$ Gaussian-parameter stack. Classical tomographic inversion — beamforming, Capon, and compressive-sensing spectral estimation — operates strictly per pixel on the covariance signature, so this baseline is the learned analogue of that regime: whatever it achieves is attributable to per-pixel mapping capacity alone, and any margin the spatial backbones ([[UNet]] and the rest of the [[Model Zoo]]) hold over it is the measured value of spatial context for this task.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{x}_p$ | Input channel vector at pixel $p$ (length $C_{\text{in}}$) |
| $\mathbf{h}^{(l)}_p$ | Hidden representation of pixel $p$ after layer $l$ (width `features[l]`) |
| $\mathbf{W}^{(l)}$ | Weight matrix of layer $l$, applied as a $1\times1$ convolution |
| $\mathbf{y}_p$ | Output parameter vector at pixel $p$ (length $3K$) |
| $L$ | Number of trunk layers, $L = \lvert\texttt{features}\rvert$ |
| $\text{norm}$ | Normalisation |
| $\text{act}$ | Activation |

---

## Architecture

$$
\begin{aligned}
\mathbf{h}^{(0)}_p &= \mathbf{x}_p \\
\mathbf{h}^{(l)}_p &= \text{act}\!\left(\text{norm}\!\left(\mathbf{W}^{(l)} \mathbf{h}^{(l-1)}_p\right)\right), \qquad l = 1, \dots, L \\
\mathbf{y}_p &= \mathbf{W}^{\text{out}} \mathbf{h}^{(L)}_p
\end{aligned}
$$

Every operation is pointwise across the spatial grid: the same MLP is applied independently at each pixel, exactly as the multihead `PixelMLP` output heads do, but here composing the entire network.

---

## Design Rationale

**The control the benchmark cannot do without.** The premise of using spatial backbones for tomographic parameter extraction is that neighbourhood context regularises the per-pixel inversion. Without a context-free baseline the benchmark cannot separate what the encode-decode architectures contribute spatially from what any sufficiently wide per-pixel mapping would achieve on the same input channels. PixelMLP closes that gap; [[Local CNN]] then adds the smallest increment of context (a fixed local window) between this model and the full backbones.

**Capacity matching caveat.** Because every parameter is applied at every pixel at full resolution, matching this model to the UNet parameter budget inflates per-pixel FLOPs and activation memory far beyond the encode-decode models (which spend most parameters at downsampled resolutions). The [[Capacity Matching]] stage will match it faithfully; the throughput and VRAM probes of the [[Benchmark Pipeline]] make the cost visible, and `skip_models` can exclude it from strictly budget-matched sweeps where the scaled width is impractical.

---

## Parameter Reference

See [[Configuration Layer]] → `PixelMLPNetConfig` (`features` — hidden widths, one per $1\times1$ layer; `activation`, `normalization`, `dropout`).

---

## Provenance

Not derived from a reference paper: this is a purpose-built scientific control, mirroring the per-pixel structure of classical TomoSAR spectral estimators (beamforming / Capon / CS) in learned form.

---

## Related Notes

- [[Local CNN]] — The local-context-only companion control
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — PixelMLPNetConfig
- [[DLR-TomoSAR Index]] — Master index
