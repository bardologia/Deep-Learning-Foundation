---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - AttentionUNet
  - Attention Gate
family: attention
registry_key: attention_unet
summary: UNet with attention-gated skip connections that suppress spatially irrelevant encoder activations.
---

# Attention UNet

`AttentionUNet` (`models/backbone/attention_unet.py`) extends the standard [[UNet]] by replacing direct skip connection concatenation with attention-gated skip connections ([[AttentionUNet_Oktay2018_1804.03999.pdf|Oktay et al., 2018]]). An attention gate learns to suppress spatially irrelevant activations in the skip features before merging them with the decoder.

---

## Summary

The encoder and bottleneck are identical to [[UNet]]. At each decoder level the order is **gate $\rightarrow$ upsample $\rightarrow$ concatenate $\rightarrow$ refine**: an `AttentionGate` module first uses the *coarse* (pre-upsample) decoder feature as the gating signal to compute per-pixel attention coefficients on the coarse grid, resamples them back to the skip resolution, multiplies them into the skip feature map, and applies an output transform; only afterwards is the decoder feature upsampled and concatenated with the gated skip.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{g} \in \mathbb{R}^{B \times G \times H_g \times W_g}$ | Gating signal: coarse decoder feature *before* upsampling |
| $\mathbf{s} \in \mathbb{R}^{B \times S \times H_s \times W_s}$ | Skip connection from encoder (fine grid) |
| $\tilde{\mathbf{s}}$ | Attention-weighted, output-transformed skip connection |
| $\mathbf{u}$ | Upsampled decoder feature, obtained from the same coarse feature $\mathbf{g}$ used as the gating signal |
| $\mathbf{d}_l$ | Refined decoder feature at level $l$ |
| $\psi_{\mathbf{g}}, \psi_{\mathbf{s}}$ | Gate and skip projections into the shared $D$-channel space on the coarse grid |
| $\mathbf{a}$ | Additive-attention activation $\text{ReLU}(\psi_{\mathbf{g}} + \psi_{\mathbf{s}})$ |
| $q$ | Single-channel attention score on the coarse grid |
| $\alpha_{\text{coarse}}$ | Attention coefficient map on the coarse $H_g \times W_g$ grid |
| $\alpha \in [0,1]^{B \times 1 \times H_s \times W_s}$ | Per-pixel attention coefficient after bilinear resampling to the skip grid |
| $B$ | Batch size |
| $G, S$ | Gate and skip channel counts |
| $H_g, W_g$ | Coarse (gate) spatial dimensions |
| $H_s, W_s$ | Fine (skip) spatial dimensions, $H_s = 2 H_g$, $W_s = 2 W_g$ |
| $D$ | Intermediate attention channels, $D = \max(1, \lfloor S \cdot r \rfloor)$ |
| $r$ | `attention_intermediate_ratio` (default: `0.5`) |
| $\text{Conv}_{A \to B,\, k\times k}$ | Convolution from $A$ to $B$ channels with kernel $k\times k$ |
| $\text{Norm}$ | Normalisation layer |
| $\sigma$ | Sigmoid activation |
| $\text{Upsample}, \text{Up}(\cdot)$ | Bilinear or transposed-convolution upsampling |
| $\text{cat}$ | Channel-wise concatenation |
| $\text{ConvBlock}$ | Decoder refinement convolution block |
| $W_g$ | Gate projection: $1\times1$ conv $G \to D$ (+ norm), preserves grid |
| $W_x$ | Skip projection: kernel-$2$ stride-$2$ conv $S \to D$ (+ norm), downsamples $\mathbf{s}$ to the coarse gate grid (fusing the paper's $1\times1$ transform and the explicit downsampling into one operation) |
| $W$ | Output transform: $1\times1$ conv $S \to S$ (+ norm) on the gated skip |

---

## Architecture

This is the additive grid-attention gate of Oktay et al. (2018). The gating signal is the **coarse, pre-upsample** decoder feature $\mathbf{g} \in \mathbb{R}^{B \times G \times H_g \times W_g}$ (the deeper-level feature *before* it is upsampled), and the skip connection is $\mathbf{s} \in \mathbb{R}^{B \times S \times H_s \times W_s}$ with $H_s = 2 H_g$, $W_s = 2 W_g$. Attention is computed on the *coarse* $H_g \times W_g$ grid and the resulting coefficient map is then resampled back to the skip resolution.

The gate and the stride-$2$ downsampled skip are projected into a shared $D$-channel space at the coarse grid:

$$
\psi_{\mathbf{g}} = \text{Norm}(\text{Conv}_{G \to D,\, 1\times1}(\mathbf{g})), \qquad
\psi_{\mathbf{s}} = \text{Norm}(\text{Conv}_{S \to D,\, 2\times2,\, \text{stride } 2}(\mathbf{s}))
$$

so that both $\psi_{\mathbf{g}}$ and $\psi_{\mathbf{s}}$ live on the $H_g \times W_g$ grid ($\psi_{\mathbf{g}}$ is bilinearly resampled to $\psi_{\mathbf{s}}$'s size only if a residual size mismatch remains). Additive attention with $\text{ReLU}$ and a $1\times1$ scoring convolution $\psi$ followed by $\sigma$ yields a coarse coefficient map:

$$
\begin{aligned}
\mathbf{a} &= \text{ReLU}(\psi_{\mathbf{g}} + \psi_{\mathbf{s}}) \\
q &= \text{Conv}_{D \to 1,\, 1\times1}(\mathbf{a}) \\
\alpha_{\text{coarse}} &= \sigma(q)
\end{aligned}
$$

The coarse map is **bilinearly resampled up** to the skip resolution, multiplied into the skip, and passed through the output transform $W$:

$$
\alpha = \text{Upsample}_{H_g \times W_g \,\to\, H_s \times W_s}^{\text{bilinear}}(\alpha_{\text{coarse}}), \qquad
\tilde{\mathbf{s}} = W\!\left(\alpha \cdot \mathbf{s}\right)
$$

The decoder then **upsamples the coarse feature** and concatenates it with the gated skip (gate $\rightarrow$ upsample $\rightarrow$ concatenate $\rightarrow$ refine):

$$
\mathbf{u} = \text{Upsample}(\mathbf{g}), \qquad
\mathbf{d}_l = \text{ConvBlock}_{2S \to S}(\text{cat}[\tilde{\mathbf{s}}, \mathbf{u}])
$$

---

## Design Rationale

> **Why attention gates on skip connections.** In standard U-Net, all skip features are concatenated equally regardless of relevance; attention gates let the decoder's coarse-scale context down-weight skip features in low-relevance regions.

For TomoSAR, different spatial regions have different scattering complexity (e.g., layover vs. single-reflection areas). Attention gates allow the model to down-weight skip features in regions where the decoder's coarser-scale context (gate signal) predicts low relevance, focusing representation capacity on the most informative regions.

**Versus self-attention.** Attention gates cost $O(H \cdot W)$ compute per level (from the gate/skip projections, the $1\times1$ scoring convolution, and the $1\times1$ output transform), considerably cheaper than full self-attention which would require $O((H \cdot W)^2)$ memory.

---

## Parameter Reference

See [[Configuration Layer]] → `AttentionUNetConfig`.

Additional parameter:

| Parameter | Default | Description |
|---|---|---|
| `attention_intermediate_ratio` | `0.5` | Ratio $r$ controlling attention bottleneck width $D = \lfloor S \cdot r \rfloor$ |

All other parameters are identical to [[UNet]].

---

## Paper fidelity

**Review date.** 2026-06-04.

**Reference.** Oktay, O., Schlemper, J., Le Folgoc, L., Lee, M., Heinrich, M., Misawa, K., Mori, K., McDonagh, S., Hammerla, N. Y., Kainz, B., Glocker, B., Rueckert, D. (2018). *Attention U-Net: Learning Where to Look for the Pancreas*. MIDL 2018. arXiv:1804.03999. [[AttentionUNet_Oktay2018_1804.03999.pdf|PDF]]

This section records an equation-by-equation, figure-by-figure verification of `AttentionUNet` (`models/backbone/attention_unet.py`, shared blocks and builders in `models/blocks.py`) against the paper. The paper PDF is treated as ground truth. The reference architecture is 3D (volumetric CT); the implementation is the 2D adaptation for TomoSAR tomograms, so any pure $3D \rightarrow 2D$ change (e.g. trilinear $\rightarrow$ bilinear resampling, $\text{Conv3d} \rightarrow \text{Conv2d}$) is an accepted adaptation rather than a deviation.

The gate follows the paper's coarse-grid (grid-attention) formulation: (i) the gating signal $g$ is the pre-upsample decoder feature, (ii) the skip is projected with a kernel-2 stride-2 convolution ($W_x$) so attention is computed on the coarse $H_g \times W_g$ grid as in Fig. 2 and Section 3.2 ("input feature-maps are downsampled to the resolution of gating signal"), (iii) $\alpha$ is bilinearly resampled back to the skip grid (the Resampler block of Fig. 2), and (iv) an output transform $W$ ($1\times1$ conv + norm) is applied after the multiplicative gating.

### Component-verdict table

| # | Component | Paper reference | Code reference | Verdict |
|---|---|---|---|---|
| 1 | Backbone encoder/decoder (double $3\times3$ conv + ReLU per scale, factor-2 rescaling) | Fig. 1; Sec. 2 | `ConvBlock` `models/blocks.py:140-176`; `models/backbone/attention_unet.py:84-151` | MATCH |
| 2 | Gate signal $g$ = coarse decoder feature (pre-upsample) | Fig. 1, Fig. 2; Sec. 3.2 | `models/backbone/attention_unet.py:171` (gate called before `upsample`) | MATCH |
| 3 | $W_g$ ($1\times1$ conv, to $F_{int}$) on $g$ | Eq. 2; Fig. 2 | `models/backbone/attention_unet.py:19-27` | MATCH |
| 3 | $W_x$ ($1\times1$ conv on $x^l$, feature-maps downsampled to gate grid) | Eq. 2; Sec. 3.2; Fig. 2 | `models/backbone/attention_unet.py:28-37` (kernel-2 stride-2 conv) | ACCEPTED ADAPTATION |
| 4a | $\sigma_1$ = ReLU on $W_x^T x + W_g^T g$ | Eq. 1 | `models/backbone/attention_unet.py:56, 64` | MATCH |
| 4b | $\psi$ ($1\times1$ conv to single channel) | Eq. 1; Fig. 2 | `models/backbone/attention_unet.py:38-44` | MATCH |
| 4c | $\sigma_2$ = sigmoid $\rightarrow \alpha \in [0,1]$ | Eq. 2; Sec. 3.2 | `models/backbone/attention_unet.py:45` | MATCH |
| 5 | Resampling of $\alpha$ back to $x^l$ grid | Fig. 2 (Resampler, trilinear) | `models/backbone/attention_unet.py:65` (bilinear) | ACCEPTED ADAPTATION |
| 6 | Output transform $W$ ($1\times1$ conv + norm) on gated skip | Fig. 2 (right recombination block) | `models/backbone/attention_unet.py:47-55, 67` | MATCH |
| 7 | Gated skip enters decoder via concatenation before refinement conv | Fig. 1 (concatenation); Sec. 3.2 | `models/backbone/attention_unet.py:171-175` | MATCH |
| 8 | Downsampling (max-pool / 2), upsampling (/2) of backbone | Fig. 1 legend | `models/backbone/attention_unet.py:98`, `build_upsample` `models/blocks.py:79-101` | MATCH |
| 9 | Deep supervision | Sec. 3 (Implementation Details) | not implemented | DEVIATION (minor) |
| 10 | Multiplicative gating $\hat{x}_{i,c}^l = x_{i,c}^l \cdot \alpha_i^l$ | Sec. 3.2 | `models/backbone/attention_unet.py:67` | MATCH |

### Additive-attention equations

The paper defines the gate as

$$
\begin{aligned}
z_i^l &= W_x^T x_i^l + W_g^T g_i + b_g \\
a_i^l &= \sigma_1(z_i^l) \\
q_{att}^l &= \psi^T a_i^l + b_\psi \\
\alpha_i^l &= \sigma_2(q_{att}^l)
\end{aligned}
$$

with $\sigma_1 = \text{ReLU}$ and $\sigma_2 = \text{sigmoid}$. The code realises this exactly: `gate_projection` and `skip_projection` provide $W_g, W_x$, their sum passes through `relu` ($\sigma_1$), `attention_score` applies $\psi$ ($1\times1$ conv to one channel) followed by `Sigmoid` ($\sigma_2$). The bias terms $b_g, b_\psi$ are governed by `conv_bias`: the paper carries $b_g, b_\psi$, whereas here the projections are bias-free by default and only $\psi$ carries a bias (`bias=True`, `models/backbone/attention_unet.py:43`). With normalization layers after each projection this is the standard and behaviourally equivalent choice; not a deviation.

### Accepted adaptations

- **$3D \rightarrow 2D$.** All convolutions and the resampler are 2D; the paper's trilinear coefficient resampling becomes bilinear (`models/backbone/attention_unet.py:65`). This is the intended adaptation for 2D tomogram inputs and changes no architectural semantics.
- **$W_x$ as a strided convolution.** The paper performs the linear transforms with $1\times1$ convolutions and *separately* downsamples the input feature-maps to the gating resolution (Sec. 3.2, "without any spatial support ... input feature-maps are downsampled to the resolution of gating signal"). The code fuses these two steps into a single kernel-2 stride-2 convolution (`models/backbone/attention_unet.py:28-37`), which matches the official `Attention-Gated-Networks` reference implementation. The resulting attention grid is identical ($H_g \times W_g$); accepted.
- **Intermediate-channel count $F_{int}$.** Driven by `attention_intermediate_ratio` ($D = \max(1, \lfloor S \cdot r \rfloor)$, default $r = 0.5$). The paper does not fix $F_{int}$ analytically, so this is a free design choice, not a deviation.

### Deviations

- **Deep supervision (minor).** The paper's Implementation Details (Sec. 3) state "All models are trained using ... deep-supervision [16]", and Sec. 3.2 reiterates that deep supervision forces intermediate feature-maps to be semantically discriminative at each scale. The current model emits a single prediction from the finest decoder level (`models/backbone/attention_unet.py:147-151, 178`); there are no auxiliary heads at coarser decoder scales. This is a *training-objective* feature rather than a core attention-gate property, so severity is minor. Proposed fix: add optional auxiliary $1\times1$ output heads on the intermediate decoder feature-maps (one per decoder level), upsample each to the input resolution, and combine their losses with the main head (guarded by a config flag such as `deep_supervision: bool = False`). The attention-gate mechanism itself is fully faithful without it.

### Overall

The attention gate matches the paper's additive grid-attention formulation (Eq. 1-2, Fig. 2): coarse-grid gating signal, $W_g/W_x$ projections to the gate grid, $\text{ReLU} \rightarrow \psi \rightarrow \text{sigmoid}$, bilinear resampling of $\alpha$, multiplicative gating, and the $W$ output transform are all present and correctly ordered. The backbone, downsampling/upsampling, and concatenation placement are faithful. The only genuine deviation is the absence of deep supervision (minor, training-time).

---

## Related Notes

- [[UNet]] — Base architecture
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — AttentionUNetConfig
- [[Training Pipeline]] — Training context
