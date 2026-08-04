---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - LinkNet
  - additive-skip encoder-decoder
family: linknet
registry_key: linknet
summary: Compact encoder-decoder with additive (not concatenative) skip connections and a ResNet-style encoder.
---

# LinkNet

`LinkNet` (`models/backbone/link_net.py`) implements a compact encoder-decoder architecture where skip connections are additive rather than concatenative ([[LinkNet_Chaurasia2017_1707.03718.pdf|Chaurasia & Culurciello, 2017]]). The decoder adds a transformed version of the encoder skip back to the decoder feature map rather than doubling the channel count via concatenation.

---

## Summary

The encoder uses ResNet-style residual blocks. The decoder inverts each encoder block via a 1×1 projection, followed by transposed convolution for upsampling and another 1×1 projection. The skip contribution is added element-wise to the decoder output at each level.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{s}_l \in \mathbb{R}^{B \times F_l \times H_l \times W_l}$ | Encoder feature map (skip) at level $l$ |
| $\mathbf{d}_{l+1}^{\uparrow} \in \mathbb{R}^{B \times F_l \times H_l \times W_l}$ | Decoder feature from the deeper level, already upsampled to level $l$ |
| $\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3$ | Intermediate bottleneck features: $1\times1$ compression, transposed-conv upsampling, $1\times1$ expansion, respectively |
| $\text{Decode}(\cdot)$ | Decoder-block output at level $l$ |
| $B$ | Batch size |
| $H_l, W_l$ | Spatial dimensions at level $l$ |
| $l$ | Decoder / encoder level index |
| $F_l$ | Encoder channel width at level $l$ (from `features`; default `[152, 312, 624, 1248]`) |
| $c$ | `decoder_bottleneck_ratio` (default `4`); each `BottleneckDecoderBlock` compresses to $\max(1, F_l // c)$ channels |
| $F_l / c$ | Intermediate bottleneck width |
| $m, n$ | Decoder-block input and output channel counts (Fig. 3 notation) |
| $k$ | `initial_kernel_size`, kernel size of the initial conv stem (default `7`) |
| $\text{Conv}_{1\times1, A \to B}$ | $1\times1$ bottleneck projection from $A$ to $B$ channels, each followed by `Norm` and `Act` |
| $\text{ConvT}_{3\times3, \cdot, s=2}$ | Transposed $3\times3$ convolution with stride 2 (and `output_padding=1`) for $2\times$ upsampling |
| $\text{BN}$ | Normalisation layer (`normalization`; default `"batch"`) |

---

## Architecture

The encoder uses ResNet-style residual stages over a convolutional stem; the decoder inverts each encoder block and adds the skip contribution element-wise.

### Decoder Block

At level $l$, with encoder skip $\mathbf{s}_l \in \mathbb{R}^{B \times F_l \times H_l \times W_l}$ and upsampled decoder input $\mathbf{d}_{l+1}^{\uparrow} \in \mathbb{R}^{B \times F_l \times H_l \times W_l}$:

$$
\begin{aligned}
\mathbf{b}_1 &= \text{Conv}_{1\times1, F_l \to F_l/4}(\mathbf{s}_l) \\
\mathbf{b}_2 &= \text{BN}\!\left(\text{ConvT}_{3\times3, F_l/4 \to F_l/4, s=2}(\mathbf{b}_1)\right) \\
\mathbf{b}_3 &= \text{Conv}_{1\times1, F_l/4 \to F_{l-1}}(\mathbf{b}_2) \\
\text{Decode}(\mathbf{s}_l, \mathbf{d}_{l+1}^{\uparrow}) &= \mathbf{b}_3 + \mathbf{d}_{l+1}^{\uparrow}
\end{aligned}
$$

The `+` denotes element-wise addition of the skip-derived features to the decoder features (requires matching spatial size, enforced by `match_spatial_size`, and matching channels).

The block diagram above uses $c = 4$ for compactness; the general compression ratio is the configurable `decoder_bottleneck_ratio`. An initial $k \times k$ convolution stem (`initial_kernel_size`, default $k = 7$) maps the `in_channels` input to `features[0]` channels before the first encoder stage. Each encoder stage (`ResidualEncoderBlock`) downsamples by a stride-2 convolution with a parallel stride-2 1×1 residual shortcut.

---

## Design Rationale

> **Additive vs. concatenative skips.** Concatenation doubles the decoder's input channel count at each level; additive skips preserve the channel count, yielding a significantly lighter decoder.

Concatenation doubles the decoder's input channel count at each level, increasing parameters. Additive skips preserve the channel count, yielding a significantly lighter decoder. This is motivated by the observation (LinkNet paper) that low-level encoder features primarily need to add residual spatial detail to the decoder, not replace its content.

---

## Parameter Reference

See [[Configuration Layer]] → `LinkNetConfig`.

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| `features` | $F_l$ | `[152, 312, 624, 1248]` | Encoder channel widths per stage |
| `initial_kernel_size` | $k$ | `7` | Kernel size of the initial conv stem |
| `decoder_bottleneck_ratio` | $c$ | `4` | Compression ratio inside each decoder block |
| `dropout` | — | `0.15` | Encoder block dropout |
| `activation` | — | `"relu"` | Activation function |
| `normalization` | — | `"batch"` | Normalisation layer |
| `conv_bias` | — | `False` | Bias flag for all conv / transposed-conv layers |
| `init_mode` | — | `"default"` | Weight-initialisation mode passed to `initialize_weights` |
| `in_channels` | — | `1` | Input channel count |
| `out_channels` | — | `6` | Output channel count ($3K$ for $K=2$ Gaussians) |
| `params_per_gaussian` | — | `3` | Parameters per Gaussian component; $\text{out\_channels} = K \cdot \text{params\_per\_gaussian}$ |

`LinkNetConfig` has no `bottleneck_factor` or `upsample_mode` field; upsampling is always via `nn.ConvTranspose2d` inside the decoder blocks. The config also carries per-group optimiser hyperparameters (`encoder_lr`/`decoder_lr`/`output_head_lr`, `encoder_wd`/`decoder_wd`/`output_head_wd`) consumed by `get_param_groups`, which assigns distinct learning rates and weight decays to the `initial_conv`, `encoder_stages`, `decoder_stages`, and `output_head` parameter groups.

---

## Relation to the LinkNet paper

**Reference:** Chaurasia, A. & Culurciello, E. (2017). *LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation.* arXiv:1707.03718. (Fig. 1 network, Fig. 2 encoder module, Fig. 3 decoder module, Tables I–II block specs.) [[LinkNet_Chaurasia2017_1707.03718.pdf|PDF]]
**Code:** `models/backbone/link_net.py`, `models/blocks.py` (`build_activation`, `build_norm2d`, `match_spatial_size`), `configuration/architectures/backbone.py` (`LinkNetConfig`).

The signature LinkNet idea — additive (not concatenative) skips, added before each decoder block input exactly as in Fig. 1 — is the contribution the architecture is named for and is reproduced here. The stem ($k\times k$ stride-1, no max-pool), the single-residual-block encoder depth, and the generic-decoder-block-plus-$1\times1$-head final stage differ from the paper's ResNet-18 encoder, fitting the single-channel dense-regression task that has no ImageNet-pretrained-weight pathway.

### Comparison table

| # | Dimension | Paper | Code |
|---|---|---|---|
| 1 | Initial stem | $7\times7$ conv stride 2 $+$ $3\times3$ max-pool stride 2 ($/4$); Fig. 1, Sec. III | $k\times k$ conv stride 1, no max-pool ($/1$); `LinkNet.py:124-135` |
| 2 | Encoder block depth | Two ResNet-18 basic blocks per stage (stride-2 projection block $+$ stride-1 identity block); Fig. 2, Sec. III | One `ResidualEncoderBlock` per stage, no identity sub-block; `LinkNet.py:9-57,137-150` |
| 3 | Conv–Norm–Act ordering | Post-activation: conv $\to$ BN $\to$ ReLU; Sec. III | Same; `LinkNet.py:20-39`, `build_norm2d`/`build_activation` `models/blocks.py:24-49` |
| 4 | Decoder block layout | $1\times1\,(m\!\to\!m/4)\to$ full-conv $3\times3$ $/2\,(m/4\!\to\!m/4)\to1\times1\,(m/4\!\to\!n)$, BN$+$ReLU each; Fig. 3, Table II | Same sequence, compression ratio $c$ (default 4) via `max(1, m//c)`; `LinkNet.py:60-103` |
| 5 | Skip linking | $\text{input}(\text{dec}_i)=\text{out}(\text{dec}_{i+1})+\text{out}(\text{enc}_i)$, add before decoder input; Fig. 1 | Same add point; deepest decoder fed directly by deepest encoder, no add; `LinkNet.py:192-202` |
| 6 | Final block | full-conv $3\times3$ $/2\,(64\!\to\!32)\to$ conv $3\times3\,(32\!\to\!32)\to$ full-conv $2\times2$ $/2\,(32\!\to\!N)$; Fig. 1 | Generic bottleneck decoder block $(F_0\!\to\!F_0)$ $+$ $1\times1$ head $(F_0\!\to\!C)$; `LinkNet.py:164-179` |
| 7 | ReLU around additions | ReLU after encoder residual add (Fig. 2); no ReLU on decoder skip circles (Fig. 1) | ReLU after encoder add (`LinkNet.py:56`); decoder add not followed by ReLU |
| 8 | Output resolution recovery | Recover from stem's $/4$ via decoder up-samples $+$ final block's two $\times2$; Sec. III | Stem at $/1$, encoder to $/16$, four stride-2 ConvT recover $/16\!\to\!/1$; bilinear `match_spatial_size`; `models/blocks.py:102-110`, `LinkNet.py:204` |
| 9 | Up-sampling operator | Full-convolution (transposed conv); Sec. III–IV | `nn.ConvTranspose2d` $3\times3$, stride 2, `output_padding=1`; `LinkNet.py:81-89` |
| 10 | Encoder channel widths | $64,128,256,512$ (Table I) | `features` hyperparameter, default $[152,312,624,1248]$; $[64,128,256,512]$ available as a tunable preset |

### Notes

The `BottleneckDecoderBlock` (`LinkNet.py:60-103`) follows Fig. 3: a $1\times1$ compression to $m/c$, a full $3\times3$ transposed convolution at stride 2, and a $1\times1$ expansion to $n$, each followed by normalisation and activation. The additive skip wiring in the forward pass (`LinkNet.py:192-202`) follows Fig. 1 — the encoder output at level $i$ is summed onto the decoder output coming down from level $i+1$, and that sum is the input to decoder block $i$, with the deepest decoder block fed directly by the deepest encoder block. The post-add ReLU sits inside the encoder block (`LinkNet.py:56`), consistent with Fig. 2; the decoder add carries no ReLU.

The stem (`LinkNet.py:124-135`) is a stride-1 convolution with no max-pool, so it performs no spatial reduction; the encoder reaches $/16$ through four stride-2 stages and the four stride-2 ConvT decoder blocks recover full resolution, with a bilinear `match_spatial_size` (`models/blocks.py:102-110`, called at `LinkNet.py:204`) enforcing the output spatial size against the original input. Each encoder stage is a single `ResidualEncoderBlock` (`LinkNet.py:9-57,137-150`) rather than the paper's pair of ResNet-18 basic blocks. The final stage appends one extra bottleneck decoder block ($F_0\!\to\!F_0$, `LinkNet.py:164-179`) feeding the $1\times1$ output head, in place of the paper's dedicated three-layer segmentation head.

---

## Related Notes

- [[UNet]] — Concatenative skip alternative
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — LinkNetConfig
- [[DLR-TomoSAR Index]] — Master index
