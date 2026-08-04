---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - ResUNetBackbone
family: resunet
registry_key: resunet
summary: UNet with residual encoder/decoder blocks and stride-2 downsampling; the benchmark-winning backbone.
---

# ResUNet

`ResUNet` (`models/backbone/resunet.py`) replaces the standard `ConvBlock` in the [[UNet]] encoder with residual blocks, enabling deeper architectures and more stable gradient flow.

---

## Summary

Each encoder stage, the bottleneck, and each decoder stage use a residual block with a learned skip connection (1×1 projection when channels change, or when the block is strided) instead of the plain double-convolution block. `ResUNet` subclasses `ResUNetBackbone` (`models/backbone/resunet.py:11`) with `downsample="stride"` (the default), so downsampling is performed by stride-2 convolutions inside the residual units (encoder blocks 1+ and the bottleneck), not by `MaxPool2d`. [[UNet]] and [[UNet Skip]] pool instead; `UNetSkip` is itself a `ResUNetBackbone` instantiated with `downsample="maxpool"` (`models/backbone/resunet.py:151-153`). The upsampling and concatenative skip-connection mechanisms are otherwise identical to [[UNet]].

---

## Symbols

| $symbol$ | meaning |
|---|---|
| $\mathbf{x}$ | input feature map, $\mathbf{x} \in \mathbb{R}^{B \times C_{\text{in}} \times H \times W}$ |
| $\mathbf{u}_i$ | intermediate activations of the main branch |
| $B$ | batch size |
| $H, W$ | feature-map height and width |
| $C_{\text{in}}, C_{\text{out}}$ | input and output channel counts |
| $\text{Conv}_{3\times3}$ | $3\times3$ convolution with padding 1 (no spatial size change); the first maps $C_{\text{in}} \to C_{\text{out}}$, the second $C_{\text{out}} \to C_{\text{out}}$ |
| $\text{Conv}_{3\times3}^{(s)}$ | $3\times3$ convolution carrying block stride $s$ |
| $\text{Norm}$ | normalisation layer (`normalization`; default `"batch"`) |
| $\text{Act}$ | activation function (`activation`; default `"relu"`) |
| $\text{Proj}_s$ | identity or learned $1\times1$ projection carrying stride $s$ (defined below) |
| $\text{ResBlock}_s$ | residual block at stride $s$ |
| $K$ | number of Gaussian components |

The stride superscript $s$ denotes the spatial stride applied to a convolution; the leading $\text{Norm}\to\text{Act}$ is omitted for the first encoding unit (the `first_unit` exception), as described below.

---

## Architecture

The block uses pre-activation ordering (`ResidualConvBlock`): each convolution is preceded by normalisation and activation. For input $\mathbf{x}$ with $C_{\text{in}}$ channels and output $C_{\text{out}}$:

$$
\begin{aligned}
\mathbf{u}_1 &= \text{Act}(\text{Norm}(\mathbf{x})) \\
\mathbf{u}_2 &= \text{Conv}_{3\times3}^{(s)}(\mathbf{u}_1) \\
\mathbf{u}_3 &= \text{Act}(\text{Norm}(\mathbf{u}_2)) \\
\mathbf{u}_4 &= \text{Conv}_{3\times3}(\mathbf{u}_3) \\
\text{ResBlock}_s(\mathbf{x}) &= \mathbf{u}_4 + \text{Proj}_s(\mathbf{x})
\end{aligned}
$$

**Stride.** The *first* $3\times3$ convolution of the main branch carries the block stride $s$ (denoted $\text{Conv}_{3\times3}^{(s)}$); the second convolution is always stride 1. Downsampling is therefore performed *inside* the residual unit rather than by a separate pooling layer. The stride is $s=2$ for encoder blocks 1+ and the bottleneck, and $s=1$ for encoder block 0 and all decoder blocks (`models/backbone/resunet.py:30`, `:59`). The shortcut $\text{Proj}_s$ carries the same stride $s$ (see below), so the residual addition remains dimensionally consistent.

**First-unit exception.** The leading $\text{Norm}\to\text{Act}$ is omitted for the *first* encoding unit (the one operating on the raw input), controlled by the `first_unit` flag set as `first_unit = (index == 0)` for encoder blocks (`models/backbone/resunet.py:31`; bottleneck and decoder retain full pre-activation). That unit reduces to the following stepped form at stride $s=1$:

$$
\begin{aligned}
\mathbf{v}_1 &= \text{Conv}_{3\times3}(\mathbf{x}) \\
\mathbf{v}_2 &= \text{Act}(\text{Norm}(\mathbf{v}_1)) \\
\mathbf{v}_3 &= \text{Conv}_{3\times3}(\mathbf{v}_2) \\
\text{ResBlock}_{\text{first}}(\mathbf{x}) &= \mathbf{v}_3 + \text{Proj}(\mathbf{x})
\end{aligned}
$$

This matches [[ResUNet_Zhang2018_1711.10684.pdf|Zhang et al. (2018)]] Fig. 2, where the level-1 unit begins directly with a convolution. All other units (encoder blocks 1+, bottleneck, decoder) retain the full pre-activation form above.

Optional `Dropout2d` is appended after the final convolution of the main branch when `dropout > 0`.

$$
\text{Proj}_s(\mathbf{x}) = \begin{cases}
\mathbf{x} & C_{\text{in}} = C_{\text{out}} \;\text{and}\; s = 1 \\
\text{Conv}_{1\times1, C_{\text{in}} \to C_{\text{out}}}^{(s)}(\mathbf{x}) & \text{otherwise}
\end{cases}
$$

The projection is applied when input and output channels differ *or* when the block is strided ($s \neq 1$); the $1\times1$ shortcut convolution carries the same stride $s$ so that spatial halving on the main and shortcut paths agree. The residual shortcut is summed (not concatenated) with the main branch output.

---

## Output heads

The head is a config axis, not a model variant: `ResUNetConfig.head` selects the output head built by `OutputHeadsMixin._build_output_head` on top of the shared `ResUNetBackbone.encode_decode` body, all configured by `out_channels` and `params_per_gaussian` (so $K = \texttt{out\_channels} / \texttt{params\_per\_gaussian} = 6/3 = 2$ Gaussians by default):

- [[Head Conv]] (`head = "conv"`, default): a single $1\times1$ `nn.Conv2d` from `embedding_channels = features[0]` to `out_channels`, emitting the full $3K$-channel map in one projection.
- [[Head Multihead]] (`head = "multihead"`): three `PixelMLP` heads (`head_amp`, `head_mu`, `head_sigma`), each a $1\times1$–`Act`–$1\times1$ stack mapping `embedding_channels → hidden_channels → K`; `_triple_head_forward` stacks them and reshapes to $(B, 3K, H, W)$ interleaved per parameter.
- [[Head Per-Gaussian]] (`head = "per_gaussian"`): an `nn.ModuleList` of $K$ `PixelMLP` heads, each emitting the `params_per_gaussian` channels of one Gaussian; `_per_gaussian_forward` reshapes to $(B, K\cdot\texttt{params\_per\_gaussian}, H, W)$ grouped per Gaussian.
- [[Head Set-Prediction]] (`head = "set_pred"`): the per-Gaussian heads plus an existence gate that blends each slot's amplitude toward a learned off level.

Here `hidden_channels = max(embedding_channels // 2, 16)` (`OutputHeadsMixin._build_output_head`). Every head emits the same $3K = 6$ output channels in the same interleaved ordering, so heads are interchangeable downstream. The `_resolve_gaussian_layout` check requires `out_channels` divisible by `params_per_gaussian`.

---

## Design Rationale

> **Gradient flow.** Residual connections provide a direct gradient path bypassing the non-linear layers, mitigating vanishing gradients in deep encoders.

Residual connections provide a direct gradient path that bypasses the non-linear transformation layers, mitigating vanishing gradients in deep encoders. This is particularly important when training with many encoder levels (e.g., 5+) or with slow-starting warmup schedules.

---

## Parameter Reference

See [[Configuration Layer]] → `ResUNetConfig`. The configurable fields and their defaults are identical to [[UNet]] (`ResUNetConfig` mirrors `UNetConfig`).

| Parameter | Default | Description |
|---|---|---|
| `features` | `[64, 128, 256, 512]` | Encoder channel widths |
| `bottleneck_factor` | `2` | Bottleneck channel multiplier |
| `dropout` | `0.15` | Per-block dropout |
| `activation` | `"relu"` | Activation function |
| `normalization` | `"batch"` | Normalisation layer |
| `upsample_mode` | `"convtranspose"` | Upsampling mode |
| `in_channels` | `1` | Input channel count |
| `out_channels` | `6` | Output channel count ($3K$ for $K=2$ Gaussians) |

---

## Paper fidelity

**Review date:** 2026-06-04
**Reference:** Z. Zhang, Q. Liu, Y. Wang, "Road Extraction by Deep Residual U-Net," *IEEE Geoscience and Remote Sensing Letters*, 2018 (arXiv:1711.10684). Ground truth: Table I (full stage table), Fig. 1(b) (residual unit), Fig. 2 (architecture diagram), Section II-A, Eq. 1–3. [[ResUNet_Zhang2018_1711.10684.pdf|PDF]]

> **Note on downsampling fidelity.** `ResUNet` downsamples by stride-2 convolutions inside the residual units, which is **faithful to the paper**: Section II-A explicitly states "instead of using pooling operation to downsample the feature map size, a stride of 2 is applied to the first convolution block to reduce the feature map by half," and Table I assigns stride 2 to the first conv of each downsampling level. The `MaxPool2d` downsampling variant is the [[UNet Skip]] family (`ResUNetBackbone` with `downsample="maxpool"`).

### Verdict table

| # | Dimension | Paper reference | Verdict |
|---|---|---|---|
| 1 | Pre-activation BN-ReLU-Conv ordering | Fig. 1(b), Sec. II-A2, Eq. 1–3 | MATCH |
| 2 | Identity vs 1×1 projection shortcut | Fig. 1(b) "Identity Mapping"; Eq. 1 $h(\mathbf{x}_l)$ | ACCEPTED ADAPTATION |
| 3 | Stride-2 placement (first conv of downsampling unit) | Table I, Sec. II-A3 | MATCH |
| 4 | Number of levels / bridge composition | Sec. II-A3 ("7-level"), Table I, Fig. 2 | DEVIATION (structural) |
| 5 | Decoder upsampling + concat (not add) of skips | Sec. II-A3, Fig. 2 ("Up sampling" → "Concatenate") | MATCH |
| 6 | Decoder residual units | Table I (Conv 9–14), Fig. 2 | MATCH |
| 7 | Output head: 1×1 conv + sigmoid | Sec. II-A3, Table I (Conv 15), Fig. 2 | ACCEPTED ADAPTATION |
| 8 | Level-1 / stem handling (no leading BN-ReLU) | Fig. 2 (first unit: Conv-BN-ReLU-Conv) | MATCH |
| 9 | Dropout / norm placement | Sec. II-A2 (BN, ReLU, Conv); no dropout in paper | ACCEPTED ADAPTATION |
| 10 | Channel widths per level | Table I (64, 128, 256, 512) | DEVIATION (structural) |

### Prose

**Residual unit composition (MATCH).** The paper adopts He et al.'s full pre-activation unit (Fig. 1(b), Sec. II-A2): each $3\times3$ convolution is preceded by BN then ReLU, and the unit output is $\mathcal{F}(\mathbf{x}_l) + h(\mathbf{x}_l)$ per Eq. 1. `ResidualConvBlock` implements exactly `Norm → Act → Conv → Norm → Act → Conv`, summed with the shortcut. This matches the paper's intra-unit ordering. (The sole exception is the first encoding unit, which omits the leading `Norm → Act` per the `first_unit` flag; see entry #8.)

**Shortcut (ACCEPTED ADAPTATION).** The paper's Fig. 1(b) labels the shortcut "Identity Mapping" and Eq. 1 uses $h(\mathbf{x}_l)=\mathbf{x}_l$, the identity. The paper does not discuss a projection shortcut, but with channels doubling and spatial size halving at every downsampling level the identity is dimensionally impossible; a projection is the standard ResNet remedy (He et al. type-B/C). Code (`models/blocks.py:223-231`) uses a $1\times1$ conv (carrying the stride) when $C_{\text{in}}\neq C_{\text{out}}$ or stride $\neq 1$, else identity. This is the only physically realisable reading of the paper and is therefore an accepted adaptation rather than a deviation.

**Stride-2 placement (MATCH).** Table I places stride 2 on the *first* conv of each downsampling level (Conv 3 of Level 2, Conv 5 of Level 3, Conv 7 of the bridge Level 4) and stride 1 everywhere else, including all of Level 1. Code mirrors this: encoder block 0 is stride 1, encoder blocks 1+ are stride 2 (`models/backbone/resunet.py:30`), the bottleneck is stride 2 (`models/backbone/resunet.py:59`), the $1\times1$ shortcut inherits the stride (`models/blocks.py:223-231`), and the decoder units are stride 1. The code matches the paper on this dimension.

**Number of levels / bridge (DEVIATION — structural).** The paper is a strict **7-level** network (Sec. II-A3, Table I): 3 encoding levels (Level 1–3), **1 bridge level** (Level 4), 3 decoding levels (Level 5–7), 15 conv layers total. The bridge is itself a single residual unit and *is* the deepest stage (512 channels, $28\times28$); there is no separate "bottleneck" beyond it. The code instead builds **4 encoder residual units** (from the default 4-element `features`) **plus a separate bottleneck** at `features[-1] * bottleneck_factor = 1024` channels (`models/backbone/resunet.py:25-60`). With default config this yields 5 encoder-side downsampling units and a $14\times14$ deepest map versus the paper's 4 units and $28\times28$ deepest map, i.e. an extra resolution octave and an extra learned stage. The code's "encoder block + bottleneck" decomposition does not correspond to the paper's "encoding + bridge" split. *Proposed fix:* to reproduce the paper exactly, set `features=[64,128,256]` and `bottleneck_factor=2` (bridge $=512$), giving 3 encoder units + 1 bridge + 3 decoder units. The current generalised form is a deliberate project-level adaptation; flag it as a structural divergence from the published net, resolvable by configuration.

**Decoder upsampling and skips (MATCH).** Sec. II-A3 and Fig. 2 specify, before each decoding unit, an up-sampling of the lower-level features followed by **concatenation** with the corresponding encoding features. Code upsamples (`models/backbone/resunet.py:102`), then concatenates the skip on the channel dim (`models/backbone/resunet.py:104`), then applies the residual unit — the decoder block input width is $2\times$ the skip width (`models/backbone/resunet.py:75`). Concatenation (not addition) matches the paper. The paper does not name the up-sampling operator; `build_upsample` defaults to transposed conv (`models/blocks.py:79-86`) with a bilinear option, an accepted free choice. `match_spatial_size` (`models/blocks.py:129-137`) is a benign robustness aid that is a no-op at the paper's power-of-two $224$ sizes.

**Decoder residual units (MATCH).** Table I Conv 9–14 are three residual units at 256/128/64 channels, all stride 1, restoring resolution via the preceding up-sampling. Code's decoder blocks are stride-1 residual units at the mirrored widths — structurally identical.

**Output head (ACCEPTED ADAPTATION).** Paper: Conv 15 is a $1\times1$ convolution followed by a **sigmoid** activation (Fig. 2 "Conv → Sigmod → Output"; Sec. II-A3 "a $1\times1$ convolution and a sigmoid activation layer is used to project the multi-channel feature maps to the desired segmentation"). The paper's task is **binary road segmentation** trained with MSE (Eq. 2). This project is a **regression** task ($3K$ Gaussian-mixture parameters, `out_channels=6`), for which an output-saturating sigmoid would be inappropriate. The code therefore uses a bare $1\times1$ conv with no activation (`models/backbone/resunet.py:114-118`). The $1\times1$ projection itself matches; the omission of sigmoid is a justified task-driven adaptation, not a fidelity error.

**Level-1 / stem handling (MATCH).** Fig. 2 draws the very first encoding unit *without* the leading BN-ReLU: it begins directly `Conv → BN → ReLU → Conv → Addition` on the raw input (consistent with the standard observation that BN-ReLU on a raw input is undesirable). All subsequent units use the full `BN → ReLU → Conv → BN → ReLU → Conv` pre-activation. `ResidualConvBlock` takes a `first_unit` flag (`models/blocks.py:189`, default `False`) that drops the leading `Norm`/`Act` when set; in `stride` mode the `ResUNetBackbone` constructor passes `first_unit = (index == 0)` for the encoder blocks only (`models/backbone/resunet.py:31`, bottleneck and decoder retain full pre-activation). Encoder block 0 therefore begins directly with the $3\times3$ convolution, matching Fig. 2. In `maxpool` mode (the [[UNet Skip]] family) `first_unit` is `False` for every block (`models/backbone/resunet.py:34`), so all units carry the leading `Norm → Act`.

**Dropout / norm (ACCEPTED ADAPTATION).** The paper uses BN only, no dropout. The code optionally appends `Dropout2d` after the main branch (`models/blocks.py:219`, default `0.15`) and exposes `instance`/`group`/`none` norm choices. These are standard regularisation/normalisation knobs left at defaults; out of scope as hyperparameters and dimensionally harmless.

**Channel widths (DEVIATION — structural, coupled to #4).** Table I fixes widths at 64/128/256/512. The default `features=[64,128,256,512]` plus a 1024 bottleneck overshoots the paper's deepest width by one octave, a direct consequence of the extra level in #4. Fixed by the same `features=[64,128,256]` adjustment.

### Summary

The intra-unit mechanics — full pre-activation BN-ReLU-Conv, stride-2 on the first conv of each downsampling unit, projection-with-stride shortcut, concatenative skips, $1\times1$ output projection — are **faithful** to Zhang et al. (2018) against Section II-A and Table I. The remaining divergences are (i) a **structural** level-count mismatch: the code's "4 encoder blocks + separate bottleneck" yields 5 deepening stages and a $14\times14$ deepest map, whereas the paper is a 7-level net with 3 encoding levels + a single bridge at $28\times28$ — reconcilable by setting `features=[64,128,256]`; and (ii) the justified **regression** adaptations (no sigmoid, optional dropout). The stem handling matches Fig. 2 via the `first_unit` flag. Overall: **faithful residual unit, generalised depth, two task-driven adaptations.**

---

## Related Notes

- [[UNet]] — Base architecture
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — ResUNetConfig
- [[DLR-TomoSAR Index]] — Master index
