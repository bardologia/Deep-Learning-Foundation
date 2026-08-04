---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - UNetBackbone
family: unet
registry_key: unet
summary: Baseline encoder-decoder with concatenative skip connections; reference architecture for all other backbones.
---

# UNet

`UNet` (`models/backbone/unet.py`) is the baseline encoder-decoder architecture implementing the original U-Net design ([[UNet_Ronneberger2015_1505.04597.pdf|Ronneberger et al., 2015]]) adapted for dense regression on SAR data patches.

---

## Summary

UNet employs a symmetric encoder-decoder structure with skip connections that bypass each encoder level to the corresponding decoder level. The encoder progressively doubles the number of feature channels while halving spatial resolution via max-pooling; the decoder inverts this process via upsampling and concatenation with the corresponding skip connection.

> **Skip connections**
> Each encoder level forwards its pre-pooling feature map $\mathbf{s}_l$ directly to the matching decoder level, restoring spatial detail lost during downsampling.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | Input feature map to a block ($\mathbf{x}_0$ is the network input) |
| $\mathbf{x}_l$ | Pooled output of encoder level $l$, shape $(B, F_l, H/2^{l}, W/2^{l})$ |
| $\mathbf{u}_i$ | Intermediate activations within a block |
| $\mathbf{f}$ | Output feature map of a block (same spatial size as $\mathbf{x}$) |
| $\mathbf{s}_l$ | Encoder skip-connection feature map at level $l$, computed before pooling, shape $(B, F_l, H/2^{l-1}, W/2^{l-1})$ |
| $\mathbf{b}$ | Bottleneck feature map, shape $(B, F_L \cdot r, H/2^L, W/2^L)$ |
| $\mathbf{u}_l$ | Upsampled decoder feature at level $l$ |
| $\mathbf{d}_l$ | Refined decoder feature at level $l$ ($\mathbf{d}_{L+1} = \mathbf{b}$, the bottleneck) |
| $\hat{\boldsymbol{\theta}}$ | Output parameter map, shape $(B, C_{\text{out}}, P_H, P_W)$ |
| $B, H, W$ | Batch size and input spatial dimensions |
| $P_H, P_W$ | Patch height and width |
| $L$ | Number of encoder levels (`len(features)`) |
| $F_l$ | Feature channels at level $l$ (from `features`; default `[64, 128, 256, 512]`); $F_0$ = `in_channels` (default `1`) |
| $r$ | Bottleneck channel multiplier (`bottleneck_factor`; default `2`) |
| $C_{\text{out}}$ | Output channels (`out_channels`; default `6`), equal to $\texttt{params\_per\_gaussian} \cdot K = 3K$ |
| $\text{Conv}_{3\times3}$ | 3×3 convolution with padding 1 (no spatial size change); the first conv in a block maps input to output channels, the second preserves them |
| $\text{Conv}_{1\times1}$ | Pointwise (1×1) convolution |
| $\text{Norm}$ | Normalisation layer (`normalization`; batch, instance, group, or none; default `"batch"`) |
| $\text{Act}$ | Activation function (`activation`; ReLU, GELU, SiLU, ELU, or LeakyReLU; default `"relu"`) |
| $\text{MaxPool}_{2\times2}$ | Non-overlapping $2\times2$ max-pooling (stride 2) |
| $\text{Upsample}$ | Transposed convolution ($2\times$, default) or bilinear + 1×1 conv, selected by `upsample_mode` |
| $\text{ConvBlock}$ | Double 3×3 convolution block (see below) |
| $\text{cat}$ | Channel concatenation, after `match_spatial_size` aligns the upsampled feature to the skip |
| $K$ | Number of Gaussian components ($K = 2$ by default) |

---

## Architecture

### ConvBlock

Each encoder and decoder stage uses a double 3×3 convolution block:

$$
\begin{aligned}
\mathbf{u}_1 &= \text{Conv}_{3\times3}(\mathbf{x}) \\
\mathbf{u}_2 &= \text{Act}(\text{Norm}(\mathbf{u}_1)) \\
\mathbf{u}_3 &= \text{Conv}_{3\times3}(\mathbf{u}_2) \\
\mathbf{f} &= \text{Act}(\text{Norm}(\mathbf{u}_3))
\end{aligned}
$$

Optional dropout (`Dropout2d`) is appended after the second activation if `dropout > 0`.

### Encoder

For each level $l \in \{1, \dots, L\}$ (where $L$ = `len(features)`):

$$
\mathbf{s}_l = \text{ConvBlock}_{F_{l-1} \to F_l}(\mathbf{x}_{l-1}), \quad \mathbf{x}_l = \text{MaxPool}_{2\times2}(\mathbf{s}_l)
$$

### Bottleneck

$$
\mathbf{b} = \text{ConvBlock}_{F_L \to F_L \cdot r}(\mathbf{x}_L)
$$

With the default `bottleneck_factor = 2`, this gives `512 * 2 = 1024` bottleneck channels.

### Decoder

For each level $l \in \{L, L-1, \dots, 1\}$:

$$
\mathbf{u}_l = \text{Upsample}(\mathbf{d}_{l+1}), \quad \mathbf{d}_l = \text{ConvBlock}_{2F_l \to F_l}(\text{cat}[\mathbf{s}_l, \mathbf{u}_l])
$$

The decoder `ConvBlock` maps the concatenated $2F_l$ channels back to $F_l$.

### Output Head

$$
\hat{\boldsymbol{\theta}} = \text{Conv}_{1\times1, F_1 \to C_{\text{out}}}(\mathbf{d}_1)
$$

The finest decoder feature $\mathbf{d}_1$ has $F_1$ = `features[0]` (default `64`) channels.

---

## Parameter Reference

See [[Configuration Layer]] → `UNetConfig`. At default `in_channels = 1`, `out_channels = 6` the model has approximately 31.0M parameters. The blocks composing the encoder, bottleneck, decoder, and head (`ConvBlock`, `Encoder`, `Decoder`) are the shared definitions in `models/blocks.py`; see [[Model Zoo]] → Shared Building Blocks.

| Parameter | Default | Description |
|---|---|---|
| `features` | `[64, 128, 256, 512]` | Encoder channel widths |
| `bottleneck_factor` | `2` | Bottleneck channel multiplier |
| `dropout` | `0.15` | Per-ConvBlock dropout |
| `activation` | `"relu"` | Activation function |
| `normalization` | `"batch"` | Normalisation layer |
| `upsample_mode` | `"convtranspose"` | Upsampling mode |
| `init_mode` | `"default"` | Weight initialisation |

Every field above (except `shape_logger_types`) is persisted per training run by `BackboneModelConfigIO` to `model_config.json` and reloaded verbatim at inference, so the checkpoint is always reconstructed against the exact widths and depth it was trained with; see [[Model Zoo]] → Configuration Persistence.

---

## Parameter Groups

The model partitions parameters into four independently-learnable groups:

| Group | Default LR | Parameters |
|---|---|---|
| `encoder` | `3e-4` | `Encoder` layers |
| `bottleneck` | `3e-4` | `ConvBlock` bottleneck |
| `decoder` | `3e-4` | `Decoder` layers |
| `output_head` | `1e-3` | Final `Conv2d` 1×1 |

---

## Paper fidelity

**Review date:** 2026-06-04

**Reference:** Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* arXiv:1505.04597 [cs.CV]. MICCAI 2015. [[UNet_Ronneberger2015_1505.04597.pdf|PDF]]

This section records an equation-by-equation, figure-by-figure verification of `models/backbone/unet.py` (with shared blocks and builders in `models/blocks.py`) against the original paper. The paper PDF is treated as ground truth. Hyperparameter choices (channel widths, depth) are out of scope except where the paper elevates a scheme to a design point.

### Component verdict table

| # | Dimension | Paper reference | Code reference | Verdict |
|---|---|---|---|---|
| 1 | Stem / input handling | Sec. 2 ("input image tile"), Fig. 1 | `UNet.forward` `unet.py:71`; `Encoder` `blocks.py:333` | MATCH |
| 2 | Encoder block (op order, kernels) | Sec. 2 (two unpadded $3\times3$ convs, each + ReLU) | `ConvBlock` `blocks.py:140-176` | ACCEPTED ADAPTATION (BatchNorm inserted) |
| 3 | Downsampling | Sec. 2 ($2\times2$ max-pool, stride 2) | `Encoder` `blocks.py:358,365-366` | MATCH |
| 4 | Bottleneck | Sec. 2 / Fig. 1 (two $3\times3$ convs $512\to1024$) | `UNetBackbone` bottleneck `unet.py:31-38`, `bottleneck_factor` `configuration/architectures/backbone.py:15` | MATCH |
| 5 | Skip topology + merge | Sec. 2 (copy + crop, concatenate); Fig. 1 "copy and crop" | `Decoder.forward` `blocks.py:402-407`; `match_spatial_size` `blocks.py:129-137` | ACCEPTED ADAPTATION (resize instead of crop) |
| 6 | Decoder / up-convolution | Sec. 2 ($2\times2$ up-conv halving channels, then two $3\times3$ convs + ReLU) | `Decoder` `blocks.py:370-408`; `build_upsample` `blocks.py:79-101` | MATCH (default) / ACCEPTED ADAPTATION (bilinear option) |
| 7 | Output head | Sec. 2 ($1\times1$ conv, 64-vector $\to$ classes) | `unet.py:63-73` | MATCH |
| 8 | Weight initialization | Sec. 3 (Gaussian, std $\sqrt{2/N}$, ref [5] He et al.) | `initialize_weights` `blocks.py:104-126`; default `init_mode="default"` `configuration/architectures/backbone.py:21` | DEVIATION (minor) |
| 9 | Padding strategy | Sec. 2 (unpadded / valid convs + overlap-tile, Fig. 2) | `Conv2d(..., padding=1)` `blocks.py:156,165` | ACCEPTED ADAPTATION (padded "same" convs) |
| 10 | Omissions / additions | Sec. 2 (no normalization, 23 conv layers); Sec. 3 (weighted soft-max loss) | `ConvBlock` `blocks.py:140`; `Dropout2d` `blocks.py:171-172` | ACCEPTED ADAPTATION (BatchNorm, dropout added; loss out of architectural scope) |

**Overall verdict:** FAITHFUL WITH ADAPTATIONS.

### Stem and input handling (MATCH)

The paper feeds a single "input image tile" directly into the first contracting block with no separate stem (Sec. 2, Fig. 1). The code mirrors this: `UNet.forward` (`unet.py:71-73`) passes the raw input straight through `encode_decode` (defined on the shared `UNetBackbone`), whose `Encoder` first `ConvBlock` maps `in_channels` to `features[0]` (`blocks.py:348-357`). There is no pre-stem convolution, so the structures coincide.

### Encoder block composition (ACCEPTED ADAPTATION)

The paper specifies "the repeated application of two $3\times3$ convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU)" (Sec. 2). The code's `ConvBlock` (`blocks.py:151-173`) applies, in order, $\text{Conv}_{3\times3} \to \text{Norm} \to \text{Act} \to \text{Conv}_{3\times3} \to \text{Norm} \to \text{Act}$. Kernel size ($3\times3$), count (two), and the conv$\to$act ordering all match. The accepted adaptation is the insertion of a normalization layer (default `BatchNorm2d`, `blocks.py:65-66`) between each convolution and its activation. BatchNorm post-dates the 2015 paper for this design and is a standard modernization that stabilizes training; it does not alter the architectural skeleton. Disabling bias on the convs (`conv_bias=False`, `blocks.py:157,166`) is consistent with following BatchNorm, which absorbs the bias term.

### Downsampling (MATCH)

The paper uses "a $2\times2$ max pooling operation with stride 2 for downsampling" (Sec. 2). The code instantiates `nn.MaxPool2d(kernel_size=2)` (`blocks.py:358`), whose default stride equals the kernel size, giving the identical non-overlapping $2\times2$/stride-2 operation. Placement matches Fig. 1: pooling is applied after the double-conv block at each level, and the pre-pool feature map is the one stored as the skip (`blocks.py:364-366`).

### Bottleneck (MATCH)

In Fig. 1 the lowest resolution stage performs two $3\times3$ convs taking $512 \to 1024 \to 1024$ channels. The code's bottleneck is a `ConvBlock` mapping `features[-1]` to `features[-1] * bottleneck_factor` (`unet.py:31-38`), and with the default `bottleneck_factor = 2` (`configuration/architectures/backbone.py:15`) this is exactly $512 \to 1024$. The doubling of channels at the deepest level — a stated design point ("At each downsampling step we double the number of feature channels", Sec. 2) — is preserved.

### Skip topology and merge semantics (ACCEPTED ADAPTATION)

The paper combines each contracting-path feature map with the upsampled decoder feature via "a concatenation with the correspondingly cropped feature map from the contracting path" (Sec. 2); Fig. 1 labels this arrow "copy and crop". The merge is a channel-wise concatenation, and the code reproduces this exactly: `torch.cat([skip, x], dim=1)` (`blocks.py:406`), feeding a `ConvBlock` whose input width is $2F_l$ (`blocks.py:393`). The topology — one skip per encoder level, deepest-to-shallowest — is faithful (`blocks.py:365`, `decoder` consumes `skip_connections[::-1]` at `unet.py:56`).

The accepted adaptation concerns *how* the spatial sizes are reconciled. The paper crops the larger encoder feature map down to the decoder feature size, a consequence of unpadded convolutions shrinking each map. Because the code uses padded ("same") convolutions, encoder and decoder maps are already nominally equal in size, and `match_spatial_size` (`blocks.py:129-137`) only resizes (bilinear interpolation) when a mismatch arises (e.g. odd input dimensions). This is the natural and standard counterpart to cropping once padded convs are adopted; it is a direct corollary of adaptation #9 rather than an independent design change.

### Decoder and up-convolution (MATCH / ACCEPTED ADAPTATION)

The paper's expansive step is "an upsampling of the feature map followed by a $2\times2$ convolution ('up-convolution') that halves the number of feature channels" (Sec. 2). The default code path uses `nn.ConvTranspose2d(kernel_size=2, stride=2)` (`blocks.py:81-86`), the canonical learned $2\times2$ up-convolution, and the channel count is halved at each decoder level by construction (`decoder_feature_sizes` steps down through `features[::-1]`, `unet.py:40`, with the up-conv mapping `feature_sizes[index]` to `feature_sizes[index+1]`, `blocks.py:384-390`). This is a MATCH. The optional `upsample_mode="bilinear"` path (`blocks.py:87-100`), which replaces the transposed conv with `nn.Upsample` + a $1\times1$ conv, is a configurable ACCEPTED ADAPTATION (interpolation in place of a learned up-conv); it is not the default. The two post-merge $3\times3$ convs each followed by an activation are provided by the decoder `ConvBlock` (`blocks.py:391-400`), matching the paper.

### Output head (MATCH)

The paper: "At the final layer a $1\times1$ convolution is used to map each 64-component feature vector to the desired number of classes" (Sec. 2). The code's `output_head` is `nn.Conv2d(features[0], out_channels, kernel_size=1)` (`unet.py:63-67`); with the default `features[0] = 64`, the 64-channel finest decoder feature is mapped by a $1\times1$ conv to `out_channels`. Exact structural match. (The task here is dense regression of Gaussian-mixture parameters rather than class soft-max, but the head *layer* is identical; the soft-max is part of the loss, not the architecture.)

### Weight initialization (DEVIATION — minor)

Paper Section 3 is explicit: for a network of alternating convolution and ReLU layers, the initial weights should be drawn "from a Gaussian distribution with a standard deviation of $\sqrt{2/N}$, where $N$ denotes the number of incoming nodes of one neuron" (e.g. $N = 9 \cdot 64 = 576$ for a $3\times3$ conv with 64 input channels). This is precisely He/Kaiming normal initialization with `fan_in` (ref [5] is He et al. 2015), and the paper frames it as "extremely important".

The code *can* express this — `initialize_weights` (`blocks.py:104-126`) supports a `"kaiming"` mode — but two issues stand:
1. The default `init_mode` is `"default"` (`configuration/architectures/backbone.py:21`), which returns immediately (`blocks.py:105-106`) and leaves PyTorch's built-in initialization in place. PyTorch's default for `Conv2d` is Kaiming-*uniform* with `a=sqrt(5)`, which is **not** the paper's Gaussian $\sqrt{2/N}$ scheme.
2. Even when `"kaiming"` is selected, the conv branch uses `mode="fan_out"` (`blocks.py:110`), whereas the paper defines $N$ as the number of *incoming* nodes — i.e. `fan_in`. The code does use `fan_in` for `nn.Linear` (`blocks.py:117`) but not for convolutions.

Severity: minor (initialization affects optimization dynamics, not the network function or parameter count, and BatchNorm substantially reduces sensitivity to it). **Proposed fix:** set the UNet default to `init_mode="kaiming"` (`configuration/architectures/backbone.py:21`) and change the conv/conv-transpose branch to `nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")` (`blocks.py:110`) so it reproduces the paper's $\sqrt{2/N}$ with $N$ = incoming nodes. For exact fidelity the nonlinearity should track `config.activation`.

### Padding strategy (ACCEPTED ADAPTATION)

The paper uses unpadded ("valid") convolutions throughout, so every conv shrinks the feature map by a 1-pixel border; the segmentation map "only contains the pixels for which the full context is available" and arbitrarily large images are handled by the overlap-tile strategy with mirror-padding at borders (Sec. 2, Fig. 2). The output is therefore smaller than the input by a constant border (Sec. 3).

The code instead uses padded convolutions: every `Conv2d` sets `padding=1` (`blocks.py:156,165`), so spatial size is preserved and the output map matches the input tile size. This is classified as an ACCEPTED ADAPTATION — padded "same" convolutions are the now-standard modernization, removing the need for overlap-tiling and cropping, at the cost of slightly less reliable predictions at the extreme image border. It is a deliberate, well-understood trade-off rather than an error. Note the paper's tiling constraint — choose the input tile so every $2\times2$ max-pool sees an even-sized layer (Sec. 2) — is *not* enforced in code; instead `match_spatial_size` (`blocks.py:129-137`) repairs any odd-size mismatch at merge time, which is the appropriate substitute under the padded regime.

### Omissions and additions (ACCEPTED ADAPTATION)

- **Normalization (addition):** the 2015 architecture has no normalization layers; the code adds BatchNorm by default (covered under #2). Accepted modernization.
- **Dropout (addition):** the code optionally appends `Dropout2d` after the second activation of each block (`blocks.py:171-172`, default `dropout=0.15`). The paper mentions only "drop-out layers at the end of the contracting path" as implicit augmentation (Sec. 3.1), not per-block dropout. The code's broader dropout placement is a reasonable regularization choice and an accepted adaptation; it does not alter the macro-architecture.
- **Layer count:** the paper states "In total the network has 23 convolutional layers" for its specific 4-level configuration. The code's depth is governed by `features` (`configuration/architectures/backbone.py:14`) and `bottleneck_factor`, so the exact count varies with configuration; this is a hyperparameter, out of scope.
- **Loss (out of scope):** the paper's weighted pixel-wise soft-max cross-entropy with the separation-border weight map $w(\mathbf{x})$ (Eqs. 1-2, Sec. 3) is a training objective, not part of `unet.py`, and is not assessed here.

No element of the paper's architectural description is silently dropped: every described operation (double conv, ReLU, max-pool, up-conv, concat skip, $1\times1$ head) is present. The only true deviation is the initialization default (#8).

---

## Related Notes

- [[DLR-TomoSAR Index]] — Master index
- [[Model Zoo]] — Comparison with other architectures
- [[Attention UNet]] — U-Net variant with attention gates on skip connections
- [[UNet++]] — U-Net variant with nested dense skip connections
- [[Head Multihead]] — parameter-type-specific output heads, selectable on every backbone
- [[Head Per-Gaussian]] — one output head per Gaussian component, selectable on every backbone
- [[Training Pipeline]] — Training context
- [[Configuration Layer]] — UNetConfig
