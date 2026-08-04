---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - Architecture Summary
  - Model Registry
---

# Model Zoo

This note summarises all model architectures available in DLR-TomoSAR, their key properties, and guidance on when to prefer each. Model selection has two independent axes: the backbone (`backbone_name`, the registry key below) and the output head (`head` on the architecture config, `backbone_head` in the entry configs), so any backbone pairs with any head.

---

## Architecture Summary

| Key | Note | Backbone | Skip Mechanism | Default Conv Head | Parameters |
|---|---|---|---|---|---|
| `unet` | [[UNet]] | CNN encoder-decoder | Direct concatenation | Single 1×1 conv | ~31.0M |
| `attention_unet` | [[Attention UNet]] | CNN encoder-decoder | Attention-gated concat | Single 1×1 conv | ~32.4M |
| `unetplusplus` | [[UNet++]] | CNN encoder-decoder | Nested dense skips | Single 1×1 conv | ~31.1M |
| `resunet` | [[ResUNet]] | Residual CNN encoder-decoder (stride-2) | Skip + residual | Single 1×1 conv | ~32.4M |
| `unet_skip` | [[UNet Skip]] | Residual CNN encoder-decoder (MaxPool) | Skip + residual | Single 1×1 conv | ~32.4M |
| `linknet` | [[LinkNet]] | ResNet encoder | Additive skip | Single 1×1 conv | ~31.0M |
| `swin_unet` | [[SwinUNet]] | Swin Transformer | Hierarchical Swin | 1×1 conv | ~28.8M |
| `transunet` | [[TransUNet]] | CNN + ViT hybrid | Transformer patch tokens | CNN decoder | ~30.6M |
| `unetr` | [[UNETR]] | Pure ViT | Transformer skip outputs | CNN decoder | ~34.4M |
| `deeplabv3plus` | [[DeepLabV3+]] | Residual CNN, output stride 8 | Low-level fusion in decoder | ASPP + light decoder | ~10.3M |
| `segformer` | [[SegFormer]] | Hierarchical transformer (MiT-style) | Pyramid to all-MLP decoder | 1×1 conv | ~5.2M |
| `convnext_unet` | [[ConvNeXt UNet]] | ConvNeXt blocks in U-Net | Direct concatenation | Single 1×1 conv | ~19.1M |
| `dense_unet` | [[DenseUNet]] | FC-DenseNet (Tiramisu) | Dense concat + U skips | Single 1×1 conv | ~1.0M |
| `hrnet` | [[HRNet]] | Parallel multi-resolution branches | Repeated branch fusion | Concat + 1×1 conv | ~3.1M |
| `multires_unet` | [[MultiResUNet]] | MultiRes blocks in U-Net | ResPath-filtered concat | Single 1×1 conv | ~14.4M |
| `fpn` | [[FPN]] | Residual bottom-up + FPN top-down | Lateral 1×1 additions | Summed pyramid + 1×1 conv | ~7.8M |
| `u2net` | [[U2-Net]] | Nested RSU mini-U-Nets | Direct concatenation | Single 1×1 conv | ~8.3M |
| `pixel_mlp` | [[PixelMLP]] | 1×1 conv stack (no spatial context) | None (single stream) | Single 1×1 conv | ~3.2M |
| `local_cnn` | [[Local CNN]] | Full-resolution 3×3 ConvBlock stack | None (single stream) | Single 1×1 conv | ~3.0M |
| `nafnet` | [[NAFNet]] | Gated NAFBlock encoder-decoder (activation-free) | Additive skip | Single 3×3 conv | ~29.2M |

Parameter counts are approximate for default configurations with $C_{\text{in}} = 1$ (`in_channels`) and $C_{\text{out}} = 6$ (`out_channels`), corresponding to $K = C_{\text{out}} / \text{params\_per\_gaussian} = 6/3 = 2$ Gaussian components. The [[Capacity Matching]] stage of the [[Benchmark Pipeline]] rescales all of them to the UNet reference budget before benchmarking.

---

## Output Heads

Every backbone exposes the same four output heads through `OutputHeadsMixin` (`models/blocks.py`), selected by the `head` config field. Benchmark and tuning sweeps cross backbones with a `heads` list; a non-default combination is written as a compound key, e.g. `resunet-set_pred`.

| Head key | Note | Structure | When |
|---|---|---|---|
| `conv` | [[Head Conv]] | Single conv projection (default) | Baseline shared projection of all $3K$ channels |
| `multihead` | [[Head Multihead]] | 3 PixelMLP heads, one per parameter type | Systematic bias across parameter types |
| `per_gaussian` | [[Head Per-Gaussian]] | $K$ PixelMLP heads, one per slot | Systematic bias across Gaussian slots |
| `set_pred` | [[Head Set-Prediction]] | $K$ PixelMLP heads + existence gate | Slot collapse; pair with hungarian matching |

---

## Selection Guidance

### Use UNet when
The dataset is small or GPU memory is limited. The standard UNet provides a strong baseline with minimal complexity. Default starting point for any new experiment.

### Use AttentionUNet when
There is spatial heterogeneity in the scene (e.g., mixed urban–vegetated regions) where different areas require different processing. The attention gates learn to suppress irrelevant skip features region-specifically.

### Use UNet++ when
Skip connection quality is suspected to be a bottleneck. UNet++ provides graduated feature fusion across multiple encoder levels, which can reduce the semantic gap between encoder and decoder.

### Use ResUNet when
Training is unstable or very deep (more than 5 levels). Residual connections in the encoder prevent gradient vanishing and allow deeper architectures.

### Use UNet Skip when
A residual encoder-decoder with MaxPool downsampling (rather than the stride-2 downsampling of [[ResUNet]]) is wanted. `UNetSkip` is a `ResUNetBackbone` instantiated with `downsample="maxpool"` (`resunet.py`): every encoder stage is a `ResidualConvBlock` at stride 1, followed by a separate `MaxPool2d(2)`, so the residual blocks never carry the downsampling stride themselves.

### Use LinkNet when
Computational efficiency is a priority. LinkNet's additive (rather than concatenative) skip connections reduce the decoder channel count and parameter budget substantially.

### Use SwinUNet / TransUNet / UNETR when
Long-range spatial dependencies are important (e.g., large homogeneous structures spanning many pixels). Transformer-based architectures attend globally, unlike CNN-only designs that are limited by receptive field.

### Use the multihead / per_gaussian heads when
The default conv head shows systematic bias across parameter types (`head = "multihead"`) or across Gaussian slots (`head = "per_gaussian"`). The head axis is independent of the backbone, so the head-structure question can be isolated on any fixed encode-decode path — see [[Head Multihead]] and [[Head Per-Gaussian]].

### Use the set_pred head when
Attacking Gaussian slot collapse head-on. Per-slot heads plus an existence-logit gate give each slot an explicit on/off mechanism decoupled from amplitude regression, and together with `param_matching: hungarian` realise DETR-style set prediction under the unchanged $3K$-channel output contract. Pair with the hungarian matcher; under `sorted_gt` the gate merely relabels slots. See [[Head Set-Prediction]].

### Use DeepLabV3+ when
Multi-scale context at fixed dilation rates is suspected to matter (e.g., characteristic facade/layover scales). ASPP aggregates several receptive fields in parallel without extra downsampling.

### Use SegFormer when
Testing whether a strong hierarchical attention encoder with a near-trivial MLP decoder suffices — the complementary hypothesis to the heavy-decoder transformer designs.

### Use ConvNeXt UNet when
Evaluating modern convolution design (large depthwise kernels, inverted bottlenecks, LayerNorm, GELU) under the unchanged U-Net topology.

### Use DenseUNet when
Parameter efficiency through feature reuse is the question; at matched capacity it is the deepest, thinnest model in the zoo.

### Use HRNet when
Per-pixel position accuracy (peak-location error) is the priority: the full-resolution stream is never downsampled, avoiding the encode-decode round trip entirely.

### Use MultiResUNet when
Structures span several spatial scales; every block sees effective 3/5/7-pixel receptive fields in parallel, and ResPaths reduce the encoder-decoder semantic gap.

### Use FPN when
Probing how much decoder capacity the task actually needs: the FPN decoder is deliberately minimal, concentrating parameters in the encoder.

### Use U2-Net when
Intra-stage multi-scale mixing is desired: every stage is itself a small U-Net, so even full-resolution layers integrate wide context.

### Use PixelMLP when
Establishing the no-spatial-context control. Classical tomographic inversion (beamforming, Capon, CS) is strictly per-pixel; PixelMLP is its learned analogue, and the margin any spatial backbone holds over it is the measured value of spatial context. Every benchmark sweep should carry it.

### Use Local CNN when
Measuring how far local context alone goes. A fixed $(4B+1)\times(4B+1)$ receptive field with no downsampling completes the context ladder PixelMLP → Local CNN → encode-decode backbones, decomposing performance into per-pixel mapping, local smoothing, and long-range aggregation.

### Use NAFNet when
Testing the restoration hypothesis: the task is dense continuous regression, closer to image restoration than to the semantic segmentation the rest of the zoo descends from. NAFNet is the restoration family's strongest simple representative — gated blocks, channel attention, no conventional activations — and directly probes whether the segmentation prior (deep semantic bottleneck) is the wrong trade for per-pixel parameter fidelity.

---

## Beyond the Zoo

[[GammaNet Unrolled]] is the physics-native counterpart to every model above: instead of learning a generic image-to-parameters mapping, it unrolls the exact tomographic forward operator into learned proximal-gradient iterations. It lives outside this registry — its own `unrolled` model family, training pipeline, and entry point — because its input contract (per-pixel coherence measurements plus kz, not the image channel stack) and output (reflectivity profiles, not Gaussian parameters) differ from the zoo's.

---

## Common Configuration

All architectures share:
- Input: $(C_{\text{in}}, P_H, P_W)$ patch tensor
- Output: $(3K, P_H, P_W)$ parameter tensor
- Optimised with AdamW (see [[Training Pipeline]])
- Per-layer learning rate groups defined in each `*Config.get_param_groups()`

Architectures requiring image size (`swin_unet`, `transunet`, `unetr`) receive `image_size = patch_height` from the dataset configuration.

---

## Shared Building Blocks

The recurring sub-modules live in a single `models/blocks.py` and are imported by every architecture. The canonical shared blocks are:

| Block | Role | Consumers |
|---|---|---|
| `ConvBlock` | Double $3\times3$ conv $\to$ Norm $\to$ Act (optional `Dropout2d`) | `unet`, `attention_unet`, `unetplusplus`, `transunet`, `unetr`, `convnext_unet` (decoder), `multires_unet`, `u2net`, `local_cnn` |
| `ResidualConvBlock` | Pre-activation residual unit with stride and `first_unit` controls | `resunet`, `unet_skip` |
| `Encoder` / `Decoder` | Symmetric `ConvBlock` stacks with `MaxPool2d` / `build_upsample` and concatenative skips | `unet` family (`UNetBackbone`) |
| `PixelMLP` | $1\times1$ conv $\to$ Act $\to$ $1\times1$ conv pointwise head | multihead / per-Gaussian / set-prediction output heads |
| `OutputHeadsMixin` | Builds the head selected by `config.head` (conv, multihead, per_gaussian, set_pred), resolves the $K = C_{\text{out}}/\texttt{params\_per\_gaussian}$ layout for the Gaussian heads, and exposes `head_parameters()` for the optimizer groups | every backbone |
| `MultiHeadSelfAttention` / `TransformerBlock` | Pre-norm full self-attention and the transformer layer (LN $\to$ MSA $\to$ residual $\to$ LN $\to$ FFN $\to$ residual, with `DropPath`) | `transunet`, `unetr` |
| `PatchEmbedding` / `tokens_to_feature_map` | Strided-conv patch tokeniser and the inverse token $\to$ feature-map reshape | `transunet`, `unetr`; `tokens_to_feature_map` also used by `swin_unet` |
| `build_activation` / `build_norm2d` / `build_upsample` / `initialize_weights` / `match_spatial_size` / `DropPath` | Factory and utility helpers | all architectures |

Every backbone class inherits `OutputHeadsMixin`, sets `embedding_channels` at the end of its constructor, and calls `_build_output_head()` / `_head_forward()` so the head is a pure function of `config.head`. The default `conv` head maps the backbone embedding straight to `out_channels` with one convolution (`conv_head_kernel_size` is $1$ everywhere except NAFNet's native $3\times3$), with no Gaussian-layout resolution. The Gaussian heads call `_resolve_gaussian_layout`, which sets $K = \texttt{out\_channels} / \texttt{params\_per\_gaussian}$ and raises `ValueError` when `out_channels` is not divisible by `params_per_gaussian`; `_build_triple_heads` builds three `PixelMLP` heads each emitting $K$ channels (interleaved to $3K$), `_build_per_gaussian_heads` builds $K$ `PixelMLP` heads each emitting `params_per_gaussian` channels, and `_build_set_prediction_heads` adds the existence gate on top. Backbones without a plain `activation` config field route the head activation explicitly (`_head_activation`: `ffn_activation` for SwinUNet, SegFormer, and ConvNeXt UNet; GELU for NAFNet). Swin-specific window attention (`WindowAttention`, `SwinTransformerBlock`, `PatchMerging`, `PatchExpanding`, `SwinStage`) remains local to `models/backbone/swin_unet.py` because it is not reused elsewhere.

---

## Configuration Persistence (BackboneModelConfigIO)

Each training run persists the full architecture configuration so a checkpoint can be reloaded into a structurally identical model at inference time. `BackboneModelConfigIO` (`pipelines/shared/config/config_persistence.py`, a subclass of the shared `ConfigIO` base) serialises every dataclass field of the model config except `shape_logger_types` (its `EXCLUDED` set) to `model_config.json` (the `RunArtifacts.BACKBONE_CONFIG` filename) in the run's metadata directory, recording `model_name`, `config_type` (the config class name), and the field/value map under `config`.

`ConfigIO.load` reconstructs the config strictly: it normalises `model_name` (lowercase, `-`/space to `_`), instantiates the dataclass from `BACKBONE_CONFIG_REGISTRY[model_name]` (returned by `BackboneModelConfigIO._registry`), then overwrites each persisted field, raising loudly if the file is missing, the `model_name` is unknown, or a persisted field is not an attribute of the dataclass. It returns the reconstructed `config` together with the raw persisted `model_name`. Tuple-typed fields (e.g. `atrous_rates`, `rsu_heights`, `num_heads`, `sr_ratios`) are coerced back from their JSON list form when the live default is a tuple. There is no tolerant fallback: an absent `model_config.json` raises `FileNotFoundError` and forces a retrain rather than a best-effort reconstruction. The sibling classes `ProfileAutoencoderConfigIO` and `ImageAutoencoderConfigIO` in the same module persist the autoencoder architectures the same way. This makes the hyperparameters in each model note (channel widths, depths, head counts, dilation rates) the exact quantities captured per run.

---

## Related Notes

- [[DLR-TomoSAR Index]] — Master index
- [[Training Pipeline]] — Model instantiation and training loop
- [[UNet]] — Baseline architecture
- [[Attention UNet]] — Attention-gated variant
- [[Head Conv]] / [[Head Multihead]] / [[Head Per-Gaussian]] / [[Head Set-Prediction]] — the output-head axis
- [[Configuration Layer]] — UNetConfig and the shared head field
