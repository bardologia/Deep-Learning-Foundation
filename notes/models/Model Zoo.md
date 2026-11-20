---
type: model
domain: model
status: current
tags:
  - ml-template
  - ml-template/model
aliases:
  - Architecture Summary
  - Model Registry
---

# Model Zoo

This note summarises all model architectures available in ML-Template, their key properties, and guidance on when to prefer each. Model selection is one choice, the backbone (`backbone_name`, the registry key below); every backbone ends in the same single-convolution output head.

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
| `unetr` | [[UNETR]] | Pure ViT | Transformer skip outputs | CNN decoder | ~34.9M |
| `deeplabv3plus` | [[DeepLabV3+]] | Residual CNN, output stride 8 | Low-level fusion in decoder | ASPP + light decoder | ~10.3M |
| `segformer` | [[SegFormer]] | Hierarchical transformer (MiT-style) | Pyramid to all-MLP decoder | 1×1 conv | ~5.2M |
| `convnext_unet` | [[ConvNeXt UNet]] | ConvNeXt blocks in U-Net | Direct concatenation | Single 1×1 conv | ~19.1M |
| `dense_unet` | [[DenseUNet]] | FC-DenseNet (Tiramisu) | Dense concat + U skips | Single 1×1 conv | ~1.0M |
| `hrnet` | [[HRNet]] | Parallel multi-resolution branches | Repeated branch fusion | Concat + 1×1 conv | ~3.1M |
| `multires_unet` | [[MultiResUNet]] | MultiRes blocks in U-Net | ResPath-filtered concat | Single 1×1 conv | ~14.4M |
| `fpn` | [[FPN]] | Residual bottom-up + FPN top-down | Lateral 1×1 additions | Summed pyramid + 1×1 conv | ~7.8M |
| `u2net` | [[U2-Net]] | Nested RSU mini-U-Nets | Direct concatenation | Single 1×1 conv | ~8.3M |
| `pixel_mlp` | [[PixelMLP]] | 1×1 conv stack (no spatial context) | None (single stream) | Single 1×1 conv | ~30.8M |
| `local_cnn` | [[Local CNN]] | Full-resolution 3×3 ConvBlock stack | None (single stream) | Single 1×1 conv | ~31.1M |
| `nafnet` | [[NAFNet]] | Gated NAFBlock encoder-decoder (activation-free) | Additive skip | Single 3×3 conv | ~29.2M |

Parameter counts are approximate for default configurations with $C_{\text{in}} = 4$ (`in_channels`) and $C_{\text{out}} = 2$ (`out_channels`), the channel counts of the example dataset. Most defaults sit near the UNet budget of about 31M parameters, PixelMLP and Local CNN included, so that a benchmark compares architectures at similar capacity.

---

## Output head

Every backbone ends in one output convolution (`OutputHeadMixin`, `models/blocks.py`) from its embedding to `out_channels`, the number of target channels of the dataset. NAFNet uses a $3\times3$ kernel, every other backbone $1\times1$.

---

## Selection Guidance

### Use UNet when
The dataset is small or GPU memory is limited. The standard UNet provides a strong baseline with minimal complexity. Default starting point for any new experiment.

### Use AttentionUNet when
Different regions of the grid follow different input-target relations and need different processing. The attention gates learn to suppress irrelevant skip features region-specifically.

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

### Use DeepLabV3+ when
Multi-scale context at fixed dilation rates is suspected to matter (structures with a characteristic spatial scale). ASPP aggregates several receptive fields in parallel without extra downsampling.

### Use SegFormer when
Testing whether a strong hierarchical attention encoder with a near-trivial MLP decoder suffices — the complementary hypothesis to the heavy-decoder transformer designs.

### Use ConvNeXt UNet when
Evaluating modern convolution design (large depthwise kernels, inverted bottlenecks, LayerNorm, GELU) under the unchanged U-Net topology.

### Use DenseUNet when
Parameter efficiency through feature reuse is the question; at matched capacity it is the deepest, thinnest model in the zoo.

### Use HRNet when
Per-pixel position accuracy is the priority: the full-resolution stream is never downsampled, avoiding the encode-decode round trip entirely.

### Use MultiResUNet when
Structures span several spatial scales; every block sees effective 3/5/7-pixel receptive fields in parallel, and ResPaths reduce the encoder-decoder semantic gap.

### Use FPN when
Probing how much decoder capacity the task actually needs: the FPN decoder is deliberately minimal, concentrating parameters in the encoder.

### Use U2-Net when
Intra-stage multi-scale mixing is desired: every stage is itself a small U-Net, so even full-resolution layers integrate wide context.

### Use PixelMLP when
Establishing the no-spatial-context control. A per-pixel regressor sees one pixel's channel vector at a time; PixelMLP is that regime in learned form, and the margin any spatial backbone holds over it is the measured value of spatial context. Every benchmark sweep should carry it.

### Use Local CNN when
Measuring how far local context alone goes. A fixed $(4B+1)\times(4B+1)$ receptive field with no downsampling sits between PixelMLP (no context) and the encode-decode backbones (wide context), so the three together separate per-pixel mapping, local smoothing and long-range aggregation.

### Use NAFNet when
Testing the restoration hypothesis: the task is dense continuous regression, closer to image restoration than to the semantic segmentation the rest of the zoo descends from. NAFNet is the restoration family's strongest simple representative — gated blocks, channel attention, no conventional activations — and directly probes whether the segmentation prior (deep semantic bottleneck) is the wrong trade for per-pixel fidelity.

---

## Common Configuration

All architectures share:
- Input: $(C_{\text{in}}, P_H, P_W)$ patch tensor
- Output: $(C_{\text{out}}, P_H, P_W)$ target tensor, one channel per target channel
- Optimised with AdamW by default (`OptimizerConfig` in `configuration/training/general/optimization.py`)
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
| `OutputHeadMixin` | Builds the output convolution to `out_channels` and exposes `head_parameters()` for the optimizer groups | every backbone |
| `MultiHeadSelfAttention` / `TransformerBlock` | Pre-norm full self-attention and the transformer layer (LN $\to$ MSA $\to$ residual $\to$ LN $\to$ FFN $\to$ residual, with `DropPath`) | `transunet`, `unetr` |
| `PatchEmbedding` / `tokens_to_feature_map` | Strided-conv patch tokeniser and the inverse token $\to$ feature-map reshape | `transunet`, `unetr`; `tokens_to_feature_map` also used by `swin_unet` |
| `build_activation` / `build_norm2d` / `build_upsample` / `initialize_weights` / `match_spatial_size` / `DropPath` | Factory and utility helpers | all architectures |

Every backbone class inherits `OutputHeadMixin`, sets `embedding_channels` at the end of its constructor, and calls `_build_output_head()` / `_head_forward()`. The head maps the backbone embedding straight to `out_channels` with one convolution (`conv_head_kernel_size` is $1$ everywhere except NAFNet's native $3\times3$). Swin-specific window attention (`WindowAttention`, `SwinTransformerBlock`, `PatchMerging`, `PatchExpanding`, `SwinStage`) remains local to `models/backbone/swin_unet.py` because it is not reused elsewhere.

---

## Configuration Persistence (BackboneModelConfigIO)

Each training run persists the full architecture configuration so a checkpoint can be reloaded into a structurally identical model at inference time. `BackboneModelConfigIO` (`pipelines/shared/config/config_persistence.py`) serialises every dataclass field of the model config except `shape_logger_types` (its `EXCLUDED` set) to `model_config.json` (the `RunArtifacts.BACKBONE_CONFIG` filename) in the run's metadata directory, recording `model_name`, `config_type` (the config class name), and the field/value map under `config`.

`BackboneModelConfigIO.load` reconstructs the config strictly: it normalises `model_name` (lowercase, `-`/space to `_`), instantiates the dataclass from `BACKBONE_CONFIG_REGISTRY[model_name]`, then overwrites each persisted field; it fails if the file is missing, the `model_name` is unknown, a field of the dataclass is absent from the file, or a persisted field is not an attribute of the dataclass. It returns the reconstructed `config` together with the raw persisted `model_name`. Tuple-typed fields (e.g. `atrous_rates`, `rsu_heights`, `num_heads`, `sr_ratios`) are coerced back from their JSON list form when the live default is a tuple. There is no tolerant fallback: an absent `model_config.json` raises `FileNotFoundError` and forces a retrain rather than a best-effort reconstruction. This makes the hyperparameters in each model note (channel widths, depths, head counts, dilation rates) the exact quantities captured per run.

---

## Related Notes

- [[UNet]] — Baseline architecture
- [[Attention UNet]] — Attention-gated variant
