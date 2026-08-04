---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - HRNet
  - HRNetLite
  - high-resolution network
family: hrnet
registry_key: hrnet
summary: Maintains a full-resolution branch throughout and repeatedly fuses progressively lower-resolution branches.
---

# HRNet

`HRNetLite` (`models/backbone/hrnet_lite.py`) is a high-resolution network ([[HRNet_Wang2020_1908.07919.pdf|Wang et al., 2020]]): instead of an encode-then-decode hourglass, it maintains a full-resolution branch throughout the network, adds progressively lower-resolution branches, and repeatedly exchanges information across all branches.

---

## Summary

A `ResidualConvBlock` stem maps the single-channel input (`in_channels = 1`) to the full-resolution branch at `base_channels` $C$. Each stage adds one branch (stride-2 transition from the previous branch, channels doubling: $C, 2C, 4C, \dots$ for `n_branches`), applies `blocks_per_stage` residual blocks per branch, and ends with a `BranchFusion` that makes every branch the sum of all branches resampled to its resolution (strided 3×3 convolutions downward, 1×1 convolution + bilinear upsampling upward). The head upsamples all branches to full resolution, concatenates, fuses with a 3×3 convolution to $2C$ channels, and applies a $1\times1$ output head producing `out_channels` per pixel. HRNetLite emits a flat regression map of $3K$ packed Gaussian-mixture parameters (`out_channels = 6` $= 3 \times 2$ Gaussians by default, `params_per_gaussian = 3`); the default head is a plain `nn.Conv2d(base*2, out_channels, 1)` ([[Head Conv]]); the `OutputHeadsMixin` triple-head, per-Gaussian, and set-prediction heads are selectable via the `head` config field.

---

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{x}_j$ | Feature map of source branch $j$ |
| $\mathbf{y}_i$ | Fused output of target branch $i$ |
| $\mathbf{d}_k$ | Intermediate down-path feature after $k$ stride-2 steps |
| $C$ | Base channel count (`base_channels`) of the full-resolution branch |
| $C_j$ | Channels of branch $j$, $C_j = 2^j C$ |
| $i$ | Target branch index |
| $j$ | Source branch index |
| $k$ | Down-path step index, $k = 1, \dots, i - j$ |
| $T_{j \to i}$ | Transform resampling source branch $j$ to the resolution of target branch $i$ |
| $\text{Conv}_{1\times1}$ | $1\times1$ convolution |
| $\text{Conv}_{3\times3}^{s2}$ | $3\times3$ convolution with stride 2 |
| $\text{BN}$ | Batch normalization |
| $\text{Act}$ | Activation |
| $\text{up}$ | Bilinear upsampling |

---

## Architecture

### Branch Fusion

For branches $\{\mathbf{x}_j\}$ with channels $C_j = 2^j C$, the fused output of branch $i$ is

For a single source $\mathbf{x}_j$, the transform $T_{j \to i}(\mathbf{x}_j)$ is built stepwise. The down path ($j < i$) takes $i - j$ stride-2 steps, with the activation applied after every non-final step:

$$
\begin{aligned}
\mathbf{d}_0 &= \mathbf{x}_j \\
\mathbf{c}_k &= \text{Conv}_{3\times3}^{s2}(\mathbf{d}_{k-1}) \\
\mathbf{d}_k &= \text{Act}(\text{BN}(\mathbf{c}_k)), \qquad k = 1, \dots, i - j - 1 \\
\mathbf{d}_{i-j} &= \text{BN}(\text{Conv}_{3\times3}^{s2}(\mathbf{d}_{i-j-1})) \\
T_{j \to i}(\mathbf{x}_j) &= \mathbf{d}_{i-j}
\end{aligned}
$$

The same-resolution path is the identity, $T_{i \to i}(\mathbf{x}_i) = \mathbf{x}_i$. The up path ($j > i$) is a single $1\times1$ projection followed by upsampling:

$$
\begin{aligned}
\mathbf{e} &= \text{Conv}_{1\times1}(\mathbf{x}_j) \\
\mathbf{m} &= \text{BN}(\mathbf{e}) \\
T_{j \to i}(\mathbf{x}_j) &= \text{up}(\mathbf{m}), \qquad j > i
\end{aligned}
$$

The fused output of branch $i$ then aggregates all sources and applies the post-sum activation:

$$
\mathbf{y}_i = \text{Act}\Big(\sum_{j} T_{j \to i}(\mathbf{x}_j)\Big)
$$

so high-resolution detail and low-resolution context are mixed at every stage rather than once in a decoder. In the down path every $3\times3$ convolution is followed by batch normalization, and all but the **final** convolution are followed by the activation; the final step is left linear (Conv $\to$ BN) because the post-sum $\text{Act}$ supplies the nonlinearity after aggregation, exactly as in the official HRNet `_make_fuse_layers`.

---

## Design Rationale

**No resolution bottleneck.** Every other CNN in the zoo compresses to $P/8$ or $P/16$ and reconstructs; position-sensitive information (exact scatterer elevation per pixel) must survive that round trip through skip connections. HRNet never discards the full-resolution stream, which is precisely the property dense per-pixel regression rewards in human-pose and segmentation benchmarks.

> The hypothesis: per-pixel $\mu$ estimation is position-critical, so a permanently high-resolution representation should reduce peak-location error relative to hourglass designs at equal capacity.

---

## Parameter Reference

See [[Configuration Layer]] → `HRNetLiteConfig` (`base_channels`, `n_branches`, `blocks_per_stage`).

---

## Paper fidelity

**Review date:** 2026-06-04
**Reference:** Wang, J. et al. (2020). *Deep High-Resolution Representation Learning for Visual Recognition*. IEEE TPAMI. arXiv:1908.07919. Architecture in §3, Fig. 2–4, Eq. (1). [[HRNet_Wang2020_1908.07919.pdf|PDF]]
**Verdict:** Faithful HRNetV2-Lite. Topology and fusion rules match the paper; all remaining departures are justified dense-prediction / Lite-scaling adaptations. No structural deviations.

### Verdict table

| # | Dimension | Paper ref | Code ref | Verdict |
|---|-----------|-----------|----------|---------|
| 1 | Parallel multi-resolution branches maintained throughout | §3, §3.1; Fig. 2 | `HRNetLite.forward` `hrnet_lite.py:131-145` | MATCH |
| 2 | Stem (two stride-2 3×3 → $1/4$) | §3; Fig. 2 caption | `hrnet_lite.py:81-88` | ACCEPTED ADAPTATION |
| 3 | Stage transition: new branch by stride-2 from lowest-resolution branch | §3.1; Eq. (1) | `hrnet_lite.py:97-101`, `:135` | MATCH |
| 4 | Fusion exchange unit (down = stacked stride-2 3×3, up = 1×1 + upsample, same = Id, aggregate = SUM, then Act) | §3.2; Eq. (1); Fig. 3 | `BranchFusion` `hrnet_lite.py:12-64` | MATCH |
| 4a | Channel change applied at last conv of downsample stack | §3.2; Fig. 3 | `hrnet_lite.py:28-39` | MATCH |
| 4b | ReLU between stacked downsample convs (non-final steps only) | §3.2/§3.4 (BN+ReLU per conv); official `_make_fuse_layers` | `hrnet_lite.py:36-37` | MATCH |
| 4c | Upsample interpolation mode (paper: bilinear here; nearest in some impls) | §3.2; Fig. 3 | `hrnet_lite.py:59` | MATCH / ACCEPTED |
| 5 | Fusion frequency (every ~4 residual units / per stage) | §3.2 | `hrnet_lite.py:136`, `:137` | ACCEPTED ADAPTATION |
| 6 | Block type (bottleneck stage 1, basic later) | §3.4 | `ResidualConvBlock` everywhere `hrnet_lite.py:81-115` | ACCEPTED ADAPTATION |
| 7 | HRNetV2 head: upsample all → CONCAT → conv | §3.3; Fig. 4(b) | `hrnet_lite.py:121-125`, `:139-144` | MATCH |
| 8 | Channel doubling $C, 2C, 4C, \dots$ | §3.4 | `hrnet_lite.py:78` | MATCH |
| 9 | Output head | §3.3 (task linear classifier) | `hrnet_lite.py:127`, `:145` | ACCEPTED ADAPTATION |
| 10 | Full all-to-all exchange (every source→target) | Fig. 2, Fig. 3 | `BranchFusion.transforms` `hrnet_lite.py:18-46` | MATCH |

### Prose

**Topology (1, 3, 8, 10).** The implementation reproduces the defining HRNet property: a high-resolution stream is created first and lower-resolution streams are appended one per stage, all carried in parallel to the end with no encoder–decoder collapse (§3.1; `hrnet_lite.py:131-145`). Each new branch is a stride-2 $3\times3$ convolution off the current lowest-resolution branch (`hrnet_lite.py:97-101` consuming `branches[-1]` at `:135`), matching $\mathcal{N}_{sr}\!\to\!\mathcal{N}_{s+1,r+1}$ of Eq. (1). Branch widths follow $C, 2C, 4C, \dots$ ($2^j C$, `hrnet_lite.py:78`), and `BranchFusion` builds a dense $n\times n$ transform matrix (`hrnet_lite.py:18-46`) so every source feeds every target — the full exchange of Fig. 2/Fig. 3, not the partial GridNet-style exchange the paper explicitly contrasts itself against (§2).

**Fusion unit (4).** The exchange equations are implemented exactly per Eq. (1) and Fig. 3: same-resolution is `nn.Identity`; the up path is $1\times1$ conv then resize (`hrnet_lite.py:41-45`, `:57-59`); the down path stacks $(r-x)$ stride-2 $3\times3$ convolutions (`hrnet_lite.py:24-40`); outputs are summed (`hrnet_lite.py:60`) and a single activation follows the sum (`hrnet_lite.py:62`). The channel change in the down path is correctly deferred to the final conv of the stack (`is_final = step == n_steps - 1`, `hrnet_lite.py:29`), matching the paper and the reference implementation.

**Down-path activation (4b).** The paper states each $3\times3$ in a fusion path is followed by batch normalization and ReLU (§3.4), and the official `_make_fuse_layers` realizes this as BN+ReLU after every non-terminal downsample conv with the terminal conv left BN-only. The Lite down path appends `build_activation(activation)` after the norm of every **non-final** downsample step, guarded by `is_final` so the terminal step stays linear (`Conv → BN`) before the shared post-sum activation (`hrnet_lite.py:28-39`, `:62`). A two-step ($4\times$) path reads `Conv → BN → Act → Conv → BN`; a single-step path is `Conv → BN`, matching the official convention.

**Adaptation 4c.** Upsampling uses `bilinear` with `align_corners=False` (`hrnet_lite.py:59`). HRNetV2's head and fusion specify bilinear upsampling (§3.2–3.3), so this matches the paper text; it differs only from PyTorch ports that use nearest. Accepted.

**Stem (2).** The paper stem is two stride-2 $3\times3$ convolutions taking the input to $1/4$ resolution (Fig. 2 caption); the highest branch therefore runs at $1/4$. The Lite stem is a single `ResidualConvBlock` at stride 1 (`hrnet_lite.py:81-88`), so branch 0 runs at full input resolution. This is a deliberate dense-prediction adaptation for per-pixel scatterer regression — keeping the top branch at full resolution maximizes position fidelity, which is the stated rationale of this note (§ Design Rationale) — and shifts the resolution pyramid from $\{1/4,1/8,\dots\}$ to $\{1, 1/2, 1/4,\dots\}$ without altering topology. Accepted adaptation.

**Block type (6) and Lite scaling.** The paper uses bottleneck units in stage 1 and basic (two $3\times3$) units thereafter, with four residual units per branch per modularized block and four-resolution width $C,2C,4C,8C$ at $C=32/48$ (§3.4). The Lite model uses `ResidualConvBlock` (a pre-activation basic block) uniformly, `blocks_per_stage = 2`, and `n_branches = 3` ($C=48 \Rightarrow 48,96,192$) by default (`configuration/architectures/backbone.py`). Fewer branches/blocks and a single block type are explicitly within the accepted Lite-scaling envelope; the block is structurally a residual basic unit, so the per-branch computation is of the paper's family. The `ResidualConvBlock` `stride` parameter (`blocks.py:161`) defaults to 1 and `HRNetLite` never passes it (`hrnet_lite.py:81-115`), so all in-branch blocks run at stride 1. Accepted adaptation.

**Fusion frequency (5).** The paper fuses repeatedly, roughly every four residual units and at stage boundaries (§3.2). The Lite model fuses once per stage, after that stage's `blocks_per_stage` (= 2) blocks (`hrnet_lite.py:136`), plus the final head fusion. With only two blocks per stage this is effectively "fuse every two units at each stage boundary" — the same repeated-fusion principle at lower depth, consistent with Lite scaling. The paper's ablation (Table 12) confirms both across-stage and within-stage fusions help; the Lite model retains across-stage fusion but, having a single block group per stage, has no separate within-stage fusion. Accepted adaptation (a within-stage fusion could be added if depth is increased, but it is not a deviation at this depth).

**Head (7, 9).** The representation head is HRNetV2 (Fig. 4(b)): all branches are bilinearly upsampled to the highest resolution (`hrnet_lite.py:139-142`), **concatenated** (`torch.cat`, `hrnet_lite.py:144`) — not summed — then mixed by a convolution (`hrnet_lite.py:121-125`). This is exactly the V2 head; concat-vs-sum is the load-bearing distinction from the fusion units and is correct. The paper's V2 head uses a $1\times1$ mixing conv followed by a task-specific linear classifier; the Lite head uses a $3\times3$ mixing conv then a $1\times1$ output head (`hrnet_lite.py:127`, `:145`). The $3\times3$ vs $1\times1$ mix and the regression output head are accepted task adaptations (dense continuous-valued scatterer parameters rather than a softmax segmentation map).

**Summary.** No structural deviations. The fusion down path matches the paper and the official `_make_fuse_layers`, and all departures are justified dense-prediction or Lite-scaling adaptations. The model is a correct scaled-down HRNetV2.

---

## Related Notes

- [[ResUNet]] — Source of the residual block
- [[Model Zoo]] — Architecture comparison
- [[Configuration Layer]] — HRNetLiteConfig
- [[DLR-TomoSAR Index]] — Master index
