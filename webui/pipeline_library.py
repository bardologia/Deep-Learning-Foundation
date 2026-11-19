
from __future__ import annotations


class PipelineLibrary:

    def collect(self) -> list[dict]:
        return [
            {
                "key"    : "synthesis",
                "name"   : "Dataset Synthesis",
                "script" : "generate_dataset",
                "blurb"  : "Write the bundled example dataset from nothing but a seed: smooth random Gaussian-mixture parameter fields, the profiles they render along z, a primary and secondary channel stack with its derived channels, an auxiliary field, the reference cube, and the parameter run that supervises training.",
                "stages" : ["Smooth parameter fields", "Profile rendering", "Channel synthesis", "Best-K scoring", "Artifact write"],
            },
            {
                "key"    : "param",
                "name"   : "Parameter Extraction",
                "script" : "extract_params",
                "blurb"  : "Fit a K-Gaussian mixture per z profile to build the supervised parameter targets, with penalised model-order selection.",
                "stages" : ["Profile conditioning", "Prominence init (CPU)", "Adam mixture fit (JAX GPU)", "Penalised best-K", "Mean-sorted target and diagnostics"],
            },
            {
                "key"    : "dataset",
                "name"   : "Dataset (Loaders)",
                "script" : None,
                "blurb"  : "Turn processed artifacts into PyTorch DataLoaders, consumed directly by training.",
                "stages" : ["Crop and split", "Patch extraction", "Augmentation", "Normalization"],
            },
            {
                "key"    : "training",
                "name"   : "Training (Supervised backbone)",
                "script" : "train_backbone",
                "blurb"  : "Supervised backbone loop: one forward pass to per-pixel Gaussian parameters, a weight-normalised curve, parameter and curve losses, AdamW with warmup, cosine annealing and a two-phase loss curriculum, and best-epoch checkpointing.",
                "stages" : ["Curve reconstruction", "Composite loss", "AdamW step", "Schedule and curriculum", "Eval and checkpoint"],
            },
            {
                "key"    : "inference",
                "name"   : "Inference",
                "script" : "infer_backbone",
                "blurb"  : "Sliding-window prediction, overlap-add curve stitching and centrality-selected parameter stitching, then the full metric suite and report generation.",
                "stages" : ["Windowed predict", "Cube stitch", "Metrics", "Stratified errors", "Report and figures"],
            },
            {
                "key"    : "benchmark",
                "name"   : "Benchmark (Capacity-matched)",
                "script" : "benchmark",
                "blurb"  : "Fairly compare backbone architectures at equal capacity: scale every model to the reference parameter budget, size its batch to the VRAM budget under a real training probe, train and infer identically across a seed and loss-component grid, then aggregate and rank.",
                "stages" : ["Capacity match to reference", "Max-batch probe (real loss)", "Seed x loss sweep", "Train & infer each unit", "Aggregate & leaderboard"],
            },
            {
                "key"    : "cross_validate",
                "name"   : "Cross-Validation (K-fold)",
                "script" : "cross_validate",
                "blurb"  : "Guard-banded spatial K-fold cross-validation: partition the scene row into folds, train one model per fold (optionally per seed), infer on held-out val and test bands, and aggregate metrics into cross-fold mean and std.",
                "stages" : ["Fold plan & guard", "Per-fold training", "Held-out inference", "Seed & cross-fold aggregate", "Report & summary"],
            },
            {
                "key"    : "tuning",
                "name"   : "Tuning",
                "script" : "tune",
                "blurb"  : "Optuna joint hyperparameter search: TPE density-ratio proposals with constant-liar parallelism and median pruning, distributed across GPUs and resumable in chunks.",
                "stages" : ["Search space", "Joint search (TPE)", "Trial and pruning", "Best-config export"],
            },
            {
                "key"    : "feed_tuner",
                "name"   : "Feed Tuner (DataLoader)",
                "script" : "tune_dataloader",
                "blurb"  : "Sweep DataLoader settings (batch size, workers, prefetch, pin-memory) against the real training workload to find the configuration that keeps the GPU fed.",
                "stages" : ["Feed target (real dataset + model)", "Batch x workers sweep", "Loader / ceiling / end-to-end probe", "Saturate & recommend", "Prefetch-pin refine & report"],
            },
        ]
