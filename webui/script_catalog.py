
from __future__ import annotations

from project_paths          import ProjectPaths
from script_config_resolver import ScriptConfigResolver


class ScriptCatalog:

    META = {
        "generate_dataset": {
            "title"     : "Generate Dataset",
            "category"  : "Data",
            "purpose"   : "Write the bundled synthetic dataset the template trains on: a few smooth random input fields with noise, target fields given by a known nonlinear and spatially local function of them, and train, validation and test row bands, all in one dataset.json with inputs.npy and targets.npy. CPU only, a few seconds.",
        },
        "train_backbone": {
            "title"     : "Train Backbone",
            "category"  : "Training",
            "purpose"   : "Train one supervised backbone end to end, or fan out trials across GPUs: rectangular patch-shape grids under a constant pixel budget, one run per listed architecture, augmentation ladders, training-stability arms, weight-EMA ladders, or mixed-precision on/off pairs.",
        },
        "infer_backbone": {
            "title"     : "Infer Backbone",
            "category"  : "Inference",
            "purpose"   : "Backbone inference: sliding-window prediction, stitched target maps, per-channel metrics and reports. Scores every backbone run found under the path given, whether that is one run, a sweep, an experiment or a whole runs tree, and leaves alone the runs already scored on the split.",
        },
        "benchmark": {
            "title"     : "Benchmark",
            "category"  : "Experiments",
            "purpose"   : "Benchmark architecture trade-offs, each model at its own configured size, sweeping every permutation of architecture and selected loss component (one architecture + one loss component per run). Point resume_from at an existing run to finish its missing units with the configuration it recorded.",
        },
        "cross_validate": {
            "title"     : "Cross-validate",
            "category"  : "Experiments",
            "purpose"   : "Run K-fold cross-validation for a model across row folds, training and inferring each fold across GPUs. Point resume_from at an existing run to finish its missing folds, seeds and inference splits with the configuration it recorded.",
        },
        "tune": {
            "title"     : "Tune",
            "category"  : "Experiments",
            "purpose"   : "Run the Optuna hyperparameter search in chunks. Point resume_from at an existing tuning run to top its studies up with the configuration it recorded.",
        },
        "compare_trials": {
            "title"     : "Compare Trials",
            "category"  : "Analysis",
            "purpose"   : "Compare inference results across multiple training runs: metrics leaderboard, side-by-side figures, and optional GIF comparison. A trial with nested seed runs enters the comparison as one entry with seed-mean metrics and sample-std annotations; figures come from a representative seed.",
        },
        "compare_runs": {
            "title"     : "Compare Benchmark Runs",
            "category"  : "Analysis",
            "purpose"   : "Rebuild the benchmark comparison report for an existing benchmark run: seed-aggregated leaderboard, without re-running training or inference.",
        },
        "compare_seeds": {
            "title"     : "Compare Seeds",
            "category"  : "Analysis",
            "purpose"   : "Aggregate the existing inference results of the seed runs nested inside a multi-seed training directory into a seed-comparison report: across-seed mean ± std of every scalar metric with per-seed columns and links to each seed's full report. Select one or more group directories and each is compared in isolation, reports generated in sequence — pure report generation from each run's latest (or a chosen) inference, without re-running inference.",
        },
        "xray_weights": {
            "title"     : "X-Ray Weights",
            "category"  : "Analysis",
            "purpose"   : "Scan a runs directory, select one or more checkpoints, and diagnose each: dead weights, near-uniform layers, rank collapse, dead neurons, exploded or non-finite values, normalisation-scale collapse, and initialisation anomalies. Writes a console report, a markdown report with per-tensor plots, and a JSON of all metrics inside each run directory.",
        },
        "xray_activations": {
            "title"     : "X-Ray Activations",
            "category"  : "Analysis",
            "purpose"   : "X-ray the activations of trained backbone runs on real data. Hook every leaf module, run a few batches, and diagnose dead layers, dead channels, exploding and constant activations. Writes depth profiles, per-layer histograms for flagged layers, a JSON of all statistics, and a markdown report inside each run directory. The static-weight sibling of this check is X-Ray Weights.",
        },
        "measure_receptive_field": {
            "title"     : "Receptive Field",
            "category"  : "Analysis",
            "purpose"   : "Measure the effective receptive field of trained backbone runs on real data. Probe pixels spread across the split region, backpropagate the centre-pixel output magnitude to the input window, and report ERF sigma per axis, the cumulative gradient-mass ladder, a log-scale ERF heatmap, and a markdown report inside each run directory.",
        },
        "attribute_inputs": {
            "title"     : "Attribute Inputs",
            "category"  : "Analysis",
            "purpose"   : "Attribute the predictions of trained backbone runs to their input channels. Gradient attribution shares |∂ output/∂ input| across channels per output family (amp, mu, sigma) on probe windows, and an occlusion pass zeroes one normalized channel at a time to measure the prediction shift. Answers which channels and which channel kinds (amplitude vs derived-channel phase vs Aux) the model actually uses.",
        },
        "probe_layers": {
            "title"     : "Probe Layers",
            "category"  : "Analysis",
            "purpose"   : "Probe trained backbone runs layer by layer with linear readouts. Ridge probes on sampled real pixels predict the GT active Gaussian count and the dominant component position along z from each layer's features; the held-out R² by depth shows where each quantity becomes linearly decodable inside the network.",
        },
        "map_loss_landscape": {
            "title"     : "Loss Landscape",
            "category"  : "Analysis",
            "purpose"   : "Map the curve-MSE landscape around the trained weights of backbone runs along two filter-normalized random directions (Li et al. 2018). Writes 1D cuts, a 2D log-scale contour, and a sharpness scalar per direction into each run directory. The physical curve objective keeps runs trained under different losses comparable, replacing the synthetic shift/scale landscape.",
        },
        "stress_inputs": {
            "title"     : "Stress Inputs",
            "category"  : "Analysis",
            "purpose"   : "Stress trained backbone runs with controlled input degradation. Curve-MSE-vs-severity under gaussian noise on the normalized inputs and under whole-channel dropout (secondary and derived channels zeroed, averaged over random channel subsets). Complements the secondaries experiment with a post-hoc robustness axis that needs no retraining.",
        },
        "animate_training": {
            "title"     : "Animate Training",
            "category"  : "Analysis",
            "purpose"   : "Animate how a backbone run learned. Re-predicts probe pixels from every epoch snapshot saved during training (enable snapshot_every_n_epochs on the training entry) and renders one GIF per pixel of the prediction converging onto the fixed ground-truth profile.",
        },
        "export_tensorboard_plots": {
            "title"     : "Export TensorBoard Plots",
            "category"  : "Analysis",
            "purpose"   : "Scan a runs directory for training runs with TensorBoard event logs, select one or more, and export every scalar series as a publication-quality figure inside each run directory, mirroring the tag hierarchy as folders. Train and validation series of the same metric share one figure. When sibling seed runs of one trial are selected, the trial directory additionally receives one overlay figure per metric with every seed's curve.",
        },
        "export_paper_figures": {
            "title"     : "Export Paper Figures",
            "category"  : "Analysis",
            "purpose"   : "Pack inference figures from selected runs into the publication figures directory under stable names (<run>__<figure-path>) with a JSON manifest of every source. Render in paper style first by running inference with figure_style=paper; the pack then feeds a paper directory without hand-copying.",
        },
        "export_onnx": {
            "title"     : "Export ONNX",
            "category"  : "Analysis",
            "purpose"   : "Rebuild each selected backbone run's model from its persisted config, load the best checkpoint and export the forward as an ONNX graph with a dynamic batch axis, plus a manifest tying the graph to its checkpoint. Needs the onnx package (pip install -e .[export]).",
        },
        "profile_training": {
            "title"     : "Profile Training Steps",
            "category"  : "Analysis",
            "purpose"   : "Record full forward, backward and optimizer steps of one backbone under torch.profiler: a chrome trace to open in chrome://tracing or Perfetto, a per-operator time table and a machine-readable summary of the top operators.",
        },
        "export_result_tables": {
            "title"     : "Export Result Tables",
            "category"  : "Analysis",
            "purpose"   : "Export one table row per run with finished inference: a CSV carrying every scalar metric at full precision, plus LaTeX (booktabs) and markdown tables with the selected metric columns at fixed precision, ready to paste into a paper.",
        },
        "collect_reports": {
            "title"     : "Collect Reports",
            "category"  : "Analysis",
            "purpose"   : "Scan a runs directory for training runs with inference reports, select one or more, and gather each run's report into a single collector directory, renamed after the run (seed runs as <trial>_seed<N>). Selecting a trial directory collects its seed-comparison report when one exists (run Compare Seeds first), and otherwise falls back to every seed run nested beneath it. Image links are rewritten to absolute paths into the original run figures, or embedded to make each report self-contained.",
        },
        "mail_experiment_reports": {
            "title"     : "Mail Results",
            "category"  : "Analysis",
            "purpose"   : "Email gathered results, one bundle per source, with the target tab choosing what is mailed. Experiments: any path is scanned for experiments by the trial runs they hold (a sweep of trials, a cross-validation run, a benchmark run, a sequence workspace or a whole runs tree), and each one is mailed as one message carrying the seed-aggregate report of every trial, the trial-comparison reports, a cross-validation run's cross-fold report and the configuration it ran with; a missing seed aggregate, comparison or cross-fold report is written first. Runs: the runs picked at or below a path travel whole, everything each one holds (configuration, metadata, logs, docs, TensorBoard events, reports and figures) minus the checkpoints, the reconstructed cubes and any file above the per-file cap. Datasets: everything Generate Dataset wrote (layout and metadata). Run and dataset bundles are packed into zip volumes that each stay under the attachment cap, sent as one message per volume, and the arrays under data/ are left out unless you name them in the gathered paths. The SMTP password comes from an environment variable, and a dry run stages without sending.",
        },
    }

    TOPICS = {
        "compare_trials"           : "trials",
        "compare_runs"             : "trials",
        "compare_seeds"            : "trials",
        "xray_weights"             : "diagnostics",
        "xray_activations"         : "diagnostics",
        "measure_receptive_field"  : "diagnostics",
        "attribute_inputs"         : "diagnostics",
        "probe_layers"             : "diagnostics",
        "map_loss_landscape"       : "diagnostics",
        "stress_inputs"            : "diagnostics",
        "animate_training"         : "diagnostics",
        "export_tensorboard_plots" : "exports",
        "export_paper_figures"     : "exports",
        "export_result_tables"     : "exports",
        "export_onnx"              : "exports",
        "profile_training"         : "diagnostics",
        "collect_reports"          : "exports",
        "mail_experiment_reports"  : "exports",
    }

    GROUPS = {
        "compare": {
            "title"    : "Compare",
            "category" : "Analysis",
            "purpose"  : "Compare a family of trials side by side. Pick what to compare: inference results across training runs, benchmark runs, or seed replicas of one training.",
            "members"  : [
                ("compare_trials", "Inference Trials"),
                ("compare_runs",   "Benchmark Runs"),
                ("compare_seeds",  "Seed Runs"),
            ],
        },
    }

    def __init__(self, paths: ProjectPaths, resolver: ScriptConfigResolver) -> None:
        self.paths    = paths
        self.resolver = resolver

    def _group_of(self, key: str) -> tuple[str | None, dict | None, str | None]:
        for group_key, group in self.GROUPS.items():
            for member_key, label in group["members"]:
                if member_key == key:
                    return group_key, group, label
        return None, None, None

    def _variants(self, group: dict) -> list[dict]:
        variants = []
        for member_key, label in group["members"]:
            if self.paths.script_entry(member_key)["path"].exists():
                variants.append({"key": member_key, "label": label})
        return variants

    def list_scripts(self) -> list[dict]:
        entries = []
        for key in self.META:
            spec = self.paths.script_entry(key)
            if not spec["path"].exists():
                continue

            meta  = self.META[key]
            entry = self.resolver.entry_config(key)

            group_key, group, label = self._group_of(key)

            entries.append({
                "key"            : key,
                "file"           : spec["rel"],
                "title"          : meta["title"],
                "category"       : meta["category"],
                "purpose"        : meta["purpose"],
                "config_class"   : entry["class"] if entry else None,
                "topic"          : self.TOPICS.get(key),
                "group"          : group_key,
                "variant"        : label,
                "group_title"    : group["title"] if group else None,
                "group_category" : group["category"] if group else None,
                "group_purpose"  : group["purpose"] if group else None,
                "variants"       : self._variants(group) if group else [],
            })
        return entries

    def get_script(self, key: str) -> dict | None:
        spec = self.paths.script_entry(key)
        if not spec["path"].exists():
            return None

        meta    = self.META.get(key, {"title": key, "category": "Other", "purpose": ""})
        source  = spec["path"].read_text(encoding="utf-8")
        entry   = self.resolver.entry_config(key)
        command = f"python {spec['rel']}"

        group_key, group, label = self._group_of(key)

        return {
            "key"          : key,
            "file"         : spec["rel"],
            "title"        : meta["title"],
            "category"     : meta["category"],
            "purpose"      : meta["purpose"],
            "source"       : source,
            "language"     : "python",
            "config_class" : entry["class"] if entry else None,
            "command"      : command,
            "preferred"    : self.paths.preferred_interpreter(self.paths.discover_interpreters()),
            "group"        : group_key,
            "group_title"  : group["title"] if group else None,
            "variant"      : label,
            "variants"     : self._variants(group) if group else [],
        }
