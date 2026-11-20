"use strict";

class SeedAxisNote {
  static list(view, leaf) {
    if (!leaf) return [];
    try {
      const parsed = PythonLiteral.parse(view._effective(leaf));
      if (Array.isArray(parsed)) return parsed;
    } catch (e) {}
    return [];
  }

  static suffix(view, leaf, trials) {
    const n = SeedAxisNote.list(view, leaf).length;
    if (n <= 1) return "";
    if (trials === null) return ` x ${n} seeds per trial`;

    const total = trials * n;
    return ` x ${n} seeds = ${total} run${total === 1 ? "" : "s"}`;
  }

  static append(namesEl, view, leaf) {
    const seeds = SeedAxisNote.list(view, leaf);
    if (seeds.length <= 1) return;

    const note       = document.createElement("span");
    note.className   = "exp-name exp-name--more";
    note.textContent = `each trial nests seeds ${seeds.join(", ")} as <trial>/seed<N>`;
    namesEl.appendChild(note);
  }
}


class WeightSweepPanel {

  constructor(builder, spec) {
    this.builder   = builder;
    this.spec      = spec;
    this.el        = null;
    this.catalogEl = null;
    this.weightsEl = null;
    this.baseEl    = null;
  }

  get group() {
    return this.builder[this.spec.group];
  }

  leaf(field) {
    return this.group.get(field);
  }

  components() {
    return this.builder._listOf(this.leaf("components"), (value) => typeof value === "string");
  }

  weights() {
    return this.builder._numbersOf(this.leaf("weights"));
  }

  baselineOn() {
    const leaf = this.leaf("include_baseline");
    return leaf ? this.builder.view._effective(leaf) === "True" : false;
  }

  base() {
    const leaf = this.leaf("base_component");
    if (!leaf) return this.spec.base.fallback;

    const value = this.builder.view._effective(leaf).replace(/^['"]|['"]$/g, "");
    return this.spec.catalog.options.some((option) => option.key === value) ? value : this.spec.base.fallback;
  }

  emitComponents(components) {
    const leaf = this.leaf("components");
    if (!leaf) return;

    this.builder.view._setValue(leaf, PythonLiteral.render(components));
    this.repaint();
  }

  emitWeights(weights) {
    const leaf = this.leaf("weights");
    if (!leaf) return;

    const clean = weights.map((value) => Number(value)).filter((value) => Number.isFinite(value) && value > 0);
    this.builder.view._setValue(leaf, PythonLiteral.render(clean));
    this.repaint();
  }

  setBase(name) {
    const leaf = this.leaf("base_component");
    if (!leaf) return;

    this.builder.view._setValue(leaf, name);

    const components = this.components().filter((value) => value !== name);
    if (components.length) this.emitComponents(components);
    else this.repaint();
  }

  toggleComponent(key) {
    if (this.spec.base && key === this.base()) return;

    const order = this.spec.catalog.options.map((option) => option.key);
    this.builder._toggleInCatalog(order, this.components(), key, (components) => this.emitComponents(components));
  }

  _baseSelect(leaf) {
    const select     = document.createElement("select");
    select.className = "run-select exp-secondary__strategy";
    select.title     = `${this.spec.base.title} (--${leaf.path})`;

    this.spec.catalog.options.forEach((option) => {
      const opt       = document.createElement("option");
      opt.value       = option.key;
      opt.textContent = option.key;
      select.appendChild(opt);
    });
    select.addEventListener("change", () => this.setBase(select.value));

    this.baseEl = select;
    this.builder.view.controls[leaf.path] = { leaf, reset: () => this.repaint() };
    return select;
  }

  _head() {
    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">${this.spec.title}</span>`;

    const baseLeaf = this.spec.base ? this.leaf("base_component") : null;
    if (baseLeaf) head.appendChild(this._baseSelect(baseLeaf));

    const baseWeightLeaf = this.spec.base && this.spec.base.weightField ? this.leaf(this.spec.base.weightField) : null;
    if (baseWeightLeaf) head.appendChild(this.builder._sweepBaseWeight(baseWeightLeaf));

    this.spec.flags.forEach((flag) => {
      const leaf = this.leaf(flag.field);
      if (leaf) head.appendChild(this.builder._patchFlag(leaf, flag.label, flag.hint));
    });

    head.appendChild(LaunchWidgetDom.mini("Add weight", () => {
      const weights = this.weights();
      weights.push(weights.length ? weights[weights.length - 1] : 0.05);
      this.emitWeights(weights);
    }));

    return head;
  }

  _note() {
    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = this.spec.note;
    return note;
  }

  _catalogSection(panel) {
    if (!this.spec.catalog) return;

    const grid     = document.createElement("div");
    grid.className = "exp-builder__names exp-presence__strategies";
    this.catalogEl = grid;

    panel.appendChild(this.builder._subHead(this.spec.catalog.head));
    panel.appendChild(grid);

    const leaf = this.leaf("components");
    if (leaf) this.builder.view.controls[leaf.path] = { leaf, reset: () => this.repaint() };
  }

  _weightsSection(panel) {
    const grid     = document.createElement("div");
    grid.className = "exp-patch__sizes";
    this.weightsEl = grid;

    panel.appendChild(this.builder._subHead("weights"));
    panel.appendChild(grid);

    const leaf = this.leaf("weights");
    if (leaf) this.builder.view.controls[leaf.path] = { leaf, reset: () => this.repaint() };
  }

  paint() {
    if (!this.el) return;

    const base = this.spec.base ? this.base() : null;
    if (this.baseEl) this.baseEl.value = base;

    if (this.catalogEl) {
      this.catalogEl.innerHTML = "";
      const active = this.components();
      this.spec.catalog.options.forEach(({ key, label }) => {
        if (key === base) return;
        const chip = this.builder._toggleChip(label, active.includes(key), this.spec.catalog.onTitle, this.spec.catalog.offTitle, () => this.toggleComponent(key));
        this.catalogEl.appendChild(chip);
      });
    }

    const weights = this.weights();
    this.weightsEl.innerHTML = "";
    weights.forEach((weight, index) => this.weightsEl.appendChild(this.builder._weightChip(weight, index, weights, this.spec.weightTitle, (next) => this.emitWeights(next))));

    this.builder.patchFlagPaints.forEach((paint) => paint());
  }

  repaint() {
    this.paint();
    this.builder._paintSummary();
    this.builder._paintNames();
  }

  build() {
    const panel     = document.createElement("div");
    panel.className = this.spec.panelClass;

    panel.appendChild(this._head());
    panel.appendChild(this._note());
    this._catalogSection(panel);
    this._weightsSection(panel);

    this.el = panel;
    this.builder[this.spec.elProp] = panel;

    this.paint();
    return panel;
  }
}


class ExperimentBuilder {

  static MODES = [
    { group: "loss", key: "curriculum", label: "curriculum",  hint: "warmup x complete cross product, one trial each",
      el        : "columnsEl",
      available : (b) => Boolean(b.warmupLeaf && b.completeLeaf),
      summary   : (b, gpus) => {
        const total = b.variants.warmup.length * b.variants.complete.length;
        return `${b.variants.warmup.length} warmup x ${b.variants.complete.length} complete = ${total} trials${b._seedsSuffix(total)}${gpus}`;
      },
      names     : (b, model) => {
        const names = [];
        b.variants.warmup.forEach((w) => b.variants.complete.forEach((c) => names.push(`${model}_w-${w.label}_c-${c.label}`)));
        return names;
      } },

    { group: "loss", key: "warmup",     label: "single stage", hint: "curriculum disabled, check losses from the catalog and each trains alone as one trial",
      el        : "warmupCatalogEl",
      available : (b) => Boolean(b.warmupLeaf && b.completeLeaf),
      paint     : (b) => b._paintWarmupCatalog(),
      summary   : (b, gpus) => {
        const n = b.variants.warmup.length;
        return `${n} warmup loss${n === 1 ? "" : "es"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b.variants.warmup.map((w) => `${model}_nc-${w.label}`) },

    { group: "loss", key: "pair",       label: "loss pairs",  hint: "one base component plus one candidate second component per weight, curriculum disabled, one trial per pair plus an optional base-only baseline",
      at        : 3,
      el        : "pairEl",
      available : (b) => b.pair.size > 0,
      make      : (b) => b.sweeps.pair.build(),
      paint     : (b) => b.sweeps.pair.paint(),
      summary   : (b, gpus) => {
        const nCand    = b.sweeps.pair.components().length;
        const nWeight  = b.sweeps.pair.weights().length;
        const baseline = b.sweeps.pair.baselineOn();
        const total    = nCand * nWeight + (baseline ? 1 : 0);
        return `${b.sweeps.pair.base()} + ${nCand} candidate${nCand === 1 ? "" : "s"} x ${nWeight} weight${nWeight === 1 ? "" : "s"}${baseline ? " + baseline" : ""} = ${total} trial${total === 1 ? "" : "s"}${b._seedsSuffix(total)}${gpus}`;
      },
      names     : (b, model) => {
        const names = [];
        if (b.sweeps.pair.baselineOn()) names.push(`${model}_pair-baseline`);
        b.sweeps.pair.components().forEach((component) => {
          b.sweeps.pair.weights().forEach((weight) => names.push(`${model}_pair-${component}-w${weight}`));
        });
        return names;
      } },

    { group: "loss", key: "log_energy", label: "log-energy",  hint: "the log-energy curve term swept across weights on top of the standard loss in both curriculum stages, one trial per weight plus an optional no-log-energy baseline",
      at        : 13,
      el        : "logEnergyEl",
      available : (b) => b.logEnergy.size > 0,
      make      : (b) => b.sweeps.log_energy.build(),
      paint     : (b) => b.sweeps.log_energy.paint(),
      summary   : (b, gpus) => {
        const nWeight  = b.sweeps.log_energy.weights().length;
        const baseline = b.sweeps.log_energy.baselineOn();
        const total    = nWeight + (baseline ? 1 : 0);
        return `${nWeight} weight${nWeight === 1 ? "" : "s"}${baseline ? " + baseline" : ""} = ${total} trial${total === 1 ? "" : "s"}${b._seedsSuffix(total)}${gpus}`;
      },
      names     : (b, model) => {
        const names = [];
        if (b.sweeps.log_energy.baselineOn()) names.push(`${model}_logE-baseline`);
        b.sweeps.log_energy.weights().forEach((weight) => names.push(`${model}_logE-w${weight}`));
        return names;
      } },

    { group: "data", key: "secondary",  label: "secondaries", hint: "one trial per secondary-track selection",
      at        : 0,
      el        : "secondaryEl",
      available : (b) => b.secondary.size > 0,
      make      : (b) => b._secondaryPanel(),
      paint     : (b) => b._paintSecondary(),
      summary   : (b, gpus) => {
        const strategy = b._strategy();
        const fixed    = b._secondaryFixed();
        const held     = fixed.length ? ` + ${fixed.length} fixed (${fixed.join(", ")})` : "";
        if (strategy === "exhaustive") return `exhaustive, all combinations of exactly ${b._secondaryEffective("n_secondaries")} secondaries${held}${b._seedsSuffix(null)}${gpus}`;
        const sampled  = strategy === "uniform" || strategy === "gaussian";
        const count    = sampled ? `${b._secondaryEffective("n_trials")} trials` : "trial count set by the stack";
        return `${strategy}, ${b._secondaryEffective("n_secondaries")} secondaries${held}, ${count}${b._seedsSuffix(null)}${gpus}`;
      },
      pattern   : (b, model) => ({ text: `${model}_sec-${b._strategy()}-tNN_<labels>`, note: b._secondaryFixed().length ? "one per selection, labels resolved from the dataset stack, each name carrying the fixed channels too" : "one per selection, labels resolved from the dataset stack" }) },

    { group: "data", key: "patch",      label: "patch",       hint: "rectangular grid of (row x range) patch shapes, same model end to end, batch rescaled to a constant pixel budget",
      at        : 1,
      el        : "patchEl",
      available : (b) => b.patch.size > 0,
      make      : (b) => b._patchPanel(),
      paint     : (b) => b._paintPatch(),
      summary   : (b, gpus) => {
        const grid = b._patchGrid();
        if (grid.error)    return `invalid patch grid: ${grid.error}`;
        if (grid.deferred) return `grid resolved at launch: ${grid.deferred}${b._seedsSuffix(null)}${gpus}`;
        const n = grid.row.length * grid.range.length;
        return `${grid.row.length} row x ${grid.range.length} range = ${n} patch shape${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => {
        const grid = b._patchGrid();
        if (!grid.row) return [];
        const pad = (value) => String(value).padStart(3, "0");
        return grid.row.flatMap((row) => grid.range.map((col) => `${model}_p-${pad(row)}x${pad(col)}`));
      } },

    { group: "arch", key: "presence",   label: "slot presence", hint: "active-normalization x presence-balance matrix (none, A, B, AB), curriculum disabled, one trial per cell",
      at        : 4,
      el        : "presenceEl",
      available : (b) => Boolean(b.presenceTrialsLeaf),
      make      : (b) => b._presencePanel(),
      paint     : (b) => b._paintPresence(),
      summary   : (b, gpus) => {
        const n = b._presenceCells().length;
        return `${n} presence cell${n === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b._presenceCells().map((cell) => `${model}_pr-${cell}`) },

    { group: "data", key: "input",      label: "input channels", hint: "input-channel ablation, one trial per input variant on its own track scope (all tracks or the reduced selection), each variant in a no-Aux and a with-Aux version",
      at        : 5,
      el        : "inputEl",
      available : (b) => Boolean(b.inputTrialsLeaf),
      make      : (b) => b._inputPanel(),
      paint     : (b) => b._paintInput(),
      summary   : (b, gpus) => {
        const trials = b._inputTrials();
        const n      = Object.keys(trials).length;
        const nAll   = Object.values(trials).filter((spec) => spec && spec.tracks === "all").length;
        return `${n} input variant${n === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"} (${nAll} all tracks, ${n - nAll} reduced)${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b._inputCells().map((cell) => `${model}_in-${cell}`) },

    { group: "arch", key: "context",    label: "context ladder", hint: "one trial per backbone architecture on the shared base config, walking the spatial-context ladder",
      at        : 6,
      el        : "contextEl",
      available : (b) => Boolean(b.contextTrialsLeaf),
      make      : (b) => b._contextPanel(),
      paint     : (b) => b._paintContext(),
      summary   : (b, gpus) => {
        const n = b._contextTrials().length;
        return `${n} backbone${n === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b) => b._contextTrials().map((rung) => `${rung.backbone}_ctx-${rung.label}`) },

    { group: "arch", key: "reach",      label: "reach", hint: "size-matched architecture comparison at fixed reach: a 33x33 local CNN against a UNet, same 32x32 patch and same parameter count, one trial per arm",
      at        : 7,
      el        : "reachEl",
      available : (b) => b.reach.size > 0,
      make      : (b) => b._reachPanel(),
      paint     : (b) => b._paintReach(),
      summary   : (b, gpus) => {
        const n = b._reachRungs().length;
        return `${n} size-matched arm${n === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"} on a shared ${b._reachPatch()} patch${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b) => b._reachRungs().map((rung) => `${rung.backbone}_reach-${rung.label}`) },

    { group: "arch", key: "head",       label: "head x matching", hint: "one trial per output-head x parameter-matching pair on the backbone this entry names",
      at        : 8,
      el        : "headEl",
      available : (b) => b.headTrials.size > 0,
      make      : (b) => b._headPanel(),
      paint     : (b) => b._paintHead(),
      summary   : (b, gpus) => {
        const nHeads     = b._headList("heads").length;
        const nMatchings = b._headList("matchings").length;
        const total      = nHeads * nMatchings;
        return `${nHeads} head${nHeads === 1 ? "" : "s"} x ${nMatchings} matching${nMatchings === 1 ? "" : "s"} = ${total} trial${total === 1 ? "" : "s"} on ${b._headBackbone()}${b._seedsSuffix(total)}${gpus}`;
      },
      names     : (b) => {
        const names    = [];
        const backbone = b._headBackbone();
        b._headList("heads").forEach((headKey) => {
          b._headList("matchings").forEach((matching) => names.push(`${backbone}_hm-${headKey}-${matching}`));
        });
        return names;
      } },

    { group: "data", key: "augmentation", label: "augmentation", hint: "cumulative augmentation ladder on the shared base config: no augmentation, then one rung per added augmentation, then + input noise at each configured strength",
      at        : 9,
      el        : "augEl",
      available : (b) => b.augTrials.size > 0,
      make      : (b) => b._augmentationPanel(),
      paint     : (b) => b._paintAugmentation(),
      summary   : (b, gpus) => {
        const n = 3 + b._augNoiseStds().length;
        return `3 ladder rungs + ${b._augNoiseStds().length} noise rung${b._augNoiseStds().length === 1 ? "" : "s"} = ${n} trials${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b._augNames().map((rung) => `${model}_${rung}`) },

    { group: "training", key: "clamp", label: "output clamp", hint: "output-clamp ladder on the shared base config: no clamp, hard clamp, then one leaky rung per slope pair; the clamp bounds come from the run's normalization config",
      at        : 10,
      el        : "clampEl",
      available : (b) => b.clampTrials.size > 0,
      make      : (b) => b._clampPanel(),
      paint     : (b) => b._paintClamp(),
      summary   : (b, gpus) => {
        const n = 2 + b._clampSlopes().length;
        return `none + hard + ${b._clampSlopes().length} leaky = ${n} trials${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b._clampNames().map((rung) => `${model}_${rung}`) },

    { group: "training", key: "stability", label: "stability", hint: "training-stability arms: no lr warmup and no gradient clipping, then lr warmup alone per fraction of the total optimizer steps, then gradient clipping alone per norm threshold, then the first warmup fraction paired with each threshold",
      at        : 11,
      el        : "stabilityEl",
      available : (b) => b.stabilityTrials.size > 0,
      make      : (b) => b._stabilityPanel(),
      paint     : (b) => b._paintStability(),
      summary   : (b, gpus) => {
        const n = 1 + b._stabilitySteps().length + 2 * b._stabilityNorms().length;
        return `bare + ${b._stabilitySteps().length} warmup + ${b._stabilityNorms().length} clipping + ${b._stabilityNorms().length} both = ${n} trials${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b._stabilityNames().map((rung) => `${model}_${rung}`) },

    { group: "training", key: "ema", label: "weight EMA", hint: "weight-EMA arms on the shared base config: raw weights, then at each configured decay an exponential moving average with the plain decay and one with the decay warmup ramp",
      at        : 15,
      el        : "emaEl",
      available : (b) => b.emaTrials.size > 0,
      make      : (b) => b._emaPanel(),
      paint     : (b) => b._paintEma(),
      summary   : (b, gpus) => {
        const n = 1 + 2 * b._emaDecays().length;
        return `off + ${b._emaDecays().length} raw + ${b._emaDecays().length} decay-warmup rung${b._emaDecays().length === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b._emaNames().map((rung) => `${model}_${rung}`) },

    { group: "training", key: "amp", label: "mixed precision", hint: "mixed-precision pair on the shared base config: autocast off, then autocast on, nothing else changes",
      at        : 16,
      el        : "ampEl",
      available : () => true,
      make      : (b) => b._ampPanel(),
      paint     : () => {},
      summary   : (b, gpus) => `autocast off and on = 2 trials${b._seedsSuffix(2)}${gpus}`,
      names     : (b, model) => ["amp-0-off", "amp-1-on"].map((rung) => `${model}_${rung}`) },

  ];

  static MODE_GROUPS = [
    { key: "loss",     label: "loss" },
    { key: "data",     label: "data" },
    { key: "arch",     label: "architecture" },
    { key: "training", label: "training" },
  ];

  static STABILITY_WARMUP_MODES = ["linear", "cosine", "exponential", "polynomial"];
  static STABILITY_CLIP_MODES   = ["fixed", "adaptive_percentile", "adaptive_mean_std"];

  static HEAD_OPTIONS = [
    { key: "conv",         label: "conv" },
    { key: "multihead",    label: "multihead" },
    { key: "per_gaussian", label: "per gaussian" },
    { key: "set_pred",     label: "set pred" },
  ];

  static MATCHING_OPTIONS = [
    { key: "sorted_gt", label: "sorted gt" },
    { key: "hungarian", label: "hungarian" },
  ];

  static PAIR_COMPONENTS = [
    "mse_curve", "l1_curve", "huber_curve", "charbonnier_curve", "cosine_curve", "log_energy_curve",
    "total_power_relerr", "moments",
    "param_huber", "param_mse", "smoothness_tv", "param_l1",
  ];

  static SWEEPS = {
    log_energy: {
      group       : "logEnergy",
      elProp      : "logEnergyEl",
      panelClass  : "exp-secondary exp-patch exp-physics",
      title       : "log-energy weight sweep",
      note        : "each weight trains the standard loss plus the log-energy curve term at that weight, enabled in both curriculum stages so the energy supervision accompanies the cosine term from epoch 0",
      flags       : [
        { field: "include_baseline",   label: "baseline",           hint: "prepend one run of the plain standard loss without the log-energy term, as the reference" },
        { field: "replace_covariance", label: "replace covariance", hint: "drop moments from every arm including the baseline, so the sweep tests log-energy as the replacement for the covariance tail repair rather than an addition" },
      ],
      catalog     : null,
      weightTitle : "log-energy curve weight applied in both curriculum stages",
      base        : null,
    },
    pair: {
      group       : "pair",
      elProp      : "pairEl",
      panelClass  : "exp-secondary exp-patch exp-pair",
      title       : "loss pair sweep",
      note        : "each checked component joins the base component at each weight as a two-term loss with every other term disabled; curriculum stays off so the pair applies from epoch 0",
      flags       : [{ field: "include_baseline", label: "baseline", hint: "prepend one run that trains the base component alone, as the reference" }],
      catalog     : { head: "second components", options: ExperimentBuilder.PAIR_COMPONENTS.map((name) => ({ key: name, label: name })), onTitle: "Click to drop this component from the sweep", offTitle: "Click to test this component as the second term" },
      weightTitle : "weight of the second component",
      base        : { fallback: "param_l1", title: "base component held in every run", weightField: "base_weight" },
    },
  };

  static STRATEGIES = [
    { key: "uniform",     note: "each trial samples n_secondaries distinct tracks uniformly, n_trials distinct sets" },
    { key: "gaussian",    note: "indices drawn from Normal(mean, sigma) over the secondary list, mean and sigma required" },
    { key: "consecutive", note: "block of n_secondaries consecutive tracks, advanced block_step per trial until the stack ends" },
    { key: "spaced",      note: "n_secondaries tracks spaced 'spacing' apart, advanced block_step per trial until the stack ends" },
    { key: "exhaustive",  note: "brute force: every combination of exactly n_secondaries tracks, one run each; run count grows combinatorially with the stack" },
  ];

  static SECONDARY_FIELDS = [
    { key: "n_secondaries", strategies: ["uniform", "gaussian", "consecutive", "spaced", "exhaustive"] },
    { key: "fixed",         strategies: ["uniform", "gaussian", "consecutive", "spaced", "exhaustive"],
      multi: { kind: "multi", wide: true, placeholder: "add channel label, Enter", empty: "no fixed channel, every secondary is swept" } },
    { key: "n_trials",      strategies: ["uniform", "gaussian"] },
    { key: "seed",          strategies: ["uniform", "gaussian"] },
    { key: "mean",          strategies: ["gaussian"] },
    { key: "sigma",         strategies: ["gaussian"] },
    { key: "block_step",    strategies: ["consecutive", "spaced"] },
    { key: "spacing",       strategies: ["spaced"] },
  ];

  constructor(view, byPath) {
    this.view         = view;
    this.trialsLeaf   = byPath.get("trials_enabled");
    this.modeLeaf     = byPath.get("trials_mode");
    this.warmupLeaf   = byPath.get("warmup_losses");
    this.completeLeaf = byPath.get("complete_losses");
    this.modelLeaf    = byPath.get("backbone_name");
    this.gpusLeaf     = byPath.get("gpus");
    this.seedsLeaf    = byPath.get("seeds");

    this.presenceTrialsLeaf = byPath.get("presence_trials");

    this.inputTrialsLeaf   = byPath.get("input_trials");
    this.contextTrialsLeaf = byPath.get("context_trials");

    this.secondary       = new Map();
    this.patch           = new Map();
    this.pair            = new Map();
    this.logEnergy       = new Map();
    this.reach           = new Map();
    this.headTrials      = new Map();
    this.augTrials       = new Map();
    this.clampTrials     = new Map();
    this.stabilityTrials = new Map();
    this.emaTrials       = new Map();
    byPath.forEach((leaf) => {
      if (leaf.section === "secondary_trials")     this.secondary.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "patch_trials")         this.patch.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "pair_trials")          this.pair.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "log_energy_trials")    this.logEnergy.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "reach_trials")         this.reach.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "head_trials")          this.headTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "augmentation_trials")  this.augTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "clamp_trials")         this.clampTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "stability_trials")     this.stabilityTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "ema_trials")           this.emaTrials.set(leaf.path.split(".").pop(), leaf);
    });

    this.claimed = ["trials_enabled"];
    if (this.warmupLeaf)   this.claimed.push("warmup_losses");
    if (this.completeLeaf) this.claimed.push("complete_losses");
    if (this.modeLeaf) this.claimed.push("trials_mode");
    this.secondary.forEach((leaf) => this.claimed.push(leaf.path));
    this.patch.forEach((leaf) => this.claimed.push(leaf.path));
    this.pair.forEach((leaf) => this.claimed.push(leaf.path));
    this.logEnergy.forEach((leaf) => this.claimed.push(leaf.path));
    this.reach.forEach((leaf) => this.claimed.push(leaf.path));
    this.headTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.augTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.clampTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.stabilityTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.emaTrials.forEach((leaf) => this.claimed.push(leaf.path));
    if (this.presenceTrialsLeaf)   this.claimed.push(this.presenceTrialsLeaf.path);
    if (this.inputTrialsLeaf)      this.claimed.push(this.inputTrialsLeaf.path);
    if (this.contextTrialsLeaf)    this.claimed.push(this.contextTrialsLeaf.path);

    this.terms          = this._termCatalog();
    this.variants       = { warmup: [], complete: [] };
    this.lists          = {};
    this.summaryEl      = null;
    this.namesEl        = null;
    this.hintEl         = null;
    this.root           = null;
    this.columnsEl      = null;
    this.columnEls      = {};
    this.secondaryEl    = null;
    this.strategySelect = null;
    this.strategyNoteEl = null;
    this.secondaryRows  = [];
    this.patchEl         = null;
    this.patchFlagPaints = [];
    this.pairEl              = null;
    this.logEnergyEl         = null;
    this.sweeps              = {};
    Object.entries(ExperimentBuilder.SWEEPS).forEach(([key, spec]) => { this.sweeps[key] = new WeightSweepPanel(this, spec); });
    this.warmupCatalogEl     = null;
    this.warmupCatalogGridEl = null;
    this.warmupCountEl       = null;
    this.warmupCustomEl      = null;
    this.warmupCustomHeadEl  = null;
    this.presenceEl     = null;
    this.inputEl        = null;
    this.inputCellsEl   = null;
    this.contextEl      = null;
    this.contextCellsEl = null;
    this.headEl          = null;
    this.headHeadsEl     = null;
    this.headMatchingsEl = null;
    this.augEl           = null;
    this.augStdsEl       = null;
    this.clampEl         = null;
    this.clampPairsEl    = null;
    this.stabilityEl     = null;
    this.stabilitySelects = new Map();
    this.stabilityStepsEl = null;
    this.stabilityNormsEl = null;
    this.emaEl           = null;
    this.emaDecaysEl     = null;
    this.ampEl           = null;
    this.modeButtons    = new Map();
    this.modeEl         = null;
    this._paintSwitch   = null;
  }

  build() {
    this.root           = document.createElement("section");
    this.root.className = "exp-builder";

    const head     = document.createElement("header");
    head.className = "special-head";
    head.innerHTML = `<h3 class="special-head__name">Experiment fan-out</h3>`;

    const summary     = document.createElement("span");
    summary.className = "exp-builder__summary";
    this.summaryEl    = summary;
    head.appendChild(summary);
    head.appendChild(this._trialsSwitch());

    const body     = document.createElement("div");
    body.className = "exp-builder__body";
    if (this.modeLeaf) body.appendChild(this._modePicker());

    if (this.warmupLeaf && this.completeLeaf) {
      const columns     = document.createElement("div");
      columns.className = "exp-builder__columns";
      this.columnsEl    = columns;
      this.columnEls.warmup   = columns.appendChild(this._column("warmup", "warmup losses"));
      this.columnEls.complete = columns.appendChild(this._column("complete", "complete losses"));
      body.appendChild(columns);
    }

    if (this.modeLeaf) {
      if (this.warmupLeaf && this.completeLeaf) body.appendChild(this._warmupCatalogPanel());
      this._panelModes().forEach((mode) => body.appendChild(mode.make(this)));
    }

    const preview     = document.createElement("div");
    preview.className = "exp-builder__preview";
    preview.innerHTML = `<span class="exp-builder__preview-title">trial run names</span>`;
    const names       = document.createElement("div");
    names.className   = "exp-builder__names";
    this.namesEl      = names;
    preview.appendChild(names);
    body.appendChild(preview);

    this.root.appendChild(head);
    this.root.appendChild(body);

    [this.warmupLeaf, this.completeLeaf].filter(Boolean).forEach((leaf) => {
      this.view.controls[leaf.path] = { leaf, reset: () => this._reload() };
    });

    this._reload();
    return this.root;
  }

  refreshFromView() {
    if (!this.summaryEl) return;
    if (this._paintSwitch) this._paintSwitch();
    this._paintMode();
    ExperimentBuilder.MODES.forEach((mode) => { if (mode.paint) mode.paint(this); });
    this._paintSummary();
    this._paintNames();
  }

  _modeEntry(key) {
    return ExperimentBuilder.MODES.find((mode) => mode.key === key && mode.available(this)) || ExperimentBuilder.MODES.find((mode) => mode.key === key);
  }

  _modes() {
    return ExperimentBuilder.MODES.filter((mode) => mode.available(this));
  }

  _panelModes() {
    return this._modes().filter((mode) => mode.make).sort((a, b) => a.at - b.at);
  }

  _mode() {
    const modes = this._modes();
    const first = modes.length ? modes[0].key : "curriculum";
    if (!this.modeLeaf) return first;
    const value = this.view._effective(this.modeLeaf);
    return modes.some((mode) => mode.key === value) ? value : first;
  }

  _strategy() {
    const leaf = this.secondary.get("strategy");
    if (!leaf) return "consecutive";
    const value = this.view._effective(leaf);
    return ExperimentBuilder.STRATEGIES.some((strategy) => strategy.key === value) ? value : "consecutive";
  }

  _modePicker() {
    const leaf     = this.modeLeaf;
    const wrap     = document.createElement("div");
    wrap.className = "exp-picker";
    wrap.title     = `--${leaf.path}`;

    ExperimentBuilder.MODE_GROUPS.forEach((group) => {
      const modes = this._modes().filter((mode) => mode.group === group.key);
      if (!modes.length) return;

      const row     = document.createElement("div");
      row.className = "exp-picker__row";
      row.innerHTML = `<span class="exp-picker__group">${group.label}</span>`;

      const cell     = document.createElement("div");
      cell.className = "exp-picker__modes";
      modes.forEach((mode) => {
        const btn       = document.createElement("button");
        btn.type        = "button";
        btn.className   = "exp-picker__btn";
        btn.textContent = mode.label;
        btn.addEventListener("click", () => {
          this.view._setValue(leaf, mode.key);
          if (this.view._effective(this.trialsLeaf) !== "True") this.view._setValue(this.trialsLeaf, "True");
        });
        this.modeButtons.set(mode.key, btn);
        cell.appendChild(btn);
      });

      row.appendChild(cell);
      wrap.appendChild(row);
    });

    const hint     = document.createElement("p");
    hint.className = "exp-picker__hint";
    this.hintEl    = hint;
    wrap.appendChild(hint);

    this.modeEl = wrap;
    this.view.controls[leaf.path] = { leaf, reset: () => this._paintMode() };
    return wrap;
  }

  _secondaryPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">secondary selection</span>`;

    const strategyLeaf = this.secondary.get("strategy");
    if (strategyLeaf) {
      const select     = document.createElement("select");
      select.className = "run-select exp-secondary__strategy";
      select.title     = `--${strategyLeaf.path}`;
      ExperimentBuilder.STRATEGIES.forEach((strategy) => {
        const opt       = document.createElement("option");
        opt.value       = strategy.key;
        opt.textContent = strategy.key;
        select.appendChild(opt);
      });
      select.addEventListener("change", () => this.view._setValue(strategyLeaf, select.value));
      head.appendChild(select);

      this.strategySelect = select;
      this.view.controls[strategyLeaf.path] = { leaf: strategyLeaf, reset: () => this._paintSecondary() };
    }

    const note     = document.createElement("p");
    note.className = "exp-secondary__note";
    this.strategyNoteEl = note;

    const grid     = document.createElement("div");
    grid.className = "exp-secondary__grid";
    ExperimentBuilder.SECONDARY_FIELDS.forEach((field) => {
      const leaf = this.secondary.get(field.key);
      if (leaf) grid.appendChild(field.multi ? this._secondaryFixedRow(field, leaf) : this._secondaryRow(field, leaf));
    });

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(grid);
    this.secondaryEl = panel;
    return panel;
  }

  _secondaryRow(field, leaf) {
    const row     = document.createElement("div");
    row.className = "cfg-edit__row";
    row.title     = `--${leaf.path}`;

    const name     = document.createElement("div");
    name.className = "cfg-edit__name";
    name.innerHTML = `${field.key}<span>${leaf.type === "none" ? "float" : leaf.type}</span>`;

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input";
    input.type       = "number";
    input.step       = leaf.type === "int" ? "1" : "any";
    input.value      = this.view._effective(leaf) === "None" ? "" : this.view._effective(leaf);
    input.spellcheck = false;
    if (leaf.value === "None") input.placeholder = "required";
    input.addEventListener("input", () => {
      this.view.applyTypedInput(leaf, input);
    });

    row.appendChild(name);
    row.appendChild(input);

    this.secondaryRows.push({ strategies: field.strategies, row });
    this.view.controls[leaf.path] = { leaf, reset: () => {
      input.value = this.view._effective(leaf) === "None" ? "" : this.view._effective(leaf);
      input.classList.toggle("is-dirty", this.view.dirty[leaf.path] !== undefined);
    } };
    return row;
  }

  _secondaryFixedRow(field, leaf) {
    const row     = document.createElement("div");
    row.className = "cfg-edit__row cfg-edit__row--board";
    row.title     = `--${leaf.path}`;

    const name     = document.createElement("div");
    name.className = "cfg-edit__name";
    name.innerHTML = `${field.key}<span>channels in every run</span>`;

    const widget = new window.MultiValueField(this.view, leaf, field.multi).build();

    row.appendChild(name);
    row.appendChild(widget.el);

    this.secondaryRows.push({ strategies: field.strategies, row });
    this.view.controls[leaf.path] = { leaf, reset: () => {
      widget.reset();
      this._paintSecondary();
    } };
    return row;
  }

  _patchPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-patch";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">patch sweep</span>`;

    const ratioLeaf = this.patch.get("stride_ratio");
    if (ratioLeaf) head.appendChild(this._patchRatio(ratioLeaf));

    const budgetLeaf = this.patch.get("constant_pixel_budget");
    if (budgetLeaf) head.appendChild(this._patchFlag(budgetLeaf, "pixel budget", "rescale each trial's batch size and lr_reference_batch_size together so every patch shape sees the same pixels per optimizer step"));

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "each (row x range) shape trains the same model end to end; stride = round(length x stride_ratio) per axis; step 0 derives the admissible step from the backbone pyramid (2^levels) at launch; the pixel budget keeps batch x pixels and the batch-linear LR scale identical across shapes";

    const grid     = document.createElement("div");
    grid.className = "exp-patch__sizes";

    const minimumInput = this._patchGridInput("minimum", "minimum (row, col)", "smallest patch shape in pixels; 0 on an axis starts at one admissible step");
    if (minimumInput) grid.appendChild(minimumInput);

    const maximumInput = this._patchGridInput("maximum", "maximum (row, col)", "largest patch shape in pixels");
    if (maximumInput) grid.appendChild(maximumInput);

    const stepInput = this._patchGridInput("step", "step", "spacing between patch lengths in pixels; 0 derives it from the backbone pyramid at launch");
    if (stepInput) grid.appendChild(stepInput);

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(grid);
    this.patchEl = panel;

    return panel;
  }

  _patchGridInput(key, label, hint) {
    const leaf = this.patch.get(key);
    if (!leaf) return null;

    const wrap     = document.createElement("label");
    wrap.className = "exp-patch__ratio";
    wrap.title     = `${hint} (--${leaf.path})`;
    wrap.innerHTML = `<span>${label}</span>`;

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input";
    input.type       = "text";
    input.spellcheck = false;
    input.value      = this.view._effective(leaf);
    input.addEventListener("input", () => {
      this.view.applyTypedInput(leaf, input);
      this._paintSummary();
      this._paintNames();
    });

    wrap.appendChild(input);
    this.view.controls[leaf.path] = { leaf, reset: () => {
      input.value = this.view._effective(leaf);
      input.classList.toggle("is-dirty", this.view.dirty[leaf.path] !== undefined);
    } };
    return wrap;
  }

  _patchRatio(leaf) {
    const wrap     = document.createElement("label");
    wrap.className = "exp-patch__ratio";
    wrap.title     = `--${leaf.path}`;
    wrap.innerHTML = `<span>stride ratio</span>`;

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input";
    input.type       = "number";
    input.step       = "any";
    input.min        = "0";
    input.max        = "1";
    input.spellcheck = false;
    input.value      = this.view._effective(leaf);
    input.addEventListener("input", () => {
      this.view.applyTypedInput(leaf, input);
    });

    wrap.appendChild(input);
    this.view.controls[leaf.path] = { leaf, reset: () => {
      input.value = this.view._effective(leaf);
      input.classList.toggle("is-dirty", this.view.dirty[leaf.path] !== undefined);
    } };
    return wrap;
  }

  _patchFlag(leaf, label, hint) {
    const wrap     = document.createElement("label");
    wrap.className = "exp-patch__flag";
    wrap.title     = `${hint} (--${leaf.path})`;
    wrap.innerHTML = `<span>${label}</span>`;

    const toggle     = document.createElement("button");
    toggle.type      = "button";
    toggle.className = "switch";
    toggle.setAttribute("role", "switch");
    toggle.innerHTML = `<span class="switch__knob"></span>`;

    const paint = () => {
      const on = this.view._effective(leaf) === "True";
      toggle.classList.toggle("is-on", on);
      toggle.classList.toggle("is-dirty", this.view.dirty[leaf.path] !== undefined);
      toggle.setAttribute("aria-checked", String(on));
    };

    toggle.addEventListener("click", () => {
      this.view._setValue(leaf, this.view._effective(leaf) === "True" ? "False" : "True");
      paint();
    });

    wrap.appendChild(toggle);
    paint();

    this.patchFlagPaints.push(paint);
    this.view.controls[leaf.path] = { leaf, reset: paint };
    return wrap;
  }

  _patchLeafTuple(key) {
    const leaf = this.patch.get(key);
    if (!leaf) return null;

    let raw = null;
    try {
      raw = PythonLiteral.parse(this.view._effective(leaf));
    } catch (e) {
      raw = null;
    }
    if (!Array.isArray(raw) || raw.length !== 2) return null;

    const pair = raw.map((value) => Number(value));
    return pair.every((value) => Number.isFinite(value)) ? pair : null;
  }

  _patchStep() {
    const leaf = this.patch.get("step");
    if (!leaf) return 0;

    const value = Number(this.view._effective(leaf));
    return Number.isFinite(value) ? Math.round(value) : 0;
  }

  _patchGrid() {
    const step = this._patchStep();
    if (step <= 0) return { deferred: "step 0 derives the admissible step from the backbone pyramid (2^levels)" };

    const minimum = this._patchLeafTuple("minimum");
    const maximum = this._patchLeafTuple("maximum");
    if (!minimum || !maximum) return { deferred: "the grid bounds are read from the stack at launch" };

    const axis = (index, label) => {
      const start = minimum[index] > 0 ? minimum[index] : step;
      if (start % step !== 0)       return { error: `minimum ${label}=${start} is not a multiple of the admissible step ${step}` };
      if (maximum[index] < start)   return { error: `maximum ${label}=${maximum[index]} is below the first admissible size ${start}` };

      const sizes = [];
      for (let value = start; value <= maximum[index]; value += step) sizes.push(value);
      return { sizes };
    };

    const row = axis(0, "row");
    const range   = axis(1, "range");
    if (row.error) return row;
    if (range.error)   return range;

    const shapes = row.sizes.length * range.sizes.length;
    if (shapes < 2) return { error: `the grid row [${row.sizes}] x range [${range.sizes}] yields ${shapes} patch shape; a sweep needs at least 2` };

    return { row: row.sizes, range: range.sizes };
  }

  _paintPatch() {
    this.patchFlagPaints.forEach((paint) => paint());
  }

  _sweepBaseWeight(leaf) {
    const wrap     = document.createElement("label");
    wrap.className = "exp-patch__ratio";
    wrap.title     = `weight of the base component in every run (--${leaf.path})`;
    wrap.innerHTML = `<span>base weight</span>`;

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input";
    input.type       = "number";
    input.step       = "any";
    input.min        = "0";
    input.spellcheck = false;
    input.value      = this.view._effective(leaf);
    input.addEventListener("input", () => {
      this.view.applyTypedInput(leaf, input);
    });

    wrap.appendChild(input);
    this.view.controls[leaf.path] = { leaf, reset: () => {
      input.value = this.view._effective(leaf);
      input.classList.toggle("is-dirty", this.view.dirty[leaf.path] !== undefined);
    } };
    return wrap;
  }

  _presenceDefaults() {
    return this._defaultsDict(this.presenceTrialsLeaf);
  }

  _presenceTrials() {
    return this._trialsDict(this.presenceTrialsLeaf);
  }

  _presenceCells() {
    return Object.keys(this._presenceTrials());
  }

  _presencePanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">slot-presence ablation</span>`;
    const reset    = LaunchWidgetDom.mini("reset matrix", () => this._resetPresence());
    reset.classList.add("exp-presence__reset");
    head.appendChild(reset);

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "Toggle cells of the active-normalization x presence-balance matrix; every cell trains with curriculum disabled and repeats across the seeds list. Cell specs live in code (TrialCatalog.presence_trials); use reset matrix to restore the full 2x2.";

    const cellHead       = document.createElement("div");
    cellHead.className   = "exp-presence__sub";
    cellHead.textContent = "matrix cells";

    const cells          = document.createElement("div");
    cells.className      = "exp-builder__names exp-presence__cells";
    this.presenceCellsEl = cells;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(cellHead);
    panel.appendChild(cells);
    this.presenceEl = panel;

    if (this.presenceTrialsLeaf) this.view.controls[this.presenceTrialsLeaf.path] = { leaf: this.presenceTrialsLeaf, reset: () => this._repaintPresence() };

    this._paintPresence();
    return panel;
  }

  _presenceChipText(cell, spec) {
    const parts = [];
    if (spec && spec.use_active_normalization) parts.push("active norm");
    if (spec && spec.presence_balance)         parts.push("presence balance");
    return `${cell} · ${parts.join(" + ") || "both off"}`;
  }

  _paintPresence() {
    if (!this.presenceCellsEl) return;
    this._paintDictCells(this.presenceCellsEl, this._presenceTrials(), this._presenceDefaults(), (cell, spec) => this._presenceChipText(cell, spec), "Click to drop this cell", "Click to add this cell", (cell) => this._togglePresenceCell(cell));
  }

  _togglePresenceCell(cell) {
    if (!this.presenceTrialsLeaf) return;

    const ordered = this._toggleDictCell(this._presenceTrials(), this._presenceDefaults(), cell);
    if (!ordered) return;

    this.view._setValue(this.presenceTrialsLeaf, PythonLiteral.render(ordered));
    this._repaintPresence();
  }

  _resetPresence() {
    if (this.presenceTrialsLeaf) this.view._setValue(this.presenceTrialsLeaf, this.presenceTrialsLeaf.value);
    this._repaintPresence();
  }

  _repaintPresence() {
    this._paintPresence();
    this._paintSummary();
    this._paintNames();
  }

  _inputTrials() {
    return this._trialsDict(this.inputTrialsLeaf);
  }

  _inputCells() {
    return Object.keys(this._inputTrials());
  }

  _inputSpecText(spec) {
    const on    = (key) => spec && Object.prototype.hasOwnProperty.call(spec, key) ? Boolean(spec[key]) : null;
    const amps  = [];
    if (on("use_primary")     !== false) amps.push("primary");
    if (on("use_secondaries") !== false) amps.push("secondaries");

    const parts = [];
    parts.push(`${spec && spec.tracks} tracks`);
    parts.push(amps.length ? `${amps.join("+")} amp` : "no amplitude");
    parts.push(on("use_derived") === false ? "no derived" : "derived");
    if (on("use_aux") !== null) parts.push(on("use_aux") ? "aux" : "no aux");
    return parts.join(", ");
  }

  _inputPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">input-channel ablation</span>`;
    const reset    = LaunchWidgetDom.mini("reset variants", () => this._resetInput());
    reset.classList.add("exp-presence__reset");
    head.appendChild(reset);

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "Each variant toggles which input channels feed the model and scopes its channel list: tracks=all expands the secondaries to every channel the dataset records in data/dataset.json, tracks=reduced keeps the configured selection, and every base variant comes as an explicit no-Aux and with-Aux version. Drop variants to trim the fan-out. Adding variants and channel representations live in code (TrialCatalog.input_trials); use reset variants to restore the default.";

    const cellHead       = document.createElement("div");
    cellHead.className    = "exp-presence__sub";
    cellHead.textContent  = "input variants";

    const cells         = document.createElement("div");
    cells.className     = "exp-builder__names exp-presence__cells";
    this.inputCellsEl   = cells;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(cellHead);
    panel.appendChild(cells);
    this.inputEl = panel;

    if (this.inputTrialsLeaf) this.view.controls[this.inputTrialsLeaf.path] = { leaf: this.inputTrialsLeaf, reset: () => this._repaintInput() };

    this._paintInput();
    return panel;
  }

  _paintInput() {
    if (!this.inputCellsEl) return;
    this.inputCellsEl.innerHTML = "";
    const trials = this._inputTrials();
    Object.keys(trials).forEach((cell) => this.inputCellsEl.appendChild(this._inputCellChip(cell, trials[cell])));
  }

  _inputCellChip(cell, spec) {
    const text = this._inputSpecText(spec);
    return this._cellChip(`${cell} · ${text}`, text, "Remove variant", () => this._removeInputCell(cell));
  }

  _removeInputCell(cell) {
    if (!this.inputTrialsLeaf) return;
    let raw;
    try {
      raw = PythonLiteral.parse(this.view._effective(this.inputTrialsLeaf));
    } catch (e) {
      return;
    }
    if (!raw || typeof raw !== "object" || Object.keys(raw).length <= 1) return;
    delete raw[cell];
    this.view._setValue(this.inputTrialsLeaf, PythonLiteral.render(raw));
    this._repaintInput();
  }

  _resetInput() {
    if (this.inputTrialsLeaf) this.view._setValue(this.inputTrialsLeaf, this.inputTrialsLeaf.value);
    this._repaintInput();
  }

  _repaintInput() {
    this._paintInput();
    this._paintSummary();
    this._paintNames();
  }

  _contextTrials() {
    return this._listOf(this.contextTrialsLeaf, () => true);
  }

  _reachRungs() {
    return this._listOf(this.reach.get("rungs"), (rung) => rung && rung.label && rung.backbone);
  }

  _reachPatch() {
    const leaf = this.reach.get("patch_size");
    if (!leaf) return "";
    try {
      const raw = PythonLiteral.parse(this.view._effective(leaf));
      return Array.isArray(raw) ? `${raw[0]}x${raw[1]}` : "";
    } catch (e) {
      return "";
    }
  }

  _contextPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">context ladder</span>`;
    const reset    = LaunchWidgetDom.mini("reset backbones", () => this._resetContext());
    reset.classList.add("exp-presence__reset");
    head.appendChild(reset);

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "One training run per rung on the shared base config, walking the spatial-context ladder at constant depth: ten-block local CNNs whose 3x3 / 1x1 block mix sets the receptive field (RF 1, 9, 29, 41 by default), every rung size matched to the default UNet. Every rung is a label plus a backbone from the registry and model overrides; add rungs by editing context_trials, use reset backbones to restore the default.";

    const cellHead       = document.createElement("div");
    cellHead.className    = "exp-presence__sub";
    cellHead.textContent  = "backbones";

    const cells         = document.createElement("div");
    cells.className     = "exp-builder__names exp-presence__cells";
    this.contextCellsEl = cells;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(cellHead);
    panel.appendChild(cells);
    this.contextEl = panel;

    if (this.contextTrialsLeaf) this.view.controls[this.contextTrialsLeaf.path] = { leaf: this.contextTrialsLeaf, reset: () => this._repaintContext() };

    this._paintContext();
    return panel;
  }

  _paintContext() {
    if (!this.contextCellsEl) return;
    this.contextCellsEl.innerHTML = "";
    this._contextTrials().forEach((rung) => this.contextCellsEl.appendChild(this._contextChip(rung)));
  }

  _contextChip(rung) {
    return this._cellChip(`${rung.label} · ${rung.backbone}`, null, "Remove rung", () => this._removeContextCell(rung.label));
  }

  _removeContextCell(label) {
    if (!this.contextTrialsLeaf) return;
    const raw = this._contextTrials();
    if (raw.length <= 1) return;
    this.view._setValue(this.contextTrialsLeaf, PythonLiteral.render(raw.filter((entry) => entry.label !== label)));
    this._repaintContext();
  }

  _resetContext() {
    if (this.contextTrialsLeaf) this.view._setValue(this.contextTrialsLeaf, this.contextTrialsLeaf.value);
    this._repaintContext();
  }

  _repaintContext() {
    this._paintContext();
    this._paintSummary();
    this._paintNames();
  }

  _reachPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">reach comparison</span>`;
    const reset    = LaunchWidgetDom.mini("reset arms", () => this._resetReach());
    reset.classList.add("exp-presence__reset");
    head.appendChild(reset);

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "Size-matched architecture comparison at fixed reach: every arm trains on the same square patch with the same parameter budget (default a 33x33 local CNN against a UNet, both about 31.0 M), so the architecture is the only free variable. The scheduler counts every arm before launching and refuses the sweep when they drift beyond the match tolerance.";

    const cellHead      = document.createElement("div");
    cellHead.className  = "exp-presence__sub";
    cellHead.textContent = "arms";

    const cells       = document.createElement("div");
    cells.className   = "exp-builder__names exp-presence__cells";
    this.reachCellsEl = cells;

    const grid     = document.createElement("div");
    grid.className = "exp-secondary__grid";
    ["patch_size", "patch_stride", "in_channels", "match_tolerance"].forEach((key) => {
      const leaf = this.reach.get(key);
      if (leaf) grid.appendChild(this._reachRow(key, leaf));
    });

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(cellHead);
    panel.appendChild(cells);
    panel.appendChild(grid);
    this.reachEl = panel;

    const rungsLeaf = this.reach.get("rungs");
    if (rungsLeaf) this.view.controls[rungsLeaf.path] = { leaf: rungsLeaf, reset: () => this._repaintReach() };

    this._paintReach();
    return panel;
  }

  _reachRow(key, leaf) {
    const row     = document.createElement("div");
    row.className = "cfg-edit__row";
    row.title     = `--${leaf.path}`;

    const name     = document.createElement("div");
    name.className = "cfg-edit__name";
    name.innerHTML = `${key}<span>${leaf.type}</span>`;

    const numeric    = key === "in_channels" || key === "match_tolerance";
    const input      = document.createElement("input");
    input.className  = "cfg-edit__input";
    input.type       = numeric ? "number" : "text";
    if (numeric) input.step = key === "in_channels" ? "1" : "any";
    input.value      = this.view._effective(leaf);
    input.spellcheck = false;
    input.addEventListener("input", () => {
      this.view.applyTypedInput(leaf, input);
      this._paintSummary();
    });

    row.appendChild(name);
    row.appendChild(input);

    this.view.controls[leaf.path] = { leaf, reset: () => {
      input.value = this.view._effective(leaf);
      input.classList.toggle("is-dirty", this.view.dirty[leaf.path] !== undefined);
    } };
    return row;
  }

  _paintReach() {
    if (!this.reachCellsEl) return;
    this.reachCellsEl.innerHTML = "";
    this._reachRungs().forEach((rung) => this.reachCellsEl.appendChild(this._reachChip(rung)));
  }

  _reachChip(rung) {
    return this._cellChip(`${rung.label} · ${rung.backbone}`, null, "Remove arm", () => this._removeReachCell(rung.label));
  }

  _removeReachCell(label) {
    const leaf = this.reach.get("rungs");
    if (!leaf) return;
    const raw = this._reachRungs();
    if (raw.length <= 2) return;
    this.view._setValue(leaf, PythonLiteral.render(raw.filter((entry) => entry.label !== label)));
    this._repaintReach();
  }

  _resetReach() {
    const leaf = this.reach.get("rungs");
    if (leaf) this.view._setValue(leaf, leaf.value);
    this._repaintReach();
  }

  _repaintReach() {
    this._paintReach();
    this._paintSummary();
    this._paintNames();
  }

  _ladderNumberRow(leaf, label, hint, step, min) {
    const row     = document.createElement("div");
    row.className = "cfg-edit__row";
    row.title     = `${hint} (--${leaf.path})`;

    const name     = document.createElement("div");
    name.className = "cfg-edit__name";
    name.innerHTML = `${label}<span>${leaf.type}</span>`;

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input";
    input.type       = "number";
    input.step       = step;
    input.min        = min;
    input.spellcheck = false;
    input.value      = this.view._effective(leaf);
    input.addEventListener("input", () => {
      this.view.applyTypedInput(leaf, input);
    });

    row.appendChild(name);
    row.appendChild(input);

    this.view.controls[leaf.path] = { leaf, reset: () => {
      input.value = this.view._effective(leaf);
      input.classList.toggle("is-dirty", this.view.dirty[leaf.path] !== undefined);
    } };
    return row;
  }

  _augNoiseStds() {
    return this._numbersOf(this.augTrials.get("noise_stds"));
  }

  _augNames() {
    const names = ["aug-0-none", "aug-1-hflip", "aug-2-vflip"];
    this._augNoiseStds().forEach((std) => names.push(`aug-3-noise${this._pyG(std)}`));
    return names;
  }

  _emitAugStds(stds) {
    const leaf = this.augTrials.get("noise_stds");
    if (!leaf) return;

    const clean = stds.map((value) => Number(value)).filter((value) => Number.isFinite(value) && value > 0);
    this.view._setValue(leaf, PythonLiteral.render(clean));

    this._repaintAugmentation();
  }

  _augmentationPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">augmentation ladder</span>`;
    const reset    = LaunchWidgetDom.mini("reset ladder", () => this._resetAugmentation());
    reset.classList.add("exp-presence__reset");
    head.appendChild(reset);
    head.appendChild(LaunchWidgetDom.mini("Add std", () => {
      const stds = this._augNoiseStds();
      stds.push(stds.length ? stds[stds.length - 1] : 0.01);
      this._emitAugStds(stds);
    }));

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "Cumulative ladder on the shared base config: rung 0 trains without augmentation, rung 1 adds horizontal flips, rung 2 adds vertical flips, and one final rung per noise std adds additive Gaussian input noise on top of both flips; rot90 stays off everywhere and every rung repeats across the seeds list.";

    const grid     = document.createElement("div");
    grid.className = "exp-secondary__grid";

    const flipLeaf = this.augTrials.get("flip_probability");
    if (flipLeaf) grid.appendChild(this._ladderNumberRow(flipLeaf, "flip_probability", "probability of each enabled flip", "any", "0"));

    const noiseLeaf = this.augTrials.get("noise_probability");
    if (noiseLeaf) grid.appendChild(this._ladderNumberRow(noiseLeaf, "noise_probability", "probability of injecting noise on the noise rungs", "any", "0"));

    const stdsHead       = document.createElement("div");
    stdsHead.className   = "exp-presence__sub";
    stdsHead.textContent = "noise stds (one rung each, normalised input units)";

    const stds     = document.createElement("div");
    stds.className = "exp-patch__sizes";
    this.augStdsEl = stds;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(grid);
    panel.appendChild(stdsHead);
    panel.appendChild(stds);
    this.augEl = panel;

    const stdsLeaf = this.augTrials.get("noise_stds");
    if (stdsLeaf) this.view.controls[stdsLeaf.path] = { leaf: stdsLeaf, reset: () => this._repaintAugmentation() };

    this._paintAugmentation();
    return panel;
  }

  _paintAugmentation() {
    if (!this.augStdsEl) return;

    const stds = this._augNoiseStds();
    this.augStdsEl.innerHTML = "";
    stds.forEach((std, index) => this.augStdsEl.appendChild(this._weightChip(std, index, stds, "noise standard deviation of this rung", (next) => this._emitAugStds(next))));
  }

  _resetAugmentation() {
    this.augTrials.forEach((leaf) => this.view._setValue(leaf, leaf.value));
    this._repaintAugmentation();
  }

  _repaintAugmentation() {
    this._paintAugmentation();
    this._paintSummary();
    this._paintNames();
  }

  _clampSlopes() {
    return this._numbersOf(this.clampTrials.get("leaky_slopes"));
  }

  _clampParamSlopes() {
    return this._numbersOf(this.clampTrials.get("param_leaky_slopes"));
  }

  _clampNames() {
    const names = ["clamp-0-none", "clamp-1-hard"];
    this._clampSlopes().forEach((slope) => names.push(`clamp-2-leaky-s${this._pyG(slope)}`));
    return names;
  }

  _emitClampPairs(slopes, params) {
    const slopeLeaf = this.clampTrials.get("leaky_slopes");
    const paramLeaf = this.clampTrials.get("param_leaky_slopes");
    if (!slopeLeaf || !paramLeaf) return;

    const keep = slopes.map((slope, index) => [Number(slope), Number(params[index])]).filter((pair) => pair.every((value) => Number.isFinite(value) && value > 0));
    this.view._setValue(slopeLeaf, PythonLiteral.render(keep.map((pair) => pair[0])));
    this.view._setValue(paramLeaf, PythonLiteral.render(keep.map((pair) => pair[1])));

    this._repaintClamp();
  }

  _clampPairChip(index, slopes, params) {
    const chip     = document.createElement("div");
    chip.className = "exp-patch__chip";

    [[slopes, "curve leak slope of this leaky rung"], [params, "parameter leak slope of this leaky rung"]].forEach(([values, title], which) => {
      const input      = document.createElement("input");
      input.className  = "cfg-edit__input exp-patch__size";
      input.type       = "number";
      input.step       = "any";
      input.min        = "0";
      input.spellcheck = false;
      input.value      = String(values[index]);
      input.title      = title;
      input.addEventListener("change", () => {
        const nextSlopes = slopes.slice();
        const nextParams = params.slice();
        (which === 0 ? nextSlopes : nextParams)[index] = Number(input.value);
        this._emitClampPairs(nextSlopes, nextParams);
      });
      chip.appendChild(input);
    });

    const remove = LaunchWidgetDom.mini("×", () => {
      const nextSlopes = slopes.slice();
      const nextParams = params.slice();
      nextSlopes.splice(index, 1);
      nextParams.splice(index, 1);
      this._emitClampPairs(nextSlopes, nextParams);
    });
    remove.classList.add("exp-patch__remove");
    remove.title = "Remove slope pair";
    chip.appendChild(remove);

    return chip;
  }

  _clampPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">output-clamp ladder</span>`;
    const reset    = LaunchWidgetDom.mini("reset ladder", () => this._resetClamp());
    reset.classList.add("exp-presence__reset");
    head.appendChild(reset);
    head.appendChild(LaunchWidgetDom.mini("Add slope pair", () => {
      const slopes = this._clampSlopes();
      const params = this._clampParamSlopes();
      slopes.push(slopes.length ? slopes[slopes.length - 1] : 0.1);
      params.push(params.length ? params[params.length - 1] : 0.1);
      this._emitClampPairs(slopes, params);
    }));

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "Ladder on the shared base config: rung 0 disables the output clamp, rung 1 clamps hard (both leak slopes zero), then one leaky rung per slope pair below; the clamp floor, ceiling and amplitude bound stay at the run's normalization config in every rung.";

    const pairsHead       = document.createElement("div");
    pairsHead.className   = "exp-presence__sub";
    pairsHead.textContent = "leak slope pairs (curve / parameter, one leaky rung each)";

    const pairs      = document.createElement("div");
    pairs.className  = "exp-patch__sizes";
    this.clampPairsEl = pairs;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(pairsHead);
    panel.appendChild(pairs);
    this.clampEl = panel;

    ["leaky_slopes", "param_leaky_slopes"].forEach((key) => {
      const leaf = this.clampTrials.get(key);
      if (leaf) this.view.controls[leaf.path] = { leaf, reset: () => this._repaintClamp() };
    });

    this._paintClamp();
    return panel;
  }

  _paintClamp() {
    if (!this.clampPairsEl) return;

    const slopes = this._clampSlopes();
    const params = this._clampParamSlopes();
    this.clampPairsEl.innerHTML = "";
    slopes.forEach((slope, index) => this.clampPairsEl.appendChild(this._clampPairChip(index, slopes, params)));
  }

  _resetClamp() {
    this.clampTrials.forEach((leaf) => this.view._setValue(leaf, leaf.value));
    this._repaintClamp();
  }

  _repaintClamp() {
    this._paintClamp();
    this._paintSummary();
    this._paintNames();
  }

  _stabilitySteps() {
    return this._numbersOf(this.stabilityTrials.get("warmup_fractions"));
  }

  _stabilityNorms() {
    return this._numbersOf(this.stabilityTrials.get("max_grad_norms"));
  }

  _stabilityClipTag(norm) {
    const leaf = this.stabilityTrials.get("clip_mode");
    const mode = leaf ? String(this.view._effective(leaf)).replace(/^['"]|['"]$/g, "") : "fixed";
    return mode === "fixed" ? `n${this._pyG(norm)}` : mode;
  }

  _stabilityNames() {
    const names = ["stab-0-none"];
    const steps = this._stabilitySteps();
    steps.forEach((fraction) => names.push(`stab-1-warmup-w${this._pyG(fraction)}`));
    this._stabilityNorms().forEach((norm) => names.push(`stab-2-clip-${this._stabilityClipTag(norm)}`));
    if (steps.length) this._stabilityNorms().forEach((norm) => names.push(`stab-3-both-w${this._pyG(steps[0])}-${this._stabilityClipTag(norm)}`));
    return names;
  }

  _emitStabilitySteps(steps) {
    const leaf = this.stabilityTrials.get("warmup_fractions");
    if (!leaf) return;

    const clean = steps.map((value) => Number(value)).filter((value) => Number.isFinite(value) && value > 0 && value <= 1);
    this.view._setValue(leaf, PythonLiteral.render(clean));

    this._repaintStability();
  }

  _emitStabilityNorms(norms) {
    const leaf = this.stabilityTrials.get("max_grad_norms");
    if (!leaf) return;

    const clean = norms.map((value) => Number(value)).filter((value) => Number.isFinite(value) && value > 0);
    this.view._setValue(leaf, PythonLiteral.render(clean));

    this._repaintStability();
  }

  _stabilityPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">stability ladder</span>`;
    const reset    = LaunchWidgetDom.mini("reset ladder", () => this._resetStability());
    reset.classList.add("exp-presence__reset");
    head.appendChild(reset);
    head.appendChild(LaunchWidgetDom.mini("Add fraction", () => {
      const steps = this._stabilitySteps();
      steps.push(steps.length ? steps[steps.length - 1] : 0.03);
      this._emitStabilitySteps(steps);
    }));
    head.appendChild(LaunchWidgetDom.mini("Add norm", () => {
      const norms = this._stabilityNorms();
      norms.push(norms.length ? norms[norms.length - 1] : 1.0);
      this._emitStabilityNorms(norms);
    }));

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "Cumulative ladder on the shared base config: rung 0 trains without lr warmup and without gradient clipping, then one rung per warmup fraction of the total optimizer steps adds the ramp, then one rung per norm threshold adds gradient clipping on top of the first warmup; every rung repeats across the seeds list.";

    const grid     = document.createElement("div");
    grid.className = "exp-secondary__grid";

    const modeLeaf = this.stabilityTrials.get("warmup_mode");
    if (modeLeaf) grid.appendChild(this._stabilitySelectRow(modeLeaf, "warmup_mode", "shape of the warmup ramp", ExperimentBuilder.STABILITY_WARMUP_MODES));

    const clipLeaf = this.stabilityTrials.get("clip_mode");
    if (clipLeaf) grid.appendChild(this._stabilitySelectRow(clipLeaf, "clip_mode", "gradient-clipping strategy of the clipping rungs; only fixed sweeps several norms", ExperimentBuilder.STABILITY_CLIP_MODES));

    const stepsHead       = document.createElement("div");
    stepsHead.className   = "exp-presence__sub";
    stepsHead.textContent = "warmup fractions of the total optimizer steps (one warmup rung each; the first also underlies the clipping rungs)";

    const steps          = document.createElement("div");
    steps.className      = "exp-patch__sizes";
    this.stabilityStepsEl = steps;

    const normsHead       = document.createElement("div");
    normsHead.className   = "exp-presence__sub";
    normsHead.textContent = "max grad norms (one clipping rung each, fixed strategy only)";

    const norms          = document.createElement("div");
    norms.className      = "exp-patch__sizes";
    this.stabilityNormsEl = norms;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(grid);
    panel.appendChild(stepsHead);
    panel.appendChild(steps);
    panel.appendChild(normsHead);
    panel.appendChild(norms);
    this.stabilityEl = panel;

    ["warmup_fractions", "max_grad_norms"].forEach((key) => {
      const leaf = this.stabilityTrials.get(key);
      if (leaf) this.view.controls[leaf.path] = { leaf, reset: () => this._repaintStability() };
    });

    this._paintStability();
    return panel;
  }

  _stabilitySelectRow(leaf, label, hint, options) {
    const row     = document.createElement("div");
    row.className = "cfg-edit__row";
    row.title     = `${hint} (--${leaf.path})`;

    const name     = document.createElement("div");
    name.className = "cfg-edit__name";
    name.innerHTML = `${label}<span>str</span>`;

    const select     = document.createElement("select");
    select.className = "run-select exp-secondary__strategy";
    options.forEach((option) => {
      const opt       = document.createElement("option");
      opt.value       = option;
      opt.textContent = option;
      select.appendChild(opt);
    });
    select.addEventListener("change", () => {
      this.view._setValue(leaf, select.value);
      this._paintSummary();
      this._paintNames();
    });
    this.stabilitySelects.set(leaf.path, select);
    this.view.controls[leaf.path] = { leaf, reset: () => this._paintStability() };

    row.appendChild(name);
    row.appendChild(select);
    return row;
  }

  _paintStability() {
    this.stabilitySelects.forEach((select, path) => {
      const leaf = this.stabilityTrials.get(path.split(".").pop());
      if (leaf && document.activeElement !== select) select.value = this.view._effective(leaf);
    });

    if (!this.stabilityStepsEl) return;

    const steps = this._stabilitySteps();
    this.stabilityStepsEl.innerHTML = "";
    steps.forEach((count, index) => this.stabilityStepsEl.appendChild(this._weightChip(count, index, steps, "fraction of the total optimizer steps of this warmup rung", (next) => this._emitStabilitySteps(next))));

    const norms = this._stabilityNorms();
    this.stabilityNormsEl.innerHTML = "";
    norms.forEach((norm, index) => this.stabilityNormsEl.appendChild(this._weightChip(norm, index, norms, "gradient-norm threshold of this clipping rung", (next) => this._emitStabilityNorms(next))));
  }

  _resetStability() {
    this.stabilityTrials.forEach((leaf) => this.view._setValue(leaf, leaf.value));
    this._repaintStability();
  }

  _repaintStability() {
    this._paintStability();
    this._paintSummary();
    this._paintNames();
  }

  _emaDecays() {
    return this._numbersOf(this.emaTrials.get("decays"));
  }

  _emaNames() {
    const names = ["ema-0-off"];
    this._emaDecays().forEach((decay) => names.push(`ema-1-raw-d${this._pyG(decay)}`));
    this._emaDecays().forEach((decay) => names.push(`ema-2-warmup-d${this._pyG(decay)}`));
    return names;
  }

  _emitEmaDecays(decays) {
    const leaf = this.emaTrials.get("decays");
    if (!leaf) return;

    const clean = decays.map((value) => Number(value)).filter((value) => Number.isFinite(value) && value > 0 && value < 1);
    this.view._setValue(leaf, PythonLiteral.render(clean));

    this._paintEma();
    this._paintSummary();
    this._paintNames();
  }

  _emaPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">weight-EMA ladder</span>`;
    const reset    = LaunchWidgetDom.mini("reset ladder", () => this._resetEma());
    reset.classList.add("exp-presence__reset");
    head.appendChild(reset);
    head.appendChild(LaunchWidgetDom.mini("Add decay", () => {
      const decays = this._emaDecays();
      decays.push(decays.length ? decays[decays.length - 1] : 0.999);
      this._emitEmaDecays(decays);
    }));

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "Rung 0 trains, evaluates and checkpoints the raw weights; then one rung per decay below tracks an exponential moving average of the weights at that decay, everything else unchanged; every rung repeats across the seeds list.";

    const decaysHead       = document.createElement("div");
    decaysHead.className   = "exp-presence__sub";
    decaysHead.textContent = "EMA decays (one on-rung each, strictly between 0 and 1)";

    const decays     = document.createElement("div");
    decays.className = "exp-patch__sizes";
    this.emaDecaysEl = decays;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(decaysHead);
    panel.appendChild(decays);
    this.emaEl = panel;

    const decaysLeaf = this.emaTrials.get("decays");
    if (decaysLeaf) this.view.controls[decaysLeaf.path] = { leaf: decaysLeaf, reset: () => { this._paintEma(); this._paintSummary(); this._paintNames(); } };

    this._paintEma();
    return panel;
  }

  _paintEma() {
    if (!this.emaDecaysEl) return;

    const decays = this._emaDecays();
    this.emaDecaysEl.innerHTML = "";
    decays.forEach((decay, index) => this.emaDecaysEl.appendChild(this._weightChip(decay, index, decays, "EMA decay of this on-rung", (next) => this._emitEmaDecays(next))));
  }

  _resetEma() {
    this.emaTrials.forEach((leaf) => this.view._setValue(leaf, leaf.value));
    this._paintEma();
    this._paintSummary();
    this._paintNames();
  }

  _ampPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">mixed-precision pair</span>`;

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "Two rungs on the shared base config: rung 0 trains in full precision, rung 1 wraps the forward pass in mixed-precision autocast; nothing else changes and both rungs repeat across the seeds list.";

    panel.appendChild(head);
    panel.appendChild(note);
    this.ampEl = panel;

    return panel;
  }

  _pyG(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return String(value);
    if (number === 0) return "0";

    const exponent = Math.floor(Math.log10(Math.abs(number)));
    if (exponent < -4 || exponent >= 6) {
      const parts    = number.toExponential(5).split("e");
      const mantissa = Number(parts[0]);
      const sign     = parts[1][0] === "-" ? "-" : "+";
      const digits   = parts[1].replace(/^[+-]/, "").padStart(2, "0");
      return `${mantissa}e${sign}${digits}`;
    }
    return String(Number(number.toPrecision(6)));
  }

  _subHead(text) {
    const head       = document.createElement("div");
    head.className   = "exp-presence__sub";
    head.textContent = text;
    return head;
  }

  _catalogSection(panel, subHead, leaf, repaint) {
    panel.appendChild(this._subHead(subHead));

    const grid     = document.createElement("div");
    grid.className = "exp-builder__names exp-presence__strategies";
    panel.appendChild(grid);

    if (leaf) this.view.controls[leaf.path] = { leaf, reset: repaint };
    return grid;
  }

  _headList(key) {
    return this._listOf(this.headTrials.get(key), (value) => typeof value === "string");
  }

  _headBackbone() {
    return this.modelLeaf ? this.view._effective(this.modelLeaf) : "?";
  }

  _emitHeadList(key, values) {
    const leaf = this.headTrials.get(key);
    if (!leaf) return;

    this.view._setValue(leaf, PythonLiteral.render(values));
    this._repaintHead();
  }

  _toggleHeadOption(key, option, catalog) {
    const order = catalog.map((entry) => entry.key);
    this._toggleInCatalog(order, this._headList(key), option, (values) => this._emitHeadList(key, values));
  }

  _headPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">head x matching grid</span>`;

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "One training run per checked head x matching pair, all on the backbone this entry names in its own backbone field. The matching strategy is applied to both curriculum stages; everything else comes from the base config.";

    const headsHead       = document.createElement("div");
    headsHead.className   = "exp-presence__sub";
    headsHead.textContent = "heads";

    const heads         = document.createElement("div");
    heads.className     = "exp-builder__names exp-presence__strategies";
    this.headHeadsEl    = heads;

    const matchingsHead       = document.createElement("div");
    matchingsHead.className   = "exp-presence__sub";
    matchingsHead.textContent = "matchings";

    const matchings         = document.createElement("div");
    matchings.className     = "exp-builder__names exp-presence__strategies";
    this.headMatchingsEl    = matchings;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(headsHead);
    panel.appendChild(heads);
    panel.appendChild(matchingsHead);
    panel.appendChild(matchings);
    this.headEl = panel;

    const headsLeaf = this.headTrials.get("heads");
    if (headsLeaf) this.view.controls[headsLeaf.path] = { leaf: headsLeaf, reset: () => this._repaintHead() };

    const matchingsLeaf = this.headTrials.get("matchings");
    if (matchingsLeaf) this.view.controls[matchingsLeaf.path] = { leaf: matchingsLeaf, reset: () => this._repaintHead() };

    this._paintHead();
    return panel;
  }

  _paintHead() {
    if (!this.headEl) return;

    const paintToggles = (el, key, catalog) => {
      el.innerHTML = "";
      const active = this._headList(key);
      catalog.forEach(({ key: option, label }) => {
        const chip = this._toggleChip(label, active.includes(option), "Click to drop this option from the grid", "Click to add this option to the grid", () => this._toggleHeadOption(key, option, catalog));
        el.appendChild(chip);
      });
    };

    if (this.headHeadsEl)     paintToggles(this.headHeadsEl,     "heads",     ExperimentBuilder.HEAD_OPTIONS);
    if (this.headMatchingsEl) paintToggles(this.headMatchingsEl, "matchings", ExperimentBuilder.MATCHING_OPTIONS);
  }

  _repaintHead() {
    this._paintHead();
    this._paintSummary();
    this._paintNames();
  }

  _trialsDict(leaf) {
    return leaf ? this._dictFrom(this.view._effective(leaf)) : {};
  }

  _defaultsDict(leaf) {
    return leaf ? this._dictFrom(leaf.value) : {};
  }

  _listOf(leaf, keep) {
    if (!leaf) return [];

    let raw = null;
    try {
      raw = PythonLiteral.parse(this.view._effective(leaf));
    } catch (e) {
      raw = null;
    }
    return Array.isArray(raw) ? raw.filter(keep) : [];
  }

  _numbersOf(leaf) {
    return this._listOf(leaf, () => true).map((value) => Number(value)).filter((value) => Number.isFinite(value));
  }

  _dictFrom(text) {
    try {
      const raw = PythonLiteral.parse(text);
      return raw && typeof raw === "object" && !Array.isArray(raw) ? raw : {};
    } catch (e) {
      return {};
    }
  }

  _toggleChip(label, on, onTitle, offTitle, onClick) {
    const chip       = document.createElement("button");
    chip.type        = "button";
    chip.className   = "exp-name exp-presence__toggle" + (on ? " is-on" : "");
    chip.textContent = label;
    chip.title       = on ? onTitle : offTitle;
    chip.addEventListener("click", onClick);
    return chip;
  }

  _cellChip(text, title, removeTitle, remove) {
    const chip     = document.createElement("span");
    chip.className = "exp-name exp-presence__cell";
    if (title != null) chip.title = title;

    const body       = document.createElement("span");
    body.textContent = text;
    chip.appendChild(body);

    const drop = LaunchWidgetDom.mini("×", remove);
    drop.classList.add("exp-presence__remove");
    drop.title = removeTitle;
    chip.appendChild(drop);
    return chip;
  }

  _weightChip(weight, index, weights, title, emit) {
    const chip     = document.createElement("div");
    chip.className = "exp-patch__chip";

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input exp-patch__size";
    input.type       = "number";
    input.step       = "any";
    input.min        = "0";
    input.spellcheck = false;
    input.value      = String(weight);
    input.title      = title;
    input.addEventListener("change", () => {
      const next  = weights.slice();
      next[index] = Number(input.value);
      emit(next);
    });

    const remove = LaunchWidgetDom.mini("×", () => {
      const next = weights.slice();
      next.splice(index, 1);
      emit(next);
    });
    remove.classList.add("exp-patch__remove");
    remove.title = "Remove weight";

    chip.appendChild(input);
    chip.appendChild(remove);
    return chip;
  }

  _toggleInCatalog(order, current, key, emit) {
    if (current.includes(key)) {
      if (current.length <= 1) return;
      emit(current.filter((value) => value !== key));
      return;
    }
    emit(order.filter((value) => current.includes(value) || value === key));
  }

  _paintDictCells(el, trials, defaults, text, onTitle, offTitle, toggle) {
    el.innerHTML = "";

    const cells = { ...defaults };
    Object.entries(trials).forEach(([cell, spec]) => { if (!(cell in cells)) cells[cell] = spec; });

    Object.entries(cells).forEach(([cell, spec]) => {
      const on = Object.prototype.hasOwnProperty.call(trials, cell);
      el.appendChild(this._toggleChip(text(cell, spec), on, onTitle, offTitle, () => toggle(cell)));
    });
  }

  _toggleDictCell(trials, defaults, cell) {
    const has = (dict, key) => Object.prototype.hasOwnProperty.call(dict, key);

    if (has(trials, cell)) {
      if (Object.keys(trials).length <= 1) return null;
      delete trials[cell];
    } else {
      if (!has(defaults, cell)) return null;
      trials[cell] = defaults[cell];
    }

    const ordered = {};
    Object.keys(defaults).forEach((key) => { if (has(trials, key)) ordered[key] = trials[key]; });
    Object.keys(trials).forEach((key) => { if (!has(ordered, key)) ordered[key] = trials[key]; });
    return ordered;
  }

  _paintMode() {
    const mode = this._mode();

    this.modeButtons.forEach((btn, key) => btn.classList.toggle("is-on", key === mode));
    if (this.modeEl) this.modeEl.classList.toggle("is-dirty", this.view.dirty[this.modeLeaf.path] !== undefined);
    if (this.hintEl) this.hintEl.textContent = this._modeEntry(mode).hint;

    if (this.columnEls.complete) this.columnEls.complete.hidden = mode !== "curriculum";
    ExperimentBuilder.MODES.forEach((entry) => {
      const el = this[entry.el];
      if (el) el.hidden = mode !== entry.key;
    });
  }

  _paintSecondary() {
    if (!this.secondaryEl) return;
    const strategy = this._strategy();
    const note     = ExperimentBuilder.STRATEGIES.find((entry) => entry.key === strategy).note;
    const fixed    = this._secondaryFixed();

    if (this.strategySelect) this.strategySelect.value = strategy;
    this.strategyNoteEl.textContent = fixed.length
      ? `${note}; ${fixed.join(", ")} ${fixed.length === 1 ? "sits" : "sit"} in every run on top of the n_secondaries swept ones, and ${fixed.length === 1 ? "is" : "are"} taken out of the swept pool`
      : note;
    this.secondaryRows.forEach((entry) => (entry.row.hidden = !entry.strategies.includes(strategy)));
  }

  _secondaryFixed() {
    return this._listOf(this.secondary.get("fixed"), (value) => typeof value === "string" && value.length > 0);
  }

  _secondaryEffective(key) {
    const leaf = this.secondary.get(key);
    return leaf ? this.view._effective(leaf) : "?";
  }

  _termCatalog() {
    const prefix  = "curriculum.complete.";
    const leaves  = this.view.config.leaves.filter((leaf) => leaf.section === "curriculum.complete");
    const short   = (leaf) => leaf.path.slice(prefix.length);
    const weights = leaves.filter((leaf) => short(leaf).startsWith("weight_") && (leaf.type === "float" || leaf.type === "int"));

    const terms = [];
    leaves.forEach((use) => {
      if (use.type !== "bool" || !short(use).startsWith("use_")) return;

      const name   = short(use).slice(4);
      const weight = weights.find((leaf) => {
        const wname = short(leaf).slice(7);
        return wname === name || name.startsWith(wname) || wname.startsWith(name);
      });
      if (!weight) return;

      const fallback = Number(weight.value);
      terms.push({ key: name, useKey: short(use), weightKey: short(weight), defaultWeight: fallback > 0 ? fallback : 1.0 });
    });
    return terms;
  }

  _trialsSwitch() {
    const leaf     = this.trialsLeaf;
    const wrap     = document.createElement("div");
    wrap.className = "exp-power";
    wrap.title     = `--${leaf.path}`;

    const state       = document.createElement("span");
    state.className   = "exp-power__state";
    wrap.appendChild(state);

    const toggle   = document.createElement("button");
    toggle.type    = "button";
    toggle.className = "switch";
    toggle.setAttribute("role", "switch");
    toggle.setAttribute("aria-label", "Enable experiment fan-out");
    toggle.innerHTML = `<span class="switch__knob"></span>`;
    wrap.appendChild(toggle);

    const paint = () => {
      const on = this.view._effective(leaf) === "True";
      toggle.classList.toggle("is-on", on);
      toggle.classList.toggle("is-dirty", this.view.dirty[leaf.path] !== undefined);
      toggle.setAttribute("aria-checked", String(on));
      state.textContent = on ? "fan-out on" : "fan-out off";
      wrap.classList.toggle("is-on", on);
      this.root.classList.toggle("is-trials", on);
    };

    toggle.addEventListener("click", () => {
      const next = this.view._effective(leaf) === "True" ? "False" : "True";
      this.view._setValue(leaf, next);
      paint();
      this._paintSummary();
      this._paintNames();
    });

    this._paintSwitch = paint;
    this.view.controls[leaf.path] = { leaf, reset: () => this._reload() };
    return wrap;
  }

  _column(which, title) {
    const col     = document.createElement("div");
    col.className = "exp-col";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">${title}</span><span class="exp-col__count"></span>`;
    head.appendChild(LaunchWidgetDom.mini("Add variant", () => {
      this.variants[which].push(this._blankVariant(which));
      this._emit(which);
    }));

    const list     = document.createElement("div");
    list.className = "exp-col__list";

    this.lists[which] = { list, count: head.querySelector(".exp-col__count") };
    col.appendChild(head);
    col.appendChild(list);
    return col;
  }

  _blankVariant(which) {
    const base  = this.terms.find((term) => term.key === "param_l1") || this.terms[0];
    const label = this._uniqueLabel(which, which === "warmup" ? "warm" : "exp");
    return { label, terms: [{ useKey: base.useKey, weightKey: base.weightKey, weight: base.defaultWeight }], extras: {} };
  }

  _uniqueLabel(which, seed) {
    const taken = new Set(this.variants[which].map((variant) => variant.label));
    if (!taken.has(seed)) return seed;

    let n = 2;
    while (taken.has(`${seed}${n}`)) n += 1;
    return `${seed}${n}`;
  }

  _variantsFrom(leaf) {
    let raw = null;
    try {
      raw = PythonLiteral.parse(this.view._effective(leaf));
    } catch (e) {
      raw = null;
    }
    if (!raw || Array.isArray(raw) || typeof raw !== "object") return [];

    return Object.entries(raw).map(([label, spec]) => {
      const entry  = spec && typeof spec === "object" && !Array.isArray(spec) ? spec : {};
      const terms  = [];
      const extras = {};
      const used   = new Set();

      this.terms.forEach((term) => {
        if (entry[term.useKey] !== true) return;
        const weight = typeof entry[term.weightKey] === "number" ? entry[term.weightKey] : term.defaultWeight;
        terms.push({ useKey: term.useKey, weightKey: term.weightKey, weight });
        used.add(term.useKey);
        used.add(term.weightKey);
      });

      Object.entries(entry).forEach(([key, value]) => {
        if (!used.has(key)) extras[key] = value;
      });

      return { label, terms, extras };
    });
  }

  _dictFor(variants) {
    const out = {};
    variants.forEach((variant) => {
      const entry = { ...variant.extras };
      variant.terms.forEach((term) => {
        entry[term.useKey]    = true;
        entry[term.weightKey] = Number(term.weight) || 0;
      });
      out[variant.label || "unnamed"] = entry;
    });
    return out;
  }

  _emit(which) {
    const leaf = which === "warmup" ? this.warmupLeaf : this.completeLeaf;
    this.view._setValue(leaf, PythonLiteral.render(this._dictFor(this.variants[which])));
    this._paintAll();
  }

  _reload() {
    this.variants.warmup   = this.warmupLeaf ? this._variantsFrom(this.warmupLeaf) : [];
    this.variants.complete = this.completeLeaf ? this._variantsFrom(this.completeLeaf) : [];
    if (this._paintSwitch) this._paintSwitch();
    this._paintMode();
    ExperimentBuilder.MODES.forEach((mode) => { if (mode.paint) mode.paint(this); });
    this._paintAll();
  }

  _paintAll() {
    ["warmup", "complete"].forEach((which) => {
      if (!this.lists[which]) return;
      const { list, count } = this.lists[which];
      list.innerHTML = "";
      this.variants[which].forEach((variant, index) => list.appendChild(this._card(which, variant, index)));
      count.textContent = `${this.variants[which].length} variant${this.variants[which].length === 1 ? "" : "s"}`;
    });

    this._paintWarmupCatalog();
    this._paintSummary();
    this._paintNames();
  }

  _card(which, variant, index) {
    const card     = document.createElement("div");
    card.className = "exp-card";

    const top     = document.createElement("div");
    top.className = "exp-card__top";

    const label      = document.createElement("input");
    label.className  = "cfg-edit__input exp-card__label";
    label.value      = variant.label;
    label.spellcheck = false;
    label.title      = "variant label, used in the run name";
    label.addEventListener("change", () => {
      const wanted = label.value.trim() || variant.label;
      const taken  = new Set(this.variants[which].filter((other) => other !== variant).map((other) => other.label));
      variant.label = taken.has(wanted) ? this._uniqueLabel(which, wanted) : wanted;
      label.value   = variant.label;
      this._emit(which);
    });

    top.appendChild(label);
    top.appendChild(LaunchWidgetDom.mini("Duplicate", () => {
      const copy = JSON.parse(JSON.stringify(variant));
      copy.label = this._uniqueLabel(which, copy.label);
      this.variants[which].splice(index + 1, 0, copy);
      this._emit(which);
    }));
    top.appendChild(LaunchWidgetDom.mini("Remove", () => {
      this.variants[which].splice(index, 1);
      this._emit(which);
    }));
    card.appendChild(top);

    const terms     = document.createElement("div");
    terms.className = "exp-card__terms";
    variant.terms.forEach((term, ti) => terms.appendChild(this._termRow(which, variant, term, ti)));
    card.appendChild(terms);

    const extrasKeys = Object.keys(variant.extras);
    if (extrasKeys.length) {
      const note       = document.createElement("div");
      note.className   = "exp-card__extras";
      note.textContent = `raw keys kept: ${extrasKeys.join(", ")}`;
      card.appendChild(note);
    }

    const addTerm = LaunchWidgetDom.mini("Add term", () => {
      const unused = this.terms.find((term) => !variant.terms.some((vt) => vt.useKey === term.useKey));
      const pick   = unused || this.terms[0];
      variant.terms.push({ useKey: pick.useKey, weightKey: pick.weightKey, weight: pick.defaultWeight });
      this._emit(which);
    });
    addTerm.classList.add("exp-card__add");
    card.appendChild(addTerm);

    return card;
  }

  _termRow(which, variant, term, index) {
    const row     = document.createElement("div");
    row.className = "exp-term";

    const select     = document.createElement("select");
    select.className = "run-select exp-term__select";
    this.terms.forEach((option) => {
      const opt       = document.createElement("option");
      opt.value       = option.useKey;
      opt.textContent = option.key;
      if (option.useKey === term.useKey) opt.selected = true;
      select.appendChild(opt);
    });
    select.addEventListener("change", () => {
      const picked   = this.terms.find((option) => option.useKey === select.value);
      term.useKey    = picked.useKey;
      term.weightKey = picked.weightKey;
      term.weight    = picked.defaultWeight;
      this._emit(which);
    });

    const weight      = document.createElement("input");
    weight.className  = "cfg-edit__input exp-term__weight";
    weight.type       = "number";
    weight.step       = "any";
    weight.value      = String(term.weight);
    weight.spellcheck = false;
    weight.title      = "loss weight";
    weight.addEventListener("change", () => {
      term.weight = Number(weight.value) || 0;
      this._emit(which);
    });

    const remove = LaunchWidgetDom.mini("×", () => {
      variant.terms.splice(index, 1);
      this._emit(which);
    });
    remove.classList.add("exp-term__remove");
    remove.title = "Remove term";

    row.appendChild(select);
    row.appendChild(weight);
    row.appendChild(remove);
    return row;
  }

  _warmupCatalogPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-warmup";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">warmup loss sweep</span><span class="exp-col__count exp-warmup__count"></span>`;
    this.warmupCountEl = head.querySelector(".exp-warmup__count");

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "Check a loss to train it alone as one trial. Expand a checked card to stack extra terms onto that trial.";

    const grid     = document.createElement("div");
    grid.className = "exp-warmup__grid";
    this.warmupCatalogGridEl = grid;

    const customHead       = document.createElement("div");
    customHead.className    = "exp-presence__sub exp-warmup__customhead";
    customHead.textContent  = "other variants";
    this.warmupCustomHeadEl = customHead;

    const custom     = document.createElement("div");
    custom.className = "exp-warmup__custom";
    this.warmupCustomEl = custom;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(grid);
    panel.appendChild(customHead);
    panel.appendChild(custom);
    this.warmupCatalogEl = panel;
    return panel;
  }

  _warmupBindings() {
    const byTerm = new Map();
    const custom = [];

    this.variants.warmup.forEach((variant, index) => {
      const base  = variant.terms[0];
      const known = base && this.terms.some((term) => term.useKey === base.useKey);
      if (known && !byTerm.has(base.useKey)) byTerm.set(base.useKey, index);
      else custom.push(index);
    });

    return { byTerm, custom };
  }

  _paintWarmupCatalog() {
    if (!this.warmupCatalogGridEl) return;

    const { byTerm, custom } = this._warmupBindings();

    this.warmupCatalogGridEl.innerHTML = "";
    this.terms.forEach((term) => {
      const index   = byTerm.has(term.useKey) ? byTerm.get(term.useKey) : -1;
      const variant = index >= 0 ? this.variants.warmup[index] : null;
      this.warmupCatalogGridEl.appendChild(this._warmupLossCard(term, variant, index));
    });

    this.warmupCustomEl.innerHTML = "";
    custom.forEach((index) => this.warmupCustomEl.appendChild(this._card("warmup", this.variants.warmup[index], index)));
    const hasCustom = custom.length > 0;
    this.warmupCustomEl.hidden     = !hasCustom;
    this.warmupCustomHeadEl.hidden = !hasCustom;

    if (this.warmupCountEl) {
      const n = this.variants.warmup.length;
      this.warmupCountEl.textContent = `${n} selected = ${n} trial${n === 1 ? "" : "s"}`;
    }
  }

  _warmupLossCard(term, variant, index) {
    const on       = !!variant;
    const card     = document.createElement("div");
    card.className = "exp-loss-card" + (on ? " is-on" : "");

    const head     = document.createElement("div");
    head.className = "exp-loss-card__head";

    const toggle     = document.createElement("button");
    toggle.type      = "button";
    toggle.className = "switch exp-loss-card__switch" + (on ? " is-on" : "");
    toggle.setAttribute("role", "switch");
    toggle.setAttribute("aria-checked", String(on));
    toggle.title     = on ? "Drop this loss from the sweep" : "Add this loss as a trial";
    toggle.innerHTML = `<span class="switch__knob"></span>`;
    toggle.addEventListener("click", () => this._toggleWarmupLoss(term, index));

    const name       = document.createElement("span");
    name.className   = "exp-loss-card__name";
    name.textContent = term.key;

    head.appendChild(toggle);
    head.appendChild(name);
    card.appendChild(head);

    if (!on) return card;

    const body     = document.createElement("div");
    body.className = "exp-loss-card__body";

    const labelRow     = document.createElement("label");
    labelRow.className = "exp-loss-card__labelrow";
    labelRow.innerHTML = `<span>label</span>`;
    const label        = document.createElement("input");
    label.className    = "cfg-edit__input";
    label.value        = variant.label;
    label.spellcheck   = false;
    label.title        = "variant label, used in the run name";
    label.addEventListener("change", () => {
      const wanted = label.value.trim() || variant.label;
      const taken  = new Set(this.variants.warmup.filter((other) => other !== variant).map((other) => other.label));
      variant.label = taken.has(wanted) ? this._uniqueLabel("warmup", wanted) : wanted;
      label.value   = variant.label;
      this._emit("warmup");
    });
    labelRow.appendChild(label);
    body.appendChild(labelRow);

    const baseRow     = document.createElement("div");
    baseRow.className = "exp-loss-card__base";
    baseRow.innerHTML = `<span class="exp-loss-card__weightlabel">weight</span>`;
    const baseWeight       = document.createElement("input");
    baseWeight.className    = "cfg-edit__input";
    baseWeight.type        = "number";
    baseWeight.step        = "any";
    baseWeight.value       = String(variant.terms[0].weight);
    baseWeight.spellcheck  = false;
    baseWeight.title       = "loss weight";
    baseWeight.addEventListener("change", () => {
      variant.terms[0].weight = Number(baseWeight.value) || 0;
      this._emit("warmup");
    });
    baseRow.appendChild(baseWeight);
    body.appendChild(baseRow);

    if (variant.terms.length > 1) {
      const extras     = document.createElement("div");
      extras.className = "exp-card__terms";
      variant.terms.forEach((extra, ti) => {
        if (ti === 0) return;
        extras.appendChild(this._termRow("warmup", variant, extra, ti));
      });
      body.appendChild(extras);
    }

    const extrasKeys = Object.keys(variant.extras);
    if (extrasKeys.length) {
      const kept       = document.createElement("div");
      kept.className   = "exp-card__extras";
      kept.textContent = `raw keys kept: ${extrasKeys.join(", ")}`;
      body.appendChild(kept);
    }

    const addTerm = LaunchWidgetDom.mini("Add term", () => {
      const unused = this.terms.find((candidate) => !variant.terms.some((vt) => vt.useKey === candidate.useKey));
      const pick   = unused || this.terms[0];
      variant.terms.push({ useKey: pick.useKey, weightKey: pick.weightKey, weight: pick.defaultWeight });
      this._emit("warmup");
    });
    addTerm.classList.add("exp-card__add");
    body.appendChild(addTerm);

    card.appendChild(body);
    return card;
  }

  _toggleWarmupLoss(term, index) {
    if (index >= 0) this.variants.warmup.splice(index, 1);
    else this.variants.warmup.push({ label: this._uniqueLabel("warmup", term.key), terms: [{ useKey: term.useKey, weightKey: term.weightKey, weight: term.defaultWeight }], extras: {} });

    this._emit("warmup");
  }

  _seedsSuffix(trials) {
    return SeedAxisNote.suffix(this.view, this.seedsLeaf, trials);
  }

  _paintSummary() {
    if (this.view._effective(this.trialsLeaf) !== "True") {
      this.summaryEl.textContent = "single run at the configured values";
      return;
    }

    let gpus = "";
    if (this.gpusLeaf) {
      try {
        const parsed = PythonLiteral.parse(this.view._effective(this.gpusLeaf));
        if (Array.isArray(parsed)) gpus = ` on ${parsed.length} GPU${parsed.length === 1 ? "" : "s"}`;
      } catch (e) {
        gpus = "";
      }
    }

    this.summaryEl.textContent = this._modeEntry(this._mode()).summary(this, gpus);
  }

  _paintNames() {
    this.namesEl.innerHTML = "";
    const model = this.modelLeaf ? this.view._effective(this.modelLeaf) : "model";
    const entry = this._modeEntry(this._mode());

    const spec = entry.pattern ? entry.pattern(this, model) : null;

    if (spec) {
      const pattern       = document.createElement("span");
      pattern.className   = "exp-name";
      pattern.textContent = spec.text;
      this.namesEl.appendChild(pattern);

      const more       = document.createElement("span");
      more.className   = "exp-name exp-name--more";
      more.textContent = spec.note;
      this.namesEl.appendChild(more);

      SeedAxisNote.append(this.namesEl, this.view, this.seedsLeaf);
      return;
    }

    const names = entry.names(this, model);
    const limit = 12;
    names.slice(0, limit).forEach((name) => {
      const chip       = document.createElement("span");
      chip.className   = "exp-name";
      chip.textContent = name;
      this.namesEl.appendChild(chip);
    });

    if (names.length > limit) {
      const more       = document.createElement("span");
      more.className   = "exp-name exp-name--more";
      more.textContent = `+${names.length - limit} more`;
      this.namesEl.appendChild(more);
    }

    SeedAxisNote.append(this.namesEl, this.view, this.seedsLeaf);
  }
}


window.ExperimentBuilder = ExperimentBuilder;
