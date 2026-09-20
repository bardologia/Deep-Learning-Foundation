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

    { group: "loss", key: "physics",    label: "physics loss", hint: "physics components crossed with weights on top of the base config under the loss curriculum, one trial per pair plus an optional no-physics baseline",
      at        : 2,
      el        : "physicsEl",
      available : (b) => b.physics.size > 0,
      make      : (b) => b.sweeps.physics.build(),
      paint     : (b) => b.sweeps.physics.paint(),
      summary   : (b, gpus) => {
        const nComp    = b.sweeps.physics.components().length;
        const nWeight  = b.sweeps.physics.weights().length;
        const baseline = b.sweeps.physics.baselineOn();
        const total    = nComp * nWeight + (baseline ? 1 : 0);
        return `${nComp} component${nComp === 1 ? "" : "s"} x ${nWeight} weight${nWeight === 1 ? "" : "s"}${baseline ? " + baseline" : ""} = ${total} trial${total === 1 ? "" : "s"}${b._seedsSuffix(total)}${gpus}`;
      },
      names     : (b, model) => {
        const names = [];
        if (b.sweeps.physics.baselineOn()) names.push(`${model}_phys-baseline`);
        b.sweeps.physics.components().forEach((component) => {
          b.sweeps.physics.weights().forEach((weight) => names.push(`${model}_phys-${component}-w${weight}`));
        });
        return names;
      } },

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
        if (strategy === "exhaustive") return `exhaustive, all combinations of exactly ${b._secondaryEffective("n_secondaries")} secondaries${b._seedsSuffix(null)}${gpus}`;
        const sampled  = strategy === "uniform" || strategy === "gaussian";
        const count    = sampled ? `${b._secondaryEffective("n_trials")} trials` : "trial count set by the stack";
        return `${strategy}, ${b._secondaryEffective("n_secondaries")} secondaries, ${count}${b._seedsSuffix(null)}${gpus}`;
      },
      pattern   : (b, model) => ({ text: `${model}_sec-${b._strategy()}-tNN_<labels>`, note: "one per selection, labels resolved from the dataset stack" }) },

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

    { group: "arch", key: "head",       label: "head x matching", hint: "one trial per output-head x parameter-matching pair on one fixed backbone",
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
      make      : (b) => b._isProfileAug() ? b._profileAugPanel() : b._augmentationPanel(),
      paint     : (b) => b._isProfileAug() ? b._paintProfileAug() : b._paintAugmentation(),
      summary   : (b, gpus) => {
        const fixed = b._isProfileAug() ? 4 : 3;
        const n     = fixed + b._augNoiseStds().length;
        return `${fixed} ladder rungs + ${b._augNoiseStds().length} noise rung${b._augNoiseStds().length === 1 ? "" : "s"} = ${n} trials${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => (b._isProfileAug() ? b._profileAugNames() : b._augNames()).map((rung) => `${model}_${rung}`) },

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

    { group: "arch", key: "routing", label: "trunk routing", hint: "one trial per params/existence channel-group assignment, each in a no-Aux and a with-Aux variant, both trunks fixed at the shared parity feature ladders",
      at        : 13,
      el        : "routingEl",
      available : (b) => b.routingTrials.size > 0,
      make      : (b) => b._routingPanel(),
      paint     : (b) => b._paintRouting(),
      summary   : (b, gpus) => {
        const n = Object.keys(b._routingTrialsDict()).length;
        return `${n} routing assignment${n === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => Object.keys(b._routingTrialsDict()).map((label) => `${model}_di-${label}`) },

    { group: "arch", key: "ratio",   label: "arm ratio", hint: "budget-matched params/existence width ladders, one trial per declared split; every model is built and its split verified at launch",
      at        : 14,
      el        : "ratioEl",
      available : (b) => b.ratioTrials.size > 0,
      make      : (b) => b._ratioPanel(),
      paint     : (b) => b._paintRatio(),
      summary   : (b, gpus) => {
        const n = Object.keys(b._ratioTrialsDict()).length;
        return `${n} arm split${n === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => Object.keys(b._ratioTrialsDict()).map((label) => `${model}_dr-${label}`) },

    { group: "arch", key: "arch", label: "AE family", hint: "one trial per profile-autoencoder family rung on the shared base config; extra rung keys apply as model overrides",
      at        : 17,
      el        : "archEl",
      available : (b) => Boolean(b.archTrialsLeaf),
      make      : (b) => b._archPanel(),
      paint     : (b) => b._paintArch(),
      summary   : (b, gpus) => {
        const n = b._archRungs().length;
        return `${n} famil${n === 1 ? "y" : "ies"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b) => b._archRungs().map((rung) => `${rung.ae_model_name}_arch-${rung.label}`) },

    { group: "arch", key: "embedding", label: "latent width", hint: "one trial per embedding dimension on the configured family, everything else unchanged",
      at        : 18,
      el        : "embDimEl",
      available : (b) => b.embDimTrials.size > 0,
      make      : (b) => b._embDimPanel(),
      paint     : (b) => b._paintEmbDim(),
      summary   : (b, gpus) => {
        const n = b._embDims().length;
        return `${n} latent width${n === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b._embDims().map((dim) => `${model}_emb-d${dim}`) },

    { group: "arch", key: "capacity", label: "capacity", hint: "width x depth grid within the configured family; empty axes walk the family's tunable choices",
      at        : 19,
      el        : "capacityEl",
      available : (b) => b.capacityTrials.has("depths"),
      make      : (b) => b._capacityPanel(),
      paint     : (b) => b._paintCapacity(),
      summary   : (b, gpus) => {
        const widths = b._capacityValues("widths");
        const depths = b._capacityValues("depths");
        if (!widths.length || !depths.length) return "width x depth grid from the family's tunable choices";
        const total = widths.length * depths.length;
        return `${widths.length} width${widths.length === 1 ? "" : "s"} x ${depths.length} depth${depths.length === 1 ? "" : "s"} = ${total} trial${total === 1 ? "" : "s"}${b._seedsSuffix(total)}${gpus}`;
      },
      names     : (b, model) => {
        const widths = b._capacityValues("widths");
        const depths = b._capacityValues("depths");
        const names  = [];
        widths.forEach((width) => depths.forEach((depth) => names.push(`${model}_cap-w${width}-d${depth}`)));
        return names;
      },
      pattern   : (b, model) => {
        if (b._capacityValues("widths").length && b._capacityValues("depths").length) return null;
        return { text: `${model}_cap-w<width>-d<depth>`, note: "one per cell, axes resolved from the family's tunable choices at launch" };
      } },

    { group: "loss", key: "loss", label: "reconstruction loss", hint: "one trial per reconstruction-loss arm from the catalog; every arm fully defines the loss, all other terms disabled",
      at        : 20,
      el        : "aeLossEl",
      available : (b) => Boolean(b.aeLossTrialsLeaf),
      make      : (b) => b._aeLossPanel(),
      paint     : (b) => b._paintAeLoss(),
      summary   : (b, gpus) => {
        const n = Object.keys(b._aeLossTrials()).length;
        return `${n} loss arm${n === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => Object.keys(b._aeLossTrials()).map((label) => `${model}_loss-${label}`) },

    { group: "loss", key: "smoothness", label: "smoothness", hint: "Sobolev and latent-noise arms on top of the configured reconstruction loss, the decoder-robustness levers for JEPA coupling",
      at        : 21,
      el        : "smoothEl",
      available : (b) => b.smoothTrials.size > 0,
      make      : (b) => b._smoothnessPanel(),
      paint     : (b) => b._paintSmoothness(),
      summary   : (b, gpus) => {
        const n = b._smoothnessNames().length;
        return `${b._smoothSobolev().length} sobolev + ${b._smoothNoise().length} latent-noise${b._smoothFlag("combine") ? " + grid" : ""}${b._smoothFlag("include_baseline") ? " + baseline" : ""} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b._smoothnessNames().map((rung) => `${model}_${rung}`) },

    { group: "data", key: "sampling", label: "data mix", hint: "pixel-subsample x kept-empty-fraction grid over the training profiles",
      at        : 22,
      el        : "samplingEl",
      available : (b) => b.samplingTrials.size > 0,
      make      : (b) => b._samplingPanel(),
      paint     : (b) => b._paintSampling(),
      summary   : (b, gpus) => {
        const subs   = b._samplingValues("subsamples");
        const empty  = b._samplingValues("empty_fracs");
        const total  = subs.length * empty.length;
        return `${subs.length} subsample${subs.length === 1 ? "" : "s"} x ${empty.length} empty fraction${empty.length === 1 ? "" : "s"} = ${total} trial${total === 1 ? "" : "s"}${b._seedsSuffix(total)}${gpus}`;
      },
      names     : (b, model) => {
        const names = [];
        b._samplingValues("subsamples").forEach((sub) => b._samplingValues("empty_fracs").forEach((frac) => names.push(`${model}_smp-p${b._pyG(sub)}-e${b._pyG(frac)}`)));
        return names;
      } },

    { group: "loss", key: "embedding", label: "embedding loss", hint: "one trial per latent-loss arm from the catalog; every arm fully defines the latent terms, the decoded-curve settings stay at the base config",
      at        : 23,
      el        : "jepaEmbEl",
      available : (b) => Boolean(b.embeddingCatalogLeaf),
      make      : (b) => b._jepaEmbeddingPanel(),
      paint     : (b) => b._paintJepaEmbedding(),
      summary   : (b, gpus) => {
        const n = Object.keys(b._jepaEmbeddingTrials()).length;
        return `${n} latent-loss arm${n === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => Object.keys(b._jepaEmbeddingTrials()).map((label) => `${model}_el-${label}`) },

    { group: "loss", key: "curve", label: "decoded curve", hint: "decoded-curve penalty kinds crossed with weights, plus Sobolev arms on an mse anchor, plus an optional no-curve baseline",
      at        : 24,
      el        : "curveEl",
      available : (b) => b.curveTrials.size > 0,
      make      : (b) => b._curvePanel(),
      paint     : (b) => b._paintCurve(),
      summary   : (b, gpus) => {
        const n = b._curveNames().length;
        return `${b._curveKinds().length} kind${b._curveKinds().length === 1 ? "" : "s"} x ${b._curveWeights().length} weight${b._curveWeights().length === 1 ? "" : "s"} + ${b._curveSobolev().length} sobolev${b._curveFlag("include_baseline") ? " + baseline" : ""} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b._curveNames().map((rung) => `${model}_${rung}`) },

    { group: "data", key: "masking", label: "input masking", hint: "context-masking grid: strategies crossed with hidden-pixel ratios, plus an optional unmasked baseline",
      at        : 25,
      el        : "maskingEl",
      available : (b) => b.maskingTrials.size > 0,
      make      : (b) => b._maskingPanel(),
      paint     : (b) => b._paintMasking(),
      summary   : (b, gpus) => {
        const n = b._maskingNames().length;
        return `${b._maskingStrategies().length} strateg${b._maskingStrategies().length === 1 ? "y" : "ies"} x ${b._maskingRatios().length} ratio${b._maskingRatios().length === 1 ? "" : "s"}${b._maskingFlag("include_baseline") ? " + baseline" : ""} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b._maskingNames().map((rung) => `${model}_${rung}`) },

    { group: "training", key: "curriculum", label: "curriculum", hint: "swap-epoch x reset grid of the two-stage loss schedule, plus an optional curriculum-free baseline",
      at        : 26,
      el        : "jepaCurEl",
      available : (b) => b.curTrials.size > 0,
      make      : (b) => b._jepaCurriculumPanel(),
      paint     : (b) => b._paintJepaCurriculum(),
      summary   : (b, gpus) => {
        const n = b._jepaCurriculumNames().length;
        return `${b._curEpochs().length} swap epoch${b._curEpochs().length === 1 ? "" : "s"} x ${b._curResets().length} reset${b._curResets().length === 1 ? "" : "s"}${b._curFlag("include_baseline") ? " + baseline" : ""} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b._jepaCurriculumNames().map((rung) => `${model}_${rung}`) },

    { group: "arch", key: "coupling", label: "AE coupling", hint: "profile-encoder freeze modes, fine-tune rates and target providers, one trial per arm",
      at        : 27,
      el        : "couplingEl",
      available : (b) => b.couplingTrials.size > 0,
      make      : (b) => b._couplingPanel(),
      paint     : (b) => b._paintCoupling(),
      summary   : (b, gpus) => {
        const n = b._couplingNames().length;
        return `${b._couplingModes().length} mode${b._couplingModes().length === 1 ? "" : "s"} x ${b._couplingProviders().length} provider${b._couplingProviders().length === 1 ? "" : "s"}, ${b._couplingLrs().length} finetune lr${b._couplingLrs().length === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b._couplingNames().map((rung) => `${model}_${rung}`) },

    { group: "arch", key: "ae_run", label: "AE run", hint: "one JEPA training per listed profile-autoencoder run, comparing pretrained autoencoders by downstream performance",
      at        : 28,
      el        : "aeRunEl",
      available : (b) => b.aeRunTrials.size > 0,
      make      : (b) => b._aeRunPanel(),
      paint     : (b) => b._paintAeRun(),
      summary   : (b, gpus) => {
        const n = b._aeRuns().length;
        return `${n} autoencoder run${n === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b._aeRuns().map((run) => `${model}_${b._aeRunName(run)}`) },

    { group: "arch", key: "capacity", label: "solver capacity", hint: "proximal width x unrolled iteration count of the configured unrolled family; empty axes walk the family's tunable choices",
      at        : 29,
      el        : "solverCapacityEl",
      available : (b) => b.capacityTrials.has("iterations"),
      make      : (b) => b._solverCapacityPanel(),
      paint     : (b) => b._paintSolverCapacity(),
      summary   : (b, gpus) => {
        const widths     = b._capacityValues("widths");
        const iterations = b._capacityValues("iterations");
        if (!widths.length || !iterations.length) return "prox width x unrolled iterations grid from the family's tunable choices";
        const total = widths.length * iterations.length;
        return `${widths.length} width${widths.length === 1 ? "" : "s"} x ${iterations.length} iteration count${iterations.length === 1 ? "" : "s"} = ${total} trial${total === 1 ? "" : "s"}${b._seedsSuffix(total)}${gpus}`;
      },
      names     : (b, model) => {
        const widths     = b._capacityValues("widths");
        const iterations = b._capacityValues("iterations");
        const names      = [];
        widths.forEach((width) => iterations.forEach((count) => names.push(`${model}_cap-h${width}-i${count}`)));
        return names;
      },
      pattern   : (b, model) => {
        if (b._capacityValues("widths").length && b._capacityValues("iterations").length) return null;
        return { text: `${model}_cap-h<width>-i<iterations>`, note: "one per cell, axes resolved from the family's tunable choices at launch" };
      } },

    { group: "arch", key: "prox", label: "proximal shape", hint: "z kernel length x activation of the learned proximal operator, at the width and unrolling depth the base config fixes",
      at        : 30,
      el        : "proxEl",
      available : (b) => b.proxTrials.size > 0,
      make      : (b) => b._proxPanel(),
      paint     : (b) => b._paintProx(),
      summary   : (b, gpus) => {
        const kernels     = b._proxKernels();
        const activations = b._proxActivations();
        if (!kernels.length || !activations.length) return "kernel x activation grid from the family's tunable choices";
        const total = kernels.length * activations.length;
        return `${kernels.length} kernel${kernels.length === 1 ? "" : "s"} x ${activations.length} activation${activations.length === 1 ? "" : "s"} = ${total} trial${total === 1 ? "" : "s"}${b._seedsSuffix(total)}${gpus}`;
      },
      names     : (b, model) => {
        const names = [];
        b._proxKernels().forEach((size) => b._proxActivations().forEach((activation) => names.push(`${model}_prox-k${size}-${activation}`)));
        return names;
      },
      pattern   : (b, model) => {
        if (b._proxKernels().length && b._proxActivations().length) return null;
        return { text: `${model}_prox-k<kernel>-<activation>`, note: "one per cell, axes resolved from the family's tunable choices at launch" };
      } },

    { group: "loss", key: "curve_loss", label: "curve loss", hint: "one trial per penalty family the unrolled criterion offers on the reconstructed z profile",
      at        : 31,
      el        : "curveLossEl",
      available : (b) => b.curveLossTrials.size > 0,
      make      : (b) => b._curveLossPanel(),
      paint     : (b) => b._paintCurveLoss(),
      summary   : (b, gpus) => {
        const n = b._curveLossKinds().length;
        return `${n} penalty famil${n === 1 ? "y" : "ies"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b._curveLossKinds().map((kind) => `${model}_loss-${kind}`) },

    { group: "data", key: "noise", label: "measurement noise", hint: "noise ladder over the synthesised coherences, the only noise an unrolled run sees, plus an optional noise-free rung",
      at        : 32,
      el        : "noiseEl",
      available : (b) => b.noiseTrials.size > 0,
      make      : (b) => b._noisePanel(),
      paint     : (b) => b._paintNoise(),
      summary   : (b, gpus) => {
        const stds = b._noiseStds();
        const n    = b._noiseNames().length;
        return `${b._noiseFlag("include_baseline") ? "clean + " : ""}${stds.length} noise level${stds.length === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b._noiseNames().map((rung) => `${model}_${rung}`) },
  ];

  static MODE_GROUPS = [
    { key: "loss",     label: "loss" },
    { key: "data",     label: "data" },
    { key: "arch",     label: "architecture" },
    { key: "training", label: "training" },
  ];

  static CURVE_KINDS = ["mse", "l1", "huber", "charbonnier"];

  static CURVE_LOSS_KINDS = ["l1", "mse"];

  static PROX_ACTIVATIONS = ["relu", "leaky_relu", "gelu", "silu"];

  static MASK_STRATEGIES   = ["block", "random"];
  static MASK_FILLS        = ["zeros", "noise"];
  static MASK_LOSS_TARGETS = ["masked", "all"];

  static CURRICULUM_RESETS = ["none", "lr", "warmup", "optimizer", "all"];

  static COUPLING_MODES     = ["frozen", "finetune"];
  static COUPLING_PROVIDERS = ["stopgrad", "live"];

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

  static PHYSICS_COMPONENTS = [
    { key: "total_power", label: "total power" },
    { key: "moments",     label: "profile moments" },
  ];

  static PAIR_COMPONENTS = [
    "mse_curve", "l1_curve", "huber_curve", "charbonnier_curve", "cosine_curve", "log_energy_curve",
    "total_power_relerr", "moments",
    "param_huber", "param_mse", "smoothness_tv", "param_l1",
  ];

  static SWEEPS = {
    physics: {
      group       : "physics",
      elProp      : "physicsEl",
      panelClass  : "exp-secondary exp-patch exp-physics",
      title       : "physics loss sweep",
      note        : "each checked component trains once per weight on top of the base config, with every other physics term disabled in both stages; the loss curriculum stays on, so the tested term joins only after the swap epoch",
      flags       : [{ field: "include_baseline", label: "baseline", hint: "prepend one run of the plain base config with every physics term disabled, as the reference" }],
      catalog     : { head: "components", options: ExperimentBuilder.PHYSICS_COMPONENTS, onTitle: "Click to drop this component from the sweep", offTitle: "Click to sweep this component" },
      weightTitle : "physics loss weight applied to the tested component",
      base        : null,
    },
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
    this.modelLeaf    = byPath.get("backbone_name") || byPath.get("model_name") || byPath.get("ae_model_name");
    this.gpusLeaf     = byPath.get("gpus");
    this.seedsLeaf    = byPath.get("seeds");

    this.presenceTrialsLeaf = byPath.get("presence_trials");

    this.inputTrialsLeaf   = byPath.get("input_trials");
    this.contextTrialsLeaf = byPath.get("context_trials");

    this.archTrialsLeaf       = byPath.get("arch_trials");
    this.aeLossTrialsLeaf     = byPath.get("loss_trials");
    this.embeddingCatalogLeaf = byPath.get("embedding_trials");

    this.secondary       = new Map();
    this.patch           = new Map();
    this.physics         = new Map();
    this.pair            = new Map();
    this.logEnergy       = new Map();
    this.reach           = new Map();
    this.headTrials      = new Map();
    this.augTrials       = new Map();
    this.clampTrials     = new Map();
    this.stabilityTrials = new Map();
    this.emaTrials       = new Map();
    this.routingTrials   = new Map();
    this.ratioTrials     = new Map();
    this.embDimTrials    = new Map();
    this.capacityTrials  = new Map();
    this.smoothTrials    = new Map();
    this.samplingTrials  = new Map();
    this.curveTrials     = new Map();
    this.maskingTrials   = new Map();
    this.curTrials       = new Map();
    this.couplingTrials  = new Map();
    this.aeRunTrials     = new Map();
    this.proxTrials      = new Map();
    this.curveLossTrials = new Map();
    this.noiseTrials     = new Map();
    byPath.forEach((leaf) => {
      if (leaf.section === "secondary_trials")     this.secondary.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "patch_trials")         this.patch.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "physics_trials")       this.physics.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "pair_trials")          this.pair.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "log_energy_trials")    this.logEnergy.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "reach_trials")         this.reach.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "head_trials")          this.headTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "augmentation_trials")  this.augTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "clamp_trials")         this.clampTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "stability_trials")     this.stabilityTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "ema_trials")           this.emaTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "routing_trials")       this.routingTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "ratio_trials")         this.ratioTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "embedding_trials")     this.embDimTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "capacity_trials")      this.capacityTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "smoothness_trials")    this.smoothTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "sampling_trials")      this.samplingTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "curve_trials")         this.curveTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "masking_trials")       this.maskingTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "curriculum_trials")    this.curTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "coupling_trials")      this.couplingTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "ae_run_trials")        this.aeRunTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "prox_trials")          this.proxTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "curve_loss_trials")    this.curveLossTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "noise_trials")         this.noiseTrials.set(leaf.path.split(".").pop(), leaf);
    });

    this.claimed = ["trials_enabled"];
    if (this.warmupLeaf)   this.claimed.push("warmup_losses");
    if (this.completeLeaf) this.claimed.push("complete_losses");
    if (this.modeLeaf) this.claimed.push("trials_mode");
    this.secondary.forEach((leaf) => this.claimed.push(leaf.path));
    this.patch.forEach((leaf) => this.claimed.push(leaf.path));
    this.physics.forEach((leaf) => this.claimed.push(leaf.path));
    this.pair.forEach((leaf) => this.claimed.push(leaf.path));
    this.logEnergy.forEach((leaf) => this.claimed.push(leaf.path));
    this.reach.forEach((leaf) => this.claimed.push(leaf.path));
    this.headTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.augTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.clampTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.stabilityTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.emaTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.routingTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.ratioTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.embDimTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.capacityTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.smoothTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.samplingTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.curveTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.maskingTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.curTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.couplingTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.aeRunTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.proxTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.curveLossTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.noiseTrials.forEach((leaf) => this.claimed.push(leaf.path));
    if (this.presenceTrialsLeaf)   this.claimed.push(this.presenceTrialsLeaf.path);
    if (this.inputTrialsLeaf)      this.claimed.push(this.inputTrialsLeaf.path);
    if (this.contextTrialsLeaf)    this.claimed.push(this.contextTrialsLeaf.path);
    if (this.archTrialsLeaf)       this.claimed.push(this.archTrialsLeaf.path);
    if (this.aeLossTrialsLeaf)     this.claimed.push(this.aeLossTrialsLeaf.path);
    if (this.embeddingCatalogLeaf) this.claimed.push(this.embeddingCatalogLeaf.path);

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
    this.physicsEl           = null;
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
    this.headBackboneEl  = null;
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
    this.routingEl       = null;
    this.routingCellsEl  = null;
    this.ratioEl         = null;
    this.ratioCellsEl    = null;
    this.archEl          = null;
    this.archCellsEl     = null;
    this.embDimEl        = null;
    this.embDimListEl    = null;
    this.capacityEl        = null;
    this.capacityWidthsEl  = null;
    this.capacityDepthsEl  = null;
    this.aeLossEl        = null;
    this.aeLossCellsEl   = null;
    this.smoothEl          = null;
    this.smoothSobolevEl   = null;
    this.smoothNoiseEl     = null;
    this.smoothFlagsEl     = null;
    this.samplingEl        = null;
    this.samplingSubsEl    = null;
    this.samplingEmptyEl   = null;
    this.jepaEmbEl       = null;
    this.jepaEmbCellsEl  = null;
    this.curveEl           = null;
    this.curveKindsEl      = null;
    this.curveWeightsEl    = null;
    this.curveSobolevEl    = null;
    this.curveFlagsEl      = null;
    this.maskingEl          = null;
    this.maskingStrategiesEl = null;
    this.maskingRatiosEl    = null;
    this.maskingSelects     = new Map();
    this.maskingFlagsEl     = null;
    this.jepaCurEl         = null;
    this.curEpochsEl       = null;
    this.curResetsEl       = null;
    this.curFlagsEl        = null;
    this.couplingEl        = null;
    this.couplingModesEl   = null;
    this.couplingLrsEl     = null;
    this.couplingProvidersEl = null;
    this.aeRunEl         = null;
    this.aeRunCellsEl    = null;
    this.aeRunInputEl    = null;
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
      if (leaf) grid.appendChild(this._secondaryRow(field, leaf));
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
    note.textContent = "Each variant toggles which input channels feed the model and scopes its track list: tracks=all expands the secondaries to every track in the baselines table, tracks=reduced keeps the configured selection, and every base variant comes as an explicit no-Aux and with-Aux version. Drop variants to trim the fan-out. Adding variants and channel representations live in code (TrialCatalog.input_trials); use reset variants to restore the default.";

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

  _ladderShell(title, note, reset, adds) {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">${title}</span>`;

    if (reset) {
      const button = LaunchWidgetDom.mini("reset ladder", reset);
      button.classList.add("exp-presence__reset");
      head.appendChild(button);
    }
    (adds || []).forEach((entry) => head.appendChild(LaunchWidgetDom.mini(entry[0], entry[1])));

    const noteEl       = document.createElement("p");
    noteEl.className   = "exp-secondary__note";
    noteEl.textContent = note;

    panel.appendChild(head);
    panel.appendChild(noteEl);
    return { panel, head };
  }

  _numberListSection(panel, subHead, leaf, repaint) {
    panel.appendChild(this._subHead(subHead));

    const grid     = document.createElement("div");
    grid.className = "exp-patch__sizes";
    panel.appendChild(grid);

    if (leaf) this.view.controls[leaf.path] = { leaf, reset: repaint };
    return grid;
  }

  _cellsSection(panel, subHead, leaf, repaint) {
    panel.appendChild(this._subHead(subHead));

    const cells     = document.createElement("div");
    cells.className = "exp-builder__names exp-presence__cells";
    panel.appendChild(cells);

    if (leaf) this.view.controls[leaf.path] = { leaf, reset: repaint };
    return cells;
  }

  _catalogSection(panel, subHead, leaf, repaint) {
    panel.appendChild(this._subHead(subHead));

    const grid     = document.createElement("div");
    grid.className = "exp-builder__names exp-presence__strategies";
    panel.appendChild(grid);

    if (leaf) this.view.controls[leaf.path] = { leaf, reset: repaint };
    return grid;
  }

  _emitNumbers(leaf, values, keep, repaint) {
    if (!leaf) return;

    const clean = values.map((value) => Number(value)).filter((value) => Number.isFinite(value) && keep(value));
    this.view._setValue(leaf, PythonLiteral.render(clean));
    repaint();
  }

  _emitStrings(leaf, values, repaint) {
    if (!leaf) return;

    this.view._setValue(leaf, PythonLiteral.render(values));
    repaint();
  }

  _paintNumberChips(el, values, title, emit) {
    if (!el) return;

    el.innerHTML = "";
    values.forEach((value, index) => el.appendChild(this._weightChip(value, index, values, title, emit)));
  }

  _paintCatalogChips(el, order, active, onTitle, offTitle, emit) {
    if (!el) return;

    el.innerHTML = "";
    order.forEach((key) => el.appendChild(this._toggleChip(key, active.includes(key), onTitle, offTitle, () => this._toggleInCatalog(order, active, key, emit))));
  }

  _choiceRow(leaf, label, hint, options, registry, repaint) {
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
      repaint();
    });

    registry.set(leaf.path, select);
    this.view.controls[leaf.path] = { leaf, reset: repaint };

    row.appendChild(name);
    row.appendChild(select);
    return row;
  }

  _paintChoiceRows(registry, group) {
    registry.forEach((select, path) => {
      const leaf = group.get(path.split(".").pop());
      if (leaf && document.activeElement !== select) select.value = this.view._effective(leaf);
    });
  }

  _flagsRow(panel, group, flags, repaint) {
    const row     = document.createElement("div");
    row.className = "exp-secondary__grid";

    flags.forEach((flag) => {
      const leaf = group.get(flag[0]);
      if (leaf) row.appendChild(this._ladderFlag(leaf, flag[0], flag[1], repaint));
    });

    panel.appendChild(row);
    return row;
  }

  _ladderFlag(leaf, label, hint, repaint) {
    const wrap  = this._patchFlag(leaf, label, hint);
    const paint = this.view.controls[leaf.path].reset;

    wrap.querySelector("button").addEventListener("click", repaint);
    this.view.controls[leaf.path] = { leaf, reset: () => { paint(); repaint(); } };
    return wrap;
  }

  _groupFlag(group, key) {
    const leaf = group.get(key);
    return leaf ? this.view._effective(leaf) === "True" : false;
  }

  _resetGroup(group, repaint) {
    group.forEach((leaf) => this.view._setValue(leaf, leaf.value));
    repaint();
  }

  _resetLeaf(leaf, repaint) {
    if (leaf) this.view._setValue(leaf, leaf.value);
    repaint();
  }

  _isProfileAug() {
    return this.augTrials.has("shift_probability");
  }

  _archRungs() {
    return this._listOf(this.archTrialsLeaf, (rung) => rung && rung.label && rung.ae_model_name);
  }

  _archPanel() {
    const shell = this._ladderShell("autoencoder families", "One training run per rung on the shared base config, each rung a label plus an ae_model_name from the profile-autoencoder registry; any further key on a rung is applied as a model override. Add rungs by editing arch_trials, use reset ladder to restore the default set. Every rung repeats across the seeds list.", () => this._resetLeaf(this.archTrialsLeaf, () => this._repaintArch()), null);

    this.archCellsEl = this._cellsSection(shell.panel, "families", this.archTrialsLeaf, () => this._repaintArch());
    this.archEl      = shell.panel;

    this._paintArch();
    return shell.panel;
  }

  _paintArch() {
    if (!this.archCellsEl) return;

    this.archCellsEl.innerHTML = "";
    this._archRungs().forEach((rung) => {
      this.archCellsEl.appendChild(this._cellChip(`${rung.label} · ${rung.ae_model_name}`, null, "Remove rung", () => this._removeArchRung(rung.label)));
    });
  }

  _removeArchRung(label) {
    if (!this.archTrialsLeaf) return;

    const rungs = this._archRungs();
    if (rungs.length <= 1) return;

    this.view._setValue(this.archTrialsLeaf, PythonLiteral.render(rungs.filter((rung) => rung.label !== label)));
    this._repaintArch();
  }

  _repaintArch() {
    this._paintArch();
    this._paintSummary();
    this._paintNames();
  }

  _embDims() {
    return this._numbersOf(this.embDimTrials.get("dims"));
  }

  _embDimPanel() {
    const leaf  = this.embDimTrials.get("dims");
    const shell = this._ladderShell("latent-width ladder", "One training run per embedding dimension on the configured autoencoder family, everything else unchanged; the entry's model_overrides must not pin embedding_dim. Every width repeats across the seeds list.", () => this._resetGroup(this.embDimTrials, () => this._repaintEmbDim()), [["Add width", () => {
      const dims = this._embDims();
      dims.push(dims.length ? dims[dims.length - 1] : 24);
      this._emitNumbers(leaf, dims, (value) => value > 0, () => this._repaintEmbDim());
    }]]);

    this.embDimListEl = this._numberListSection(shell.panel, "embedding dimensions (one run each)", leaf, () => this._repaintEmbDim());
    this.embDimEl     = shell.panel;

    this._paintEmbDim();
    return shell.panel;
  }

  _paintEmbDim() {
    const leaf = this.embDimTrials.get("dims");
    this._paintNumberChips(this.embDimListEl, this._embDims(), "latent width of this run", (next) => this._emitNumbers(leaf, next, (value) => value > 0, () => this._repaintEmbDim()));
  }

  _repaintEmbDim() {
    this._paintEmbDim();
    this._paintSummary();
    this._paintNames();
  }

  _capacityValues(key) {
    return this._numbersOf(this.capacityTrials.get(key));
  }

  _capacityPanel() {
    const widthLeaf = this.capacityTrials.get("widths");
    const depthLeaf = this.capacityTrials.get("depths");
    const shell     = this._ladderShell("capacity grid", "Full width by depth grid within the configured autoencoder family: the width axis drives the family's own width field (hidden_dim or seq_channels). Leaving an axis empty walks that family's tunable choices instead, resolved at launch. Every cell repeats across the seeds list.", () => this._resetGroup(this.capacityTrials, () => this._repaintCapacity()), [
      ["Add width", () => {
        const widths = this._capacityValues("widths");
        widths.push(widths.length ? widths[widths.length - 1] : 512);
        this._emitNumbers(widthLeaf, widths, (value) => value > 0, () => this._repaintCapacity());
      }],
      ["Add depth", () => {
        const depths = this._capacityValues("depths");
        depths.push(depths.length ? depths[depths.length - 1] : 4);
        this._emitNumbers(depthLeaf, depths, (value) => value > 0, () => this._repaintCapacity());
      }],
    ]);

    this.capacityWidthsEl = this._numberListSection(shell.panel, "widths (empty walks the family's tunable choices)", widthLeaf, () => this._repaintCapacity());
    this.capacityDepthsEl = this._numberListSection(shell.panel, "depths (empty walks the family's tunable choices)", depthLeaf, () => this._repaintCapacity());
    this.capacityEl       = shell.panel;

    this._paintCapacity();
    return shell.panel;
  }

  _paintCapacity() {
    const widthLeaf = this.capacityTrials.get("widths");
    const depthLeaf = this.capacityTrials.get("depths");

    this._paintNumberChips(this.capacityWidthsEl, this._capacityValues("widths"), "width of this grid column", (next) => this._emitNumbers(widthLeaf, next, (value) => value > 0, () => this._repaintCapacity()));
    this._paintNumberChips(this.capacityDepthsEl, this._capacityValues("depths"), "depth of this grid row", (next) => this._emitNumbers(depthLeaf, next, (value) => value > 0, () => this._repaintCapacity()));
  }

  _repaintCapacity() {
    this._paintCapacity();
    this._paintSummary();
    this._paintNames();
  }

  _aeLossTrials() {
    return this.aeLossTrialsLeaf ? this._dictFrom(this.view._effective(this.aeLossTrialsLeaf)) : {};
  }

  _lossSpecText(spec) {
    const terms = Object.keys(spec).filter((key) => key.startsWith("use_") && spec[key]).map((key) => {
      const name   = key.slice(4);
      const weight = spec[`weight_${name}`];
      return weight === undefined ? name : `${name} ${this._pyG(weight)}`;
    });
    return terms.length ? terms.join(" + ") : "no term";
  }

  _aeLossPanel() {
    const shell = this._ladderShell("reconstruction-loss arms", "One training run per arm on the shared base config. Every arm states its loss completely: the planner disables every reconstruction, shape, smoothness and latent-noise term first, then applies the arm's own fields, so arms never inherit each other's terms. Edit loss_trials to add arms; every arm repeats across the seeds list.", () => this._resetLeaf(this.aeLossTrialsLeaf, () => this._repaintAeLoss()), null);

    this.aeLossCellsEl = this._cellsSection(shell.panel, "loss arms", this.aeLossTrialsLeaf, () => this._repaintAeLoss());
    this.aeLossEl      = shell.panel;

    this._paintAeLoss();
    return shell.panel;
  }

  _paintAeLoss() {
    if (!this.aeLossCellsEl) return;

    this.aeLossCellsEl.innerHTML = "";
    const trials = this._aeLossTrials();
    Object.keys(trials).forEach((label) => {
      this.aeLossCellsEl.appendChild(this._trialChip(label, this._lossSpecText(trials[label]), () => this._removeTrialKey(this.aeLossTrialsLeaf, label, () => this._repaintAeLoss())));
    });
  }

  _repaintAeLoss() {
    this._paintAeLoss();
    this._paintSummary();
    this._paintNames();
  }

  _smoothSobolev() {
    return this._numbersOf(this.smoothTrials.get("sobolev_weights"));
  }

  _smoothNoise() {
    return this._numbersOf(this.smoothTrials.get("latent_noise_stds"));
  }

  _smoothFlag(key) {
    return this._groupFlag(this.smoothTrials, key);
  }

  _smoothnessNames() {
    const names = this._smoothFlag("include_baseline") ? ["smooth-baseline"] : [];

    this._smoothSobolev().forEach((weight) => names.push(`smooth-sob${this._pyG(weight)}`));
    this._smoothNoise().forEach((std) => names.push(`smooth-ln${this._pyG(std)}`));

    if (this._smoothFlag("combine")) {
      this._smoothSobolev().forEach((weight) => this._smoothNoise().forEach((std) => names.push(`smooth-sob${this._pyG(weight)}-ln${this._pyG(std)}`)));
    }
    return names;
  }

  _smoothnessPanel() {
    const sobolevLeaf = this.smoothTrials.get("sobolev_weights");
    const noiseLeaf   = this.smoothTrials.get("latent_noise_stds");
    const shell       = this._ladderShell("smoothness levers", "Decoder-robustness arms on top of the configured reconstruction loss. One rung per Sobolev weight penalises the curvature of the reconstruction residual; one rung per latent-noise std additionally decodes a jittered embedding against the clean target, which is what trains the decoder to tolerate the embedding error a coupled JEPA makes. Read the std off a JEPA run's Embedding RMSE diagnostic instead of guessing it. Every rung repeats across the seeds list.", () => this._resetGroup(this.smoothTrials, () => this._repaintSmoothness()), [
      ["Add sobolev", () => {
        const weights = this._smoothSobolev();
        weights.push(weights.length ? weights[weights.length - 1] : 0.5);
        this._emitNumbers(sobolevLeaf, weights, (value) => value > 0, () => this._repaintSmoothness());
      }],
      ["Add std", () => {
        const stds = this._smoothNoise();
        stds.push(stds.length ? stds[stds.length - 1] : 0.05);
        this._emitNumbers(noiseLeaf, stds, (value) => value > 0, () => this._repaintSmoothness());
      }],
    ]);

    this.smoothFlagsEl   = this._flagsRow(shell.panel, this.smoothTrials, [["combine", "also train the full sobolev x latent-noise grid on top of the one-at-a-time rungs"], ["include_baseline", "add a rung with both levers off"]], () => this._repaintSmoothness());
    this.smoothSobolevEl = this._numberListSection(shell.panel, "sobolev weights (one rung each, residual curvature)", sobolevLeaf, () => this._repaintSmoothness());
    this.smoothNoiseEl   = this._numberListSection(shell.panel, "latent noise stds (one rung each, normalised embedding units)", noiseLeaf, () => this._repaintSmoothness());
    this.smoothEl        = shell.panel;

    this._paintSmoothness();
    return shell.panel;
  }

  _paintSmoothness() {
    const sobolevLeaf = this.smoothTrials.get("sobolev_weights");
    const noiseLeaf   = this.smoothTrials.get("latent_noise_stds");

    this._paintNumberChips(this.smoothSobolevEl, this._smoothSobolev(), "sobolev weight of this rung", (next) => this._emitNumbers(sobolevLeaf, next, (value) => value > 0, () => this._repaintSmoothness()));
    this._paintNumberChips(this.smoothNoiseEl, this._smoothNoise(), "latent noise std of this rung", (next) => this._emitNumbers(noiseLeaf, next, (value) => value > 0, () => this._repaintSmoothness()));
  }

  _repaintSmoothness() {
    this._paintSmoothness();
    this._paintSummary();
    this._paintNames();
  }

  _samplingValues(key) {
    return this._numbersOf(this.samplingTrials.get(key));
  }

  _samplingPanel() {
    const subsLeaf  = this.samplingTrials.get("subsamples");
    const emptyLeaf = this.samplingTrials.get("empty_fracs");
    const shell     = this._ladderShell("data-mix grid", "Full grid over the two dataset knobs: the fraction of scene pixels kept as training profiles, and the fraction of empty profiles kept in that sample. Together they set how much data the autoencoder sees and how much of it is background, which is the trade the reconstruction quality is most sensitive to. Every cell repeats across the seeds list.", () => this._resetGroup(this.samplingTrials, () => this._repaintSampling()), [
      ["Add subsample", () => {
        const subs = this._samplingValues("subsamples");
        subs.push(subs.length ? subs[subs.length - 1] : 1.0);
        this._emitNumbers(subsLeaf, subs, (value) => value > 0 && value <= 1, () => this._repaintSampling());
      }],
      ["Add fraction", () => {
        const fracs = this._samplingValues("empty_fracs");
        fracs.push(fracs.length ? fracs[fracs.length - 1] : 0.05);
        this._emitNumbers(emptyLeaf, fracs, (value) => value >= 0 && value <= 1, () => this._repaintSampling());
      }],
    ]);

    this.samplingSubsEl  = this._numberListSection(shell.panel, "pixel subsamples (one grid column each, in (0, 1])", subsLeaf, () => this._repaintSampling());
    this.samplingEmptyEl = this._numberListSection(shell.panel, "kept empty fractions (one grid row each, in [0, 1])", emptyLeaf, () => this._repaintSampling());
    this.samplingEl      = shell.panel;

    this._paintSampling();
    return shell.panel;
  }

  _paintSampling() {
    const subsLeaf  = this.samplingTrials.get("subsamples");
    const emptyLeaf = this.samplingTrials.get("empty_fracs");

    this._paintNumberChips(this.samplingSubsEl, this._samplingValues("subsamples"), "pixel subsample of this grid column", (next) => this._emitNumbers(subsLeaf, next, (value) => value > 0 && value <= 1, () => this._repaintSampling()));
    this._paintNumberChips(this.samplingEmptyEl, this._samplingValues("empty_fracs"), "kept empty fraction of this grid row", (next) => this._emitNumbers(emptyLeaf, next, (value) => value >= 0 && value <= 1, () => this._repaintSampling()));
  }

  _repaintSampling() {
    this._paintSampling();
    this._paintSummary();
    this._paintNames();
  }

  _profileAugNames() {
    const names = ["aug-0-none", "aug-1-flip", "aug-2-shift", "aug-3-scale"];
    this._augNoiseStds().forEach((std) => names.push(`aug-4-noise${this._pyG(std)}`));
    return names;
  }

  _profileAugPanel() {
    const stdsLeaf = this.augTrials.get("noise_stds");
    const shell    = this._ladderShell("profile augmentation ladder", "Cumulative ladder on the shared base config: rung 0 trains without augmentation, rung 1 adds profile flips, rung 2 adds height shifts, rung 3 adds amplitude scaling, and one final rung per noise std adds input noise on top of all three. The noise rungs train against the clean profile, so they are denoising rungs rather than noise reproduction. Every rung repeats across the seeds list.", () => this._resetGroup(this.augTrials, () => this._repaintProfileAug()), [["Add std", () => {
      const stds = this._augNoiseStds();
      stds.push(stds.length ? stds[stds.length - 1] : 0.01);
      this._emitNumbers(stdsLeaf, stds, (value) => value > 0, () => this._repaintProfileAug());
    }]]);

    const grid     = document.createElement("div");
    grid.className = "exp-secondary__grid";

    [
      ["flip_probability",      "probability of flipping the profile from the flip rung on"],
      ["shift_probability",     "probability of shifting the profile from the shift rung on"],
      ["max_shift",             "largest absolute height-bin shift of the shift augmentation"],
      ["amp_scale_probability", "probability of rescaling the amplitude from the scale rung on"],
      ["noise_probability",     "probability of injecting noise on the noise rungs"],
    ].forEach((entry) => {
      const leaf = this.augTrials.get(entry[0]);
      if (leaf) grid.appendChild(this._ladderNumberRow(leaf, entry[0], entry[1], "any", "0"));
    });

    shell.panel.appendChild(grid);
    this.augStdsEl = this._numberListSection(shell.panel, "noise stds (one rung each, normalised profile units)", stdsLeaf, () => this._repaintProfileAug());
    this.augEl     = shell.panel;

    this._paintProfileAug();
    return shell.panel;
  }

  _paintProfileAug() {
    const stdsLeaf = this.augTrials.get("noise_stds");
    this._paintNumberChips(this.augStdsEl, this._augNoiseStds(), "noise standard deviation of this rung", (next) => this._emitNumbers(stdsLeaf, next, (value) => value > 0, () => this._repaintProfileAug()));
  }

  _repaintProfileAug() {
    this._paintProfileAug();
    this._paintSummary();
    this._paintNames();
  }

  _jepaEmbeddingTrials() {
    return this.embeddingCatalogLeaf ? this._dictFrom(this.view._effective(this.embeddingCatalogLeaf)) : {};
  }

  _jepaEmbeddingPanel() {
    const shell = this._ladderShell("latent-loss arms", "One training run per arm on the shared base config. Every arm states its latent loss completely: the planner disables every embedding term first, then applies the arm's own fields, so arms never inherit each other's terms. The decoded-curve settings stay at the base config, which is what isolates the choice of latent penalty and collapse regulariser; sweep the curve terms with the decoded curve mode instead. Every arm repeats across the seeds list.", () => this._resetLeaf(this.embeddingCatalogLeaf, () => this._repaintJepaEmbedding()), null);

    this.jepaEmbCellsEl = this._cellsSection(shell.panel, "latent-loss arms", this.embeddingCatalogLeaf, () => this._repaintJepaEmbedding());
    this.jepaEmbEl      = shell.panel;

    this._paintJepaEmbedding();
    return shell.panel;
  }

  _paintJepaEmbedding() {
    if (!this.jepaEmbCellsEl) return;

    this.jepaEmbCellsEl.innerHTML = "";
    const trials = this._jepaEmbeddingTrials();
    Object.keys(trials).forEach((label) => {
      this.jepaEmbCellsEl.appendChild(this._trialChip(label, this._lossSpecText(trials[label]), () => this._removeTrialKey(this.embeddingCatalogLeaf, label, () => this._repaintJepaEmbedding())));
    });
  }

  _repaintJepaEmbedding() {
    this._paintJepaEmbedding();
    this._paintSummary();
    this._paintNames();
  }

  _curveKinds() {
    return this._listOf(this.curveTrials.get("kinds"), (value) => typeof value === "string");
  }

  _curveWeights() {
    return this._numbersOf(this.curveTrials.get("weights"));
  }

  _curveSobolev() {
    return this._numbersOf(this.curveTrials.get("sobolev_weights"));
  }

  _curveFlag(key) {
    return this._groupFlag(this.curveTrials, key);
  }

  _curveNames() {
    const names = this._curveFlag("include_baseline") ? ["curve-off"] : [];

    this._curveKinds().forEach((kind) => this._curveWeights().forEach((weight) => names.push(`curve-${kind}${this._pyG(weight)}`)));
    this._curveSobolev().forEach((weight) => names.push(`curve-sob${this._pyG(weight)}`));
    return names;
  }

  _curvePanel() {
    const kindsLeaf   = this.curveTrials.get("kinds");
    const weightsLeaf = this.curveTrials.get("weights");
    const sobolevLeaf = this.curveTrials.get("sobolev_weights");
    const shell       = this._ladderShell("decoded-curve arms", "Arms on the decoded profile, on top of whatever latent terms the base config enables. The grid crosses each penalty kind with each weight; the Sobolev rungs instead fix an mse curve term at weight 1 and add the curvature penalty, which is the lever against decoding jitter. Every arm repeats across the seeds list.", () => this._resetGroup(this.curveTrials, () => this._repaintCurve()), [
      ["Add weight", () => {
        const weights = this._curveWeights();
        weights.push(weights.length ? weights[weights.length - 1] : 1.0);
        this._emitNumbers(weightsLeaf, weights, (value) => value > 0, () => this._repaintCurve());
      }],
      ["Add sobolev", () => {
        const weights = this._curveSobolev();
        weights.push(weights.length ? weights[weights.length - 1] : 1.0);
        this._emitNumbers(sobolevLeaf, weights, (value) => value > 0, () => this._repaintCurve());
      }],
    ]);

    this.curveFlagsEl   = this._flagsRow(shell.panel, this.curveTrials, [["include_baseline", "add a rung with the decoded-curve terms off, training on the latent terms alone"]], () => this._repaintCurve());
    this.curveKindsEl   = this._catalogSection(shell.panel, "penalty kinds (crossed with the weights below)", kindsLeaf, () => this._repaintCurve());
    this.curveWeightsEl = this._numberListSection(shell.panel, "curve weights", weightsLeaf, () => this._repaintCurve());
    this.curveSobolevEl = this._numberListSection(shell.panel, "sobolev weights (one rung each, on an mse curve term at weight 1)", sobolevLeaf, () => this._repaintCurve());
    this.curveEl        = shell.panel;

    this._paintCurve();
    return shell.panel;
  }

  _paintCurve() {
    const kindsLeaf   = this.curveTrials.get("kinds");
    const weightsLeaf = this.curveTrials.get("weights");
    const sobolevLeaf = this.curveTrials.get("sobolev_weights");

    this._paintCatalogChips(this.curveKindsEl, ExperimentBuilder.CURVE_KINDS, this._curveKinds(), "drop this penalty kind from the grid", "add this penalty kind to the grid", (next) => this._emitStrings(kindsLeaf, next, () => this._repaintCurve()));
    this._paintNumberChips(this.curveWeightsEl, this._curveWeights(), "weight of the decoded-curve term in this arm", (next) => this._emitNumbers(weightsLeaf, next, (value) => value > 0, () => this._repaintCurve()));
    this._paintNumberChips(this.curveSobolevEl, this._curveSobolev(), "sobolev weight of this rung", (next) => this._emitNumbers(sobolevLeaf, next, (value) => value > 0, () => this._repaintCurve()));
  }

  _repaintCurve() {
    this._paintCurve();
    this._paintSummary();
    this._paintNames();
  }

  _maskingStrategies() {
    return this._listOf(this.maskingTrials.get("strategies"), (value) => typeof value === "string");
  }

  _maskingRatios() {
    return this._numbersOf(this.maskingTrials.get("ratios"));
  }

  _maskingFlag(key) {
    return this._groupFlag(this.maskingTrials, key);
  }

  _maskingNames() {
    const names = this._maskingFlag("include_baseline") ? ["mask-off"] : [];

    this._maskingStrategies().forEach((strategy) => this._maskingRatios().forEach((ratio) => names.push(`mask-${strategy}${this._pyG(ratio)}`)));
    return names;
  }

  _maskingPanel() {
    const strategiesLeaf = this.maskingTrials.get("strategies");
    const ratiosLeaf     = this.maskingTrials.get("ratios");
    const shell          = this._ladderShell("input-masking grid", "Hides part of every input patch before the forward pass, turning the run into context-to-target prediction. The grid crosses each strategy with each hidden-pixel ratio; the fill and the loss target below apply to every arm, and restricting the loss to the masked pixels is what makes the run a genuine JEPA rather than a masked-input regularisation. Masking needs a coupled profile autoencoder. Every arm repeats across the seeds list.", () => this._resetGroup(this.maskingTrials, () => this._repaintMasking()), [["Add ratio", () => {
      const ratios = this._maskingRatios();
      ratios.push(ratios.length ? ratios[ratios.length - 1] : 0.5);
      this._emitNumbers(ratiosLeaf, ratios, (value) => value > 0 && value < 1, () => this._repaintMasking());
    }]]);

    this.maskingFlagsEl = this._flagsRow(shell.panel, this.maskingTrials, [["include_baseline", "add an unmasked rung"]], () => this._repaintMasking());

    const grid     = document.createElement("div");
    grid.className = "exp-secondary__grid";

    const fillLeaf = this.maskingTrials.get("fill");
    if (fillLeaf) grid.appendChild(this._choiceRow(fillLeaf, "fill", "value written into hidden pixels on every arm", ExperimentBuilder.MASK_FILLS, this.maskingSelects, () => this._repaintMasking()));

    const lossLeaf = this.maskingTrials.get("loss_on");
    if (lossLeaf) grid.appendChild(this._choiceRow(lossLeaf, "loss_on", "pixels the embedding loss is evaluated on for every arm", ExperimentBuilder.MASK_LOSS_TARGETS, this.maskingSelects, () => this._repaintMasking()));

    shell.panel.appendChild(grid);

    this.maskingStrategiesEl = this._catalogSection(shell.panel, "strategies (crossed with the ratios below)", strategiesLeaf, () => this._repaintMasking());
    this.maskingRatiosEl     = this._numberListSection(shell.panel, "hidden-pixel ratios (strictly between 0 and 1)", ratiosLeaf, () => this._repaintMasking());
    this.maskingEl           = shell.panel;

    this._paintMasking();
    return shell.panel;
  }

  _paintMasking() {
    this._paintChoiceRows(this.maskingSelects, this.maskingTrials);

    const strategiesLeaf = this.maskingTrials.get("strategies");
    const ratiosLeaf     = this.maskingTrials.get("ratios");

    this._paintCatalogChips(this.maskingStrategiesEl, ExperimentBuilder.MASK_STRATEGIES, this._maskingStrategies(), "drop this strategy from the grid", "add this strategy to the grid", (next) => this._emitStrings(strategiesLeaf, next, () => this._repaintMasking()));
    this._paintNumberChips(this.maskingRatiosEl, this._maskingRatios(), "fraction of the patch pixels hidden in this arm", (next) => this._emitNumbers(ratiosLeaf, next, (value) => value > 0 && value < 1, () => this._repaintMasking()));
  }

  _repaintMasking() {
    this._paintMasking();
    this._paintSummary();
    this._paintNames();
  }

  _curEpochs() {
    return this._numbersOf(this.curTrials.get("swap_epochs"));
  }

  _curResets() {
    return this._listOf(this.curTrials.get("resets"), (value) => typeof value === "string");
  }

  _curFlag(key) {
    return this._groupFlag(this.curTrials, key);
  }

  _jepaCurriculumNames() {
    const names = this._curFlag("include_baseline") ? ["cur-off"] : [];

    this._curEpochs().forEach((epoch) => this._curResets().forEach((reset) => names.push(`cur-e${epoch}-${reset}`)));
    return names;
  }

  _jepaCurriculumPanel() {
    const epochsLeaf = this.curTrials.get("swap_epochs");
    const resetsLeaf = this.curTrials.get("resets");
    const shell      = this._ladderShell("curriculum grid", "Training starts on the warmup stage of the active loss regime and swaps to the main loss at the swap epoch. The grid crosses each swap epoch with each reset combination, which decides how much of the optimizer state survives the swap; the run name still encodes the main loss, so a swept curriculum does not rename the runs. Every arm repeats across the seeds list.", () => this._resetGroup(this.curTrials, () => this._repaintJepaCurriculum()), [["Add epoch", () => {
      const epochs = this._curEpochs();
      epochs.push(epochs.length ? epochs[epochs.length - 1] : 15);
      this._emitNumbers(epochsLeaf, epochs, (value) => value > 0, () => this._repaintJepaCurriculum());
    }]]);

    this.curFlagsEl  = this._flagsRow(shell.panel, this.curTrials, [["include_baseline", "add a curriculum-free rung"]], () => this._repaintJepaCurriculum());
    this.curEpochsEl = this._numberListSection(shell.panel, "swap epochs (crossed with the resets below)", epochsLeaf, () => this._repaintJepaCurriculum());
    this.curResetsEl = this._catalogSection(shell.panel, "state resets applied at the swap", resetsLeaf, () => this._repaintJepaCurriculum());
    this.jepaCurEl   = shell.panel;

    this._paintJepaCurriculum();
    return shell.panel;
  }

  _paintJepaCurriculum() {
    const epochsLeaf = this.curTrials.get("swap_epochs");
    const resetsLeaf = this.curTrials.get("resets");

    this._paintNumberChips(this.curEpochsEl, this._curEpochs(), "epoch at which this arm swaps to the main loss", (next) => this._emitNumbers(epochsLeaf, next, (value) => value > 0, () => this._repaintJepaCurriculum()));
    this._paintCatalogChips(this.curResetsEl, ExperimentBuilder.CURRICULUM_RESETS, this._curResets(), "drop this reset from the grid", "add this reset to the grid", (next) => this._emitStrings(resetsLeaf, next, () => this._repaintJepaCurriculum()));
  }

  _repaintJepaCurriculum() {
    this._paintJepaCurriculum();
    this._paintSummary();
    this._paintNames();
  }

  _couplingModes() {
    return this._listOf(this.couplingTrials.get("modes"), (value) => typeof value === "string");
  }

  _couplingLrs() {
    return this._numbersOf(this.couplingTrials.get("finetune_lrs"));
  }

  _couplingProviders() {
    return this._listOf(this.couplingTrials.get("target_providers"), (value) => typeof value === "string");
  }

  _couplingNames() {
    const names = [];
    const modes = this._couplingModes();

    this._couplingProviders().forEach((provider) => {
      if (modes.includes("frozen") && provider !== "live") names.push(`cpl-${provider}-frozen`);
      if (modes.includes("finetune")) this._couplingLrs().forEach((rate) => names.push(`cpl-${provider}-ft${this._pyG(rate)}`));
    });
    return names;
  }

  _couplingPanel() {
    const modesLeaf     = this.couplingTrials.get("modes");
    const lrsLeaf       = this.couplingTrials.get("finetune_lrs");
    const providersLeaf = this.couplingTrials.get("target_providers");
    const shell         = this._ladderShell("coupling arms", "How the pretrained profile encoder is attached. Each provider contributes one frozen arm, plus one fine-tune arm per learning rate; a frozen encoder keeps the embedding target fixed, while fine-tuning lets the target drift with the backbone, which is the collapse risk stopgrad exists to bound. The live provider trains the encoder through the embedding loss itself, so it has no frozen arm and is planned on the fine-tune rates alone. Every arm repeats across the seeds list.", () => this._resetGroup(this.couplingTrials, () => this._repaintCoupling()), [["Add lr", () => {
      const rates = this._couplingLrs();
      rates.push(rates.length ? rates[rates.length - 1] : 3e-5);
      this._emitNumbers(lrsLeaf, rates, (value) => value > 0, () => this._repaintCoupling());
    }]]);

    this.couplingModesEl     = this._catalogSection(shell.panel, "encoder modes", modesLeaf, () => this._repaintCoupling());
    this.couplingProvidersEl = this._catalogSection(shell.panel, "target providers", providersLeaf, () => this._repaintCoupling());
    this.couplingLrsEl       = this._numberListSection(shell.panel, "finetune learning rates (one arm each, used only by the finetune mode)", lrsLeaf, () => this._repaintCoupling());
    this.couplingEl          = shell.panel;

    this._paintCoupling();
    return shell.panel;
  }

  _paintCoupling() {
    const modesLeaf     = this.couplingTrials.get("modes");
    const lrsLeaf       = this.couplingTrials.get("finetune_lrs");
    const providersLeaf = this.couplingTrials.get("target_providers");

    this._paintCatalogChips(this.couplingModesEl, ExperimentBuilder.COUPLING_MODES, this._couplingModes(), "drop this mode from the sweep", "add this mode to the sweep", (next) => this._emitStrings(modesLeaf, next, () => this._repaintCoupling()));
    this._paintCatalogChips(this.couplingProvidersEl, ExperimentBuilder.COUPLING_PROVIDERS, this._couplingProviders(), "drop this provider from the sweep", "add this provider to the sweep", (next) => this._emitStrings(providersLeaf, next, () => this._repaintCoupling()));
    this._paintNumberChips(this.couplingLrsEl, this._couplingLrs(), "learning rate of this fine-tune arm", (next) => this._emitNumbers(lrsLeaf, next, (value) => value > 0, () => this._repaintCoupling()));
  }

  _repaintCoupling() {
    this._paintCoupling();
    this._paintSummary();
    this._paintNames();
  }

  _aeRuns() {
    return this._listOf(this.aeRunTrials.get("runs"), (value) => typeof value === "string" && value.length > 0);
  }

  _aeRunName(run) {
    return `pae-${run.replace(/[^A-Za-z0-9._-]+/g, "_")}`;
  }

  _aeRunPanel() {
    const runsLeaf = this.aeRunTrials.get("runs");
    const shell    = this._ladderShell("autoencoder runs", "One JEPA training per listed profile-autoencoder run, everything else shared, which compares pretrained autoencoders by the downstream task instead of by their own reconstruction loss. Paste the run folder names as they sit under the profile-autoencoder log root; the embedding size and architecture come from each run's saved config. Every run repeats across the seeds list.", () => this._resetGroup(this.aeRunTrials, () => this._repaintAeRun()), null);

    const row     = document.createElement("div");
    row.className = "cfg-edit__row";
    row.title     = runsLeaf ? `run folder to append (--${runsLeaf.path})` : "run folder to append";

    const name     = document.createElement("div");
    name.className = "cfg-edit__name";
    name.innerHTML = `add run<span>str</span>`;

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input";
    input.type       = "text";
    input.spellcheck = false;
    input.placeholder = "mlp_ae_20260827_1132";
    input.addEventListener("change", () => {
      const wanted = input.value.trim();
      if (!wanted) return;

      const runs = this._aeRuns();
      if (!runs.includes(wanted)) this._emitStrings(runsLeaf, runs.concat([wanted]), () => this._repaintAeRun());
      input.value = "";
    });

    row.appendChild(name);
    row.appendChild(input);
    shell.panel.appendChild(row);

    this.aeRunInputEl = input;
    this.aeRunCellsEl = this._cellsSection(shell.panel, "coupled runs (one arm each, named after the run)", runsLeaf, () => this._repaintAeRun());
    this.aeRunEl      = shell.panel;

    this._paintAeRun();
    return shell.panel;
  }

  _paintAeRun() {
    if (!this.aeRunCellsEl) return;

    const runsLeaf = this.aeRunTrials.get("runs");
    const runs     = this._aeRuns();

    this.aeRunCellsEl.innerHTML = "";
    runs.forEach((run) => {
      this.aeRunCellsEl.appendChild(this._cellChip(`${this._aeRunName(run)} · ${run}`, run, "Remove run", () => this._emitStrings(runsLeaf, runs.filter((other) => other !== run), () => this._repaintAeRun())));
    });
  }

  _repaintAeRun() {
    this._paintAeRun();
    this._paintSummary();
    this._paintNames();
  }

  _solverCapacityPanel() {
    const widthLeaf = this.capacityTrials.get("widths");
    const countLeaf = this.capacityTrials.get("iterations");
    const shell     = this._ladderShell("solver capacity grid", "Full grid over the two capacity knobs of an unrolled solver: the channel width of the learned proximal blocks, and how many proximal-gradient iterations the network unrolls. The iteration axis is the one an unrolled network has and a feed-forward one does not, since every extra iteration is another gradient step against the measurements followed by another learned refinement. Leaving an axis empty walks the family's tunable choices instead, resolved at launch. Every cell repeats across the seeds list.", () => this._resetGroup(this.capacityTrials, () => this._repaintSolverCapacity()), [
      ["Add width", () => {
        const widths = this._capacityValues("widths");
        widths.push(widths.length ? widths[widths.length - 1] : 16);
        this._emitNumbers(widthLeaf, widths, (value) => value > 0, () => this._repaintSolverCapacity());
      }],
      ["Add iterations", () => {
        const counts = this._capacityValues("iterations");
        counts.push(counts.length ? counts[counts.length - 1] : 8);
        this._emitNumbers(countLeaf, counts, (value) => value > 0, () => this._repaintSolverCapacity());
      }],
    ]);

    this.solverWidthsEl     = this._numberListSection(shell.panel, "prox hidden widths (empty walks the family's tunable choices)", widthLeaf, () => this._repaintSolverCapacity());
    this.solverIterationsEl = this._numberListSection(shell.panel, "unrolled iterations (empty walks the family's tunable choices)", countLeaf, () => this._repaintSolverCapacity());
    this.solverCapacityEl   = shell.panel;

    this._paintSolverCapacity();
    return shell.panel;
  }

  _paintSolverCapacity() {
    const widthLeaf = this.capacityTrials.get("widths");
    const countLeaf = this.capacityTrials.get("iterations");

    this._paintNumberChips(this.solverWidthsEl, this._capacityValues("widths"), "prox width of this grid column", (next) => this._emitNumbers(widthLeaf, next, (value) => value > 0, () => this._repaintSolverCapacity()));
    this._paintNumberChips(this.solverIterationsEl, this._capacityValues("iterations"), "unrolled iteration count of this grid row", (next) => this._emitNumbers(countLeaf, next, (value) => value > 0, () => this._repaintSolverCapacity()));
  }

  _repaintSolverCapacity() {
    this._paintSolverCapacity();
    this._paintSummary();
    this._paintNames();
  }

  _proxKernels() {
    return this._numbersOf(this.proxTrials.get("kernel_sizes"));
  }

  _proxActivations() {
    return this._listOf(this.proxTrials.get("activations"), (value) => typeof value === "string");
  }

  _proxPanel() {
    const kernelLeaf     = this.proxTrials.get("kernel_sizes");
    const activationLeaf = this.proxTrials.get("activations");
    const shell          = this._ladderShell("proximal shape grid", "Shape of the learned proximal operator rather than its size: the kernel length sets how far along z one correction reaches, and the activation sets the nonlinearity the correction is built from. Kernel lengths must be odd so the convolution stays centred on its z bin. Leaving an axis empty walks the family's tunable choices instead, resolved at launch. Every cell repeats across the seeds list.", () => this._resetGroup(this.proxTrials, () => this._repaintProx()), [
      ["Add kernel", () => {
        const kernels = this._proxKernels();
        kernels.push(kernels.length ? kernels[kernels.length - 1] : 5);
        this._emitNumbers(kernelLeaf, kernels, (value) => value > 0, () => this._repaintProx());
      }],
    ]);

    this.proxKernelsEl     = this._numberListSection(shell.panel, "kernel lengths in z bins (one grid column each, odd values)", kernelLeaf, () => this._repaintProx());
    this.proxActivationsEl = this._catalogSection(shell.panel, "activations (one grid row each)", activationLeaf, () => this._repaintProx());
    this.proxEl            = shell.panel;

    this._paintProx();
    return shell.panel;
  }

  _paintProx() {
    const kernelLeaf     = this.proxTrials.get("kernel_sizes");
    const activationLeaf = this.proxTrials.get("activations");

    this._paintNumberChips(this.proxKernelsEl, this._proxKernels(), "kernel length of this grid column", (next) => this._emitNumbers(kernelLeaf, next, (value) => value > 0, () => this._repaintProx()));
    this._paintCatalogChips(this.proxActivationsEl, ExperimentBuilder.PROX_ACTIVATIONS, this._proxActivations(), "drop this activation from the grid", "add this activation to the grid", (next) => this._emitStrings(activationLeaf, next, () => this._repaintProx()));
  }

  _repaintProx() {
    this._paintProx();
    this._paintSummary();
    this._paintNames();
  }

  _curveLossKinds() {
    return this._listOf(this.curveLossTrials.get("kinds"), (value) => typeof value === "string");
  }

  _curveLossPanel() {
    const kindsLeaf = this.curveLossTrials.get("kinds");
    const shell     = this._ladderShell("curve-loss arms", "One training run per penalty family applied to the reconstructed z profile, everything else shared. The unrolled criterion is a single masked per-pixel curve error, so an arm names one family and never combines two.", () => this._resetGroup(this.curveLossTrials, () => this._repaintCurveLoss()), null);

    this.curveLossKindsEl = this._catalogSection(shell.panel, "penalty families (one arm each)", kindsLeaf, () => this._repaintCurveLoss());
    this.curveLossEl      = shell.panel;

    this._paintCurveLoss();
    return shell.panel;
  }

  _paintCurveLoss() {
    const kindsLeaf = this.curveLossTrials.get("kinds");

    this._paintCatalogChips(this.curveLossKindsEl, ExperimentBuilder.CURVE_LOSS_KINDS, this._curveLossKinds(), "drop this penalty from the sweep", "add this penalty to the sweep", (next) => this._emitStrings(kindsLeaf, next, () => this._repaintCurveLoss()));
  }

  _repaintCurveLoss() {
    this._paintCurveLoss();
    this._paintSummary();
    this._paintNames();
  }

  _noiseStds() {
    return this._numbersOf(this.noiseTrials.get("stds"));
  }

  _noiseFlag(key) {
    return this._groupFlag(this.noiseTrials, key);
  }

  _noiseNames() {
    const names = this._noiseFlag("include_baseline") ? ["noise-clean"] : [];

    this._noiseStds().forEach((std) => names.push(`noise-s${this._pyG(std)}`));
    return names;
  }

  _noisePanel() {
    const stdsLeaf = this.noiseTrials.get("stds");
    const shell    = this._ladderShell("measurement-noise ladder", "One rung per standard deviation of the circular complex noise added to the synthesised coherences. This is the only noise an unrolled run sees: the input stack is discarded and the targets are rebuilt from the ground-truth Gaussians, so the dataset augmentation noise never reaches the objective. Every rung repeats across the seeds list.", () => this._resetGroup(this.noiseTrials, () => this._repaintNoise()), [
      ["Add std", () => {
        const stds = this._noiseStds();
        stds.push(stds.length ? stds[stds.length - 1] : 0.03);
        this._emitNumbers(stdsLeaf, stds, (value) => value > 0, () => this._repaintNoise());
      }],
    ]);

    this.noiseFlagsEl = this._flagsRow(shell.panel, this.noiseTrials, [["include_baseline", "prepend one noise-free rung as the reference"]], () => this._repaintNoise());
    this.noiseStdsEl  = this._numberListSection(shell.panel, "noise standard deviations (one rung each, measurement units)", stdsLeaf, () => this._repaintNoise());
    this.noiseEl      = shell.panel;

    this._paintNoise();
    return shell.panel;
  }

  _paintNoise() {
    const stdsLeaf = this.noiseTrials.get("stds");

    this._paintNumberChips(this.noiseStdsEl, this._noiseStds(), "noise standard deviation of this rung", (next) => this._emitNumbers(stdsLeaf, next, (value) => value > 0, () => this._repaintNoise()));
  }

  _repaintNoise() {
    this._paintNoise();
    this._paintSummary();
    this._paintNames();
  }

  _headList(key) {
    return this._listOf(this.headTrials.get(key), (value) => typeof value === "string");
  }

  _headBackbone() {
    const leaf = this.headTrials.get("backbone");
    return leaf ? this.view._effective(leaf) : "unet";
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

    const backboneLeaf = this.headTrials.get("backbone");
    if (backboneLeaf) {
      const input      = document.createElement("input");
      input.className  = "cfg-edit__input exp-secondary__strategy";
      input.spellcheck = false;
      input.title      = `--${backboneLeaf.path} · backbone trained in every grid cell`;
      input.addEventListener("change", () => {
        this.view._setValue(backboneLeaf, input.value.trim());
        this._repaintHead();
      });
      this.headBackboneEl = input;
      head.appendChild(input);
      this.view.controls[backboneLeaf.path] = { leaf: backboneLeaf, reset: () => this._repaintHead() };
    }

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "One training run per checked head x matching pair, all on the one backbone named in the corner field. The matching strategy is applied to both curriculum stages; everything else comes from the base config.";

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

    if (this.headBackboneEl && document.activeElement !== this.headBackboneEl) this.headBackboneEl.value = this._headBackbone();

    const paintToggles = (el, key, catalog) => {
      el.innerHTML = "";
      const active = this._headList(key);
      catalog.forEach(({ key: option, label }) => {
        const chip = this._toggleChip(label, active.includes(option), "Click to drop this option from the grid", "Click to add this option to the grid", () => this._toggleHeadOption(key, option, catalog));
        el.appendChild(chip);
      });
    };

    if (this.headHeadsEl)     paintToggles(this.headHeadsEl,     "heads",     this._headOptions());
    if (this.headMatchingsEl) paintToggles(this.headMatchingsEl, "matchings", ExperimentBuilder.MATCHING_OPTIONS);
  }

  _headOptions() {
    if (this.view.key === "train_dual") return ExperimentBuilder.HEAD_OPTIONS.filter((option) => option.key === "set_pred");
    return ExperimentBuilder.HEAD_OPTIONS;
  }

  _repaintHead() {
    this._paintHead();
    this._paintSummary();
    this._paintNames();
  }

  _routingTrialsDict() {
    return this._trialsDict(this.routingTrials.get("trials"));
  }

  _ratioTrialsDict() {
    return this._trialsDict(this.ratioTrials.get("trials"));
  }

  _trialsDict(leaf) {
    return leaf ? this._dictFrom(this.view._effective(leaf)) : {};
  }

  _defaultsDict(leaf) {
    return leaf ? this._dictFrom(leaf.value) : {};
  }

  _routingSpecText(spec) {
    const groups = (key) => (Array.isArray(spec && spec[key]) ? spec[key].join("+") : "?");
    return `${groups("params")} | ${groups("existence")}`;
  }

  _ratioSpecText(spec) {
    const ladder = (key) => (Array.isArray(spec && spec[key]) ? spec[key].join("/") : "?");
    return `params ${ladder("params")} · existence ${ladder("existence")}`;
  }

  _routingPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">trunk routing</span>`;
    const reset    = LaunchWidgetDom.mini("reset assignments", () => this._resetRouting());
    reset.classList.add("exp-presence__reset");
    head.appendChild(reset);

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "One training run per params | existence channel-group assignment, each assignment doubled as a no-Aux and a with-Aux variant; the Aux input channel follows the selection, and both trunks stay at the shared feature ladders below. Drop assignments to trim the fan-out; adding assignments lives in code (_default_dual_routing_trials); use reset assignments to restore the default grid.";

    const grid     = document.createElement("div");
    grid.className = "exp-secondary__grid";
    ["params_features", "existence_features"].forEach((key) => {
      const leaf = this.routingTrials.get(key);
      if (leaf) grid.appendChild(this._trialTextRow(key, leaf, () => this._repaintRouting()));
    });

    const cellHead       = document.createElement("div");
    cellHead.className   = "exp-presence__sub";
    cellHead.textContent = "assignments";

    const cells         = document.createElement("div");
    cells.className     = "exp-builder__names exp-presence__cells";
    this.routingCellsEl = cells;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(grid);
    panel.appendChild(cellHead);
    panel.appendChild(cells);
    this.routingEl = panel;

    const trialsLeaf = this.routingTrials.get("trials");
    if (trialsLeaf) this.view.controls[trialsLeaf.path] = { leaf: trialsLeaf, reset: () => this._repaintRouting() };

    this._paintRouting();
    return panel;
  }

  _ratioPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">arm ratio</span>`;
    const reset    = LaunchWidgetDom.mini("reset splits", () => this._resetRatio());
    reset.classList.add("exp-presence__reset");
    head.appendChild(reset);

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "One training run per declared params/existence split, every trial spending the same parameter budget; the launcher builds each model and refuses ladders whose realized split or total budget drifts beyond the match tolerance. Drop splits to trim the fan-out; retuning ladders lives in code (_default_dual_ratio_trials); use reset splits to restore the default grid.";

    const grid     = document.createElement("div");
    grid.className = "exp-secondary__grid";
    const tolerance = this.ratioTrials.get("match_tolerance");
    if (tolerance) grid.appendChild(this._trialTextRow("match_tolerance", tolerance, () => this._repaintRatio()));

    const cellHead       = document.createElement("div");
    cellHead.className   = "exp-presence__sub";
    cellHead.textContent = "splits";

    const cells       = document.createElement("div");
    cells.className   = "exp-builder__names exp-presence__cells";
    this.ratioCellsEl = cells;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(grid);
    panel.appendChild(cellHead);
    panel.appendChild(cells);
    this.ratioEl = panel;

    const trialsLeaf = this.ratioTrials.get("trials");
    if (trialsLeaf) this.view.controls[trialsLeaf.path] = { leaf: trialsLeaf, reset: () => this._repaintRatio() };

    this._paintRatio();
    return panel;
  }

  _trialTextRow(label, leaf, repaint) {
    const row     = document.createElement("div");
    row.className = "cfg-edit__row";
    row.title     = `--${leaf.path}`;

    const name     = document.createElement("div");
    name.className = "cfg-edit__name";
    name.innerHTML = `${label}<span>${leaf.type}</span>`;

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input";
    input.type       = "text";
    input.value      = this.view._effective(leaf);
    input.spellcheck = false;
    input.addEventListener("input", () => {
      this.view.applyTypedInput(leaf, input);
    });

    row.appendChild(name);
    row.appendChild(input);

    this.view.controls[leaf.path] = { leaf, reset: () => {
      input.value = this.view._effective(leaf);
      input.classList.toggle("is-dirty", this.view.dirty[leaf.path] !== undefined);
      repaint();
    } };
    return row;
  }

  _trialChip(label, text, remove) {
    return this._cellChip(`${label} · ${text}`, text, "Remove trial", remove);
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

  _removeTrialKey(leaf, label, repaint) {
    if (!leaf) return;
    let raw;
    try {
      raw = PythonLiteral.parse(this.view._effective(leaf));
    } catch (e) {
      return;
    }
    if (!raw || typeof raw !== "object" || Object.keys(raw).length <= 1) return;
    delete raw[label];
    this.view._setValue(leaf, PythonLiteral.render(raw));
    repaint();
  }

  _paintRouting() {
    if (!this.routingCellsEl) return;
    this.routingCellsEl.innerHTML = "";
    const trials = this._routingTrialsDict();
    Object.keys(trials).forEach((label) => {
      this.routingCellsEl.appendChild(this._trialChip(label, this._routingSpecText(trials[label]), () => this._removeTrialKey(this.routingTrials.get("trials"), label, () => this._repaintRouting())));
    });
  }

  _paintRatio() {
    if (!this.ratioCellsEl) return;
    this.ratioCellsEl.innerHTML = "";
    const trials = this._ratioTrialsDict();
    Object.keys(trials).forEach((label) => {
      this.ratioCellsEl.appendChild(this._trialChip(label, this._ratioSpecText(trials[label]), () => this._removeTrialKey(this.ratioTrials.get("trials"), label, () => this._repaintRatio())));
    });
  }

  _resetRouting() {
    const leaf = this.routingTrials.get("trials");
    if (leaf) this.view._setValue(leaf, leaf.value);
    this._repaintRouting();
  }

  _resetRatio() {
    const leaf = this.ratioTrials.get("trials");
    if (leaf) this.view._setValue(leaf, leaf.value);
    this._repaintRatio();
  }

  _repaintRouting() {
    this._paintRouting();
    this._paintSummary();
    this._paintNames();
  }

  _repaintRatio() {
    this._paintRatio();
    this._paintSummary();
    this._paintNames();
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

    if (this.strategySelect) this.strategySelect.value = strategy;
    this.strategyNoteEl.textContent = ExperimentBuilder.STRATEGIES.find((entry) => entry.key === strategy).note;
    this.secondaryRows.forEach((entry) => (entry.row.hidden = !entry.strategies.includes(strategy)));
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


class AblationBuilder {
  constructor(view, byPath) {
    this.view   = view;
    this.byPath = byPath;

    this.trialsLeaf   = byPath.get("trials_enabled");
    this.modeLeaf     = byPath.get("trials_mode");
    this.modelLeaf    = byPath.get("backbone_name") || byPath.get("model_name");
    this.gpusLeaf     = byPath.get("gpus");
    this.seedsLeaf    = byPath.get("seeds");
    this.featuresLeaf = byPath.get("ablation_features");
    this.catalogLeaf  = byPath.get("ablation_catalog");
    this.fullLeaf     = byPath.get("ablation_include_full");

    this.claimed = [];
    if (this.trialsLeaf)   this.claimed.push("trials_enabled");
    if (this.modeLeaf)     this.claimed.push("trials_mode");
    if (this.featuresLeaf) this.claimed.push(this.featuresLeaf.path);
    if (this.catalogLeaf)  this.claimed.push(this.catalogLeaf.path);
    if (this.fullLeaf)     this.claimed.push(this.fullLeaf.path);

    this.root         = null;
    this.summaryEl    = null;
    this.namesEl      = null;
    this.menuEl       = null;
    this.listEl       = null;
    this.fullEl       = null;
    this.datalistId   = "ablation-cfg-paths";
    this._paintSwitch = null;
  }

  get available() {
    return !!(this.featuresLeaf && this.catalogLeaf && this.trialsLeaf && this.modeLeaf);
  }

  build() {
    this.root           = document.createElement("section");
    this.root.className = "exp-builder exp-ablation-card";

    const head     = document.createElement("header");
    head.className = "special-head";
    head.innerHTML = `<h3 class="special-head__name">Ablation study</h3>`;

    const hint       = document.createElement("span");
    hint.className   = "special-head__hint";
    hint.textContent = "full model degrades one feature at a time, in order, down to the baseline";
    head.appendChild(hint);

    const summary     = document.createElement("span");
    summary.className = "exp-builder__summary";
    this.summaryEl    = summary;
    head.appendChild(summary);

    head.appendChild(this._switch());

    const body     = document.createElement("div");
    body.className = "exp-builder__body";
    body.appendChild(this._panel());

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
    this.root.appendChild(this._pathDatalist());

    if (this.featuresLeaf) this.view.controls[this.featuresLeaf.path] = { leaf: this.featuresLeaf, reset: () => this._repaint() };
    if (this.fullLeaf)     this.view.controls[this.fullLeaf.path]     = { leaf: this.fullLeaf,     reset: () => this._repaint() };

    this._repaint();
    if (this._paintSwitch) this._paintSwitch();
    return this.root;
  }

  refreshFromView() {
    if (!this.summaryEl) return;
    if (this._paintSwitch) this._paintSwitch();
    this._paint();
    this._paintSummary();
    this._paintNames();
  }

  _active() {
    return this.view._effective(this.trialsLeaf) === "True" && this.view._effective(this.modeLeaf) === "ablation";
  }

  _switch() {
    const toggle     = document.createElement("button");
    toggle.type      = "button";
    toggle.className = "switch";
    toggle.setAttribute("role", "switch");
    toggle.title     = "run an ablation study";
    toggle.innerHTML = `<span class="switch__knob"></span>`;

    const paint = () => {
      const on = this._active();
      toggle.classList.toggle("is-on", on);
      toggle.setAttribute("aria-checked", String(on));
      this.root.classList.toggle("is-trials", on);
    };

    toggle.addEventListener("click", () => {
      if (this._active()) {
        this.view._setValue(this.trialsLeaf, "False");
      } else {
        this.view._setValue(this.modeLeaf, "ablation");
        this.view._setValue(this.trialsLeaf, "True");
      }
      paint();
    });

    this._paintSwitch = paint;
    return toggle;
  }

  _catalog() {
    if (!this.catalogLeaf) return {};
    try {
      const raw = PythonLiteral.parse(this.view._effective(this.catalogLeaf));
      return raw && typeof raw === "object" && !Array.isArray(raw) ? raw : {};
    } catch (e) {
      return {};
    }
  }

  _features() {
    if (!this.featuresLeaf) return [];
    try {
      const raw = PythonLiteral.parse(this.view._effective(this.featuresLeaf));
      return Array.isArray(raw) ? raw : [];
    } catch (e) {
      return [];
    }
  }

  _full() {
    if (!this.fullLeaf) return true;
    return this.view._effective(this.fullLeaf) === "True";
  }

  _specText(feature) {
    const degrade = feature && feature.degrade ? feature.degrade : {};
    const parts   = Object.entries(degrade).map(([path, value]) => `${path.split(".").pop()}=${PythonLiteral.render(value)}`);
    return parts.length ? parts.join(", ") : "no change";
  }

  _featureKeys(feature) {
    const keys = [];
    const seen = new Set();
    [feature.enable || {}, feature.degrade || {}].forEach((dict) => {
      Object.keys(dict).forEach((path) => {
        if (seen.has(path)) return;
        seen.add(path);
        keys.push(path);
      });
    });
    return keys;
  }

  _pathDatalist() {
    const list = document.createElement("datalist");
    list.id    = this.datalistId;
    [...this.byPath.keys()].forEach((path) => {
      const opt   = document.createElement("option");
      opt.value   = path;
      list.appendChild(opt);
    });
    return list;
  }

  _panel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence exp-ablation";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">ablation order</span>`;

    const full = LaunchWidgetDom.mini("full-model run", () => this._toggleFull());
    full.classList.add("exp-ablation__full");
    this.fullEl = full;
    head.appendChild(full);

    const reset = LaunchWidgetDom.mini("reset features", () => this._reset());
    reset.classList.add("exp-presence__reset");
    head.appendChild(reset);

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "Run 0 is the full model with every selected feature enabled (the before values); each later run degrades one more feature in the order below to its after value, ending at the baseline with all of them degraded. Edit the before and after of any config path, add or remove paths, or rename the feature. Pick from the catalog, or add a custom feature by config path.";

    const menuHead      = document.createElement("div");
    menuHead.className   = "exp-presence__sub";
    menuHead.textContent = "feature catalog";

    const menu       = document.createElement("div");
    menu.className   = "exp-builder__names exp-ablation__menu";
    this.menuEl      = menu;

    const listHead      = document.createElement("div");
    listHead.className   = "exp-presence__sub";
    listHead.textContent = "degradation order (full to baseline)";

    const list       = document.createElement("div");
    list.className   = "exp-ablation__list";
    this.listEl      = list;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(menuHead);
    panel.appendChild(menu);
    panel.appendChild(listHead);
    panel.appendChild(list);
    panel.appendChild(this._customForm());
    return panel;
  }

  _customForm() {
    const form     = document.createElement("div");
    form.className = "exp-ablation__custom";

    const label = document.createElement("input");
    label.type        = "text";
    label.placeholder = "label";
    label.className   = "exp-ablation__field exp-ablation__field--label";

    const group = document.createElement("input");
    group.type        = "text";
    group.placeholder = "group";
    group.className   = "exp-ablation__field exp-ablation__field--group";

    const path = document.createElement("input");
    path.type        = "text";
    path.placeholder = "config.path";
    path.className   = "exp-ablation__field exp-ablation__field--path";
    path.setAttribute("list", this.datalistId);

    const before = document.createElement("input");
    before.type        = "text";
    before.placeholder = "before";
    before.className   = "exp-ablation__field exp-ablation__field--value";

    const after = document.createElement("input");
    after.type        = "text";
    after.placeholder = "after";
    after.className   = "exp-ablation__field exp-ablation__field--value";

    const add = LaunchWidgetDom.mini("add custom feature", () => {
      if (this._addCustom(label.value, group.value, path.value, before.value, after.value)) {
        label.value = "";
        group.value = "";
        path.value  = "";
        before.value = "";
        after.value  = "";
      }
    });

    form.appendChild(label);
    form.appendChild(group);
    form.appendChild(path);
    form.appendChild(before);
    form.appendChild(after);
    form.appendChild(add);
    return form;
  }

  _coerceLiteral(text) {
    const trimmed = String(text).trim();
    if (trimmed === "") return "";
    if (trimmed.toLowerCase() === "true")  return true;
    if (trimmed.toLowerCase() === "false") return false;
    if (trimmed.toLowerCase() === "none")  return null;
    if (!isNaN(Number(trimmed))) return Number(trimmed);
    return trimmed;
  }

  _addFeature(key) {
    const feature = this._catalog()[key];
    if (!feature) return;

    const features = this._features();
    if (features.some((entry) => entry.label === key)) return;

    features.push(feature);
    this._setFeatures(features);
  }

  _addCustom(label, group, path, before, after) {
    const cleanLabel = String(label).trim();
    const cleanGroup = String(group).trim() || "custom";
    const cleanPath  = String(path).trim();
    if (!cleanLabel || !cleanPath) {
      Toast.show("custom feature needs a label and a config path", "error");
      return false;
    }

    const features = this._features();
    if (features.some((entry) => entry.label === cleanLabel)) {
      Toast.show(`feature '${cleanLabel}' is already in the order`, "error");
      return false;
    }

    const enable  = {};
    const degrade = {};
    if (String(before).trim() !== "") enable[cleanPath]  = this._coerceLiteral(before);
    degrade[cleanPath] = this._coerceLiteral(after);

    features.push({ label: cleanLabel, group: cleanGroup, enable, degrade });
    this._setFeatures(features);
    return true;
  }

  _renameFeature(index, label) {
    const clean    = String(label).trim();
    if (!clean) return;
    const features = this._features();
    if (!features[index]) return;
    if (features.some((entry, i) => i !== index && entry.label === clean)) {
      Toast.show(`feature '${clean}' is already in the order`, "error");
      this._repaint();
      return;
    }
    features[index].label = clean;
    this._setFeatures(features);
  }

  _setPathValue(index, side, path, value) {
    const features = this._features();
    const feature  = features[index];
    if (!feature) return;
    if (!feature[side]) feature[side] = {};
    feature[side][path] = value;
    this._setFeatures(features);
  }

  _addPath(index, path, before, after) {
    const cleanPath = String(path).trim();
    if (!cleanPath) {
      Toast.show("a config path is required", "error");
      return false;
    }

    const features = this._features();
    const feature  = features[index];
    if (!feature) return false;
    if (!feature.enable)  feature.enable  = {};
    if (!feature.degrade) feature.degrade = {};
    if (String(before).trim() !== "") feature.enable[cleanPath] = this._coerceLiteral(before);
    feature.degrade[cleanPath] = this._coerceLiteral(after);
    this._setFeatures(features);
    return true;
  }

  _removePath(index, path) {
    const features = this._features();
    const feature  = features[index];
    if (!feature) return;
    if (feature.enable)  delete feature.enable[path];
    if (feature.degrade) delete feature.degrade[path];
    this._setFeatures(features);
  }

  _move(index, delta) {
    const features = this._features();
    const target   = index + delta;
    if (target < 0 || target >= features.length) return;

    const moved = features.splice(index, 1)[0];
    features.splice(target, 0, moved);
    this._setFeatures(features);
  }

  _remove(index) {
    const features = this._features();
    features.splice(index, 1);
    this._setFeatures(features);
  }

  _toggleFull() {
    if (!this.fullLeaf) return;
    const next = this._full() ? "False" : "True";
    this.view._setValue(this.fullLeaf, next);
    this._repaint();
  }

  _setFeatures(features) {
    if (!this.featuresLeaf) return;
    this.view._setValue(this.featuresLeaf, PythonLiteral.render(features));
    this._repaint();
  }

  _reset() {
    if (this.featuresLeaf) this.view._setValue(this.featuresLeaf, this.featuresLeaf.value);
    if (this.fullLeaf)     this.view._setValue(this.fullLeaf, this.fullLeaf.value);
    this._repaint();
  }

  _paint() {
    this._paintMenu();
    this._paintList();
    if (this.fullEl) this.fullEl.classList.toggle("is-on", this._full());
  }

  _paintMenu() {
    if (!this.menuEl) return;
    this.menuEl.innerHTML = "";

    const chosen  = new Set(this._features().map((feature) => feature.label));
    const catalog = this._catalog();

    Object.keys(catalog).forEach((key) => {
      const chip       = document.createElement("button");
      chip.type        = "button";
      chip.className   = "exp-name exp-ablation__option";
      chip.textContent = key;
      chip.title       = this._specText(catalog[key]);
      chip.disabled    = chosen.has(key);
      chip.classList.toggle("is-on", chosen.has(key));
      chip.addEventListener("click", () => this._addFeature(key));
      this.menuEl.appendChild(chip);
    });
  }

  _paintList() {
    if (!this.listEl) return;
    this.listEl.innerHTML = "";

    const features = this._features();

    features.forEach((feature, index) => this.listEl.appendChild(this._row(feature, index, features.length)));

    if (!features.length) {
      const empty       = document.createElement("p");
      empty.className   = "exp-ablation__empty";
      empty.textContent = "no features selected; pick from the catalog above";
      this.listEl.appendChild(empty);
    }
  }

  _row(feature, index, total) {
    const row     = document.createElement("div");
    row.className = "exp-ablation__row";

    const head     = document.createElement("div");
    head.className = "exp-ablation__rowhead";

    const isBaseline = index === total - 1;
    const step       = document.createElement("span");
    step.className   = "exp-ablation__step";
    step.textContent = isBaseline ? `${index + 1} base` : `${index + 1}`;
    head.appendChild(step);

    const label       = document.createElement("input");
    label.type        = "text";
    label.className   = "exp-ablation__labelinput";
    label.value       = feature.label || "";
    label.spellcheck  = false;
    label.title       = "feature label, used in the trial run name";
    label.addEventListener("change", () => this._renameFeature(index, label.value));
    head.appendChild(label);

    const group       = document.createElement("span");
    group.className   = "exp-ablation__group";
    group.textContent = feature.group || "custom";
    head.appendChild(group);

    const up = LaunchWidgetDom.mini("↑", () => this._move(index, -1));
    up.classList.add("exp-ablation__move");
    up.disabled = index === 0;
    head.appendChild(up);

    const down = LaunchWidgetDom.mini("↓", () => this._move(index, 1));
    down.classList.add("exp-ablation__move");
    down.disabled = index === total - 1;
    head.appendChild(down);

    const remove = LaunchWidgetDom.mini("×", () => this._remove(index));
    remove.classList.add("exp-presence__remove");
    remove.title = "remove feature";
    head.appendChild(remove);

    row.appendChild(head);

    const paths     = document.createElement("div");
    paths.className = "exp-ablation__paths";

    const keys = this._featureKeys(feature);
    if (!keys.length) {
      const none       = document.createElement("p");
      none.className   = "exp-ablation__pathnote";
      none.textContent = "no config paths yet — add one below";
      paths.appendChild(none);
    }
    keys.forEach((path) => paths.appendChild(this._pathRow(feature, index, path)));
    paths.appendChild(this._addPathForm(index));

    row.appendChild(paths);
    return row;
  }

  _pathRow(feature, index, path) {
    const item     = document.createElement("div");
    item.className = "exp-ablation__path";

    const name       = document.createElement("span");
    name.className   = "exp-ablation__pathname";
    name.textContent = path;
    name.title       = path;
    item.appendChild(name);

    const beforeVal = feature.enable ? feature.enable[path] : undefined;
    const before    = this._valueEditor(path, beforeVal, (v) => this._setPathValue(index, "enable", path, v));
    before.classList.add("exp-ablation__val", "exp-ablation__val--before");
    before.title = `before (full model) · --${path}`;
    item.appendChild(before);

    const arrow       = document.createElement("span");
    arrow.className   = "exp-ablation__arrow";
    arrow.textContent = "→";
    item.appendChild(arrow);

    const afterVal = feature.degrade ? feature.degrade[path] : undefined;
    const after    = this._valueEditor(path, afterVal, (v) => this._setPathValue(index, "degrade", path, v));
    after.classList.add("exp-ablation__val", "exp-ablation__val--after");
    after.title = `after (degraded) · --${path}`;
    item.appendChild(after);

    const rm = LaunchWidgetDom.mini("×", () => this._removePath(index, path));
    rm.classList.add("exp-ablation__pathremove");
    rm.title = "remove this config path";
    item.appendChild(rm);

    return item;
  }

  _addPathForm(index) {
    const form     = document.createElement("div");
    form.className = "exp-ablation__addpath";

    const path = document.createElement("input");
    path.type        = "text";
    path.placeholder = "config.path";
    path.className   = "exp-ablation__field exp-ablation__field--path";
    path.setAttribute("list", this.datalistId);

    let before  = this._addValueField(false, "before", "");
    let after   = this._addValueField(false, "after", "");
    let isModel = false;

    const swap = (next) => {
      const freshBefore = this._addValueField(next, "before", before.value);
      const freshAfter  = this._addValueField(next, "after", after.value);
      form.replaceChild(freshBefore, before);
      form.replaceChild(freshAfter, after);
      before  = freshBefore;
      after   = freshAfter;
      isModel = next;
    };

    path.addEventListener("input", () => {
      const next = path.value.trim() === "backbone_name" && !!(this.view.modelFamilies && this.view.modelFamilies.length);
      if (next !== isModel) swap(next);
    });

    const add = LaunchWidgetDom.mini("add path", () => {
      if (this._addPath(index, path.value, before.value, after.value)) {
        path.value = "";
        swap(false);
        before.value = "";
        after.value  = "";
      }
    });
    add.classList.add("exp-ablation__addbtn");

    form.appendChild(path);
    form.appendChild(before);
    form.appendChild(after);
    form.appendChild(add);
    return form;
  }

  _addValueField(isModel, placeholder, current) {
    if (isModel) {
      const select = this._backboneSelect(current, () => {});
      select.classList.add("exp-ablation__field", "exp-ablation__field--value");
      return select;
    }

    const input       = document.createElement("input");
    input.type        = "text";
    input.placeholder = placeholder;
    input.className   = "exp-ablation__field exp-ablation__field--value";
    input.value       = current === undefined || current === null ? "" : String(current);
    return input;
  }

  _valueEditor(path, value, onCommit) {
    const leaf = this.byPath.get(path);
    const type = leaf ? leaf.type : null;

    if (path === "backbone_name" && this.view.modelFamilies && this.view.modelFamilies.length) {
      return this._backboneSelect(value, onCommit);
    }

    if (type === "bool") {
      const btn       = document.createElement("button");
      btn.type        = "button";
      btn.className   = "switch exp-ablation__switch";
      btn.setAttribute("role", "switch");
      btn.innerHTML   = `<span class="switch__knob"></span>`;
      const paint = (state) => {
        btn.classList.toggle("is-on", state);
        btn.setAttribute("aria-checked", String(state));
      };
      paint(value === true);
      btn.addEventListener("click", () => {
        const next = !btn.classList.contains("is-on");
        paint(next);
        onCommit(next);
      });
      return btn;
    }

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input exp-ablation__input";
    input.spellcheck = false;

    if (type === "int" || type === "float") {
      input.type  = "number";
      input.step  = type === "int" ? "1" : "any";
      input.value = value === undefined || value === null ? "" : String(value);
      input.addEventListener("change", () => {
        const raw = input.value.trim();
        onCommit(raw === "" ? null : Number(raw));
      });
    } else {
      const structured = value !== null && typeof value === "object";
      input.type        = "text";
      input.value       = value === undefined || value === null ? "" : (structured ? PythonLiteral.render(value) : String(value));
      input.placeholder = value === undefined ? "unset" : "";
      input.addEventListener("change", () => {
        if (!structured) {
          onCommit(input.value);
          return;
        }
        try {
          const parsed = PythonLiteral.parse(input.value);
          input.classList.remove("is-invalid");
          input.title = "";
          onCommit(parsed);
        } catch (e) {
          input.classList.add("is-invalid");
          input.title = `not a valid Python literal: ${e.message || e}`;
        }
      });
    }
    return input;
  }

  _backboneSelect(value, onCommit) {
    const select     = document.createElement("select");
    select.className = "run-select exp-ablation__input exp-ablation__select";

    const current = value === undefined || value === null ? "" : String(value);
    const known   = new Set();

    (this.view.modelFamilies || []).forEach((family) => {
      const group   = document.createElement("optgroup");
      group.label   = family.family;
      family.models.forEach((model) => {
        const opt       = document.createElement("option");
        opt.value       = model.key;
        opt.textContent = model.recommended ? `${model.name} (recommended)` : model.name || model.key;
        group.appendChild(opt);
        known.add(model.key);
      });
      select.appendChild(group);
    });

    if (current && !known.has(current)) {
      const opt       = document.createElement("option");
      opt.value       = current;
      opt.textContent = current;
      select.insertBefore(opt, select.firstChild);
    }

    select.value = current;
    select.addEventListener("change", () => onCommit(select.value));
    return select;
  }

  _gpusSuffix() {
    if (!this.gpusLeaf) return "";
    try {
      const parsed = PythonLiteral.parse(this.view._effective(this.gpusLeaf));
      if (Array.isArray(parsed)) return ` on ${parsed.length} GPU${parsed.length === 1 ? "" : "s"}`;
    } catch (e) {
      return "";
    }
    return "";
  }

  _paintSummary() {
    if (!this.summaryEl) return;
    const n    = this._features().length;
    const runs = n + (this._full() ? 1 : 0);
    this.summaryEl.textContent = `${n} feature${n === 1 ? "" : "s"} = ${runs} trial${runs === 1 ? "" : "s"} (full to baseline)${SeedAxisNote.suffix(this.view, this.seedsLeaf, runs)}${this._gpusSuffix()}`;
  }

  _paintNames() {
    if (!this.namesEl) return;
    this.namesEl.innerHTML = "";

    const model    = this.modelLeaf ? this.view._effective(this.modelLeaf) : "model";
    const features = this._features();
    const total    = features.length;
    const names    = [];

    if (this._full()) names.push(`${model}_abl-0-full`);
    features.forEach((feature, index) => {
      const step = index + 1;
      if (step === total) names.push(`${model}_abl-${step}-baseline`);
      else names.push(`${model}_abl-${step}-no_${feature.label}`);
    });

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

  _repaint() {
    this._paint();
    this._paintSummary();
    this._paintNames();
  }
}

window.ExperimentBuilder = ExperimentBuilder;
window.AblationBuilder   = AblationBuilder;
