"use strict";

class PythonLiteral {
  static parse(text) {
    const reader = new PythonLiteral(String(text));
    const value  = reader._value();
    reader._ws();
    if (reader.pos !== reader.text.length) throw new Error(`trailing input at ${reader.pos}`);
    return value;
  }

  static render(value) {
    if (value === null || value === undefined) return "None";
    if (value === true)  return "True";
    if (value === false) return "False";
    if (typeof value === "number") return String(value);
    if (typeof value === "string") return `'${value.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`;
    if (Array.isArray(value)) return `[${value.map((item) => PythonLiteral.render(item)).join(", ")}]`;
    return `{${Object.entries(value).map(([key, item]) => `${PythonLiteral.render(key)}: ${PythonLiteral.render(item)}`).join(", ")}}`;
  }

  constructor(text) {
    this.text = text;
    this.pos  = 0;
  }

  _ws() {
    while (this.pos < this.text.length && /\s/.test(this.text[this.pos])) this.pos += 1;
  }

  _value() {
    this._ws();
    const ch = this.text[this.pos];

    if (ch === "{") return this._dict();
    if (ch === "[") return this._seq("]");
    if (ch === "(") return this._seq(")");
    if (ch === "'" || ch === '"') return this._string(ch);
    return this._atom();
  }

  _dict() {
    this.pos += 1;
    const out = {};

    while (true) {
      this._ws();
      if (this.pos >= this.text.length) throw new Error("unterminated dict");
      if (this.text[this.pos] === "}") { this.pos += 1; return out; }

      const key = this._value();
      this._ws();
      if (this.text[this.pos] !== ":") throw new Error(`expected ':' at ${this.pos}`);
      this.pos += 1;

      out[String(key)] = this._value();
      this._ws();
      if (this.text[this.pos] === ",") this.pos += 1;
    }
  }

  _seq(close) {
    this.pos += 1;
    const out = [];

    while (true) {
      this._ws();
      if (this.pos >= this.text.length) throw new Error("unterminated sequence");
      if (this.text[this.pos] === close) { this.pos += 1; return out; }

      out.push(this._value());
      this._ws();
      if (this.text[this.pos] === ",") this.pos += 1;
    }
  }

  _string(quote) {
    this.pos += 1;
    let out = "";

    while (this.pos < this.text.length) {
      const ch = this.text[this.pos];
      if (ch === "\\") { out += this.text[this.pos + 1]; this.pos += 2; continue; }
      if (ch === quote) { this.pos += 1; return out; }
      out += ch;
      this.pos += 1;
    }
    throw new Error("unterminated string");
  }

  _atom() {
    const start = this.pos;
    while (this.pos < this.text.length && !/[\s,\]})\:]/.test(this.text[this.pos])) this.pos += 1;
    const token = this.text.slice(start, this.pos);

    if (token === "True")  return true;
    if (token === "False") return false;
    if (token === "None")  return null;
    if (token === "") throw new Error(`empty token at ${start}`);

    const number = Number(token);
    return Number.isNaN(number) ? token : number;
  }
}


class LaunchWidgetDom {
  static mini(label, onClick) {
    const btn       = document.createElement("button");
    btn.type        = "button";
    btn.className   = "btn btn--mini";
    btn.textContent = label;
    btn.addEventListener("click", onClick);
    return btn;
  }
}


class ModelTogglePanel {
  constructor(view, leaf) {
    this.view     = view;
    this.leaf     = leaf;
    this.families = view.modelFamilies || [];
    this.keys     = this.families.flatMap((family) => family.models.map((model) => model.key));
    this.chips    = new Map();
    this.countEl  = null;
  }

  build() {
    const root     = document.createElement("section");
    root.className = "model-panel";

    const head     = document.createElement("header");
    head.className = "special-head";
    head.innerHTML = `<h3 class="special-head__name">Models in run</h3><span class="special-head__hint">toggled-off models are skipped</span>`;

    const count     = document.createElement("span");
    count.className = "model-panel__count";
    this.countEl    = count;

    head.appendChild(count);
    head.appendChild(LaunchWidgetDom.mini("All on",  () => this._emit(new Set())));
    head.appendChild(LaunchWidgetDom.mini("All off", () => this._emit(new Set(this.keys))));
    root.appendChild(head);

    const body     = document.createElement("div");
    body.className = "model-panel__families";
    this.families.forEach((family) => body.appendChild(this._family(family)));
    root.appendChild(body);

    this.view.controls[this.leaf.path] = { leaf: this.leaf, reset: () => this._paint() };
    this._paint();
    return root;
  }

  _family(family) {
    const block     = document.createElement("div");
    block.className = "model-family";

    const name       = document.createElement("div");
    name.className   = "model-family__name";
    name.textContent = family.family;

    const grid     = document.createElement("div");
    grid.className = "model-family__grid";
    family.models.forEach((model) => grid.appendChild(this._chip(model)));

    block.appendChild(name);
    block.appendChild(grid);
    return block;
  }

  _chip(model) {
    const chip     = document.createElement("button");
    chip.type      = "button";
    chip.className = "model-chip";
    chip.title     = `--${this.leaf.path} · ${model.key}`;
    chip.innerHTML = `<span class="model-chip__name">${model.name}</span><span class="model-chip__meta">${model.params || ""}</span>`;
    chip.addEventListener("click", () => this._toggle(model.key));
    this.chips.set(model.key, chip);
    return chip;
  }

  _skipped() {
    try {
      const parsed = PythonLiteral.parse(this.view._effective(this.leaf));
      return new Set(Array.isArray(parsed) ? parsed.map(String) : []);
    } catch (e) {
      return new Set();
    }
  }

  _toggle(key) {
    const skipped = this._skipped();
    if (skipped.has(key)) skipped.delete(key);
    else skipped.add(key);
    this._emit(skipped);
  }

  _emit(skipped) {
    const ordered = this.keys.filter((key) => skipped.has(key));
    this.view._setValue(this.leaf, PythonLiteral.render(ordered));
    this._paint();
  }

  _paint() {
    const skipped = this._skipped();

    this.chips.forEach((chip, key) => {
      const on = !skipped.has(key);
      chip.classList.toggle("is-on", on);
      chip.setAttribute("aria-pressed", String(on));
    });

    const active = this.keys.filter((key) => !skipped.has(key)).length;
    this.countEl.textContent = `${active}/${this.keys.length} active`;
  }
}


class ModelCardPanel {
  constructor(view, leaf, headLeaf = null) {
    this.view      = view;
    this.leaf      = leaf;
    this.headLeaf  = headLeaf;
    this.families  = view.modelFamilies || [];
    this.heads     = view.modelHeads || [];
    this.cards     = new Map();
    this.headChips = new Map();
    this.currentEl = null;
  }

  build() {
    const root     = document.createElement("section");
    root.className = "model-panel";

    const isAe     = this.leaf.path.endsWith("ae_model_name");
    const head     = document.createElement("header");
    head.className = "special-head";
    head.innerHTML = isAe
      ? `<h3 class="special-head__name">Autoencoder</h3><span class="special-head__hint">the autoencoder to train</span>`
      : `<h3 class="special-head__name">Model</h3><span class="special-head__hint">the architecture to train</span>`;

    const current   = document.createElement("span");
    current.className = "model-panel__count";
    this.currentEl  = current;
    head.appendChild(current);
    root.appendChild(head);

    const body     = document.createElement("div");
    body.className = "model-panel__families";
    this.families.forEach((family) => body.appendChild(this._family(family)));
    root.appendChild(body);

    if (this.headLeaf && this.heads.length) root.appendChild(this._headPicker());

    this.view.controls[this.leaf.path] = { leaf: this.leaf, reset: () => this._paint() };
    if (this.headLeaf) this.view.controls[this.headLeaf.path] = { leaf: this.headLeaf, reset: () => this._paint() };
    this._paint();
    return root;
  }

  _headPicker() {
    const block     = document.createElement("div");
    block.className = "model-family";

    const name       = document.createElement("div");
    name.className   = "model-family__name";
    name.textContent = "Output head";

    const grid     = document.createElement("div");
    grid.className = "model-family__grid";
    this.heads.forEach((headSpec) => grid.appendChild(this._headChip(headSpec)));

    block.appendChild(name);
    block.appendChild(grid);
    return block;
  }

  _headChip(headSpec) {
    const chip     = document.createElement("button");
    chip.type      = "button";
    chip.className = "model-chip";
    chip.title     = headSpec.when || "";

    chip.innerHTML = `<span class="model-chip__name">${headSpec.name}</span><span class="model-chip__meta">${headSpec.structure || ""}</span>`;

    chip.addEventListener("click", () => this._selectHead(headSpec.key));
    this.headChips.set(headSpec.key, chip);
    return chip;
  }

  _selectHead(key) {
    this.view._setValue(this.headLeaf, key);
    this._paint();
  }

  _family(family) {
    const block     = document.createElement("div");
    block.className = "model-family";

    const name       = document.createElement("div");
    name.className   = "model-family__name";
    name.textContent = family.family;

    const grid     = document.createElement("div");
    grid.className = "model-pick__grid";
    family.models.forEach((model) => grid.appendChild(this._card(model)));

    block.appendChild(name);
    block.appendChild(grid);
    return block;
  }

  _card(model) {
    const card     = document.createElement("button");
    card.type      = "button";
    card.className = "model-pick";
    card.title     = `--${this.leaf.path} ${model.key}`;

    const badge = model.recommended ? `<span class="model-pick__badge">recommended</span>` : "";
    const meta  = [model.head, model.skip].filter(Boolean).join("  ·  ");

    card.innerHTML =
      `<span class="model-pick__top"><span class="model-pick__name">${model.name}</span>${badge}<span class="model-pick__params">${model.params || ""}</span></span>` +
      `<span class="model-pick__meta">${meta}</span>`;

    card.addEventListener("click", () => this._select(model.key));
    this.cards.set(model.key, card);
    return card;
  }

  _select(key) {
    this.view._setValue(this.leaf, key);
    this._paint();
  }

  _paint() {
    const current = this.view._effective(this.leaf);
    let label     = current;

    this.cards.forEach((card, key) => {
      const on = key === current;
      card.classList.toggle("is-on", on);
      card.setAttribute("aria-pressed", String(on));
      if (on) label = card.querySelector(".model-pick__name").textContent;
    });

    if (this.headLeaf && this.headChips.size) {
      const currentHead = this.view._effective(this.headLeaf);
      let headLabel     = currentHead;

      this.headChips.forEach((chip, key) => {
        const on = key === currentHead;
        chip.classList.toggle("is-on", on);
        chip.setAttribute("aria-pressed", String(on));
        if (on) headLabel = chip.querySelector(".model-chip__name").textContent;
      });

      label = `${label} · ${headLabel} head`;
    }

    this.currentEl.textContent = label;
  }
}


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


class ExperimentBuilder {

  static MODES = [
    { key: "curriculum", label: "curriculum",  hint: "warmup x complete cross product, one trial each",
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

    { key: "warmup",     label: "single stage", hint: "curriculum disabled, check losses from the catalog and each trains alone as one trial",
      el        : "warmupCatalogEl",
      available : (b) => Boolean(b.warmupLeaf && b.completeLeaf),
      paint     : (b) => b._paintWarmupCatalog(),
      summary   : (b, gpus) => {
        const n = b.variants.warmup.length;
        return `${n} warmup loss${n === 1 ? "" : "es"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b.variants.warmup.map((w) => `${model}_nc-${w.label}`) },

    { key: "physics",    label: "physics loss", hint: "physics components crossed with weights on top of the base config under the loss curriculum, one trial per pair plus an optional no-physics baseline",
      at        : 2,
      el        : "physicsEl",
      available : (b) => b.physics.size > 0,
      make      : (b) => b._physicsPanel(),
      paint     : (b) => b._paintPhysics(),
      summary   : (b, gpus) => {
        const nComp    = b._physicsComponents().length;
        const nWeight  = b._physicsWeights().length;
        const baseline = b._physicsBaselineOn();
        const total    = nComp * nWeight + (baseline ? 1 : 0);
        return `${nComp} component${nComp === 1 ? "" : "s"} x ${nWeight} weight${nWeight === 1 ? "" : "s"}${baseline ? " + baseline" : ""} = ${total} trial${total === 1 ? "" : "s"}${b._seedsSuffix(total)}${gpus}`;
      },
      names     : (b, model) => {
        const names = [];
        if (b._physicsBaselineOn()) names.push(`${model}_phys-baseline`);
        b._physicsComponents().forEach((component) => {
          b._physicsWeights().forEach((weight) => names.push(`${model}_phys-${component}-w${weight}`));
        });
        return names;
      } },

    { key: "pair",       label: "loss pairs",  hint: "one base component plus one candidate second component per weight, curriculum disabled, one trial per pair plus an optional base-only baseline",
      at        : 3,
      el        : "pairEl",
      available : (b) => b.pair.size > 0,
      make      : (b) => b._pairPanel(),
      paint     : (b) => b._paintPair(),
      summary   : (b, gpus) => {
        const nCand    = b._pairCandidates().length;
        const nWeight  = b._pairWeights().length;
        const baseline = b._pairBaselineOn();
        const total    = nCand * nWeight + (baseline ? 1 : 0);
        return `${b._pairBase()} + ${nCand} candidate${nCand === 1 ? "" : "s"} x ${nWeight} weight${nWeight === 1 ? "" : "s"}${baseline ? " + baseline" : ""} = ${total} trial${total === 1 ? "" : "s"}${b._seedsSuffix(total)}${gpus}`;
      },
      names     : (b, model) => {
        const names = [];
        if (b._pairBaselineOn()) names.push(`${model}_pair-baseline`);
        b._pairCandidates().forEach((component) => {
          b._pairWeights().forEach((weight) => names.push(`${model}_pair-${component}-w${weight}`));
        });
        return names;
      } },

    { key: "secondary",  label: "secondaries", hint: "one trial per secondary-track selection",
      at        : 0,
      el        : "secondaryEl",
      available : (b) => b.secondary.size > 0,
      make      : (b) => b._secondaryPanel(),
      paint     : (b) => b._paintSecondary(),
      summary   : (b, gpus) => {
        const strategy = b._strategy();
        const sampled  = strategy === "uniform" || strategy === "gaussian";
        const count    = sampled ? `${b._secondaryEffective("n_trials")} trials` : "trial count set by the stack";
        return `${strategy}, ${b._secondaryEffective("n_secondaries")} secondaries, ${count}${b._seedsSuffix(null)}${gpus}`;
      },
      pattern   : (b, model) => ({ text: `${model}_sec-${b._strategy()}-tNN_<labels>`, note: "one per selection, labels resolved from the dataset stack" }) },

    { key: "patch",      label: "patch",       hint: "one trial per patch size, same model end to end",
      at        : 1,
      el        : "patchEl",
      available : (b) => b.patch.size > 0,
      make      : (b) => b._patchPanel(),
      paint     : (b) => b._paintPatch(),
      summary   : (b, gpus) => {
        const n = b._patchSizes().length;
        return `${n} patch size${n === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b._patchSizes().map((size) => `${model}_p-${size}`) },

    { key: "presence",   label: "slot presence", hint: "active-normalization x presence-balance matrix (none, A, B, AB), curriculum disabled, one trial per cell",
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

    { key: "input",      label: "input channels", hint: "input-channel ablation, one trial per input variant on its own track scope (all tracks or the reduced selection)",
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

    { key: "context",    label: "context ladder", hint: "one trial per backbone architecture on the shared base config, walking the spatial-context ladder",
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

    { key: "reach",      label: "reach", hint: "size-matched architecture comparison at fixed reach: a 33x33 local CNN against a UNet, same 32x32 patch and same parameter count, one trial per arm",
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

    { key: "head",       label: "head x matching", hint: "one trial per output-head x parameter-matching pair on one fixed backbone",
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

    { key: "augmentation", label: "augmentation", hint: "flips-only augmentation on/off on the shared base config, one trial per state",
      at        : 9,
      el        : "augEl",
      available : (b) => Boolean(b.augTrialsLeaf),
      make      : (b) => b._augmentationPanel(),
      paint     : (b) => b._paintAugmentation(),
      summary   : (b, gpus) => {
        const n = b._augCells().length;
        return `${n} augmentation state${n === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => b._augCells().map((cell) => `${model}_aug-${cell}`) },

    { key: "normalization", label: "normalization", hint: "cumulative normalization ladder: pass amp, then ifg phase, then output amp, then output sigma switch from initial to final strategy",
      at        : 10,
      el        : "normEl",
      available : (b) => b.normTrials.size > 0,
      make      : (b) => b._normalizationPanel(),
      paint     : (b) => b._paintNormalization(),
      summary   : (b, gpus) => `baseline + 4 ladder rungs = 5 trials${b._seedsSuffix(5)}${gpus}`,
      names     : (b, model) => ["nrm-0-initial", "nrm-1-pass_mag", "nrm-2-ifg_phase", "nrm-3-out_amp", "nrm-4-out_sigma"].map((rung) => `${model}_${rung}`) },

    { key: "routing", label: "trunk routing", hint: "one trial per params/existence channel-group assignment, both trunks fixed at the shared parity feature ladders",
      at        : 11,
      el        : "routingEl",
      available : (b) => b.routingTrials.size > 0,
      make      : (b) => b._routingPanel(),
      paint     : (b) => b._paintRouting(),
      summary   : (b, gpus) => {
        const n = Object.keys(b._routingTrialsDict()).length;
        return `${n} routing assignment${n === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => Object.keys(b._routingTrialsDict()).map((label) => `${model}_di-${label}`) },

    { key: "ratio",   label: "arm ratio", hint: "budget-matched params/existence width ladders, one trial per declared split; every model is built and its split verified at launch",
      at        : 12,
      el        : "ratioEl",
      available : (b) => b.ratioTrials.size > 0,
      make      : (b) => b._ratioPanel(),
      paint     : (b) => b._paintRatio(),
      summary   : (b, gpus) => {
        const n = Object.keys(b._ratioTrialsDict()).length;
        return `${n} arm split${n === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"}${b._seedsSuffix(n)}${gpus}`;
      },
      names     : (b, model) => Object.keys(b._ratioTrialsDict()).map((label) => `${model}_dr-${label}`) },
  ];

  static NORM_PRESETS = ["min_max", "min_max_log1p", "robust_iqr", "robust_iqr_log1p", "fixed_div_pi", "zscore", "zscore_log1p"];

  static NORM_GROUPS = [
    { key: "pass_mag",  label: "pass amplitude",   step: 1 },
    { key: "ifg_phase", label: "ifg phase",        step: 2 },
    { key: "out_amp",   label: "output amplitude", step: 3 },
    { key: "out_sigma", label: "output sigma",     step: 4 },
  ];

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
    { key: "total_power",      label: "total power" },
    { key: "moments",          label: "profile moments" },
    { key: "coherence_resyn",  label: "coherence resyn" },
    { key: "covariance_match", label: "covariance match" },
    { key: "capon_cycle",      label: "capon cycle" },
  ];

  static PAIR_COMPONENTS = [
    "mse_curve", "l1_curve", "huber_curve", "charbonnier_curve", "cosine_curve",
    "total_power_relerr", "moments", "coherence_resyn", "covariance_match", "capon_cycle",
    "param_huber", "param_mse", "param_legacy", "smoothness_tv", "param_l1",
  ];

  static STRATEGIES = [
    { key: "uniform",     note: "each trial samples n_secondaries distinct tracks uniformly, n_trials distinct sets" },
    { key: "gaussian",    note: "indices drawn from Normal(mean, sigma) over the secondary list, mean and sigma required" },
    { key: "consecutive", note: "block of n_secondaries consecutive tracks, advanced block_step per trial until the stack ends" },
    { key: "spaced",      note: "n_secondaries tracks spaced 'spacing' apart, advanced block_step per trial until the stack ends" },
  ];

  static SECONDARY_FIELDS = [
    { key: "n_secondaries", strategies: ["uniform", "gaussian", "consecutive", "spaced"] },
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
    this.modelLeaf    = byPath.get("backbone_name") || byPath.get("model_name");
    this.gpusLeaf     = byPath.get("gpus");
    this.seedsLeaf    = byPath.get("seeds");

    this.presenceTrialsLeaf = byPath.get("presence_trials");

    this.inputTrialsLeaf   = byPath.get("input_trials");
    this.contextTrialsLeaf = byPath.get("context_trials");
    this.augTrialsLeaf     = byPath.get("augmentation_trials");

    this.secondary     = new Map();
    this.patch         = new Map();
    this.physics       = new Map();
    this.pair          = new Map();
    this.reach         = new Map();
    this.headTrials    = new Map();
    this.normTrials    = new Map();
    this.routingTrials = new Map();
    this.ratioTrials   = new Map();
    byPath.forEach((leaf) => {
      if (leaf.section === "secondary_trials")     this.secondary.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "patch_trials")         this.patch.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "physics_trials")       this.physics.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "pair_trials")          this.pair.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "reach_trials")         this.reach.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "head_trials")          this.headTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "normalization_trials") this.normTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "routing_trials")       this.routingTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "ratio_trials")         this.ratioTrials.set(leaf.path.split(".").pop(), leaf);
    });

    this.claimed = ["trials_enabled", "warmup_losses", "complete_losses"];
    if (this.modeLeaf) this.claimed.push("trials_mode");
    this.secondary.forEach((leaf) => this.claimed.push(leaf.path));
    this.patch.forEach((leaf) => this.claimed.push(leaf.path));
    this.physics.forEach((leaf) => this.claimed.push(leaf.path));
    this.pair.forEach((leaf) => this.claimed.push(leaf.path));
    this.reach.forEach((leaf) => this.claimed.push(leaf.path));
    this.headTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.normTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.routingTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.ratioTrials.forEach((leaf) => this.claimed.push(leaf.path));
    if (this.presenceTrialsLeaf)   this.claimed.push(this.presenceTrialsLeaf.path);
    if (this.inputTrialsLeaf)      this.claimed.push(this.inputTrialsLeaf.path);
    if (this.contextTrialsLeaf)    this.claimed.push(this.contextTrialsLeaf.path);
    if (this.augTrialsLeaf)        this.claimed.push(this.augTrialsLeaf.path);

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
    this.patchSizesEl    = null;
    this.patchFlagPaints = [];
    this.physicsEl           = null;
    this.physicsComponentsEl = null;
    this.physicsWeightsEl    = null;
    this.pairEl              = null;
    this.pairBaseSelect      = null;
    this.pairCandidatesEl    = null;
    this.pairWeightsEl       = null;
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
    this.augCellsEl      = null;
    this.normEl          = null;
    this.normSelects     = new Map();
    this.routingEl       = null;
    this.routingCellsEl  = null;
    this.ratioEl         = null;
    this.ratioCellsEl    = null;
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

    const hint     = document.createElement("span");
    hint.className = "special-head__hint";
    this.hintEl    = hint;
    head.appendChild(hint);

    const summary     = document.createElement("span");
    summary.className = "exp-builder__summary";
    this.summaryEl    = summary;
    head.appendChild(summary);
    if (this.modeLeaf) head.appendChild(this._modeSegment());
    head.appendChild(this._trialsSwitch());

    const body     = document.createElement("div");
    body.className = "exp-builder__body";

    const columns     = document.createElement("div");
    columns.className = "exp-builder__columns";
    this.columnsEl    = columns;
    this.columnEls.warmup   = columns.appendChild(this._column("warmup", "warmup losses"));
    this.columnEls.complete = columns.appendChild(this._column("complete", "complete losses"));
    body.appendChild(columns);

    if (this.modeLeaf) {
      body.appendChild(this._warmupCatalogPanel());
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

    [this.warmupLeaf, this.completeLeaf].forEach((leaf) => {
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
    return ExperimentBuilder.MODES.find((mode) => mode.key === key);
  }

  _modes() {
    return ExperimentBuilder.MODES.filter((mode) => mode.available(this));
  }

  _panelModes() {
    return this._modes().filter((mode) => mode.make).sort((a, b) => a.at - b.at);
  }

  _mode() {
    if (!this.modeLeaf) return "curriculum";
    const value = this.view._effective(this.modeLeaf);
    return this._modes().some((mode) => mode.key === value) ? value : "curriculum";
  }

  _strategy() {
    const leaf = this.secondary.get("strategy");
    if (!leaf) return "consecutive";
    const value = this.view._effective(leaf);
    return ExperimentBuilder.STRATEGIES.some((strategy) => strategy.key === value) ? value : "consecutive";
  }

  _modeSegment() {
    const leaf     = this.modeLeaf;
    const wrap     = document.createElement("div");
    wrap.className = "exp-mode";
    wrap.title     = `--${leaf.path}`;

    this._modes().forEach((mode) => {
      const btn       = document.createElement("button");
      btn.type        = "button";
      btn.className   = "exp-mode__btn";
      btn.textContent = mode.label;
      btn.addEventListener("click", () => {
        this.view._setValue(leaf, mode.key);
        if (this.view._effective(this.trialsLeaf) !== "True") this.view._setValue(this.trialsLeaf, "True");
      });
      this.modeButtons.set(mode.key, btn);
      wrap.appendChild(btn);
    });

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
    input.value      = leaf.value === "None" ? "" : leaf.value;
    input.spellcheck = false;
    if (leaf.value === "None") input.placeholder = "required";
    input.addEventListener("input", () => {
      input.classList.toggle("is-dirty", input.value !== leaf.value && input.value !== "");
      this.view._setValue(leaf, input.value);
    });

    row.appendChild(name);
    row.appendChild(input);

    this.secondaryRows.push({ strategies: field.strategies, row });
    this.view.controls[leaf.path] = { leaf, reset: () => {
      input.value = leaf.value === "None" ? "" : leaf.value;
      input.classList.remove("is-dirty");
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

    const findLeaf = this.patch.get("find_max_batch");
    if (findLeaf) head.appendChild(this._patchFlag(findLeaf, "max batch", "probe the largest batch that fits the VRAM budget before each trial run"));

    const scaleLeaf = this.patch.get("scale_lr");
    if (scaleLeaf) head.appendChild(this._patchFlag(scaleLeaf, "scale lr", "scale the learning rate linearly by resolved batch / lr_reference_batch_size"));

    head.appendChild(LaunchWidgetDom.mini("Add size", () => {
      const sizes = this._patchSizes();
      sizes.push(sizes.length ? sizes[sizes.length - 1] : 64);
      this._emitPatchSizes(sizes);
    }));

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "each size trains the same model end to end; stride = round(size x stride_ratio); max batch probes the largest OOM-safe batch per trial, scale lr rescales the learning rate to the batch the probe resolved";

    const sizes     = document.createElement("div");
    sizes.className = "exp-patch__sizes";
    this.patchSizesEl = sizes;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(sizes);
    this.patchEl = panel;

    const sizesLeaf = this.patch.get("sizes");
    if (sizesLeaf) this.view.controls[sizesLeaf.path] = { leaf: sizesLeaf, reset: () => this._paintPatch() };
    return panel;
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
    input.value      = leaf.value;
    input.addEventListener("input", () => {
      input.classList.toggle("is-dirty", input.value !== leaf.value && input.value !== "");
      this.view._setValue(leaf, input.value);
    });

    wrap.appendChild(input);
    this.view.controls[leaf.path] = { leaf, reset: () => {
      input.value = leaf.value;
      input.classList.remove("is-dirty");
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

  _patchSizes() {
    const leaf = this.patch.get("sizes");
    if (!leaf) return [];

    let raw = null;
    try {
      raw = PythonLiteral.parse(this.view._effective(leaf));
    } catch (e) {
      raw = null;
    }
    if (!Array.isArray(raw)) return [];

    return raw.map((value) => Number(value)).filter((value) => Number.isFinite(value));
  }

  _emitPatchSizes(sizes) {
    const leaf = this.patch.get("sizes");
    if (!leaf) return;

    const clean = sizes.map((value) => Math.round(Number(value))).filter((value) => Number.isFinite(value) && value >= 1);
    this.view._setValue(leaf, PythonLiteral.render(clean));

    this._paintPatch();
    this._paintSummary();
    this._paintNames();
  }

  _patchChip(size, index, sizes) {
    const chip     = document.createElement("div");
    chip.className = "exp-patch__chip";

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input exp-patch__size";
    input.type       = "number";
    input.step       = "1";
    input.min        = "1";
    input.spellcheck = false;
    input.value      = String(size);
    input.title      = "patch size in pixels (square)";
    input.addEventListener("change", () => {
      const next  = sizes.slice();
      next[index] = Number(input.value);
      this._emitPatchSizes(next);
    });

    const remove = LaunchWidgetDom.mini("×", () => {
      const next = sizes.slice();
      next.splice(index, 1);
      this._emitPatchSizes(next);
    });
    remove.classList.add("exp-patch__remove");
    remove.title = "Remove size";

    chip.appendChild(input);
    chip.appendChild(remove);
    return chip;
  }

  _paintPatch() {
    if (!this.patchSizesEl) return;

    const sizes = this._patchSizes();
    this.patchSizesEl.innerHTML = "";
    sizes.forEach((size, index) => this.patchSizesEl.appendChild(this._patchChip(size, index, sizes)));

    this.patchFlagPaints.forEach((paint) => paint());
  }

  _physicsPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-patch exp-physics";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">physics loss sweep</span>`;

    const baselineLeaf = this.physics.get("include_baseline");
    if (baselineLeaf) head.appendChild(this._patchFlag(baselineLeaf, "baseline", "prepend one run of the plain base config with every physics term disabled, as the reference"));

    head.appendChild(LaunchWidgetDom.mini("Add weight", () => {
      const weights = this._physicsWeights();
      weights.push(weights.length ? weights[weights.length - 1] : 0.05);
      this._emitPhysicsWeights(weights);
    }));

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "each checked component trains once per weight on top of the base config, with every other physics term disabled in both stages; the loss curriculum stays on, so the tested term joins only after the swap epoch";

    const componentsHead       = document.createElement("div");
    componentsHead.className   = "exp-presence__sub";
    componentsHead.textContent = "components";

    const components         = document.createElement("div");
    components.className     = "exp-builder__names exp-presence__strategies";
    this.physicsComponentsEl = components;

    const weightsHead       = document.createElement("div");
    weightsHead.className   = "exp-presence__sub";
    weightsHead.textContent = "weights";

    const weights         = document.createElement("div");
    weights.className     = "exp-patch__sizes";
    this.physicsWeightsEl = weights;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(componentsHead);
    panel.appendChild(components);
    panel.appendChild(weightsHead);
    panel.appendChild(weights);
    this.physicsEl = panel;

    const componentsLeaf = this.physics.get("components");
    if (componentsLeaf) this.view.controls[componentsLeaf.path] = { leaf: componentsLeaf, reset: () => this._repaintPhysics() };

    const weightsLeaf = this.physics.get("weights");
    if (weightsLeaf) this.view.controls[weightsLeaf.path] = { leaf: weightsLeaf, reset: () => this._repaintPhysics() };

    this._paintPhysics();
    return panel;
  }

  _physicsComponents() {
    return this._listOf(this.physics.get("components"), (value) => typeof value === "string");
  }

  _physicsWeights() {
    return this._numbersOf(this.physics.get("weights"));
  }

  _physicsBaselineOn() {
    const leaf = this.physics.get("include_baseline");
    return leaf ? this.view._effective(leaf) === "True" : false;
  }

  _emitPhysicsComponents(components) {
    const leaf = this.physics.get("components");
    if (!leaf) return;

    this.view._setValue(leaf, PythonLiteral.render(components));
    this._repaintPhysics();
  }

  _emitPhysicsWeights(weights) {
    const leaf = this.physics.get("weights");
    if (!leaf) return;

    const clean = weights.map((value) => Number(value)).filter((value) => Number.isFinite(value) && value > 0);
    this.view._setValue(leaf, PythonLiteral.render(clean));
    this._repaintPhysics();
  }

  _togglePhysicsComponent(key) {
    const order = ExperimentBuilder.PHYSICS_COMPONENTS.map((entry) => entry.key);
    this._toggleInCatalog(order, this._physicsComponents(), key, (components) => this._emitPhysicsComponents(components));
  }

  _paintPhysics() {
    if (!this.physicsEl) return;

    if (this.physicsComponentsEl) {
      this.physicsComponentsEl.innerHTML = "";
      const active = this._physicsComponents();
      ExperimentBuilder.PHYSICS_COMPONENTS.forEach(({ key, label }) => {
        const chip = this._toggleChip(label, active.includes(key), "Click to drop this component from the sweep", "Click to sweep this component", () => this._togglePhysicsComponent(key));
        this.physicsComponentsEl.appendChild(chip);
      });
    }

    if (this.physicsWeightsEl) {
      const weights = this._physicsWeights();
      this.physicsWeightsEl.innerHTML = "";
      weights.forEach((weight, index) => this.physicsWeightsEl.appendChild(this._weightChip(weight, index, weights, "physics loss weight applied to the tested component", (next) => this._emitPhysicsWeights(next))));
    }

    this.patchFlagPaints.forEach((paint) => paint());
  }

  _repaintPhysics() {
    this._paintPhysics();
    this._paintSummary();
    this._paintNames();
  }

  _pairPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-patch exp-pair";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">loss pair sweep</span>`;

    const baseLeaf = this.pair.get("base_component");
    if (baseLeaf) {
      const select     = document.createElement("select");
      select.className = "run-select exp-secondary__strategy";
      select.title     = `base component held in every run (--${baseLeaf.path})`;
      ExperimentBuilder.PAIR_COMPONENTS.forEach((name) => {
        const opt       = document.createElement("option");
        opt.value       = name;
        opt.textContent = name;
        select.appendChild(opt);
      });
      select.addEventListener("change", () => this._setPairBase(select.value));
      head.appendChild(select);

      this.pairBaseSelect = select;
      this.view.controls[baseLeaf.path] = { leaf: baseLeaf, reset: () => this._repaintPair() };
    }

    const baseWeightLeaf = this.pair.get("base_weight");
    if (baseWeightLeaf) head.appendChild(this._pairBaseWeight(baseWeightLeaf));

    const baselineLeaf = this.pair.get("include_baseline");
    if (baselineLeaf) head.appendChild(this._patchFlag(baselineLeaf, "baseline", "prepend one run that trains the base component alone, as the reference"));

    head.appendChild(LaunchWidgetDom.mini("Add weight", () => {
      const weights = this._pairWeights();
      weights.push(weights.length ? weights[weights.length - 1] : 0.05);
      this._emitPairWeights(weights);
    }));

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "each checked component joins the base component at each weight as a two-term loss with every other term disabled; curriculum stays off so the pair applies from epoch 0";

    const candidatesHead       = document.createElement("div");
    candidatesHead.className   = "exp-presence__sub";
    candidatesHead.textContent = "second components";

    const candidates      = document.createElement("div");
    candidates.className  = "exp-builder__names exp-presence__strategies";
    this.pairCandidatesEl = candidates;

    const weightsHead       = document.createElement("div");
    weightsHead.className   = "exp-presence__sub";
    weightsHead.textContent = "weights";

    const weights      = document.createElement("div");
    weights.className  = "exp-patch__sizes";
    this.pairWeightsEl = weights;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(candidatesHead);
    panel.appendChild(candidates);
    panel.appendChild(weightsHead);
    panel.appendChild(weights);
    this.pairEl = panel;

    const componentsLeaf = this.pair.get("components");
    if (componentsLeaf) this.view.controls[componentsLeaf.path] = { leaf: componentsLeaf, reset: () => this._repaintPair() };

    const weightsLeaf = this.pair.get("weights");
    if (weightsLeaf) this.view.controls[weightsLeaf.path] = { leaf: weightsLeaf, reset: () => this._repaintPair() };

    this._paintPair();
    return panel;
  }

  _pairBaseWeight(leaf) {
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
    input.value      = leaf.value;
    input.addEventListener("input", () => {
      input.classList.toggle("is-dirty", input.value !== leaf.value && input.value !== "");
      this.view._setValue(leaf, input.value);
    });

    wrap.appendChild(input);
    this.view.controls[leaf.path] = { leaf, reset: () => {
      input.value = leaf.value;
      input.classList.remove("is-dirty");
    } };
    return wrap;
  }

  _pairBase() {
    const leaf = this.pair.get("base_component");
    if (!leaf) return "param_l1";

    const value = this.view._effective(leaf).replace(/^['"]|['"]$/g, "");
    return ExperimentBuilder.PAIR_COMPONENTS.includes(value) ? value : "param_l1";
  }

  _pairCandidates() {
    return this._listOf(this.pair.get("components"), (value) => typeof value === "string");
  }

  _pairWeights() {
    return this._numbersOf(this.pair.get("weights"));
  }

  _pairBaselineOn() {
    const leaf = this.pair.get("include_baseline");
    return leaf ? this.view._effective(leaf) === "True" : false;
  }

  _emitPairCandidates(candidates) {
    const leaf = this.pair.get("components");
    if (!leaf) return;

    this.view._setValue(leaf, PythonLiteral.render(candidates));
    this._repaintPair();
  }

  _emitPairWeights(weights) {
    const leaf = this.pair.get("weights");
    if (!leaf) return;

    const clean = weights.map((value) => Number(value)).filter((value) => Number.isFinite(value) && value > 0);
    this.view._setValue(leaf, PythonLiteral.render(clean));
    this._repaintPair();
  }

  _setPairBase(name) {
    const leaf = this.pair.get("base_component");
    if (!leaf) return;

    this.view._setValue(leaf, name);

    const candidates = this._pairCandidates().filter((value) => value !== name);
    if (candidates.length) this._emitPairCandidates(candidates);
    else this._repaintPair();
  }

  _togglePairCandidate(name) {
    if (name === this._pairBase()) return;
    this._toggleInCatalog(ExperimentBuilder.PAIR_COMPONENTS, this._pairCandidates(), name, (candidates) => this._emitPairCandidates(candidates));
  }

  _paintPair() {
    if (!this.pairEl) return;

    const base = this._pairBase();
    if (this.pairBaseSelect) this.pairBaseSelect.value = base;

    if (this.pairCandidatesEl) {
      this.pairCandidatesEl.innerHTML = "";
      const active = this._pairCandidates();
      ExperimentBuilder.PAIR_COMPONENTS.forEach((name) => {
        if (name === base) return;
        const chip = this._toggleChip(name, active.includes(name), "Click to drop this component from the sweep", "Click to test this component as the second term", () => this._togglePairCandidate(name));
        this.pairCandidatesEl.appendChild(chip);
      });
    }

    if (this.pairWeightsEl) {
      const weights = this._pairWeights();
      this.pairWeightsEl.innerHTML = "";
      weights.forEach((weight, index) => this.pairWeightsEl.appendChild(this._weightChip(weight, index, weights, "weight of the second component", (next) => this._emitPairWeights(next))));
    }

    this.patchFlagPaints.forEach((paint) => paint());
  }

  _repaintPair() {
    this._paintPair();
    this._paintSummary();
    this._paintNames();
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
    note.textContent = "Toggle cells of the active-normalization x presence-balance matrix; every cell trains with curriculum disabled and repeats across the seeds list. Cell specs live in code (_default_presence_trials); use reset matrix to restore the full 2x2.";

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
    parts.push(on("use_interferograms") === false ? "no ifg" : "ifg");
    if (on("use_dem") === true) parts.push("dem");
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
    note.textContent = "Each variant toggles which input channels feed the model and scopes its track list: tracks=all expands the secondaries to every track in the baselines table, tracks=reduced keeps the configured selection. Drop variants to trim the fan-out. Adding variants and channel representations live in code (_default_input_trials); use reset variants to restore the default.";

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
    input.value      = leaf.value;
    input.spellcheck = false;
    input.addEventListener("input", () => {
      input.classList.toggle("is-dirty", input.value !== leaf.value && input.value !== "");
      this.view._setValue(leaf, input.value);
      this._paintSummary();
    });

    row.appendChild(name);
    row.appendChild(input);

    this.view.controls[leaf.path] = { leaf, reset: () => {
      input.value = leaf.value;
      input.classList.remove("is-dirty");
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

  _augDefaults() {
    return this._defaultsDict(this.augTrialsLeaf);
  }

  _augTrials() {
    return this._trialsDict(this.augTrialsLeaf);
  }

  _augCells() {
    return Object.keys(this._augTrials());
  }

  _augChipText(cell, enabled) {
    return `${cell} · ${enabled ? "flips h+v @ 0.5" : "no augmentation"}`;
  }

  _augmentationPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">augmentation on/off</span>`;
    const reset    = LaunchWidgetDom.mini("reset states", () => this._resetAugmentation());
    reset.classList.add("exp-presence__reset");
    head.appendChild(reset);

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "Toggle the two augmentation states on the shared base config: on trains with horizontal and vertical flips at probability 0.5 (rot90 and noise forced off), off zeroes every augmentation probability; every state repeats across the seeds list. State specs live in code (_default_augmentation_trials); use reset states to restore the on/off pair.";

    const cellHead       = document.createElement("div");
    cellHead.className   = "exp-presence__sub";
    cellHead.textContent = "augmentation states";

    const cells     = document.createElement("div");
    cells.className = "exp-builder__names exp-presence__cells";
    this.augCellsEl = cells;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(cellHead);
    panel.appendChild(cells);
    this.augEl = panel;

    if (this.augTrialsLeaf) this.view.controls[this.augTrialsLeaf.path] = { leaf: this.augTrialsLeaf, reset: () => this._repaintAugmentation() };

    this._paintAugmentation();
    return panel;
  }

  _paintAugmentation() {
    if (!this.augCellsEl) return;
    this._paintDictCells(this.augCellsEl, this._augTrials(), this._augDefaults(), (cell, enabled) => this._augChipText(cell, enabled), "Click to drop this state", "Click to add this state", (cell) => this._toggleAugCell(cell));
  }

  _toggleAugCell(cell) {
    if (!this.augTrialsLeaf) return;

    const ordered = this._toggleDictCell(this._augTrials(), this._augDefaults(), cell);
    if (!ordered) return;

    this.view._setValue(this.augTrialsLeaf, PythonLiteral.render(ordered));
    this._repaintAugmentation();
  }

  _resetAugmentation() {
    if (this.augTrialsLeaf) this.view._setValue(this.augTrialsLeaf, this.augTrialsLeaf.value);
    this._repaintAugmentation();
  }

  _repaintAugmentation() {
    this._paintAugmentation();
    this._paintSummary();
    this._paintNames();
  }

  _normValue(key) {
    const leaf = this.normTrials.get(key);
    return leaf ? this.view._effective(leaf) : "?";
  }

  _normalizationPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">normalization ladder</span>`;
    const reset    = LaunchWidgetDom.mini("reset ladder", () => this._resetNormalization());
    reset.classList.add("exp-presence__reset");
    head.appendChild(reset);

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "Cumulative ladder from the all-initial baseline to the final strategies: rung 1 switches pass amplitude, rung 2 adds ifg phase, rung 3 adds output amplitude, rung 4 adds output sigma (out_mu keeps the base config throughout). Every rung sets all four channel strategies explicitly and repeats across the seeds list.";

    const grid     = document.createElement("div");
    grid.className = "exp-secondary__grid";

    ExperimentBuilder.NORM_GROUPS.forEach((group) => {
      const initialLeaf = this.normTrials.get(`initial_${group.key}`);
      const finalLeaf   = this.normTrials.get(`final_${group.key}`);
      if (!initialLeaf || !finalLeaf) return;

      const row     = document.createElement("div");
      row.className = "cfg-edit__row";

      const label       = document.createElement("span");
      label.className   = "cfg-edit__key";
      label.textContent = `${group.label} (rung ${group.step})`;
      row.appendChild(label);

      [initialLeaf, finalLeaf].forEach((leaf, index) => {
        const select     = document.createElement("select");
        select.className = "run-select exp-secondary__strategy";
        select.title     = `--${leaf.path}`;
        ExperimentBuilder.NORM_PRESETS.forEach((preset) => {
          const opt       = document.createElement("option");
          opt.value       = preset;
          opt.textContent = preset;
          select.appendChild(opt);
        });
        select.addEventListener("change", () => {
          this.view._setValue(leaf, select.value);
          this._repaintNormalization();
        });
        this.normSelects.set(leaf.path, select);
        this.view.controls[leaf.path] = { leaf, reset: () => this._repaintNormalization() };
        row.appendChild(select);

        if (index === 0) {
          const arrow       = document.createElement("span");
          arrow.className   = "exp-presence__sub";
          arrow.textContent = "->";
          row.appendChild(arrow);
        }
      });

      grid.appendChild(row);
    });

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(grid);
    this.normEl = panel;

    this._paintNormalization();
    return panel;
  }

  _paintNormalization() {
    if (!this.normEl) return;

    this.normSelects.forEach((select, path) => {
      const leaf = this.normTrials.get(path.split(".").pop());
      if (leaf && document.activeElement !== select) select.value = this.view._effective(leaf);
    });
  }

  _resetNormalization() {
    this.normTrials.forEach((leaf) => this.view._setValue(leaf, leaf.value));
    this._repaintNormalization();
  }

  _repaintNormalization() {
    this._paintNormalization();
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
    note.textContent = "One training run per params | existence channel-group assignment, both trunks held at the shared feature ladders below. Drop assignments to trim the fan-out; adding assignments lives in code (_default_dual_routing_trials); use reset assignments to restore the default grid.";

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
    input.value      = leaf.value;
    input.spellcheck = false;
    input.addEventListener("input", () => {
      input.classList.toggle("is-dirty", input.value !== leaf.value);
      this.view._setValue(leaf, input.value);
    });

    row.appendChild(name);
    row.appendChild(input);

    this.view.controls[leaf.path] = { leaf, reset: () => {
      input.value = leaf.value;
      input.classList.remove("is-dirty");
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
    const toggle   = document.createElement("button");
    toggle.type    = "button";
    toggle.className = "switch";
    toggle.setAttribute("role", "switch");
    toggle.title     = `--${leaf.path}`;
    toggle.innerHTML = `<span class="switch__knob"></span>`;

    const paint = () => {
      const on = this.view._effective(leaf) === "True";
      toggle.classList.toggle("is-on", on);
      toggle.classList.toggle("is-dirty", this.view.dirty[leaf.path] !== undefined);
      toggle.setAttribute("aria-checked", String(on));
      this.root.classList.toggle("is-trials", on);
    };

    toggle.addEventListener("click", () => {
      const next = this.view._effective(leaf) === "True" ? "False" : "True";
      this.view._setValue(leaf, next);
      paint();
    });

    this._paintSwitch = paint;
    this.view.controls[leaf.path] = { leaf, reset: () => this._reload() };
    return toggle;
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
    this.variants.warmup   = this._variantsFrom(this.warmupLeaf);
    this.variants.complete = this._variantsFrom(this.completeLeaf);
    if (this._paintSwitch) this._paintSwitch();
    this._paintMode();
    ExperimentBuilder.MODES.forEach((mode) => { if (mode.paint) mode.paint(this); });
    this._paintAll();
  }

  _paintAll() {
    ["warmup", "complete"].forEach((which) => {
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

    if (entry.pattern) {
      const spec = entry.pattern(this, model);

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
      window.toast("custom feature needs a label and a config path", "error");
      return false;
    }

    const features = this._features();
    if (features.some((entry) => entry.label === cleanLabel)) {
      window.toast(`feature '${cleanLabel}' is already in the order`, "error");
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
      window.toast(`feature '${clean}' is already in the order`, "error");
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
      window.toast("a config path is required", "error");
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


class NumberField {

  constructor(view, leaf, short, spec = null) {
    this.view    = view;
    this.leaf    = leaf;
    this.short   = short || leaf.path.split(".").pop();
    this.integer = leaf.type === "int";
    this.default = Number.isFinite(Number(leaf.value)) ? Number(leaf.value) : 0;
    this.log     = false;
    this.range   = this._resolve(spec);
    this.input   = null;
    this.chips   = new Map();
    this.reset   = () => this._paint();
  }

  build() {
    const el     = document.createElement("div");
    el.className = "numfield";

    const top     = document.createElement("div");
    top.className = "numfield__top";

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input numfield__input";
    input.type       = "number";
    input.step       = this.integer ? "1" : "any";
    input.spellcheck = false;
    this.input       = input;

    input.addEventListener("input", () => {
      const raw = input.value;
      const v   = Number(raw);
      const out = raw === "" ? "" : (v === this.default ? this.leaf.value : raw);
      this.view._setValue(this.leaf, out);
      this._mark();
    });
    top.appendChild(input);
    el.appendChild(top);

    const presets     = document.createElement("div");
    presets.className = "numfield__presets";
    this.range.presets.forEach((value) => presets.appendChild(this._chip(value)));
    top.appendChild(presets);

    this._paint();
    return { el, input, reset: this.reset };
  }

  _resolve(spec) {
    const r = spec
      ? { min: spec.min, max: spec.max, step: spec.step || 1, log: Boolean(spec.log), presets: spec.presets ? spec.presets.slice() : this._spanPresets(spec) }
      : this._fallback();

    this.log = r.log;
    r.min = Math.min(r.min, this.default);
    r.max = Math.max(r.max, this.default);
    r.presets.push(this.default);
    r.presets = this._cleanPresets(r.presets, r);
    return r;
  }

  _spanPresets(spec) {
    const step = spec.step || 1;
    const span = spec.max - spec.min;
    const snap = (x) => spec.min + Math.round((x - spec.min) / step) * step;
    return [0, 0.25, 0.5, 0.75, 1].map((f) => snap(spec.min + span * f));
  }

  _fallback() {
    if (!this.integer && this.default > 0 && this.default <= 1) {
      return { min: 0, max: 1, step: 0.01, log: false, presets: [0, 0.25, 0.5, 0.75, 1] };
    }
    const base    = Math.abs(this.default) || (this.integer ? 10 : 1);
    const max     = this._nice(base * 4);
    const min     = this.default < 0 ? -max : 0;
    const span    = max - min || 1;
    const step    = this.integer ? 1 : Math.pow(10, Math.floor(Math.log10(span)) - 2) || 0.01;
    const presets = [min, min + span * 0.25, min + span * 0.5, min + span * 0.75, max].map((x) => (this.integer ? Math.round(x) : this._nice(x)));
    return { min, max, step, log: false, presets };
  }

  _nice(x) {
    if (x === 0) return 0;
    const unit = Math.pow(10, Math.floor(Math.log10(Math.abs(x)))) / 10;
    return Math.round(x / unit) * unit;
  }

  _cleanPresets(list, r) {
    const within = list.filter((x) => Number.isFinite(x) && x >= r.min - 1e-9 && x <= r.max + 1e-9);
    const seen   = new Map();
    within.forEach((x) => {
      const key = this.integer ? String(Math.round(x)) : this._fmt(x);
      if (!seen.has(key)) seen.set(key, Number(key));
    });
    return [...seen.values()].sort((a, b) => a - b).slice(0, 8);
  }

  _fmt(v) {
    if (this.integer) return String(Math.round(v));
    if (v === 0) return "0";
    return String(Number(v.toPrecision(this.log ? 2 : 6)));
  }

  _chip(value) {
    const chip       = document.createElement("button");
    chip.type        = "button";
    chip.className   = "numfield__chip";
    chip.textContent = this._fmt(value);
    chip.title       = `set ${this.short} = ${this._fmt(value)}`;
    chip.addEventListener("click", () => {
      this.input.value = this._fmt(value);
      const out = value === this.default ? this.leaf.value : this._fmt(value);
      this.view._setValue(this.leaf, out);
      this._mark();
    });
    this.chips.set(value, chip);
    return chip;
  }

  _mark() {
    const cur   = Number(this.view._effective(this.leaf));
    const dirty = this.view.dirty[this.leaf.path] !== undefined;
    this.input.classList.toggle("is-dirty", dirty);
    this.chips.forEach((chip, key) => {
      const tol = this.integer ? 0.5 : Math.max(1e-12, Math.abs(cur) * 1e-6);
      chip.classList.toggle("is-active", Number.isFinite(cur) && Math.abs(Number(key) - cur) < tol);
    });
  }

  _paint() {
    const eff = this.view._effective(this.leaf);
    this.input.value = eff === "None" ? "" : eff;
    this._mark();
  }
}


class GpuCardSelect {
  constructor(host, opts = {}) {
    this.host     = host;
    this.multi    = Boolean(opts.multi);
    this.onChange = typeof opts.onChange === "function" ? opts.onChange : () => {};
    this.selected = new Set((opts.initial || []).map(Number).filter(Number.isFinite));
    this.gpus     = [];
    this.chips    = new Map();
    this.manual   = null;
    this.loaded   = false;
  }

  async load() {
    let payload = null;
    try {
      payload = await window.apiGet("/api/system");
    } catch (e) {
      payload = null;
    }
    this.gpus   = payload && Array.isArray(payload.gpus) ? payload.gpus : [];
    this.loaded = true;
    this._render();
    return this;
  }

  value() {
    return [...this.selected].sort((a, b) => a - b);
  }

  set(indices) {
    this.selected = new Set((indices || []).map(Number).filter(Number.isFinite));
    if (this.loaded) this._render();
  }

  _indices() {
    const detected = this.gpus.map((gpu) => Number(gpu.index)).filter(Number.isFinite);
    const merged   = new Set([...detected, ...this.selected]);
    return [...merged].sort((a, b) => a - b);
  }

  _render() {
    this.host.innerHTML = "";
    this.chips.clear();
    this.host.classList.add("gpu-picker__board");

    const indices = this._indices();
    if (!indices.length) {
      this._renderManual();
      return;
    }

    this.host.classList.add("gpu-picker__board--chips");
    indices.forEach((index) => this.host.appendChild(this._chip(index)));
    this._paint();
  }

  _chip(index) {
    const info = this.gpus.find((gpu) => Number(gpu.index) === index) || null;

    const chip        = document.createElement("button");
    chip.type         = "button";
    chip.className    = "gpu-chip";
    chip.title        = `cuda:${index}`;
    chip.dataset.tone = info ? (info.others ? "busy" : info.mine ? "mine" : "free") : "offline";

    const name = info ? info.name : "not detected";
    const meta = [];
    if (info && info.util != null)      meta.push(`${info.util}%`);
    if (info && info.mem_total != null) meta.push(`${Math.round(info.mem_used || 0)}/${Math.round(info.mem_total)} MiB`);

    chip.innerHTML =
      `<span class="gpu-chip__id">GPU ${index}</span>` +
      `<span class="gpu-chip__name">${name}</span>` +
      `<span class="gpu-chip__meta">${meta.join("  ·  ")}</span>`;

    chip.addEventListener("click", () => this._toggle(index));
    this.chips.set(index, chip);
    return chip;
  }

  _toggle(index) {
    if (!this.multi) {
      this.selected = new Set([index]);
    } else if (this.selected.has(index)) {
      if (this.selected.size <= 1) return;
      this.selected.delete(index);
    } else {
      this.selected.add(index);
    }
    this._paint();
    this.onChange(this.value());
  }

  _paint() {
    this.chips.forEach((chip, index) => {
      const on = this.selected.has(index);
      chip.classList.toggle("is-on", on);
      chip.setAttribute("aria-pressed", String(on));
    });
  }

  _renderManual() {
    this.host.classList.remove("gpu-picker__board--chips");

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input";
    input.value      = this.value().join(", ");
    input.spellcheck = false;
    input.title      = this.multi
      ? "no GPUs detected on this host — enter device indices, e.g. 0, 1"
      : "no GPUs detected on this host — enter a device index, e.g. 0";
    input.addEventListener("input", () => {
      const parts = input.value.split(",").map((token) => Number(token.trim())).filter(Number.isFinite);
      this.selected = new Set(this.multi ? parts : parts.slice(0, 1));
      this.onChange(this.value());
    });

    this.manual = input;
    this.host.appendChild(input);
  }
}


class GpuPicker {
  constructor(view, leaf) {
    this.view     = view;
    this.leaf     = leaf;
    this.multi    = leaf.type === "list";
    this.el       = null;
    this.note     = null;
    this.cards    = null;
    this.inactive = false;
    this.why      = "";
    this.reset    = () => this._syncFromLeaf();
  }

  build() {
    this.el           = document.createElement("div");
    this.el.className  = "picker gpu-picker";

    const board = document.createElement("div");
    this.el.appendChild(board);

    const note     = document.createElement("p");
    note.className = "picker__note";
    note.textContent = "detecting GPUs...";
    this.note      = note;
    this.el.appendChild(note);

    this.cards = new GpuCardSelect(board, {
      multi    : this.multi,
      initial  : this._parse(this.view._effective(this.leaf)),
      onChange : (indices) => this._emit(indices),
    });
    this.cards.load().then(() => this._paintNote());

    return { el: this.el, input: this.el, reset: this.reset, setInactive: (on, why) => this.setInactive(on, why) };
  }

  setInactive(on, why) {
    this.inactive = Boolean(on);
    this.why      = why || "";
    if (this.el) this.el.classList.toggle("is-inactive", this.inactive);
    this._paintNote();
  }

  _parse(raw) {
    if (this.multi) {
      try {
        const parsed = PythonLiteral.parse(raw);
        return Array.isArray(parsed) ? parsed.map(Number).filter(Number.isFinite) : [];
      } catch (e) {
        return [];
      }
    }
    const value = Number(raw);
    return Number.isFinite(value) ? [value] : [];
  }

  _emit(indices) {
    const value = this.multi ? PythonLiteral.render(indices) : (indices.length ? String(indices[0]) : "");
    this.view._setValue(this.leaf, value);
    this._paintNote();
  }

  _syncFromLeaf() {
    if (this.cards) this.cards.set(this._parse(this.leaf.value));
    this._paintNote();
  }

  _paintNote() {
    if (!this.note) return;

    if (this.inactive) {
      this.note.textContent = this.why;
      this.note.classList.remove("is-dirty");
      return;
    }

    const ordered = this.cards ? this.cards.value() : [];
    const head    = this.multi ? `${ordered.length} GPU${ordered.length === 1 ? "" : "s"}` : `GPU ${ordered[0]}`;
    this.note.textContent = ordered.length
      ? `${head} selected · CUDA_VISIBLE_DEVICES=${ordered.join(",")}`
      : (this.multi ? "select at least one GPU" : "select a GPU");
    this.note.classList.toggle("is-dirty", this.view.dirty[this.leaf.path] !== undefined);
  }
}



class MultiValueField {
  constructor(view, leaf, spec) {
    this.view  = view;
    this.leaf  = leaf;
    this.spec  = spec;
    this.el    = null;
    this.chips = null;
    this.input = null;
    this.count = null;
    this.reset = () => this._paint();
  }

  build() {
    this.el           = document.createElement("div");
    this.el.className = "picker multivalue" + (this.spec.wide ? " multivalue--wide" : "");

    const chips     = document.createElement("div");
    chips.className = "multivalue__chips";
    this.chips      = chips;
    this.el.appendChild(chips);

    if (!this.spec.choices) {
      const entry     = document.createElement("input");
      entry.className = "cfg-edit__input multivalue__entry" + (this.spec.wide ? " multivalue__entry--wide" : "");
      entry.type      = "text";
      entry.spellcheck = false;
      entry.placeholder = this.spec.placeholder || "add value, Enter";
      entry.addEventListener("keydown", (event) => this._onKey(event));
      entry.addEventListener("blur", () => this._commitEntry());
      this.input = entry;
      this.el.appendChild(entry);
    }

    const note     = document.createElement("p");
    note.className = "picker__note";
    this.count     = note;
    this.el.appendChild(note);

    this._paint();
    return { el: this.el, input: this.input || this.el, reset: this.reset };
  }

  _values() {
    try {
      const parsed = PythonLiteral.parse(this.view._effective(this.leaf));
      return Array.isArray(parsed) ? parsed.slice() : [];
    } catch (e) {
      return [];
    }
  }

  _cast(token) {
    if (!this.spec.numeric) return token;
    const value = Number(token);
    if (!Number.isFinite(value)) return null;
    return this.spec.integer ? Math.trunc(value) : value;
  }

  _emit(values) {
    this.view._setValue(this.leaf, PythonLiteral.render(values));
    this._paint();
  }

  _onKey(event) {
    if (event.key !== "Enter" && event.key !== ",") return;
    event.preventDefault();
    this._commitEntry();
  }

  _commitEntry() {
    if (!this.input) return;
    const tokens = this.input.value.split(",").map((part) => part.trim()).filter(Boolean);
    if (!tokens.length) return;

    const values = this._values();
    tokens.forEach((token) => {
      const cast = this._cast(token);
      if (cast !== null && !values.some((existing) => existing === cast)) values.push(cast);
    });

    this.input.value = "";
    this._emit(values);
  }

  _toggleChoice(value) {
    const values = this._values();
    const index  = values.indexOf(value);
    if (index >= 0) values.splice(index, 1);
    else            values.push(value);

    const ordered = this.spec.choices.map((choice) => choice.value).filter((choice) => values.includes(choice));
    this._emit(ordered);
  }

  _removeValue(value) {
    this._emit(this._values().filter((existing) => existing !== value));
  }

  _paint() {
    const values = this._values();
    this.chips.innerHTML = "";

    if (this.spec.choices) {
      this.spec.choices.forEach((choice) => {
        const on        = values.includes(choice.value);
        const chip      = document.createElement("button");
        chip.type       = "button";
        chip.className  = "multivalue__choice" + (on ? " is-on" : "");
        chip.textContent = choice.label;
        chip.title      = `--${this.leaf.path} · ${choice.value}`;
        chip.setAttribute("aria-pressed", String(on));
        chip.addEventListener("click", () => this._toggleChoice(choice.value));
        this.chips.appendChild(chip);
      });
    } else {
      values.forEach((value) => {
        const chip      = document.createElement("span");
        chip.className  = "multivalue__chip";
        chip.title      = String(value);
        const label     = document.createElement("span");
        label.className = "multivalue__label";
        label.textContent = String(value);
        const remove    = document.createElement("button");
        remove.type     = "button";
        remove.className = "multivalue__x";
        remove.innerHTML = "&times;";
        remove.title    = "remove";
        remove.addEventListener("click", () => this._removeValue(value));
        chip.appendChild(label);
        chip.appendChild(remove);
        this.chips.appendChild(chip);
      });
    }

    if (this.count) {
      this.count.textContent = values.length
        ? `${values.length} value${values.length === 1 ? "" : "s"} · ${PythonLiteral.render(values)}`
        : (this.spec.empty || "select at least one value");
      this.count.classList.toggle("is-dirty", this.view.dirty[this.leaf.path] !== undefined);
    }
  }
}


class ConfigForm {
  constructor() {
    this.dirty         = {};
    this.controls      = {};
    this.dependents    = {};
    this.states        = [];
    this.gates         = [];
    this.sections      = [];
    this.pairs         = [];
    this.pairBase      = new Map();
    this.byPath        = new Map();
    this.activeSection = null;
    this.query         = "";
    this._section      = null;
    this.config        = null;
    this.builder       = null;
    this.modelFamilies = null;
    this.layoutEl      = null;
    this.navHost       = null;
    this.pinsEl        = null;
    this.nomatchEl     = null;
    this.countEl       = null;
    this.legacyMode    = false;
    this.legacyBtn     = null;
  }

  _buildToolbar(cfg) {
    const bar = document.createElement("div");
    bar.className = "cfg-toolbar";

    const search = document.createElement("input");
    search.className = "cfg-search";
    search.type = "search";
    search.placeholder = `Filter ${cfg.leaves.length} fields...`;
    search.spellcheck = false;
    search.addEventListener("input", () => {
      this.query = search.value.trim().toLowerCase();
      this._applyVisibility();
    });

    const count = document.createElement("span");
    count.className = "cfg-toolbar__count";
    this.countEl = count;

    const reset = document.createElement("button");
    reset.className = "btn btn--mini";
    reset.textContent = "Reset all";
    reset.addEventListener("click", () => this._resetAll());

    bar.appendChild(search);
    bar.appendChild(count);

    if (cfg.layout && cfg.layout.legacy) {
      const legacy = document.createElement("button");
      legacy.className = "btn btn--mini btn--legacy";
      legacy.textContent = "Legacy mode";
      legacy.title = "Swap every config to the legacy scaled-MSE setup (legacy term only, sorted_gt matching, curriculum and augmentation off) and show only its fields; pick a two-Gaussian parameters dataset";
      legacy.addEventListener("click", () => this._toggleLegacy());
      this.legacyBtn = legacy;
      bar.appendChild(legacy);
    }

    bar.appendChild(reset);
    return bar;
  }

  _toggleLegacy() {
    const spec = this.config.layout.legacy;
    this.legacyMode = !this.legacyMode;

    if (this.legacyMode) {
      Object.entries(spec.preset).forEach(([path, value]) => {
        const leaf = this.byPath.get(path);
        if (!leaf) return;
        if (value !== leaf.value) this.dirty[path] = value;
        else delete this.dirty[path];
      });
    } else {
      Object.keys(spec.preset).forEach((path) => delete this.dirty[path]);
    }

    this.legacyBtn.classList.toggle("is-on", this.legacyMode);
    Object.values(this.controls).forEach((c) => c.reset());
    this._refresh();
  }

  _buildPins(pinned) {
    const panel = document.createElement("section");
    panel.className = "launch-pins";

    const head = document.createElement("header");
    head.className = "launch-pins__head";
    head.innerHTML = `<h3 class="launch-pins__name">Run essentials</h3><span class="launch-pins__hint">check these before every launch</span>`;

    const grid = document.createElement("div");
    grid.className = "launch-pins__grid";
    pinned.forEach((leaf) => grid.appendChild(this._buildRow(leaf, "essentials", true)));

    panel.appendChild(head);
    panel.appendChild(grid);
    this.pinsEl = panel;
    return panel;
  }

  _renderLayout(host, cfg) {
    const layout = cfg.layout;
    this.byPath  = new Map(cfg.leaves.map((leaf) => [leaf.path, leaf]));

    const wrap = document.createElement("div");
    wrap.className = "launch-layout";
    if (layout.mode === "single") wrap.classList.add("launch-layout--single");
    this.layoutEl = wrap;

    const nav = document.createElement("nav");
    nav.className = "secnav";
    nav.setAttribute("aria-label", "Configuration sections");
    this.navHost = nav;

    const main = document.createElement("div");
    main.className = "secmain";

    const declared = [];
    if (layout.essentials.length) {
      declared.push({ key: "essentials", title: "Essentials", panels: null });
    }
    layout.sections.forEach((section) => declared.push(section));

    declared.forEach((section) => {
      this._section = section.key;

      const el = document.createElement("section");
      el.className = "launch-section";
      el.dataset.section = section.key;

      const title = document.createElement("h3");
      title.className = "launch-section__title";
      title.textContent = section.title;
      el.appendChild(title);

      const body = document.createElement("div");
      body.className = "launch-section__body";
      el.appendChild(body);

      if (section.panels === null) {
        body.appendChild(this._buildPins(layout.essentials.map((entry) => this.byPath.get(entry.path))));
      } else {
        section.panels.forEach((panel) => {
          const built = this._buildPanel(panel);
          if (built) body.appendChild(built);
        });
      }

      const record = { key: section.key, title: section.title, when: section.when || null, el, navBtn: null, badge: null };

      if (layout.mode === "sections") {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "secnav__item";
        const badge = document.createElement("span");
        badge.className = "edit-badge";
        badge.hidden = true;
        btn.innerHTML = `<span class="secnav__name">${section.title}</span>`;
        btn.appendChild(badge);
        btn.addEventListener("click", () => this._navigate(record.key));
        record.navBtn = btn;
        record.badge = badge;
        nav.appendChild(btn);
      }

      main.appendChild(el);
      this.sections.push(record);
    });

    wrap.appendChild(main);
    if (layout.mode === "sections") wrap.appendChild(nav);

    const empty = document.createElement("p");
    empty.className = "cfg-note launch-nomatch";
    empty.textContent = "No fields match this filter.";
    empty.hidden = true;
    this.nomatchEl = empty;
    main.appendChild(empty);

    host.appendChild(wrap);
    this._setActiveSection(this.activeSection || this.sections[0].key);
  }

  _buildPanel(panel) {
    if (panel.kind === "hidden") return null;
    if (panel.kind === "special") return this._buildSpecialPanel(panel);
    if (panel.kind === "pair") return this._buildPairPanel(panel);
    return this._buildFieldsPanel(panel);
  }

  _buildSpecialPanel(panel) {
    if (panel.panel === "model_card") {
      const leaf     = this.byPath.get(panel.fields[0]);
      const headLeaf = panel.fields.length > 1 ? this.byPath.get(panel.fields[1]) : null;
      if (!leaf || !this.modelFamilies || !this.modelFamilies.length) return this._buildPathsPanel("Model", panel.fields);
      return new window.ModelCardPanel(this, leaf, headLeaf).build();
    }

    if (panel.panel === "model_toggle") {
      const leaf = this.byPath.get(panel.fields[0]);
      if (!leaf || !this.modelFamilies || !this.modelFamilies.length) return this._buildPathsPanel("Models in run", panel.fields);
      return new window.ModelTogglePanel(this, leaf).build();
    }

    if (panel.panel === "experiment_builder") {
      const candidate = new window.ExperimentBuilder(this, this.byPath);
      if (!candidate.terms.length) return this._buildPathsPanel("Experiment fan-out", panel.fields);
      this.builder = candidate;
      return candidate.build();
    }

    return this._buildPathsPanel(panel.panel, panel.fields);
  }

  _buildPathsPanel(title, paths) {
    const groups = [{ title: null, fields: paths.map((path) => ({ path })) }];
    return this._buildFieldsPanel({ kind: "fields", title, groups });
  }

  _buildFieldsPanel(panel) {
    const el = document.createElement("section");
    el.className = "cfg-panel";
    el.dataset.cols = String(Math.min(panel.groups.length, 4));

    if (panel.title) {
      const head = document.createElement("header");
      head.className = "cfg-panel__head";
      head.innerHTML = `<h4 class="cfg-panel__name">${panel.title}</h4>`;
      el.appendChild(head);
    }

    el.appendChild(this._buildGroups(panel.groups));
    return el;
  }

  _buildGroups(groups, pathMap = null) {
    const body = document.createElement("div");
    body.className = "cfg-panel__groups";

    groups.forEach((group) => {
      const groupEl = document.createElement("div");
      groupEl.className = "field-group";
      if (group.title) {
        const heading = document.createElement("div");
        heading.className = "field-group__title";
        heading.textContent = group.title;
        groupEl.appendChild(heading);
      }
      const inner = document.createElement("div");
      inner.className = "field-group__grid";
      group.fields.forEach((entry) => this._buildEntry(entry, inner, pathMap));
      groupEl.appendChild(inner);
      body.appendChild(groupEl);
    });

    return body;
  }

  _mapPath(path, pathMap) {
    return pathMap ? pathMap.override + path.slice(pathMap.base.length) : path;
  }

  _buildEntry(entry, host, pathMap) {
    if (!entry.gate) {
      host.appendChild(this._buildRow(this.byPath.get(this._mapPath(entry.path, pathMap)), this._section));
      return;
    }

    const lead = this.byPath.get(this._mapPath(entry.gate, pathMap));
    const cell = document.createElement("div");
    cell.className = "band-block";

    cell.appendChild(this._buildGateRow(lead, this._gateLabel(this._shortName(lead))));

    const gatedRows = [];
    entry.fields.forEach((sub) => {
      const leaf = this.byPath.get(this._mapPath(sub.path, pathMap));
      const short = this._shortName(leaf);
      const row = short.startsWith("weight_") ? this._buildWeightRow(leaf) : this._buildRow(leaf, this._section);
      row.classList.add("cfg-edit__row--dependent");
      cell.appendChild(row);
      gatedRows.push(this.states[this.states.length - 1]);
    });

    this.gates.push({ leaf: lead, states: gatedRows });
    host.appendChild(cell);
  }

  _gateLabel(short) {
    if (short.startsWith("use_")) return short.slice(4);
    if (short !== "enabled" && short.endsWith("_enabled")) return short.slice(0, -"_enabled".length);
    return short;
  }

  _shortName(leaf) {
    return leaf.section ? leaf.path.slice(leaf.section.length + 1) : leaf.path;
  }

  _buildPairPanel(panel) {
    const el = document.createElement("section");
    el.className = "cfg-panel cfg-panel--pair";

    const head = document.createElement("header");
    head.className = "cfg-panel__head";
    head.innerHTML = `<h4 class="cfg-panel__name">${panel.title}</h4><span class="cfg-panel__hint">${panel.base} · overridden per-field by ${panel.override}</span>`;
    el.appendChild(head);

    el.appendChild(this._buildGroups(panel.groups));

    const override = document.createElement("div");
    override.className = "pair-override";

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "pair-override__head";
    toggle.setAttribute("aria-expanded", "false");
    const badge = document.createElement("span");
    badge.className = "edit-badge";
    toggle.innerHTML = `<span class="pair-override__chev">&rsaquo;</span><h4 class="pair-override__name">${panel.override} overrides</h4>`;
    toggle.appendChild(badge);

    const note = document.createElement("p");
    note.className = "pair-override__note";
    note.textContent = `Fields where both stages share a default inherit ${panel.base} edits; editing here pins the ${panel.override} value.`;

    const startAt = this.states.length;
    const body = this._buildGroups(panel.groups, { base: panel.base, override: panel.override });
    body.classList.add("pair-override__body");
    body.hidden = true;

    const record = { base: panel.base, override: panel.override, badge, body, toggle, container: override, open: false, states: this.states.slice(startAt) };

    const inheritPath = panel.base.split(".").slice(0, -1).concat("inherit").join(".");
    record.states.forEach(({ leaf }) => {
      this.pairBase.set(leaf.path, { base: panel.base + leaf.path.slice(panel.override.length), inherit: inheritPath });
    });
    toggle.addEventListener("click", () => {
      record.open = !record.open;
      this._applyVisibility();
    });
    this.pairs.push(record);

    override.appendChild(toggle);
    override.appendChild(note);
    override.appendChild(body);
    el.appendChild(override);
    return el;
  }

  _buildRow(leaf, sectionKey, pinned = false) {
    const short = this._shortName(leaf);

    const row = document.createElement("div");
    row.className = "cfg-edit__row";
    row.title = `--${leaf.path}`;

    const label = document.createElement("div");
    label.className = "cfg-edit__name";
    label.textContent = short;
    label.title = `${leaf.type} · --${leaf.path}`;
    row.appendChild(label);

    let control;
    const spec    = leaf.editable ? this._widgetSpec(leaf) : null;
    const kind    = spec ? spec.kind : null;
    const choices = Array.isArray(leaf.choices) && leaf.choices.length ? leaf.choices : (kind === "choice" ? spec.options : null);
    if (kind === "gpu" && window.GpuPicker) {
      control = new window.GpuPicker(this, leaf).build();
      row.classList.add("cfg-edit__row--board");
    } else if (kind === "multi" && window.MultiValueField) {
      control = new window.MultiValueField(this, leaf, spec).build();
      row.classList.add("cfg-edit__row--board");
    } else if (kind === "dataset" && window.DatasetPicker) {
      control = new window.DatasetPicker(this, leaf, spec).build();
      row.classList.add(spec.multi ? "cfg-edit__row--board" : "cfg-edit__row--wide");
    } else if (choices) {
      control = this._choiceControl(leaf, choices, spec ? spec.default_label : null);
      row.classList.add("cfg-edit__row--choice");
    } else if (!leaf.editable) {
      control = this._textControl(leaf);
      control.input.disabled = true;
      control.input.classList.add("is-locked");
      control.input.title = "not overridable from the command line";
    } else if (leaf.type === "bool") {
      control = this._switchControl(leaf);
      row.classList.add("cfg-edit__row--bool");
    } else if (leaf.type === "int" || leaf.type === "float") {
      control = new window.NumberField(this, leaf, short, kind === "number" ? spec : null).build();
      row.classList.add("cfg-edit__row--num");
    } else {
      control = this._textControl(leaf);
    }

    row.appendChild(control.el);
    this.controls[leaf.path] = { leaf, reset: control.reset, input: control.input, setInactive: control.setInactive };
    this.states.push({ leaf, row, sectionKey: sectionKey !== undefined ? sectionKey : this._section, pinned });
    return row;
  }

  _buildGateRow(lead, label) {
    const row = document.createElement("div");
    row.className = "cfg-edit__row cfg-edit__row--bool cfg-edit__row--gate";
    row.title = `--${lead.path}`;

    const name = document.createElement("div");
    name.className = "cfg-edit__name";
    name.textContent = label;
    row.appendChild(name);

    const toggle = this._switchControl(lead);
    row.appendChild(toggle.el);
    this.controls[lead.path] = { leaf: lead, reset: toggle.reset, input: toggle.input };
    this.states.push({ leaf: lead, row, sectionKey: this._section });
    return row;
  }

  _buildWeightRow(weight) {
    const row = document.createElement("div");
    row.className = "cfg-edit__row cfg-edit__row--dependent cfg-edit__row--gateweight";
    row.title = `--${weight.path}`;

    const name = document.createElement("div");
    name.className = "cfg-edit__name";
    name.textContent = "weight";
    row.appendChild(name);

    const control = this._numberControl(weight);
    control.input.title = `--${weight.path}`;
    row.appendChild(control.input);

    this.controls[weight.path] = { leaf: weight, reset: control.reset, input: control.input };
    this.states.push({ leaf: weight, row, sectionKey: this._section });
    return row;
  }

  _widgetSpec(leaf) {
    if (!this.config || !this.config.layout) return null;
    return this.config.layout.widgets[leaf.path] || null;
  }

  _effective(leaf) {
    if (this.dirty[leaf.path] !== undefined) return this.dirty[leaf.path];
    const inherited = this._inherited(leaf);
    return inherited !== undefined ? inherited : leaf.value;
  }

  _inherited(leaf) {
    if (!this.pairBase || this.dirty[leaf.path] !== undefined) return undefined;

    const link = this.pairBase.get(leaf.path);
    if (!link) return undefined;

    const inheritLeaf = this.byPath.get(link.inherit);
    if (inheritLeaf && (this.dirty[link.inherit] !== undefined ? this.dirty[link.inherit] : inheritLeaf.value) !== "True") return undefined;

    const base = this.byPath.get(link.base);
    if (!base || leaf.value !== base.value) return undefined;

    return this.dirty[link.base];
  }

  _leafByPath(path) {
    return this.byPath.get(path) || null;
  }

  _choiceControl(leaf, choices, defaultLabel = null) {
    const select = document.createElement("select");
    select.className = "cfg-edit__input picker__select";

    const current   = String(leaf.value);
    const effective = String(this._effective(leaf));
    const options   = [...new Set([current, effective, ...choices].filter((value) => choices.includes(value) || value === current || value === effective))];
    options.forEach((value) => {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value === "default" && defaultLabel ? defaultLabel : value;
      select.appendChild(opt);
    });
    select.value = effective;
    select.classList.toggle("is-dirty", this.dirty[leaf.path] !== undefined);

    select.addEventListener("change", () => {
      select.classList.toggle("is-dirty", select.value !== leaf.value);
      this._setValue(leaf, select.value);
      this._fireDependents(leaf.path, select.value);
    });

    const reset = () => {
      select.value = String(this._effective(leaf));
      select.classList.toggle("is-dirty", this.dirty[leaf.path] !== undefined);
    };
    return { el: select, input: select, reset };
  }

  _onDependency(path, fn) {
    (this.dependents[path] = this.dependents[path] || []).push(fn);
  }

  _fireDependents(path, value) {
    (this.dependents[path] || []).forEach((fn) => fn(value));
  }

  _setValue(leaf, value) {
    const changed = value !== leaf.value || this._inherited(leaf) !== undefined;
    if (changed) this.dirty[leaf.path] = value;
    else delete this.dirty[leaf.path];
    this._refresh();
  }

  _unsetValue(leaf) {
    delete this.dirty[leaf.path];
    this._refresh();
  }

  _textControl(leaf) {
    const input = document.createElement("input");
    input.className = "cfg-edit__input";
    input.value = this._effective(leaf);
    input.spellcheck = false;
    input.classList.toggle("is-dirty", this.dirty[leaf.path] !== undefined);
    input.addEventListener("input", () => {
      input.classList.toggle("is-dirty", input.value !== leaf.value);
      this._setValue(leaf, input.value);
      this._fireDependents(leaf.path, input.value);
    });
    const reset = () => {
      input.value = this._effective(leaf);
      input.classList.toggle("is-dirty", this.dirty[leaf.path] !== undefined);
    };
    return { el: input, input, reset };
  }

  _numberControl(leaf) {
    const input = document.createElement("input");
    input.className = "cfg-edit__input";
    input.type = "number";
    input.step = leaf.type === "int" ? "1" : "any";
    input.value = this._effective(leaf);
    input.spellcheck = false;
    input.classList.toggle("is-dirty", this.dirty[leaf.path] !== undefined);
    input.addEventListener("input", () => {
      const empty   = input.value === "";
      const invalid = !empty && leaf.type === "int" && !/^-?\d+$/.test(input.value);

      input.classList.toggle("is-dirty", !empty && input.value !== leaf.value);
      input.classList.toggle("is-invalid", invalid);
      input.title = invalid ? "not a whole number; this field takes an integer" : "";

      if (empty || invalid) this._unsetValue(leaf);
      else this._setValue(leaf, input.value);
    });
    const reset = () => {
      input.value = this._effective(leaf);
      input.classList.toggle("is-dirty", this.dirty[leaf.path] !== undefined);
      input.classList.remove("is-invalid");
    };
    return { el: input, input, reset };
  }

  _switchControl(leaf) {
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "switch";
    toggle.setAttribute("role", "switch");
    toggle.innerHTML = `<span class="switch__knob"></span>`;

    const paint = () => {
      const on = this._effective(leaf) === "True";
      toggle.classList.toggle("is-on", on);
      toggle.classList.toggle("is-dirty", this.dirty[leaf.path] !== undefined);
      toggle.setAttribute("aria-checked", String(on));
    };
    toggle.addEventListener("click", () => {
      const next = this._effective(leaf) === "True" ? "False" : "True";
      this._setValue(leaf, next);
      paint();
    });
    paint();

    const reset = () => paint();
    return { el: toggle, input: toggle, reset };
  }

  _resetField(path) {
    const control = this.controls[path];
    delete this.dirty[path];
    if (control) control.reset();
    this._refresh();
  }

  _resetAll() {
    this.dirty = {};
    this.legacyMode = false;
    if (this.legacyBtn) this.legacyBtn.classList.remove("is-on");
    Object.values(this.controls).forEach((c) => c.reset());
    this._refresh();
  }

  _navigate(key) {
    this._setActiveSection(key);
  }

  _sectionHidden(section) {
    if (!section.when) return false;
    const leaf = this.byPath.get(section.when.field);
    return leaf ? !section.when.in.includes(this._effective(leaf)) : false;
  }

  _setActiveSection(key) {
    const target = this.sections.find((section) => section.key === key && !this._sectionHidden(section));
    const fallback = this.sections.find((section) => !this._sectionHidden(section));
    this.activeSection = (target || fallback || this.sections[0]).key;

    this.sections.forEach((section) => {
      if (section.navBtn) section.navBtn.classList.toggle("is-active", section.key === this.activeSection);
    });
    this._applyVisibility();
  }

  _refreshGpuRegime() {
    const control = this.controls["gpu"];
    const seeds   = this.byPath.get("seeds");
    if (!control || !control.setInactive || !seeds) return;

    let parsed = [];
    try {
      const value = PythonLiteral.parse(this._effective(seeds));
      if (Array.isArray(value)) parsed = value;
    } catch (e) {
      parsed = [];
    }

    const fanout = parsed.length > 1;
    control.setInactive(fanout, `unused with ${parsed.length} seeds · each seed trains on its own GPU from the gpus pool`);

    this.states.forEach(({ leaf, row }) => {
      if (leaf.path !== "gpu") return;
      row.classList.toggle("cfg-edit__row--inactive", fanout);
      row.title = fanout ? "--gpu · unused: multi-seed runs fan out across the gpus pool" : `--${leaf.path}`;
    });
  }

  _refreshGates() {
    this._refreshGpuRegime();

    this.states.forEach(({ row }) => {
      delete row.dataset.gated;
    });

    this.gates.forEach((gate) => {
      const open = this._effective(gate.leaf) === "True";
      if (!open) gate.states.forEach(({ row }) => (row.dataset.gated = "1"));
    });

    if (this.activeSection) {
      const active = this.sections.find((section) => section.key === this.activeSection);
      if (active && this._sectionHidden(active)) {
        this._setActiveSection(this.sections.find((section) => !this._sectionHidden(section)).key);
        return;
      }
    }

    this._applyVisibility();
  }

  _applyVisibility() {
    const searching = Boolean(this.query);
    const legacy    = this.legacyMode && this.config && this.config.layout && this.config.layout.legacy ? this.config.layout.legacy : null;
    const legacySec = legacy ? new Set(legacy.sections) : null;
    const legacyFld = legacy ? new Set(legacy.expose) : null;
    const stacked   = searching || Boolean(legacy);
    if (this.layoutEl) this.layoutEl.classList.toggle("is-searching", stacked);

    this.states.forEach(({ leaf, row, sectionKey, pinned }) => {
      const matchesQuery  = !searching || leaf.path.toLowerCase().includes(this.query);
      const matchesLegacy = !legacy || pinned || sectionKey === "essentials" || legacySec.has(sectionKey) || legacyFld.has(leaf.path);
      row.hidden = !matchesQuery || !matchesLegacy || row.dataset.gated === "1";
    });

    this.pairs.forEach((pair) => {
      if (pair.container) pair.container.hidden = Boolean(legacy);
      const wantOpen = !legacy && (pair.open || (searching && pair.states.some(({ row }) => !row.hidden)));
      pair.body.hidden = !wantOpen;
      pair.toggle.setAttribute("aria-expanded", String(wantOpen));
      pair.toggle.classList.toggle("is-open", wantOpen);
    });

    this._sweepShells(Boolean(legacy));

    let anyVisible = false;
    this.sections.forEach((section) => {
      const whenHidden = this._sectionHidden(section);
      const hasRows    = this.states.some(({ row, sectionKey }) => sectionKey === section.key && !row.hidden);
      if (section.navBtn) section.navBtn.hidden = whenHidden || (Boolean(legacy) && !hasRows);

      const single = this.config && this.config.layout && this.config.layout.mode === "single";
      const show   = !whenHidden && (stacked ? hasRows : (single || section.key === this.activeSection));
      section.el.hidden = !show;
      anyVisible = anyVisible || (show && (!stacked || hasRows));
    });

    if (this.nomatchEl) this.nomatchEl.hidden = !searching || anyVisible;
  }

  _sweepShells(on) {
    if (!this.layoutEl) return;

    this.layoutEl.querySelectorAll(".band-block, .field-group, .cfg-panel").forEach((el) => {
      if (!on) {
        el.style.display = "";
        return;
      }
      const rows = el.querySelectorAll(".cfg-edit__row");
      if (rows.length) el.style.display = [...rows].some((row) => !row.hidden) ? "" : "none";
    });
  }

  _refreshPairs() {
    this.pairs.forEach((pair) => {
      pair.states.forEach(({ leaf, row }) => {
        row.classList.toggle("is-inherited", this._inherited(leaf) !== undefined);

        const control = this.controls[leaf.path];
        if (control && this.dirty[leaf.path] === undefined && document.activeElement !== control.input) control.reset();
      });

      const differ = pair.states.filter(({ leaf }) => {
        const base = this.byPath.get(pair.base + leaf.path.slice(pair.override.length));
        return base && this._effective(leaf) !== this._effective(base);
      }).length;
      pair.badge.hidden = differ === 0;
      pair.badge.textContent = differ ? `${differ} differ` : "";
    });
  }

  _refreshBadges() {
    const counts = new Map();
    this.states.forEach(({ leaf, sectionKey }) => {
      if (this.dirty[leaf.path] !== undefined) counts.set(sectionKey, (counts.get(sectionKey) || 0) + 1);
    });

    this.sections.forEach((section) => {
      if (!section.badge) return;
      const n = counts.get(section.key) || 0;
      section.badge.hidden = n === 0;
      section.badge.textContent = n ? `${n}` : "";
    });

    this._refreshPairs();
  }
}

window.PythonLiteral     = PythonLiteral;
window.GpuCardSelect     = GpuCardSelect;
window.GpuPicker         = GpuPicker;
window.NumberField       = NumberField;
window.MultiValueField   = MultiValueField;
window.LaunchWidgetDom   = LaunchWidgetDom;
window.ModelTogglePanel  = ModelTogglePanel;
window.ModelCardPanel    = ModelCardPanel;
window.ExperimentBuilder = ExperimentBuilder;
window.AblationBuilder   = AblationBuilder;
window.ConfigForm        = ConfigForm;
