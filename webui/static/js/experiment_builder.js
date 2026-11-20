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


class ExperimentBuilder {

  static MODES = [
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
    { key: "data",     label: "data" },
    { key: "arch",     label: "architecture" },
    { key: "training", label: "training" },
  ];

  static STABILITY_WARMUP_MODES = ["linear", "cosine", "exponential", "polynomial"];
  static STABILITY_CLIP_MODES   = ["fixed", "adaptive_percentile", "adaptive_mean_std"];

  constructor(view, byPath) {
    this.view         = view;
    this.trialsLeaf   = byPath.get("trials_enabled");
    this.modeLeaf     = byPath.get("trials_mode");
    this.modelLeaf    = byPath.get("backbone_name");
    this.gpusLeaf     = byPath.get("gpus");
    this.seedsLeaf    = byPath.get("seeds");

    this.contextTrialsLeaf = byPath.get("context_trials");

    this.patch           = new Map();
    this.reach           = new Map();
    this.augTrials       = new Map();
    this.stabilityTrials = new Map();
    this.emaTrials       = new Map();
    byPath.forEach((leaf) => {
      if (leaf.section === "patch_trials")         this.patch.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "reach_trials")         this.reach.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "augmentation_trials")  this.augTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "stability_trials")     this.stabilityTrials.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "ema_trials")           this.emaTrials.set(leaf.path.split(".").pop(), leaf);
    });

    this.claimed = ["trials_enabled"];
    if (this.modeLeaf) this.claimed.push("trials_mode");
    this.patch.forEach((leaf) => this.claimed.push(leaf.path));
    this.reach.forEach((leaf) => this.claimed.push(leaf.path));
    this.augTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.stabilityTrials.forEach((leaf) => this.claimed.push(leaf.path));
    this.emaTrials.forEach((leaf) => this.claimed.push(leaf.path));
    if (this.contextTrialsLeaf)    this.claimed.push(this.contextTrialsLeaf.path);

    this.summaryEl      = null;
    this.namesEl        = null;
    this.hintEl         = null;
    this.root           = null;
    this.patchEl         = null;
    this.patchFlagPaints = [];
    this.warmupCustomEl      = null;
    this.warmupCustomHeadEl  = null;
    this.contextEl      = null;
    this.contextCellsEl = null;
    this.augEl           = null;
    this.augStdsEl       = null;
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

    if (this.modeLeaf) {
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
    const first = modes.length ? modes[0].key : "augmentation";
    if (!this.modeLeaf) return first;
    const value = this.view._effective(this.modeLeaf);
    return modes.some((mode) => mode.key === value) ? value : first;
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

  _paintMode() {
    const mode = this._mode();

    this.modeButtons.forEach((btn, key) => btn.classList.toggle("is-on", key === mode));
    if (this.modeEl) this.modeEl.classList.toggle("is-dirty", this.view.dirty[this.modeLeaf.path] !== undefined);
    if (this.hintEl) this.hintEl.textContent = this._modeEntry(mode).hint;

    ExperimentBuilder.MODES.forEach((entry) => {
      const el = this[entry.el];
      if (el) el.hidden = mode !== entry.key;
    });
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

  _reload() {
    if (this._paintSwitch) this._paintSwitch();
    this._paintMode();
    ExperimentBuilder.MODES.forEach((mode) => { if (mode.paint) mode.paint(this); });
    this._paintAll();
  }

  _paintAll() {
    this._paintSummary();
    this._paintNames();
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
