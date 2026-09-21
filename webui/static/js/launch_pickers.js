"use strict";

class DatasetPicker {
  static CUSTOM = "__custom__";

  static MULTI_MODES = {
    datasets:     { endpoint: "/api/fs/datasets", key: "datasets", noun: "datasets", hint: "nothing selected = every dataset is processed", badge: (d) => (d.has_params ? null : { text: "no params", tone: "warn" }) },
    runs:         { endpoint: "/api/fs/runs",     key: "runs",     noun: "runs",     hint: "nothing selected = every run is processed",     badge: (d) => (d.n_seeds ? (d.own_inference ? { text: "seed report", tone: "ok" } : { text: "no seed report", tone: "warn" }) : !d.has_checkpoint ? { text: "no checkpoint", tone: "warn" } : d.has_inference ? { text: "inferred", tone: "ok" } : null) },
    runs_compare: { endpoint: "/api/fs/runs",     key: "runs",     noun: "runs",     hint: "select 2 or more runs to compare",            params: "units=1", badge: (d) => (d.n_seeds ? { text: `${d.n_seeds} seed${d.n_seeds > 1 ? "s" : ""}`, tone: d.has_inference ? "ok" : "warn" } : d.has_inference ? { text: "inferred", tone: "ok" } : { text: "no inference", tone: "warn" }) },
    run_groups:   { endpoint: "/api/fs/run_groups", key: "groups", noun: "groups",   hint: "nothing selected = runs_dir itself is treated as one seed group", badge: (d) => ({ text: `${d.n_runs} runs`, tone: d.n_runs >= 2 ? "ok" : "warn" }) },
    experiments:  { endpoint: "/api/fs/experiments", key: "experiments", noun: "experiments", hint: "nothing selected = every experiment found is mailed; missing seed aggregates and comparisons are written first", badge: (d) => ({ text: `${d.kind} · ${d.n_trials} trial${d.n_trials === 1 ? "" : "s"} · ${d.n_comparisons ? `${d.n_comparisons} comparison${d.n_comparisons === 1 ? "" : "s"}` : "no comparison yet"}`, tone: d.n_comparisons >= 1 ? "ok" : "warn" }) },
    param_trials: { endpoint: "/api/fs/param_trials", key: "trials", noun: "trials", hint: "select 2 or more trials to compare across datasets", badge: (d) => (d.dataset ? { text: d.dataset, tone: "ok" } : null) },
  };

  constructor(view, leaf, spec) {
    this.view = view;
    this.leaf = leaf;
    this.spec = spec;
    this.el = null;
    this.select = null;
    this.custom = null;
    this.note = null;
    this.board = null;
    this.count = null;
    this.hint = null;
    this.filter = null;
    this.body = null;
    this.items = [];
    this.options = [];
    this.selected = new Set();
    this.stale = [];
    this.reloadTimer = null;
    this.baseLabel = "";
    this.multiLoadSeq = 0;
  }

  _baseList() {
    if (!this.spec.multiBase) {
      const base = this._base();
      return base ? [base] : [];
    }
    const source = this.spec.baseFrom ? this.view._leafByPath(this.spec.baseFrom) : null;
    const raw = source ? this.view._effective(source) : "";
    try {
      const parsed = window.PythonLiteral.parse(raw);
      if (Array.isArray(parsed)) return parsed.map(String).filter(Boolean);
    } catch (e) {}
    return raw ? [String(raw)] : [];
  }

  build() {
    this.el = document.createElement("div");
    this.el.className = "picker";

    if (this.spec.multi) this._buildMulti();
    else this._buildSingle();

    if (this.spec.datasetFrom) {
      this.view._onDependency(this.spec.datasetFrom, () => this._scheduleSingleReload());
    }

    return { el: this.el, input: this.select, reset: () => this._reset() };
  }

  _buildSingle() {
    this.select = document.createElement("select");
    this.select.className = "cfg-edit__input picker__select";

    this.custom = document.createElement("input");
    this.custom.className = "cfg-edit__input picker__custom";
    this.custom.spellcheck = false;
    this.custom.hidden = true;
    this.custom.placeholder = "absolute path";

    this.note = document.createElement("span");
    this.note.className = "picker__note";

    this.select.addEventListener("change", () => this._onSelect());
    this.custom.addEventListener("input", () => {
      this.custom.classList.toggle("is-dirty", this.custom.value !== this.leaf.value);
      this.view._setValue(this.leaf, this.custom.value);
      this.view._fireDependents(this.leaf.path, this.custom.value);
      this._paintAnalysisWarning();
    });

    this.el.appendChild(this.select);
    this.el.appendChild(this.custom);
    this.el.appendChild(this.note);

    if (this.spec.baseFrom) {
      this.view._onDependency(this.spec.baseFrom, () => this._scheduleSingleReload());
    }

    this._renderSingleOptions([]);
    this._loadSingle();
  }

  _scheduleSingleReload() {
    clearTimeout(this.singleReloadTimer);
    this.singleReloadTimer = setTimeout(() => this._loadSingle(), 350);
  }

  _base() {
    if (this.spec.baseFromParent) {
      const current = this.view._effective(this.leaf);
      const trimmed = String(current).replace(/\/+$/, "");
      const cut = trimmed.lastIndexOf("/");
      return cut > 0 ? trimmed.slice(0, cut) : trimmed;
    }
    if (this.spec.baseFrom) {
      const source = this.view._leafByPath(this.spec.baseFrom);
      return source ? this.view._effective(source) : "";
    }
    return "";
  }

  async _loadSingle() {
    const seq = (this.singleLoadSeq = (this.singleLoadSeq || 0) + 1);

    if (this.spec.mode === "params") {
      const dataset = this.view._effective(this.view._leafByPath(this.spec.datasetFrom));
      await this._fetchParams(dataset, seq);
      return;
    }
    if (this.spec.mode === "runs") {
      await this._fetchRuns(this._base(), seq);
      return;
    }
    if (this.spec.mode === "resume_runs") {
      await this._fetchResumeRuns(this._base(), seq);
      return;
    }
    await this._fetchDatasets(this._base(), seq);
  }

  _singleStale(seq) {
    return seq !== this.singleLoadSeq;
  }

  async _fetchRuns(base, seq) {
    if (!base) {
      this.note.textContent = "set a runs directory first";
      this._renderSingleOptions([]);
      return;
    }
    this.note.textContent = "listing...";
    const res = await Api.get(`/api/fs/runs?base=${encodeURIComponent(base)}`);
    if (this._singleStale(seq)) return;
    if (!res.ok) {
      this.note.textContent = res.error || "could not list runs";
      this._renderSingleOptions([]);
      return;
    }
    let items = res.runs || [];
    if (this.spec.checkpointOnly) items = items.filter((r) => r.has_checkpoint);
    this.note.textContent = items.length ? `${items.length} run${items.length > 1 ? "s" : ""} in ${res.base}` : `no completed runs in ${res.base}`;
    this._renderSingleOptions(items.map((r) => ({ value: r.name, label: r.name + (r.has_checkpoint ? "" : "  (no checkpoint)") })));
  }

  async _fetchResumeRuns(base, seq) {
    if (!base) {
      this.note.textContent = "set a log directory first, or type the run path";
      this._renderSingleOptions([]);
      return;
    }
    this.note.textContent = "listing...";
    const res = await Api.get(`/api/run-config/runs?base=${encodeURIComponent(base)}`);
    if (this._singleStale(seq)) return;
    if (!res.ok) {
      this.note.textContent = res.error || "could not list runs";
      this._renderSingleOptions([]);
      return;
    }
    const items = res.runs || [];
    this.note.textContent = items.length
      ? `${items.length} run${items.length > 1 ? "s" : ""} under ${res.base} recorded a configuration to finish from`
      : `no run under ${res.base} recorded a configuration; type the path if it lives elsewhere`;
    this._renderSingleOptions(items.map((r) => ({ value: r.path, label: `${r.name}  (${r.source})` })));
  }

  async _fetchDatasets(base, seq) {
    this.note.textContent = "listing...";
    const res = await Api.get(`/api/fs/datasets?base=${encodeURIComponent(base || "")}`);
    if (this._singleStale(seq)) return;
    if (!res.ok) {
      this.note.textContent = res.error || "could not list datasets";
      this._renderSingleOptions([]);
      return;
    }
    const items = (res.datasets || []).filter((d) => (this.spec.validOnly ? d.is_dataset : true));
    this.note.textContent = items.length ? `${items.length} in ${res.base}` : `no datasets in ${res.base}`;
    this._renderSingleOptions(items.map((d) => ({ value: d.path, label: d.name + (d.has_params ? "" : "  (no params)") })));
  }

  async _fetchParams(dataset, seq) {
    if (!dataset) {
      this._clearAnalysisState();
      this.note.textContent = "select a dataset first";
      this._renderSingleOptions([]);
      return;
    }
    this.note.textContent = "listing...";
    const res = await Api.get(`/api/fs/params?dataset=${encodeURIComponent(dataset)}`);
    if (this._singleStale(seq)) return;
    if (!res.ok) {
      this._clearAnalysisState();
      this.note.textContent = res.error || "could not list params";
      this._renderSingleOptions([]);
      return;
    }
    const items = res.files || [];
    this.unanalysed = new Set(items.filter((f) => f.analysed === false).map((f) => f.path));
    this.noteBase = items.length ? `${items.length} under ${res.params_root}` : "no extracted params yet";
    this.note.textContent = this.noteBase;
    this._renderSingleOptions(items.map((f) => ({ value: f.path, label: f.name + (this.spec.warnUnanalysed && f.analysed === false ? "  (not analysed)" : "") })));
    this._paintAnalysisWarning();
  }

  _clearAnalysisState() {
    this.unanalysed = null;
    this.noteBase = "";
    this.note.classList.remove("is-warn");
  }

  _paintAnalysisWarning() {
    if (!this.spec.warnUnanalysed || !this.unanalysed) return;
    const flagged = this.unanalysed.has(String(this.view._effective(this.leaf) || ""));
    this.note.classList.toggle("is-warn", flagged);
    if (flagged) this.note.textContent = "no resolution curves in this run: run Param inference (analyze_param_extraction) on it first, or the launch will abort";
    else this.note.textContent = this.noteBase || "";
  }

  _renderSingleOptions(items) {
    const current = this.view._effective(this.leaf);
    const known = new Set(items.map((it) => it.value));

    this.options = items.slice();
    this.select.innerHTML = "";

    if (current && !known.has(current)) {
      this.select.appendChild(this._option(current, `${this._tail(current)} (current)`));
    }
    items.forEach((it) => this.select.appendChild(this._option(it.value, it.label)));
    this.select.appendChild(this._option(DatasetPicker.CUSTOM, "Custom path..."));

    const isCustom = !this.custom.hidden && this.custom.value && !known.has(this.custom.value);
    this.select.value = isCustom ? DatasetPicker.CUSTOM : current;
    if (this.select.value !== DatasetPicker.CUSTOM) this.custom.hidden = true;

    this.select.classList.toggle("is-dirty", this.view.dirty[this.leaf.path] !== undefined);
  }

  _onSelect() {
    if (this.select.value === DatasetPicker.CUSTOM) {
      this.custom.hidden = false;
      if (!this.custom.value) this.custom.value = this.view._effective(this.leaf);
      this.custom.focus();
      this.view._setValue(this.leaf, this.custom.value);
      this.view._fireDependents(this.leaf.path, this.custom.value);
      this._paintAnalysisWarning();
      return;
    }
    this.custom.hidden = true;
    this.select.classList.toggle("is-dirty", this.select.value !== this.leaf.value);
    this.view._setValue(this.leaf, this.select.value);
    this.view._fireDependents(this.leaf.path, this.select.value);
    this._paintAnalysisWarning();
  }

  _buildMulti() {
    const mode = DatasetPicker.MULTI_MODES[this.spec.mode];

    this.board = document.createElement("div");
    this.board.className = "picker__board";

    const head = document.createElement("div");
    head.className = "picker__board-head";

    this.count = document.createElement("span");
    this.count.className = "picker__count";

    this.hint = document.createElement("span");
    this.hint.className = "picker__hint";

    this.filter = document.createElement("input");
    this.filter.className = "picker__filter";
    this.filter.spellcheck = false;
    this.filter.placeholder = `filter ${mode.noun}...`;
    this.filter.addEventListener("input", () => this._renderItems());

    head.appendChild(this.count);
    head.appendChild(this.hint);
    head.appendChild(this.filter);
    head.appendChild(window.LaunchWidgetDom.mini("All", () => this._setAll(true)));
    head.appendChild(window.LaunchWidgetDom.mini("None", () => this._setAll(false)));
    if (this.spec.pendingButton) {
      const pending = window.LaunchWidgetDom.mini("Pending", () => this._setPending());
      pending.title = "select runs with a checkpoint and no inference yet";
      head.appendChild(pending);
    }
    if (this.spec.seedButton) {
      const seeds = window.LaunchWidgetDom.mini("Seeds", () => this._setSeeds());
      seeds.title = "select only multi-seed run groups";
      head.appendChild(seeds);
    }
    head.appendChild(window.LaunchWidgetDom.mini("Reload", () => this._loadMulti()));

    this.dropBtn = window.LaunchWidgetDom.mini("Drop missing", () => this._dropStale());
    this.dropBtn.title = "remove the selected names that the current base does not list";
    this.dropBtn.hidden = true;
    head.appendChild(this.dropBtn);

    this.body = document.createElement("div");
    this.body.className = "picker__items";

    this.board.appendChild(head);
    this.board.appendChild(this.body);
    this.el.appendChild(this.board);

    if (this.spec.baseFrom) {
      this.view._onDependency(this.spec.baseFrom, () => this._scheduleReload());
    }
    if (this.spec.markerFrom) {
      this.view._onDependency(this.spec.markerFrom, () => this._scheduleReload());
    }
    if (this.spec.splitFrom) {
      this.view._onDependency(this.spec.splitFrom, () => this._scheduleReload());
    }

    this._loadMulti();
  }

  _scheduleReload() {
    clearTimeout(this.reloadTimer);
    this.reloadTimer = setTimeout(() => this._loadMulti(), 350);
  }

  _selectedNames() {
    try {
      const parsed = window.PythonLiteral.parse(this.view._effective(this.leaf));
      return Array.isArray(parsed) ? parsed.map(String) : [];
    } catch (e) {
      return [];
    }
  }

  async _loadMulti() {
    const mode  = DatasetPicker.MULTI_MODES[this.spec.mode];
    const bases = this._baseList();
    const seq   = ++this.multiLoadSeq;

    this.count.textContent = "listing...";
    this.hint.textContent = "";

    const parts = (bases.length ? bases : [""]).map((b) => `base=${encodeURIComponent(b)}`);
    if (mode.params) parts.push(mode.params);
    else if (this.spec.units) parts.push("units=1");
    if (this.spec.markerFrom) {
      const markerLeaf = this.view._leafByPath(this.spec.markerFrom);
      parts.push(`marker=${encodeURIComponent(markerLeaf ? this.view._effective(markerLeaf) : "")}`);
    }
    if (this.spec.splitFrom) {
      const splitLeaf = this.view._leafByPath(this.spec.splitFrom);
      parts.push(`split=${encodeURIComponent(splitLeaf ? this.view._effective(splitLeaf) : "")}`);
    }

    const res = await Api.get(`${mode.endpoint}?${parts.join("&")}`);
    if (this._multiStale(seq)) return;

    if (!res.ok) {
      this.items = [];
      this.count.textContent = `no ${mode.noun}`;
      this.body.innerHTML = `<p class="picker__empty">${res.error || `could not list ${mode.noun}`}</p>`;
      return;
    }

    this.baseLabel = res.base || bases.join(", ");

    this.items = (res[mode.key] || []).filter((d) => (this.spec.validOnly ? d.is_dataset : true));

    this._syncSelection();
    this._renderItems();
    this._paintSummary();
  }

  _multiStale(seq) {
    return seq !== this.multiLoadSeq;
  }

  _syncSelection() {
    const listed   = new Set(this.items.map((d) => d.name));
    const selected = this._selectedNames();

    this.stale    = selected.filter((n) => !listed.has(n));
    this.selected = new Set(selected.filter((n) => listed.has(n)));
  }

  _renderItems() {
    const mode = DatasetPicker.MULTI_MODES[this.spec.mode];
    const query = this.filter.value.trim().toLowerCase();

    this.body.innerHTML = "";

    if (!this.items.length) {
      this.body.appendChild(Object.assign(document.createElement("p"), { className: "picker__empty", textContent: `no ${mode.noun} in ${this.baseLabel || this._baseList().join(", ")}` }));
      return;
    }

    let shown = 0;
    this.items.forEach((d) => {
      if (query && !d.name.toLowerCase().includes(query)) return;
      shown += 1;

      const row = document.createElement("label");
      row.className = "picker__item";
      row.classList.toggle("is-selected", this.selected.has(d.name));

      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = this.selected.has(d.name);
      box.addEventListener("change", () => {
        if (box.checked) this.selected.add(d.name);
        else this.selected.delete(d.name);
        row.classList.toggle("is-selected", box.checked);
        this._commitMulti();
      });

      const name = document.createElement("span");
      name.className = "picker__item-name";
      name.textContent = d.name;
      name.title = d.path || d.name;

      row.appendChild(box);
      row.appendChild(name);

      const badge = mode.badge(d);
      if (badge) {
        const pill = document.createElement("span");
        pill.className = `picker__badge picker__badge--${badge.tone}`;
        pill.textContent = badge.text;
        row.appendChild(pill);
      }

      this.body.appendChild(row);
    });

    if (!shown) {
      this.body.appendChild(Object.assign(document.createElement("p"), { className: "picker__empty", textContent: `no ${mode.noun} match "${query}"` }));
    }
  }

  _visibleNames() {
    const query = this.filter.value.trim().toLowerCase();
    return this.items.map((d) => d.name).filter((n) => (!query || n.toLowerCase().includes(query)));
  }

  _setAll(on) {
    if (!on) this.stale = [];
    this._visibleNames().forEach((n) => (on ? this.selected.add(n) : this.selected.delete(n)));
    this._renderItems();
    this._commitMulti();
  }

  _setPending() {
    const visible = new Set(this._visibleNames());
    this.items.forEach((d) => {
      if (!visible.has(d.name)) return;
      if (d.has_checkpoint && !d.has_inference) this.selected.add(d.name);
      else this.selected.delete(d.name);
    });
    this._renderItems();
    this._commitMulti();
  }

  _setSeeds() {
    const visible = new Set(this._visibleNames());
    this.items.forEach((d) => {
      if (!visible.has(d.name)) return;
      if (d.n_seeds) this.selected.add(d.name);
      else this.selected.delete(d.name);
    });
    this._renderItems();
    this._commitMulti();
  }

  _dropStale() {
    if (!this.stale.length) return;

    this.stale = [];
    this._commitMulti();
  }

  _commitMulti() {
    const names = this.items.map((d) => d.name).filter((n) => this.selected.has(n)).concat(this.stale);

    this.view._setValue(this.leaf, window.PythonLiteral.render(names));
    this._paintSummary();
  }

  _paintSummary() {
    const mode = DatasetPicker.MULTI_MODES[this.spec.mode];
    const picked = this.items.filter((d) => this.selected.has(d.name)).length;

    this.count.textContent = picked || this.stale.length ? `${picked + this.stale.length} of ${this.items.length + this.stale.length} selected` : `all ${this.items.length} ${mode.noun}`;
    this.hint.textContent = this.stale.length ? `kept ${this.stale.join(", ")}: not under ${this.baseLabel}` : picked ? "" : (this.spec.hint || mode.hint);
    if (this.dropBtn) this.dropBtn.hidden = !this.stale.length;
    this.board.classList.toggle("is-dirty", this.view.dirty[this.leaf.path] !== undefined);
  }

  _reset() {
    if (this.spec.multi) {
      this._syncSelection();
      this._renderItems();
      this._paintSummary();
      return;
    }
    this.custom.hidden = true;
    this.custom.value = "";
    this.custom.classList.remove("is-dirty");
    this._renderSingleOptions(this.options);
    if (this.spec.mode === "params") this._loadSingle();
  }

  _option(value, label) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = label;
    return opt;
  }

  _tail(path) {
    const trimmed = String(path).replace(/\/+$/, "");
    const cut = trimmed.lastIndexOf("/");
    return cut >= 0 ? trimmed.slice(cut + 1) : trimmed;
  }
}

class EntryFormControl {

  constructor(view) {
    this.view = view;
  }

  _entryKey() {
    if (this.view.key) return this.view.key;
    const editorKey = this.view.editorKey || "";
    const cut = editorKey.indexOf(":");
    return cut >= 0 ? editorKey.slice(cut + 1) : "";
  }

  _entryTitle() {
    if (this.view.detail) return this.view.detail.title;

    const key     = this._entryKey();
    const catalog = this.view.scripts || [];
    const script  = catalog.find((entry) => entry.key === key);
    return script ? script.title : key;
  }
}

class RunConfigCopy extends EntryFormControl {
  static DATASET_ENTRIES = ["generate_dataset", "extract_params", "inject_external_params"];

  constructor(view) {
    super(view);
    this.base = null;
    this.baseKey = null;
    this.run = "";
    this.loadSeq = 0;
    this.timer = null;
  }

  _defaultBase() {
    return RunConfigCopy.DATASET_ENTRIES.includes(this._entryKey()) ? ResultsSources.datasets() : ResultsSources.runs();
  }

  build() {
    const key = this._entryKey();
    if (key !== this.baseKey) {
      this.baseKey = key;
      this.base = null;
      this.run = "";
    }

    const dir = document.createElement("input");
    dir.className = "cfg-edit__input rail-copy__dir";
    dir.spellcheck = false;
    dir.placeholder = RunConfigCopy.DATASET_ENTRIES.includes(key) ? "dataset directory" : "runs directory";
    dir.value = this.base !== null ? this.base : this._defaultBase();

    const select = document.createElement("select");
    select.className = "run-select";

    const note = document.createElement("span");
    note.className = "picker__note";

    const apply = document.createElement("button");
    apply.className = "btn btn--ghost rail-copy";
    apply.textContent = "Copy configs";
    apply.title = "Replace every field with the selected run's resolved config, so you only change what should differ; the run's own name and resume state are kept as they are";
    apply.disabled = true;

    dir.addEventListener("input", () => {
      this.base = dir.value;
      clearTimeout(this.timer);
      this.timer = setTimeout(() => this._load(select, note, apply), 350);
    });
    select.addEventListener("change", () => {
      this.run = select.value;
      apply.disabled = !select.value;
    });
    apply.addEventListener("click", () => this._apply(select.value, apply));

    this._load(select, note, apply);
    return { dir, select, note, apply };
  }

  async _load(select, note, apply) {
    const base = this.base !== null ? this.base : this._defaultBase();
    const seq  = ++this.loadSeq;

    note.textContent = "listing...";
    const res = await Api.get(`/api/run-config/runs?base=${encodeURIComponent(base)}`);
    if (seq !== this.loadSeq) return;

    select.innerHTML = "";
    if (!res.ok) {
      note.textContent = res.error || "could not list runs";
      apply.disabled = true;
      return;
    }

    const runs = res.runs || [];
    note.textContent = runs.length ? `${runs.length} run${runs.length > 1 ? "s" : ""} with a stored config` : "no runs with a stored config here";

    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = runs.length ? "select a run..." : "no runs found";
    select.appendChild(blank);

    runs.forEach((run) => {
      const opt = document.createElement("option");
      opt.value = run.path;
      opt.textContent = run.name;
      if (run.path === this.run) opt.selected = true;
      select.appendChild(opt);
    });

    apply.disabled = !select.value;
  }

  async _apply(path, apply) {
    if (!path) return;

    apply.disabled = true;
    try {
      const res = await Api.get(`/api/run-config?path=${encodeURIComponent(path)}`);
      if (!res.ok) {
        Toast.show(res.error || "Could not read the run's config", "error");
        return;
      }

      const outcome  = this.view.applyRunConfig(res.config);
      const identity = res.identity;
      const parts    = [`Copied ${res.run}: ${outcome.applied} override${outcome.applied === 1 ? "" : "s"} set`];
      if (identity.length) parts.push(`${identity.length} identity field${identity.length === 1 ? "" : "s"} kept: ${identity.join(", ")}`);
      parts.push(...ConfigForm.outcomeParts(outcome));
      Toast.show(parts.join(" · "), ConfigForm.outcomeKind(outcome));
    } finally {
      apply.disabled = false;
    }
  }
}


class ConfigTemplates extends EntryFormControl {

  constructor(view) {
    super(view);
    this.entries   = [];
    this.selected  = "";
    this.loadSeq   = 0;
    this.selectEl  = null;
    this.noteEl    = null;
    this.importBtn = null;
    this.saveBtn   = null;
    this.deleteBtn = null;
  }

  build() {
    const select = document.createElement("select");
    select.className = "run-select";
    select.addEventListener("change", () => {
      this.selected = select.value;
      this._paintActions();
    });

    const note = document.createElement("span");
    note.className = "picker__note";

    const actions = document.createElement("div");
    actions.className = "rail-template__actions";

    const load = document.createElement("button");
    load.className = "btn btn--ghost rail-template";
    load.textContent = "Import";
    load.title = "Set the form to this template's overrides; every field the template does not carry goes back to its default";
    load.addEventListener("click", () => this._import());

    const store = document.createElement("button");
    store.className = "btn btn--ghost rail-template";
    store.textContent = "Save as template";
    store.title = "Store the overrides currently set on this form under a name, to import on any launch page later";
    store.addEventListener("click", () => this._save());

    const remove = document.createElement("button");
    remove.className = "btn btn--mini rail-template__delete";
    remove.textContent = "×";
    remove.title = "Delete the selected template";
    remove.addEventListener("click", () => this._delete());

    actions.append(load, store, remove);

    this.selectEl  = select;
    this.noteEl    = note;
    this.importBtn = load;
    this.saveBtn   = store;
    this.deleteBtn = remove;

    this._load();
    return { select, note, actions };
  }

  async _load() {
    const seq = ++this.loadSeq;

    this.noteEl.textContent = "loading...";
    const res = await Api.get("/api/config-templates");
    if (seq !== this.loadSeq) return;

    if (res.error) {
      this.entries = [];
      this.noteEl.textContent = res.error;
      this._paintOptions();
      return;
    }

    this.entries = res.templates || [];
    this._paintOptions();
  }

  _paintOptions() {
    const key  = this._entryKey();
    const mine = this.entries.filter((entry) => entry.script === key);
    const rest = this.entries.filter((entry) => entry.script !== key);

    this.selectEl.innerHTML = "";

    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = this.entries.length ? "select a template..." : "no templates saved yet";
    this.selectEl.appendChild(blank);

    if (mine.length) this.selectEl.appendChild(this._group("this entry point", mine, false));
    if (rest.length) this.selectEl.appendChild(this._group("other entry points", rest, true));

    this.selectEl.value = this.entries.some((entry) => entry.template_id === this.selected) ? this.selected : "";
    this.selected = this.selectEl.value;

    this.noteEl.textContent = this.entries.length ? `${mine.length} for this entry point, ${rest.length} other` : "save the current overrides to start one";
    this._paintActions();
  }

  _group(label, entries, tagged) {
    const group = document.createElement("optgroup");
    group.label = label;

    entries.forEach((entry) => {
      const count  = Object.keys(entry.overrides).length;
      const option = document.createElement("option");
      option.value = entry.template_id;
      option.textContent = `${entry.name}  ·  ${count} override${count === 1 ? "" : "s"}${tagged ? `  ·  ${entry.script}` : ""}`;
      group.appendChild(option);
    });

    return group;
  }

  _paintActions() {
    const entry = this._selected();

    this.importBtn.disabled = !entry;
    this.deleteBtn.disabled = !entry;
    this.importBtn.classList.toggle("is-armed", !!entry && entry.script === this._entryKey());
  }

  _selected() {
    return this.entries.find((entry) => entry.template_id === this.selected) || null;
  }

  _import() {
    const entry = this._selected();
    if (!entry) return;

    if (!this.view.config) {
      Toast.show("The configuration is still loading", "error");
      return;
    }

    const outcome = this.view.applyRunConfig(entry.overrides);
    const parts   = [`Imported "${entry.name}": ${outcome.applied} override${outcome.applied === 1 ? "" : "s"} set`];
    if (entry.script !== this._entryKey()) parts.push(`saved on ${entry.script}`);
    parts.push(...ConfigForm.outcomeParts(outcome));
    Toast.show(parts.join(" · "), ConfigForm.outcomeKind(outcome));
  }

  async _save() {
    const key = this._entryKey();
    if (!key || !this.view.config) {
      Toast.show("The configuration is still loading", "error");
      return;
    }

    const raw = window.prompt("Name this config template:", "");
    if (raw === null) return;

    const name = raw.trim();
    if (!name) {
      Toast.show("A template needs a name", "error");
      return;
    }

    const clash = this.entries.find((entry) => entry.script === key && entry.name === name);
    if (clash && !window.confirm(`A template named "${name}" already exists for ${key}. Replace it?`)) return;

    const overrides = { ...this.view.dirty };
    const count     = Object.keys(overrides).length;

    this.saveBtn.disabled = true;
    try {
      const res = await Api.post("/api/config-templates", { script_key: key, title: this._entryTitle(), name, overrides, replace: !!clash });
      if (!res.ok) {
        Toast.show(res.error || "Save failed", "error");
        return;
      }

      this.selected = res.entry.template_id;
      await this._load();
      Toast.show(`${res.replaced ? "Replaced" : "Saved"} template "${name}" with ${count} override${count === 1 ? "" : "s"}`, "ok");
    } finally {
      this.saveBtn.disabled = false;
    }
  }

  async _delete() {
    const entry = this._selected();
    if (!entry) return;
    if (!window.confirm(`Delete template "${entry.name}"?`)) return;

    const res = await Api.post(`/api/config-templates/${entry.template_id}/delete`, {});
    if (!res.ok) {
      Toast.show(res.error || "Delete failed", "error");
      return;
    }

    this.selected = "";
    await this._load();
    Toast.show(`Deleted template "${entry.name}"`, "ok");
  }
}

window.DatasetPicker   = DatasetPicker;
window.RunConfigCopy   = RunConfigCopy;
window.ConfigTemplates = ConfigTemplates;
