"use strict";

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
  constructor(view, leaf, headLeaf = null, headGate = null) {
    this.view      = view;
    this.leaf      = leaf;
    this.headLeaf  = headLeaf;
    this.headGate  = headGate;
    this.families  = view.modelFamilies || [];
    this.heads     = view.modelHeads || [];
    this.cards     = new Map();
    this.headChips = new Map();
    this.currentEl = null;
    this.headHint  = null;
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

    if (this.headLeaf && this.heads.length) body.appendChild(this._headPicker());
    root.appendChild(body);

    this.view.controls[this.leaf.path] = { leaf: this.leaf, reset: () => this._paint() };
    if (this.headLeaf) this.view.controls[this.headLeaf.path] = { leaf: this.headLeaf, reset: () => this._paint() };
    if (this.headLeaf && this.headGate) this.view.repainters.push(() => this._paint());
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

    if (this.headGate) {
      const hint       = document.createElement("p");
      hint.className   = "model-family__hint";
      hint.textContent = this.headGate.hint;
      hint.hidden      = true;
      this.headHint    = hint;
      block.appendChild(hint);
    }

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
    this.view._fireDependents(this.leaf.path, key);
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
      const locked      = Boolean(this.headGate) && this.view._whenHolds(this.headGate.when);
      let headLabel     = currentHead;

      this.headChips.forEach((chip, key) => {
        const on      = key === currentHead;
        const allowed = !locked || this.headGate.only.includes(key);
        chip.classList.toggle("is-on", on);
        chip.classList.toggle("is-locked", !allowed);
        chip.disabled = !allowed;
        chip.setAttribute("aria-pressed", String(on));
        if (on) headLabel = chip.querySelector(".model-chip__name").textContent;
      });

      if (this.headHint) this.headHint.hidden = !locked;

      const invalid = locked && !this.headGate.only.includes(currentHead);
      this.currentEl.classList.toggle("model-panel__count--warn", invalid);
      label = `${label} · ${headLabel} head${invalid ? " (invalid with a profile AE)" : ""}`;
    }

    this.currentEl.textContent = label;
  }
}


class ArchOverridesPanel {
  constructor(view, leaf, modelLeaf, panel) {
    this.view      = view;
    this.leaf      = leaf;
    this.modelLeaf = modelLeaf;
    this.modelBy   = (panel && panel.modelBy) || null;
    this.title     = (panel && panel.title) || "Architecture";
    this.exclude   = (panel && panel.exclude) || [];
    this.bodyEl    = null;
    this.countEl   = null;
    this.hintEl    = null;
  }

  build() {
    const root     = document.createElement("section");
    root.className = "model-panel arch-panel";

    const head     = document.createElement("header");
    head.className = "special-head";
    head.innerHTML = `<h3 class="special-head__name"></h3><span class="special-head__hint">fields of the selected model; untouched fields train at their defaults</span>`;
    head.querySelector(".special-head__name").textContent = this.title;
    this.hintEl = head.querySelector(".special-head__hint");

    const count     = document.createElement("span");
    count.className = "model-panel__count";
    this.countEl    = count;
    head.appendChild(count);
    root.appendChild(head);

    const body     = document.createElement("div");
    body.className = "arch-panel__grid";
    this.bodyEl    = body;
    root.appendChild(body);

    this.view.controls[this.leaf.path] = { leaf: this.leaf, reset: () => this._paint() };
    this._modelPaths().forEach((path) => this.view._onDependency(path, () => this._paint()));

    this._paint();
    return root;
  }

  _modelPaths() {
    if (!this.modelBy) return [this.modelLeaf.path];
    return [this.modelBy.field, ...new Set(Object.values(this.modelBy.models))];
  }

  _activeModelLeaf() {
    if (!this.modelBy) return this.modelLeaf;
    const type = String(this.view._effective(this.view.byPath.get(this.modelBy.field)));
    const path = this.modelBy.models[type];
    return path ? this.view.byPath.get(path) : null;
  }

  _model() {
    const modelLeaf = this._activeModelLeaf();
    if (!modelLeaf) return null;
    const key = String(this.view._effective(modelLeaf));
    for (const family of this.view.modelFamilies || []) {
      for (const model of family.models || []) {
        if (model.key === key) return model;
      }
    }
    return null;
  }

  _overrides() {
    try {
      const parsed = window.PythonLiteral.parse(this.view._effective(this.leaf));
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
    } catch (e) {}
    return {};
  }

  _has(name) {
    return Object.prototype.hasOwnProperty.call(this._overrides(), name);
  }

  _set(name, value, defaultValue) {
    const overrides = this._overrides();
    if (value === undefined || this._matchesDefault(value, defaultValue)) delete overrides[name];
    else overrides[name] = value;

    this.view._setValue(this.leaf, window.PythonLiteral.render(overrides));
    this._refreshCount();
  }

  _matchesDefault(value, defaultValue) {
    if (defaultValue === undefined) return false;
    return window.PythonLiteral.render(value) === window.PythonLiteral.render(defaultValue);
  }

  _excluded(name) {
    return this.exclude.some((pattern) => pattern.startsWith("*") ? name.endsWith(pattern.slice(1)) : name === pattern);
  }

  _fields(model) {
    const arch = Array.isArray(model.arch) ? model.arch : [];
    return arch.filter((field) => !this._excluded(field.name));
  }

  _label(value) {
    return typeof value === "string" ? value : window.PythonLiteral.render(value);
  }

  _refreshCount() {
    const n = Object.keys(this._overrides()).length;
    this.countEl.textContent = n ? `${n} overridden` : "all defaults";
  }

  _paint() {
    this.bodyEl.innerHTML = "";

    if (!this._activeModelLeaf()) {
      this._paintRaw();
      return;
    }
    this.hintEl.textContent = "fields of the selected model; untouched fields train at their defaults";

    const model  = this._model();
    const fields = model ? this._fields(model) : [];

    if (!fields.length) {
      const empty = document.createElement("p");
      empty.className = "cfg-note";
      empty.textContent = "no architecture fields published for this model";
      this.bodyEl.appendChild(empty);
      this._refreshCount();
      return;
    }

    const overrides = this._overrides();
    fields.forEach((field) => this.bodyEl.appendChild(this._row(field, overrides)));

    const known  = new Set(fields.map((field) => field.name));
    const strays = Object.keys(overrides).filter((name) => !known.has(name));
    if (strays.length) this.bodyEl.appendChild(this._strayBlock(strays, overrides));

    this._refreshCount();
  }

  _paintRaw() {
    this.hintEl.textContent = "this training type publishes no model card; edit the override dict directly";

    const row     = document.createElement("div");
    row.className = "cfg-edit__row arch-row arch-row--raw";
    row.title     = `--${this.leaf.path}`;

    const label       = document.createElement("div");
    label.className   = "cfg-edit__name";
    label.textContent = this.leaf.path;
    row.appendChild(label);

    const control = this.view._textControl(this.leaf);
    row.appendChild(control.el);
    this.bodyEl.appendChild(row);
    this._refreshCount();
  }

  _row(field, overrides) {
    const row     = document.createElement("div");
    row.className = "cfg-edit__row arch-row";
    row.title     = `${field.hint ? field.hint + " · " : ""}${this.leaf.path}['${field.name}']`;

    const label       = document.createElement("div");
    label.className   = "cfg-edit__name";
    label.innerHTML   = `${field.name}<span class="arch-row__default">default ${this._label(field.default)}</span>`;
    row.appendChild(label);

    const overridden = Object.prototype.hasOwnProperty.call(overrides, field.name);
    row.classList.toggle("is-overridden", overridden);

    if (field.kind === "bool")      row.appendChild(this._boolControl(field, row));
    else if (field.options)         row.appendChild(this._optionsControl(field, row, overrides, overridden));
    else if (field.kind === "list") row.appendChild(this._listControl(field, row, overrides, overridden));
    else if (field.kind === "str")  row.appendChild(this._textControl(field, row, overrides, overridden));
    else                            row.appendChild(this._numberControl(field, row, overrides, overridden));

    return row;
  }

  _boolControl(field, row) {
    const toggle     = document.createElement("button");
    toggle.type      = "button";
    toggle.className = "switch";
    toggle.setAttribute("role", "switch");
    toggle.innerHTML = `<span class="switch__knob"></span>`;

    const paint = () => {
      const has = this._has(field.name);
      const on  = has ? Boolean(this._overrides()[field.name]) : Boolean(field.default);
      toggle.classList.toggle("is-on", on);
      toggle.classList.toggle("is-dirty", has);
      toggle.setAttribute("aria-checked", String(on));
      row.classList.toggle("is-overridden", has);
    };

    toggle.addEventListener("click", () => {
      const has = this._has(field.name);
      const on  = has ? Boolean(this._overrides()[field.name]) : Boolean(field.default);
      this._set(field.name, !on, field.default);
      paint();
    });

    paint();
    return toggle;
  }

  _optionsControl(field, row, overrides, overridden) {
    const select     = document.createElement("select");
    select.className = "cfg-edit__input picker__select";

    const current = String(overridden ? overrides[field.name] : field.default);
    const values  = field.options.includes(current) ? [...field.options] : [current, ...field.options];
    values.forEach((value) => {
      const option       = document.createElement("option");
      option.value       = value;
      option.textContent = value === String(field.default) ? `${value} (default)` : value;
      select.appendChild(option);
    });

    select.value = current;
    select.classList.toggle("is-dirty", overridden);

    select.addEventListener("change", () => {
      this._set(field.name, select.value, field.default);
      const has = select.value !== String(field.default);
      select.classList.toggle("is-dirty", has);
      row.classList.toggle("is-overridden", has);
    });

    return select;
  }

  _listControl(field, row, overrides, overridden) {
    const wrap     = document.createElement("div");
    wrap.className = "arch-row__num arch-row__list";

    const input       = document.createElement("input");
    input.className   = "cfg-edit__input";
    input.spellcheck  = false;
    input.placeholder = this._label(field.default);
    if (overridden) input.value = this._label(overrides[field.name]);
    input.classList.toggle("is-dirty", overridden);

    const apply = (raw) => {
      if (raw === "") {
        this._set(field.name, undefined, field.default);
        input.classList.remove("is-dirty", "is-invalid");
        row.classList.remove("is-overridden");
        return;
      }

      let parsed;
      try {
        parsed = window.PythonLiteral.parse(raw);
      } catch (e) {
        input.classList.add("is-invalid");
        return;
      }

      const defaults = Array.isArray(field.default) ? field.default : [];
      const allowed  = defaults.length ? [...new Set(defaults.map((item) => typeof item))] : ["number"];
      const valid    = Array.isArray(parsed) && parsed.length > 0 && parsed.every((item) => allowed.includes(typeof item));
      input.classList.toggle("is-invalid", !valid);
      if (!valid) return;

      this._set(field.name, parsed, field.default);
      const has = !this._matchesDefault(parsed, field.default);
      input.classList.toggle("is-dirty", has);
      row.classList.toggle("is-overridden", has);
    };

    input.addEventListener("input", () => apply(input.value.trim()));
    wrap.appendChild(input);

    if (Array.isArray(field.presets) && field.presets.length) {
      const chips     = document.createElement("div");
      chips.className = "arch-row__presets";
      field.presets.forEach((preset) => {
        const chip       = document.createElement("button");
        chip.type        = "button";
        chip.className   = "arch-chip";
        chip.textContent = this._label(preset);
        chip.addEventListener("click", () => {
          input.value = this._matchesDefault(preset, field.default) ? "" : this._label(preset);
          apply(input.value);
        });
        chips.appendChild(chip);
      });
      wrap.appendChild(chips);
    }

    return wrap;
  }

  _textControl(field, row, overrides, overridden) {
    const input       = document.createElement("input");
    input.className   = "cfg-edit__input";
    input.spellcheck  = false;
    input.placeholder = String(field.default);
    if (overridden) input.value = String(overrides[field.name]);
    input.classList.toggle("is-dirty", overridden);

    input.addEventListener("input", () => {
      const raw = input.value.trim();
      this._set(field.name, raw === "" ? undefined : raw, field.default);
      const has = raw !== "" && raw !== String(field.default);
      input.classList.toggle("is-dirty", has);
      row.classList.toggle("is-overridden", has);
    });

    return input;
  }

  _numberControl(field, row, overrides, overridden) {
    const wrap     = document.createElement("div");
    wrap.className = "arch-row__num";

    const input       = document.createElement("input");
    input.className   = "cfg-edit__input";
    input.type        = "number";
    input.step        = field.kind === "int" ? "1" : "any";
    input.spellcheck  = false;
    input.placeholder = String(field.default);
    if (overridden) input.value = String(overrides[field.name]);
    input.classList.toggle("is-dirty", overridden);

    const apply = (raw) => {
      if (raw === "") {
        this._set(field.name, undefined, field.default);
        input.classList.remove("is-dirty", "is-invalid");
        row.classList.remove("is-overridden");
        return;
      }

      const invalid = field.kind === "int" ? !/^-?\d+$/.test(raw) : Number.isNaN(Number(raw));
      input.classList.toggle("is-invalid", invalid);
      if (invalid) return;

      const value = Number(raw);
      this._set(field.name, value, field.default);
      const has = value !== field.default;
      input.classList.toggle("is-dirty", has);
      row.classList.toggle("is-overridden", has);
    };

    input.addEventListener("input", () => apply(input.value.trim()));
    wrap.appendChild(input);

    if (Array.isArray(field.presets) && field.presets.length) {
      const chips     = document.createElement("div");
      chips.className = "arch-row__presets";
      field.presets.forEach((preset) => {
        const chip       = document.createElement("button");
        chip.type        = "button";
        chip.className   = "arch-chip";
        chip.textContent = String(preset);
        chip.addEventListener("click", () => {
          input.value = preset === field.default ? "" : String(preset);
          apply(input.value);
        });
        chips.appendChild(chip);
      });
      wrap.appendChild(chips);
    }

    return wrap;
  }

  _strayBlock(strays, overrides) {
    const block     = document.createElement("div");
    block.className = "arch-panel__stray";

    const label       = document.createElement("span");
    label.className   = "arch-panel__stray-label";
    label.textContent = "overrides outside this model's field list (kept):";
    block.appendChild(label);

    strays.forEach((name) => {
      const chip     = document.createElement("button");
      chip.type      = "button";
      chip.className = "arch-chip arch-chip--stray";
      chip.title     = "click to drop this override";
      chip.innerHTML = `${name}: ${this._label(overrides[name])}<span aria-hidden="true"> ×</span>`;
      chip.addEventListener("click", () => {
        this._set(name, undefined, undefined);
        this._paint();
      });
      block.appendChild(chip);
    });

    return block;
  }
}

window.ModelTogglePanel   = ModelTogglePanel;
window.ModelCardPanel     = ModelCardPanel;
window.ArchOverridesPanel = ArchOverridesPanel;
