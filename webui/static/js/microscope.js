"use strict";

class MicroscopeView {
  static FLIP_LABELS    = { none: "original", row: "row flip", col: "col flip", both: "both" };
  static STATUS_RETRIES = 5;
  static PAGES = [
    { key: "prediction",  title: "Prediction" },
    { key: "fields",      title: "Target fields" },
    { key: "attribution", title: "Attribution" },
    { key: "robustness",  title: "Robustness" },
    { key: "features",    title: "Feature maps" },
  ];

  constructor(root) {
    this.root        = root;
    this.info        = null;
    this.pixel       = null;
    this.layer       = "";
    this.block       = "";
    this.blocks      = [];
    this.layerTypes  = {};
    this.built       = false;
    this.pollToken   = 0;
    this.attribToken = 0;
    this.ablToken    = 0;
    this.occlToken   = 0;
    this.flipToken   = 0;
    this.runs        = [];
    this.selectedId  = null;
    this.readyPath   = null;
    this.runStrip    = null;
    this.attrib      = null;
    this.gridChannel = -1;
    this.fieldsData  = null;
    this.fieldsIndex = 0;
    this.fieldsToken = 0;
    this.vitalsData  = null;
    this.vitalsToken = 0;
    this.pager       = null;
    this.pageLoaded  = new Set();
  }

  enter() {
    if (!this.built) {
      this._build();
      this.built = true;
    }
    this._refreshRuns();
    this._pollStatus();
  }

  figureTag() {
    const run   = this.runs.find((entry) => entry.id === this.selectedId);
    const parts = [run ? run.run : ""];

    if (this.pixel) parts.push(`row${String(this.pixel.row).padStart(4, "0")}_col${String(this.pixel.col).padStart(4, "0")}`);

    return parts.filter(Boolean).join("_");
  }

  _build() {
    this.root.innerHTML = `
      <div class="cube-pick">
        <div class="run-strip" id="probe-strip" aria-label="Trained backbone runs"></div>
        <p class="cube-hint" id="probe-hint">${SharedCharts.waitLine("Scanning the runs directory for loadable backbone runs…")}</p>
        <div class="cube-progress" id="probe-progress" hidden>
          <div class="cube-progress__track"><i class="cube-progress__fill" id="probe-progress-fill"></i></div>
          <p class="cube-progress__label" id="probe-progress-label">loading&hellip;</p>
        </div>
      </div>

      <div class="ms-stage" id="probe-stage" hidden>
        <section class="ms-card ms-scene">
          <div class="ms-scene__map">
            <div class="fl-map__frame">
              <span class="cube-axis cube-axis--y">row &darr;</span>
              <div class="fl-map__wrap" id="probe-map-wrap">
                <img id="probe-map-img" alt="First target channel of the split" />
                <div class="fl-marks" id="probe-marks"></div>
              </div>
              <span class="cube-axis cube-axis--x">col &rarr;</span>
            </div>
          </div>
          <div class="ms-scene__info">
            <div class="ms-card__head">
              <h3 class="cube-panel-title">Split</h3>
              <span class="ms-badge" id="probe-model-badge"></span>
            </div>
            <div class="ms-meta" id="probe-meta"></div>
            <p class="cube-coords" id="probe-coords">Click the map to probe a pixel.</p>
            <div class="fl-pixrow" role="group" aria-label="Probe a pixel by index">
              <span class="cube-jump__cell">
                <label for="probe-row">row</label>
                <input type="number" id="probe-row" class="cube-jump__input" min="0" step="1" inputmode="numeric" />
              </span>
              <span class="cube-jump__cell">
                <label for="probe-col">col</label>
                <input type="number" id="probe-col" class="cube-jump__input" min="0" step="1" inputmode="numeric" />
              </span>
              <button type="button" class="btn btn--mini" id="probe-go">Probe</button>
            </div>
          </div>
        </section>

        <div class="ms-main">
          <div class="ms-idle" id="probe-idle">Click a pixel on the map, or type row / col indices, to put it under the microscope.</div>

          <div class="ms-layout" id="probe-cards" hidden>
            <div class="ms-pages">

            <section class="ms-page" data-page="prediction">
            <section class="ms-card ms-card--snug">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Prediction</h3>
                <span class="ms-card__note">every target channel at the probed pixel, in target units</span>
              </div>
              <table class="ms-slots" id="probe-targets"></table>
              <div class="ms-subhead">
                <h4 class="cube-panel-title">Inputs</h4>
                <span class="ms-card__note">raw input channel values at the probed pixel</span>
              </div>
              <table class="ms-slots" id="probe-inputs"></table>
            </section>
            </section>

            <section class="ms-page" data-page="fields" hidden>
            <section class="ms-card">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Local target fields</h3>
                <span class="ms-card__note">one forward pass over the input window: prediction, target and absolute error around the pixel</span>
              </div>
              <div class="ms-ctlrow">
                <span class="ms-ctlrow__label">target</span>
                <div class="ms-chiprow" id="probe-fields-targets"></div>
              </div>
              <div class="ms-fieldrow" id="probe-fields-maps"></div>
            </section>
            </section>

            <section class="ms-page" data-page="attribution" hidden>
            <section class="ms-card">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Input attribution</h3>
                <span class="ms-card__note">share of |&part;output/&part;input| per input channel, every target channel side by side</span>
              </div>
              <div class="ms-bars" id="probe-bars"></div>
              <div class="ms-legend" id="probe-bars-legend"></div>
              <div class="ms-subhead">
                <h4 class="cube-panel-title">Spatial gradients</h4>
                <span class="ms-card__note">|&part;output/&part;input| over the input window; pick an input channel or view all</span>
              </div>
              <div class="ms-ctlrow">
                <span class="ms-ctlrow__label">channel</span>
                <div class="ms-chiprow" id="probe-grid-channels"></div>
              </div>
              <div class="ms-gridwrap"><div class="ms-grid" id="probe-grid"></div></div>
              <div class="ms-subhead">
                <h4 class="cube-panel-title">Effective receptive field</h4>
                <span class="ms-card__note">share of each target's gradient energy within a radius of the probed pixel</span>
              </div>
              <canvas id="probe-erf" data-fig="effective receptive field" width="640" height="180" hidden></canvas>
              <p class="ms-wifacts" id="probe-erf-facts"></p>
            </section>
            </section>

            <section class="ms-page" data-page="robustness" hidden>
            <section class="ms-card">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Channel ablation</h3>
                <span class="ms-card__note">zero one input channel at a time and re-run: causal damage to the prediction, unlike the attribution page's local gradients</span>
              </div>
              <div class="ms-abl" id="probe-ablation"></div>
            </section>

            <section class="ms-card ms-card--third">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Occlusion sensitivity</h3>
                <span class="ms-card__note">slide a zero-patch across the window and re-run: where the model's evidence for this pixel sits</span>
              </div>
              <div class="ms-fieldrow" id="probe-occlusion"></div>
              <p class="ms-wifacts" id="probe-occl-facts"></p>
            </section>

            <section class="ms-card ms-card--twothird">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Symmetry</h3>
                <span class="ms-card__note">re-run the model on flipped windows and flip the outputs back; a flip-equivariant model repeats itself exactly</span>
              </div>
              <table class="ms-slots" id="probe-flips"></table>
              <p class="ms-wifacts" id="probe-flips-facts"></p>
            </section>
            </section>

            <section class="ms-page" data-page="features" hidden>
            <section class="ms-card">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Feature maps</h3>
                <span class="ms-card__note" id="probe-feat-note"></span>
              </div>
              <div class="ms-arch" id="probe-arch" aria-label="Model blocks in forward order"></div>
              <div class="ms-arch__layers" id="probe-arch-layers"></div>
              <div class="ms-feat"><img id="probe-features" alt="Feature maps" hidden /></div>
              <div class="ms-subhead">
                <h4 class="cube-panel-title">Learned kernels</h4>
                <span class="ms-card__note" id="probe-kern-note"></span>
              </div>
              <div class="ms-feat"><img id="probe-kernels" alt="Learned convolution kernels" hidden /></div>
              <div class="ms-subhead">
                <h4 class="cube-panel-title">Layer vitals</h4>
                <span class="ms-card__note">one hooked forward pass at the probed window; click a row to open that layer's feature maps</span>
              </div>
              <p class="ms-wifacts" id="probe-vitals-summary"></p>
              <div class="ms-vitals"><table class="ms-slots ms-vitals__table" id="probe-vitals"></table></div>
            </section>
            </section>

            </div>

            <nav class="secnav" id="probe-nav" aria-label="Microscope analyses"></nav>
          </div>
        </div>
      </div>`;

    this.refs = {
      strip         : this.root.querySelector("#probe-strip"),
      hint          : this.root.querySelector("#probe-hint"),
      progress      : this.root.querySelector("#probe-progress"),
      fill          : this.root.querySelector("#probe-progress-fill"),
      label         : this.root.querySelector("#probe-progress-label"),
      stage         : this.root.querySelector("#probe-stage"),
      badge         : this.root.querySelector("#probe-model-badge"),
      meta          : this.root.querySelector("#probe-meta"),
      mapImg        : this.root.querySelector("#probe-map-img"),
      marks         : this.root.querySelector("#probe-marks"),
      coords        : this.root.querySelector("#probe-coords"),
      rowInput      : this.root.querySelector("#probe-row"),
      colInput      : this.root.querySelector("#probe-col"),
      goBtn         : this.root.querySelector("#probe-go"),
      idle          : this.root.querySelector("#probe-idle"),
      cards         : this.root.querySelector("#probe-cards"),
      nav           : this.root.querySelector("#probe-nav"),
      pages         : [...this.root.querySelectorAll(".ms-page")],
      targets       : this.root.querySelector("#probe-targets"),
      inputs        : this.root.querySelector("#probe-inputs"),
      fieldTargets  : this.root.querySelector("#probe-fields-targets"),
      fieldMaps     : this.root.querySelector("#probe-fields-maps"),
      bars          : this.root.querySelector("#probe-bars"),
      barsLegend    : this.root.querySelector("#probe-bars-legend"),
      gridChips     : this.root.querySelector("#probe-grid-channels"),
      grid          : this.root.querySelector("#probe-grid"),
      erf           : this.root.querySelector("#probe-erf"),
      erfFacts      : this.root.querySelector("#probe-erf-facts"),
      ablation      : this.root.querySelector("#probe-ablation"),
      occlusion     : this.root.querySelector("#probe-occlusion"),
      occlFacts     : this.root.querySelector("#probe-occl-facts"),
      flips         : this.root.querySelector("#probe-flips"),
      flipsFacts    : this.root.querySelector("#probe-flips-facts"),
      arch          : this.root.querySelector("#probe-arch"),
      archLayers    : this.root.querySelector("#probe-arch-layers"),
      features      : this.root.querySelector("#probe-features"),
      featNote      : this.root.querySelector("#probe-feat-note"),
      kernels       : this.root.querySelector("#probe-kernels"),
      kernNote      : this.root.querySelector("#probe-kern-note"),
      vitalsSum     : this.root.querySelector("#probe-vitals-summary"),
      vitals        : this.root.querySelector("#probe-vitals"),
    };

    this.refs.mapImg.addEventListener("click", (ev) => this._onMapClick(ev));
    this.refs.goBtn.addEventListener("click", () => this._probeInputs());
    this.refs.rowInput.addEventListener("keydown", (ev) => { if (ev.key === "Enter") this._probeInputs(); });
    this.refs.colInput.addEventListener("keydown", (ev) => { if (ev.key === "Enter") this._probeInputs(); });
    this.refs.features.addEventListener("load", () => { this.refs.features.hidden = false; });
    this.refs.features.addEventListener("error", () => {
      this.refs.features.hidden = true;
      this.refs.featNote.textContent = `${this.layer} captured no activation at this pixel`;
    });
    this.refs.kernels.addEventListener("load", () => { this.refs.kernels.hidden = false; });
    this.refs.kernels.addEventListener("error", () => {
      this.refs.kernels.hidden = true;
      this.refs.kernNote.textContent = `${this.layer} rendered no kernels`;
    });

    this.pager = new SectionPager(this.refs.nav, this.refs.pages, MicroscopeView.PAGES, (key) => this._repaintPage(key));
  }

  _pageLoaders(key) {
    if (key === "fields")      return [() => this._loadFields()];
    if (key === "attribution") return [() => this._loadAttribution()];
    if (key === "robustness")  return [() => this._loadAblation(), () => this._loadOcclusion(), () => this._loadFlips()];
    if (key === "features")    return [() => this._loadFeatures(), () => this._loadVitals()];
    return [];
  }

  _ensurePage(key) {
    if (!this.pixel || this.pageLoaded.has(key)) return;

    this.pageLoaded.add(key);
    this._pageLoaders(key).forEach((load) => load());
  }

  _repaintPage(key) {
    this._ensurePage(key);

    if (key === "attribution" && this.attrib) this._renderErf(this.attrib);
  }

  async _refreshRuns() {
    const base = ResultsSources.runs();
    const data = await Api.get(`/api/probe/runs?base=${encodeURIComponent(base)}`);

    if (!data || !data.ok) {
      this.refs.hint.textContent = (data && data.error) || "Backend unreachable.";
      return;
    }

    this.runs = data.runs || [];
    this._renderStrip();

    if (!this.runs.length) {
      this.refs.hint.textContent = `No loadable runs under ${data.root} (a loadable run holds best_model.pt and meta/model_config.json). Train a backbone run, or point the runs directory in the Results tab at the right place.`;
      return;
    }

    if (!this.info) this.refs.hint.textContent = "Pick a trained backbone run to put under the microscope.";
  }

  _renderStrip() {
    if (!this.runStrip) {
      this.runStrip = new RunStrip(this.refs.strip, {
        stateFor : (run) => run.id === this.selectedId,
        onPick   : (run) => this._load(run.id),
      });
    }
    this.runStrip.render(this.runs);
  }

  async _load(path) {
    const out = await Api.post("/api/probe/load", { path });
    if (!out.ok) {
      this.refs.hint.textContent = out.error || "load failed";
      return;
    }

    this.selectedId = path;
    this.readyPath  = null;
    this._renderStrip();

    this.refs.progress.hidden = false;
    this._pollStatus();
  }

  async _pollStatus() {
    const token = ++this.pollToken;
    let failures = 0;
    const tick  = async () => {
      if (token !== this.pollToken) return;

      const st = await Api.get("/api/probe/status");

      if (!st.state) {
        failures += 1;
        this.refs.hint.textContent = `probe status unavailable — ${st.error} (attempt ${failures} of ${MicroscopeView.STATUS_RETRIES})`;
        if (failures < MicroscopeView.STATUS_RETRIES) setTimeout(tick, 1000);
        else this.refs.progress.hidden = true;
        return;
      }
      failures = 0;

      if (st.state === "loading") {
        this.refs.progress.hidden  = false;
        this.refs.fill.style.width = `${Math.round((st.progress || 0) * 100)}%`;
        this.refs.label.textContent = st.stage || "loading";
        setTimeout(tick, 500);
        return;
      }

      this.refs.progress.hidden = true;

      if (st.state === "error") {
        this.refs.hint.textContent = st.error;
        return;
      }
      if (st.state === "ready" && st.info) {
        if (st.path && st.path !== this.selectedId) {
          this.selectedId = st.path;
          this._renderStrip();
        }
        if (this.readyPath !== st.path) {
          await this._onReady(st.info);
          this.readyPath = st.path;
        }
      }
    };
    tick();
  }

  async _onReady(info) {
    this.info        = info;
    this.pixel       = null;
    this.attrib      = null;
    this.gridChannel = -1;
    this.fieldsData  = null;
    this.fieldsIndex = 0;
    this.vitalsData  = null;
    this.pageLoaded.clear();

    this.refs.stage.hidden = false;
    this.refs.idle.hidden  = false;
    this.refs.cards.hidden = true;
    this.refs.marks.innerHTML = "";

    this.refs.hint.textContent   = `Loaded ${info.backbone}, probing the ${info.split} split; the map shows the target '${info.target_channels[0]}'.`;
    this.refs.badge.textContent  = info.backbone;
    this.refs.coords.textContent = "Click the map to probe a pixel.";
    this.refs.mapImg.src         = `/api/probe/map?t=${Date.now()}`;

    this.refs.rowInput.max = info.row_size - 1;
    this.refs.colInput.max = info.col_size - 1;

    this.refs.meta.innerHTML = [
      `<span class="ms-chip"><b>in</b> ${info.input_channels.length} ch</span>`,
      `<span class="ms-chip"><b>out</b> ${info.target_channels.length} ch</span>`,
      `<span class="ms-chip"><b>region</b> ${info.row_size}&times;${info.col_size} px</span>`,
      `<span class="ms-chip"><b>window</b> ${info.patch[0]}&times;${info.patch[1]} px</span>`,
      `<span class="ms-chip"><b>split</b> ${SharedCharts.esc(info.split)}</span>`,
    ].join("");

    const layers = await Api.get("/api/probe/layers");

    if (!layers.ok) {
      this.refs.hint.textContent = layers.error || "layer list unavailable";
      return;
    }

    this._buildArch(layers.layers);
  }

  _blockKey(name) {
    const parts = name.split(".");
    return parts.length > 1 && /^\d+$/.test(parts[1]) ? `${parts[0]}.${parts[1]}` : parts[0];
  }

  _layerCategory(type) {
    const t = type.toLowerCase();
    if (t.includes("conv")) return "conv";
    if (t.includes("norm")) return "norm";
    if (/relu|gelu|silu|elu|sigmoid|tanh|softmax|mish|swish|activation/.test(t)) return "act";
    if (/linear|attention|embed|mlp/.test(t)) return "dense";
    return "aux";
  }

  _buildArch(layers) {
    this.blocks     = [];
    this.layerTypes = {};

    const index = new Map();
    layers.forEach((layer) => {
      this.layerTypes[layer.name] = layer.type;
      const key = this._blockKey(layer.name);
      if (!index.has(key)) {
        index.set(key, { key, layers: [] });
        this.blocks.push(index.get(key));
      }
      index.get(key).layers.push(layer);
    });

    this.block = this.blocks.length ? this.blocks[0].key : "";
    this.layer = this.blocks.length ? this.blocks[0].layers[0].name : "";

    this._renderArch();
    this._renderArchLayers();
    this._noteFeatureLayer();
  }

  _renderArch() {
    this.refs.arch.innerHTML = "";
    this.blocks.forEach((block) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "ms-arch__block" + (block.key === this.block ? " is-active" : "");
      btn.innerHTML = `<span class="ms-arch__name">${SharedCharts.esc(block.key)}</span><span class="ms-arch__count">${block.layers.length} layer${block.layers.length === 1 ? "" : "s"}</span>`;
      btn.addEventListener("click", () => {
        this.block = block.key;
        this.layer = block.layers[0].name;
        this._renderArch();
        this._renderArchLayers();
        this._noteFeatureLayer();
        this._loadFeatures();
        this._renderVitals();
      });
      this.refs.arch.appendChild(btn);
    });
  }

  _renderArchLayers() {
    this.refs.archLayers.innerHTML = "";
    const block = this.blocks.find((b) => b.key === this.block);
    if (!block) return;

    block.layers.forEach((layer) => {
      const short = layer.name === block.key ? layer.name : layer.name.slice(block.key.length + 1);
      const chip  = document.createElement("button");
      chip.type = "button";
      chip.className = "ms-arch__layer" + (layer.name === this.layer ? " is-active" : "");
      chip.dataset.cat = this._layerCategory(layer.type);
      chip.title = layer.name;
      chip.innerHTML = `${SharedCharts.esc(short)} <em>${SharedCharts.esc(layer.type)}</em>`;
      chip.addEventListener("click", () => {
        this.layer = layer.name;
        this._renderArchLayers();
        this._noteFeatureLayer();
        this._loadFeatures();
        this._renderVitals();
      });
      this.refs.archLayers.appendChild(chip);
    });
  }

  _noteFeatureLayer() {
    if (!this.layer) {
      this.refs.featNote.textContent = "";
      return;
    }
    const position = Object.keys(this.layerTypes).indexOf(this.layer) + 1;
    this.refs.featNote.textContent = `${this.layer} (${this.layerTypes[this.layer]}) — layer ${position}/${Object.keys(this.layerTypes).length}, 16 strongest channels over the probed window`;
  }

  _onMapClick(ev) {
    if (!this.info) return;
    const rect = this.refs.mapImg.getBoundingClientRect();
    const row  = Math.round(((ev.clientY - rect.top) / rect.height) * (this.info.row_size - 1));
    const col  = Math.round(((ev.clientX - rect.left) / rect.width) * (this.info.col_size - 1));
    this._probe(row, col);
  }

  _probeInputs() {
    const row = parseInt(this.refs.rowInput.value, 10);
    const col = parseInt(this.refs.colInput.value, 10);
    if (Number.isFinite(row) && Number.isFinite(col)) this._probe(row, col);
  }

  async _probe(row, col) {
    row = Math.max(0, Math.min(this.info.row_size - 1, row));
    col = Math.max(0, Math.min(this.info.col_size - 1, col));

    this.pixel                   = { row, col };
    this.refs.rowInput.value     = row;
    this.refs.colInput.value     = col;
    this.refs.coords.textContent = `row ${row} · col ${col} (absolute ${row + this.info.row_offset} · ${col + this.info.col_offset})`;
    this._renderMark(row, col);

    const out = await Api.post("/api/probe/predict", { row, col });
    if (!out.ok) {
      this.refs.coords.textContent = out.error;
      return;
    }

    this.refs.idle.hidden  = true;
    this.refs.cards.hidden = false;

    this._renderPrediction(out);

    this.pageLoaded.clear();
    this._ensurePage(this.pager.page);
  }

  _renderMark(row, col) {
    const x                   = (col / (this.info.col_size - 1)) * 100;
    const y                   = (row / (this.info.row_size - 1)) * 100;
    this.refs.marks.innerHTML = `<i class="fl-mark" style="left:${x}%;top:${y}%;background:#e11d48"></i>`;
  }

  _renderPrediction(out) {
    const targets = out.targets.map((t) => `<tr><td class="ms-slots__key">${SharedCharts.esc(t.name)}</td><td>${SharedCharts.fmt(t.prediction)}</td><td>${SharedCharts.fmt(t.target)}</td><td>${t.error >= 0 ? "+" : ""}${SharedCharts.fmt(t.error)}</td></tr>`);
    const inputs  = out.inputs.map((i) => `<tr><td class="ms-slots__key">${SharedCharts.esc(i.name)}</td><td>${SharedCharts.fmt(i.value)}</td></tr>`);

    this.refs.targets.innerHTML = `<thead><tr><th class="ms-slots__key">TARGET</th><th>PREDICTION</th><th>TARGET VALUE</th><th>ERROR</th></tr></thead><tbody>${targets.join("")}</tbody>`;
    this.refs.inputs.innerHTML  = `<thead><tr><th class="ms-slots__key">INPUT</th><th>VALUE</th></tr></thead><tbody>${inputs.join("")}</tbody>`;
  }

  async _loadAttribution() {
    const token = ++this.attribToken;

    this.refs.bars.innerHTML       = SharedCharts.waitPanel("computing gradients…");
    this.refs.barsLegend.innerHTML = "";
    this.refs.gridChips.innerHTML  = "";
    this.refs.grid.innerHTML       = "";
    this.refs.erf.hidden           = true;
    this.refs.erfFacts.innerHTML   = "";

    const out = await Api.post("/api/probe/attribution", { row: this.pixel.row, col: this.pixel.col });
    if (token !== this.attribToken) return;

    if (!out.ok) {
      this.attrib = null;
      this.refs.bars.innerHTML = `<p class="cube-hint">${SharedCharts.esc(out.error)}</p>`;
      return;
    }

    this.attrib = out;
    SharedCharts.targetBars(this.refs.bars, this.refs.barsLegend, out.channels, out.targets, "no dependence at this pixel");
    this._renderGridChips();
    this._renderGrid();
    this._renderErf(out);
  }

  _renderErf(out) {
    const live = out.targets.filter((t) => t.radial);

    if (!live.length) {
      this.refs.erf.hidden = true;
      this.refs.erfFacts.innerHTML = "no gradient energy at this pixel; every target channel is flat here";
      return;
    }

    this.refs.erf.hidden = false;
    SharedCharts.lineChart(
      this.refs.erf,
      live[0].radial.radii,
      live.map((t) => ({ values: t.radial.cumulative, color: SharedCharts.targetColor(out.targets.indexOf(t)), width: 1.5, label: t.name })),
      { xUnit: "px" },
    );

    const parts = out.targets.map((t) => t.radial
      ? `${SharedCharts.esc(t.name)} r<sub>50</sub> <b>${t.radial.r50} px</b> / r<sub>90</sub> <b>${t.radial.r90} px</b>`
      : `${SharedCharts.esc(t.name)}: no dependence`);

    this.refs.erfFacts.innerHTML = parts.join(" &middot; ");
  }

  _renderGridChips() {
    this.refs.gridChips.innerHTML = "";

    const pick = (index) => () => {
      this.gridChannel = index;
      this._renderGridChips();
      this._renderGrid();
    };

    this.refs.gridChips.appendChild(SharedCharts.chip("all channels", this.gridChannel === -1, pick(-1)));
    this.attrib.channels.forEach((label, index) => {
      this.refs.gridChips.appendChild(SharedCharts.chip(label, this.gridChannel === index, pick(index)));
    });
  }

  _renderGrid() {
    SharedCharts.gradientGrid(this.refs.grid, this.attrib, this.gridChannel);
  }

  async _loadAblation() {
    const token = ++this.ablToken;

    this.refs.ablation.innerHTML = SharedCharts.waitPanel("re-running the model without each channel…");

    const out = await Api.post("/api/probe/ablation", { row: this.pixel.row, col: this.pixel.col });
    if (token !== this.ablToken) return;

    if (!out.ok) {
      this.refs.ablation.innerHTML = `<p class="cube-hint">${SharedCharts.esc(out.error)}</p>`;
      return;
    }

    SharedCharts.ablationRows(this.refs.ablation, out.channels);
  }

  async _loadOcclusion() {
    const token = ++this.occlToken;

    this.refs.occlFacts.innerHTML = "";
    this.refs.occlusion.innerHTML = SharedCharts.waitPanel("sliding an occluder across the window…");

    const out = await Api.post("/api/probe/occlusion", { row: this.pixel.row, col: this.pixel.col });
    if (token !== this.occlToken) return;

    if (!out.ok) {
      this.refs.occlusion.innerHTML = `<p class="cube-hint">${SharedCharts.esc(out.error)}</p>`;
      return;
    }

    this.refs.occlusion.innerHTML = `
      <figure class="ms-field ms-field--wide">
        <span class="ms-cell__frame">
          <img src="data:image/png;base64,${out.map.png}" alt="Occlusion sensitivity map" />
          ${SharedCharts.dot(out.center_cell, out.grid)}
        </span>
        <figcaption>squared prediction shift per occluded cell &middot; ${SharedCharts.fmt(out.map.min)} &rarr; ${SharedCharts.fmt(out.map.max)}</figcaption>
      </figure>`;

    const dRow = (out.worst.row + 0.5) * out.occluder[0] - out.center[0];
    const dCol = (out.worst.col + 0.5) * out.occluder[1] - out.center[1];
    const dist = Math.hypot(dRow, dCol);

    this.refs.occlFacts.innerHTML = `occluder ${out.occluder[0]}&times;${out.occluder[1]} px &middot; peak shift <b>${out.worst.delta.toExponential(2)}</b> (normalized target units) about ${dist.toFixed(0)} px from the probed pixel`;
  }

  async _loadFields() {
    const token = ++this.fieldsToken;

    this.refs.fieldTargets.innerHTML = "";
    this.refs.fieldMaps.innerHTML    = SharedCharts.waitPanel("computing target fields…");

    const out = await Api.post("/api/probe/fields", { row: this.pixel.row, col: this.pixel.col });
    if (token !== this.fieldsToken) return;

    if (!out.ok) {
      this.fieldsData = null;
      this.refs.fieldMaps.innerHTML = `<p class="cube-hint">${SharedCharts.esc(out.error)}</p>`;
      return;
    }

    this.fieldsData = out;
    this._renderFieldsChips();
    this._renderFieldsMaps();
  }

  _renderFieldsChips() {
    this.refs.fieldTargets.innerHTML = "";

    this.fieldsData.targets.forEach((target, index) => {
      this.refs.fieldTargets.appendChild(SharedCharts.chip(target.name, this.fieldsIndex === index, () => {
        if (index === this.fieldsIndex) return;
        this.fieldsIndex = index;
        this._renderFieldsChips();
        this._renderFieldsMaps();
      }));
    });
  }

  _fieldFigure(payload, title) {
    return `
      <figure class="ms-field">
        <span class="ms-cell__frame">
          <img src="data:image/png;base64,${payload.png}" alt="${SharedCharts.esc(title)}" />
          ${SharedCharts.dot(this.fieldsData.center, this.fieldsData.patch)}
        </span>
        <figcaption>${SharedCharts.esc(title)} &middot; ${SharedCharts.fmt(payload.min)} &rarr; ${SharedCharts.fmt(payload.max)}</figcaption>
      </figure>`;
  }

  _renderFieldsMaps() {
    const target = this.fieldsData.targets[this.fieldsIndex] || this.fieldsData.targets[0];

    this.refs.fieldMaps.innerHTML = [
      this._fieldFigure(target.prediction, `${target.name} prediction`),
      this._fieldFigure(target.target,     `${target.name} target`),
      this._fieldFigure(target.error,      `${target.name} absolute error`),
    ].join("");
  }

  async _loadFlips() {
    const token = ++this.flipToken;

    this.refs.flips.innerHTML      = "";
    this.refs.flipsFacts.innerHTML = SharedCharts.waitLine("re-running the model on flipped windows…");

    const out = await Api.post("/api/probe/flips", { row: this.pixel.row, col: this.pixel.col });
    if (token !== this.flipToken) return;

    if (!out.ok) {
      this.refs.flipsFacts.innerHTML = SharedCharts.esc(out.error);
      return;
    }

    const labels = MicroscopeView.FLIP_LABELS;
    const head   = `<thead><tr><th class="ms-slots__key">FLIP</th>${out.channels.map((name) => `<th>${SharedCharts.esc(name)}</th>`).join("")}<th>SHIFT</th></tr></thead>`;
    const rows   = out.entries.map((e) => `<tr><td class="ms-slots__key">${labels[e.flip]}</td>${e.values.map((v) => `<td>${SharedCharts.fmt(v)}</td>`).join("")}<td>${e.delta.toExponential(2)}</td></tr>`);

    this.refs.flips.innerHTML      = `${head}<tbody>${rows.join("")}</tbody>`;
    this.refs.flipsFacts.innerHTML = "values in target units; the shift is the squared change of the normalized prediction against the original window";
  }

  _loadFeatures() {
    this._loadKernels();
    if (!this.pixel || !this.layer) return;
    this.refs.features.dataset.fig = `feature maps ${this.layer}`;
    this.refs.features.src = `/api/probe/features?row=${this.pixel.row}&col=${this.pixel.col}&layer=${encodeURIComponent(this.layer)}&t=${Date.now()}`;
  }

  _loadKernels() {
    if (!this.layer) return;

    const type = this.layerTypes[this.layer] || "";
    if (!type.toLowerCase().includes("conv")) {
      this.refs.kernels.hidden = true;
      this.refs.kernNote.textContent = `${this.layer} (${type}) holds no convolution kernels — pick a conv layer above`;
      return;
    }

    this.refs.kernNote.textContent = `${this.layer} — signed weights, red positive and blue negative on a shared scale`;
    this.refs.kernels.dataset.fig = `kernels ${this.layer}`;
    this.refs.kernels.src = `/api/probe/kernels?layer=${encodeURIComponent(this.layer)}&t=${Date.now()}`;
  }

  async _loadVitals() {
    const token = ++this.vitalsToken;

    this.refs.vitalsSum.innerHTML = SharedCharts.waitLine("recording activation statistics…");
    this.refs.vitals.innerHTML    = "";

    const out = await Api.post("/api/probe/vitals", { row: this.pixel.row, col: this.pixel.col });
    if (token !== this.vitalsToken) return;

    if (!out.ok) {
      this.refs.vitalsSum.innerHTML = SharedCharts.esc(out.error);
      return;
    }

    this.vitalsData = out;
    this._renderVitals();
  }

  _renderVitals() {
    if (!this.vitalsData) return;

    SharedCharts.vitalsTable(this.refs.vitalsSum, this.refs.vitals, this.vitalsData, {
      currentLayer : this.layer,
      onPick       : (name) => {
        if (!(name in this.layerTypes)) return;
        this.block = this._blockKey(name);
        this.layer = name;
        this._renderArch();
        this._renderArchLayers();
        this._noteFeatureLayer();
        this._loadFeatures();
        this._renderVitals();
      },
    });
  }
}


class SectionPager {
  constructor(nav, pages, defs, onShow) {
    this.nav    = nav;
    this.pages  = pages;
    this.onShow = onShow;
    this.page   = defs[0].key;

    defs.forEach((def) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "secnav__item";
      btn.dataset.page = def.key;
      btn.innerHTML = `<span class="secnav__name">${def.title}</span>`;
      btn.addEventListener("click", () => this.show(def.key));
      nav.appendChild(btn);
    });

    this.show(this.page);
  }

  show(key) {
    this.page = key;
    this.nav.querySelectorAll(".secnav__item").forEach((btn) => btn.classList.toggle("is-active", btn.dataset.page === key));
    this.pages.forEach((el) => { el.hidden = el.dataset.page !== key; });
    this.onShow(key);
  }
}

window.MicroscopeView = MicroscopeView;
window.SectionPager   = SectionPager;
