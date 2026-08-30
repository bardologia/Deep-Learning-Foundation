"use strict";

class SurveyView {
  static STATUS_RETRIES = 5;
  static FLIP_LABELS = { row: "row flip", range: "range flip", both: "both flips" };
  static PAGES = [
    { key: "prediction",  title: "Prediction" },
    { key: "attribution", title: "Attribution" },
    { key: "robustness",  title: "Robustness" },
    { key: "vitals",      title: "Layer vitals" },
  ];

  constructor(root) {
    this.root         = root;
    this.built        = false;
    this.pollToken    = 0;
    this.runs         = [];
    this.selectedId   = null;
    this.runStrip     = null;
    this.renderedPath = null;
    this.resultData   = null;
    this.gridChannel  = -1;
    this.pager        = null;
    this.cfg          = { split: "test", device: "cpu", tiles: 64, erf: 24, threads: 64 };
  }

  enter() {
    if (!this.built) {
      this._build();
      this.built = true;
    }
    this._refreshRuns();
    this._pollStatus();
  }

  _build() {
    this.root.innerHTML = `
      <div class="cube-pick">
        <div class="run-strip" id="survey-strip" aria-label="Trained backbone and dual runs"></div>
        <p class="cube-hint" id="survey-hint">Scanning the runs directory for loadable backbone and dual runs&hellip;</p>
        <div class="cube-progress sv-progress" id="survey-progress" hidden>
          <div class="cube-progress__track"><i class="cube-progress__fill" id="survey-progress-fill"></i></div>
          <p class="cube-progress__label" id="survey-progress-label">starting&hellip;</p>
          <button type="button" class="btn btn--mini" id="survey-cancel">Cancel</button>
        </div>
      </div>

      <section class="ms-card sv-setup" id="survey-setup" hidden>
        <div class="ms-card__head">
          <h3 class="cube-panel-title">Survey setup</h3>
          <span class="ms-badge" id="survey-setup-run"></span>
        </div>
        <div class="ms-ctlrow">
          <span class="ms-ctlrow__label">split</span>
          <div class="cube-spaces" id="survey-split" role="group" aria-label="Split">
            <button type="button" class="cube-space" data-value="train">train</button>
            <button type="button" class="cube-space" data-value="val">val</button>
            <button type="button" class="cube-space is-active" data-value="test">test</button>
          </div>
        </div>
        <div class="ms-ctlrow">
          <span class="ms-ctlrow__label">device</span>
          <div class="cube-spaces" id="survey-device" role="group" aria-label="Device">
            <button type="button" class="cube-space is-active" data-value="cpu">cpu</button>
            <button type="button" class="cube-space" data-value="cuda">cuda</button>
          </div>
        </div>
        <div class="ms-ctlrow">
          <span class="ms-ctlrow__label">tile cap</span>
          <input type="number" id="survey-tiles" class="cube-jump__input" min="0" step="1" value="64" />
          <span class="ms-card__note">evenly strided tiles to survey &mdash; 0 surveys every tile of the region</span>
        </div>
        <div class="ms-ctlrow">
          <span class="ms-ctlrow__label">ERF samples</span>
          <input type="number" id="survey-erf-samples" class="cube-jump__input" min="0" step="1" value="24" />
          <span class="ms-card__note">pixels sampled for the receptive-field profile &mdash; 0 skips that phase</span>
        </div>
        <div class="ms-ctlrow">
          <span class="ms-ctlrow__label">CPU threads</span>
          <input type="number" id="survey-threads" class="cube-jump__input" min="1" step="1" value="64" />
          <span class="ms-card__note">torch thread cap while the survey runs; restored afterwards</span>
        </div>
        <div class="ms-ctlrow">
          <button type="button" class="btn" id="survey-start-btn">Start survey</button>
          <span class="ms-card__note">re-runs the model many times per tile &mdash; progress and cancel appear above</span>
        </div>
      </section>

      <div class="ms-main">
        <div class="sv-results" id="survey-cards" hidden>
          <section class="ms-card">
            <div class="ms-card__head">
              <h3 class="cube-panel-title">Surveyed model</h3>
              <span class="ms-badge" id="survey-badge"></span>
            </div>
            <div class="ms-meta" id="survey-meta"></div>
          </section>

          <div class="ms-layout">
            <div class="ms-pages">

            <section class="ms-page" data-page="prediction">
            <section class="ms-card ms-card--snug">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Prediction fit</h3>
                <span class="ms-card__note">per-pixel curve errors over the whole surveyed region</span>
              </div>
              <table class="ms-slots" id="survey-fit"></table>
              <p class="ms-wifacts" id="survey-detection"></p>
              <p class="ms-card__note ms-witablenote" id="survey-match-note" hidden>hungarian-matched per-parameter mean absolute error over every active ground-truth component</p>
              <table class="ms-slots" id="survey-match"></table>
            </section>
            </section>

            <section class="ms-page" data-page="attribution" hidden>
            <section class="ms-card">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Input attribution</h3>
                <span class="ms-card__note">share of |&part;output/&part;input| per input channel, gradients of the summed outputs over every pixel</span>
              </div>
              <div class="ms-bars" id="survey-bars"></div>
              <div class="ms-legend" id="survey-bars-legend"></div>
              <div class="ms-subhead">
                <h4 class="cube-panel-title">Mean spatial gradients</h4>
                <span class="ms-card__note" id="survey-grid-note"></span>
              </div>
              <div class="ms-ctlrow" id="survey-grid-ctl">
                <span class="ms-ctlrow__label">channel</span>
                <div class="ms-chiprow" id="survey-grid-channels"></div>
              </div>
              <div class="ms-gridwrap"><div class="ms-grid" id="survey-grid"></div></div>
              <div class="ms-subhead">
                <h4 class="cube-panel-title">Effective receptive field</h4>
                <span class="ms-card__note" id="survey-erf-note"></span>
              </div>
              <canvas id="survey-erf" width="640" height="180" hidden></canvas>
              <p class="ms-wifacts" id="survey-erf-facts"></p>
            </section>
            </section>

            <section class="ms-page" data-page="robustness" hidden>
            <section class="ms-card">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Channel ablation</h3>
                <span class="ms-card__note">zero one input channel at a time and re-run &mdash; mean causal damage over every pixel</span>
              </div>
              <div class="ms-abl" id="survey-ablation"></div>
            </section>

            <section class="ms-card ms-card--third">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Symmetry</h3>
                <span class="ms-card__note">mean disagreement between the model and its flipped self over every pixel</span>
              </div>
              <table class="ms-slots" id="survey-flips"></table>
            </section>

            <section class="ms-card ms-card--twothird">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Noise robustness</h3>
                <span class="ms-card__note">Gaussian noise added to every channel, one fresh draw per tile &mdash; mean damage vs &sigma;</span>
              </div>
              <canvas id="survey-noise" width="640" height="200"></canvas>
              <p class="ms-wifacts" id="survey-noise-facts"></p>
            </section>

            <section class="ms-card ms-card--snug">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Occlusion distance profile</h3>
                <span class="ms-card__note">mean damage to a pixel when a zero-patch lands at a given distance from it</span>
              </div>
              <canvas id="survey-occl" width="640" height="200"></canvas>
              <p class="ms-wifacts" id="survey-occl-facts"></p>
            </section>
            </section>

            <section class="ms-page" data-page="vitals" hidden>
            <section class="ms-card">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Layer vitals</h3>
                <span class="ms-card__note">activation statistics accumulated over every surveyed tile</span>
              </div>
              <p class="ms-wifacts" id="survey-vitals-summary"></p>
              <div class="ms-vitals"><table class="ms-slots ms-vitals__table" id="survey-vitals"></table></div>
            </section>
            </section>

            </div>

            <nav class="secnav" id="survey-nav" aria-label="Survey analyses"></nav>
          </div>
        </div>
      </div>`;

    this.refs = {
      strip      : this.root.querySelector("#survey-strip"),
      hint       : this.root.querySelector("#survey-hint"),
      progress   : this.root.querySelector("#survey-progress"),
      fill       : this.root.querySelector("#survey-progress-fill"),
      label      : this.root.querySelector("#survey-progress-label"),
      cancelBtn  : this.root.querySelector("#survey-cancel"),
      setup      : this.root.querySelector("#survey-setup"),
      setupRun   : this.root.querySelector("#survey-setup-run"),
      splitBtns  : this.root.querySelector("#survey-split"),
      deviceBtns : this.root.querySelector("#survey-device"),
      tilesInput : this.root.querySelector("#survey-tiles"),
      erfInput   : this.root.querySelector("#survey-erf-samples"),
      thrInput   : this.root.querySelector("#survey-threads"),
      startBtn   : this.root.querySelector("#survey-start-btn"),
      cards      : this.root.querySelector("#survey-cards"),
      nav        : this.root.querySelector("#survey-nav"),
      pages      : [...this.root.querySelectorAll(".ms-page")],
      badge      : this.root.querySelector("#survey-badge"),
      meta       : this.root.querySelector("#survey-meta"),
      fit        : this.root.querySelector("#survey-fit"),
      detection  : this.root.querySelector("#survey-detection"),
      matchNote  : this.root.querySelector("#survey-match-note"),
      match      : this.root.querySelector("#survey-match"),
      bars       : this.root.querySelector("#survey-bars"),
      barsLegend : this.root.querySelector("#survey-bars-legend"),
      gridNote   : this.root.querySelector("#survey-grid-note"),
      gridCtl    : this.root.querySelector("#survey-grid-ctl"),
      gridChips  : this.root.querySelector("#survey-grid-channels"),
      grid       : this.root.querySelector("#survey-grid"),
      erfNote    : this.root.querySelector("#survey-erf-note"),
      erf        : this.root.querySelector("#survey-erf"),
      erfFacts   : this.root.querySelector("#survey-erf-facts"),
      ablation   : this.root.querySelector("#survey-ablation"),
      flips      : this.root.querySelector("#survey-flips"),
      noise      : this.root.querySelector("#survey-noise"),
      noiseFacts : this.root.querySelector("#survey-noise-facts"),
      occl       : this.root.querySelector("#survey-occl"),
      occlFacts  : this.root.querySelector("#survey-occl-facts"),
      vitalsSum  : this.root.querySelector("#survey-vitals-summary"),
      vitals     : this.root.querySelector("#survey-vitals"),
    };

    this.refs.cancelBtn.addEventListener("click", async () => {
      await Api.post("/api/survey/cancel", {});
    });

    this._bindChips(this.refs.splitBtns, "split");
    this._bindChips(this.refs.deviceBtns, "device");
    this.refs.startBtn.addEventListener("click", () => this._startSurvey());

    this.pager = new SectionPager(this.refs.nav, this.refs.pages, SurveyView.PAGES, (key) => this._repaintPage(key));
  }

  _repaintPage(key) {
    if (!this.resultData) return;
    if (key === "attribution") this._renderAttribution(this.resultData);
    if (key === "robustness") {
      this._renderNoise(this.resultData);
      this._renderOcclusion(this.resultData);
    }
  }

  _bindChips(group, key) {
    group.querySelectorAll(".cube-space").forEach((btn) => {
      btn.addEventListener("click", () => {
        group.querySelectorAll(".cube-space").forEach((b) => b.classList.toggle("is-active", b === btn));
        this.cfg[key] = btn.dataset.value;
      });
    });
  }

  _readNumbers() {
    const read = (input, fallback, min) => {
      const value = parseInt(input.value, 10);
      const safe  = Number.isFinite(value) ? Math.max(min, value) : fallback;
      input.value = safe;
      return safe;
    };

    this.cfg.tiles   = read(this.refs.tilesInput, 64, 0);
    this.cfg.erf     = read(this.refs.erfInput, 24, 0);
    this.cfg.threads = read(this.refs.thrInput, 64, 1);
  }

  async _refreshRuns() {
    const data = await Api.get(`/api/survey/runs?base=${encodeURIComponent(ResultsSources.runs())}`);

    if (!data || !data.ok) {
      this.refs.hint.textContent = (data && data.error) || "Backend unreachable.";
      return;
    }

    this.runs = data.runs || [];
    this._renderStrip();

    if (!this.runs.length) {
      this.refs.hint.textContent = `No loadable runs under ${data.root}. Train a backbone or dual run, or point the runs directory in the Results tab at the right place.`;
      return;
    }

    if (this.renderedPath === null) this.refs.hint.textContent = "Pick a run to survey the entire test region. This re-runs the model many times per tile and takes a few minutes.";
  }

  _renderStrip() {
    if (!this.runStrip) {
      this.runStrip = new RunStrip(this.refs.strip, {
        stateFor : (run) => run.id === this.selectedId,
        onPick   : (run) => this._pick(run),
      });
    }
    this.runStrip.render(this.runs);
  }

  _pick(run) {
    this.selectedId = run.id;
    this._renderStrip();

    this.refs.setupRun.textContent = run.run || run.id;
    this.refs.setup.hidden         = false;
    this.refs.hint.textContent     = "Set the survey up below, then start it.";
  }

  async _startSurvey() {
    if (!this.selectedId) return;
    this._readNumbers();

    const out = await Api.post("/api/survey/start", {
      path        : this.selectedId,
      split       : this.cfg.split,
      device      : this.cfg.device,
      tiles       : this.cfg.tiles,
      erf_samples : this.cfg.erf,
      threads     : this.cfg.threads,
    });
    if (!out.ok) {
      this.refs.hint.textContent = out.error || "survey failed to start";
      return;
    }

    this.renderedPath      = null;
    this.refs.setup.hidden = true;
    this.refs.cards.hidden = true;
    this._pollStatus();
  }

  async _pollStatus() {
    const token = ++this.pollToken;
    let failures = 0;
    const tick  = async () => {
      if (token !== this.pollToken) return;

      const st = await Api.get("/api/survey/status");

      if (!st.state) {
        failures += 1;
        this.refs.hint.textContent = `survey status unavailable — ${st.error} (attempt ${failures} of ${SurveyView.STATUS_RETRIES})`;
        if (failures < SurveyView.STATUS_RETRIES) setTimeout(tick, 1000);
        else this.refs.progress.hidden = true;
        return;
      }
      failures = 0;

      if (st.state === "running") {
        this.refs.progress.hidden   = false;
        this.refs.setup.hidden      = true;
        this.refs.fill.style.width  = `${Math.round((st.progress || 0) * 100)}%`;
        this.refs.label.textContent = `${st.stage || "running"} — ${Math.round((st.progress || 0) * 100)}%`;
        if (st.path && st.path !== this.selectedId) this._adoptPath(st.path);
        setTimeout(tick, 800);
        return;
      }

      this.refs.progress.hidden = true;

      if (st.state === "error") {
        this.refs.hint.textContent = st.error;
        this.refs.setup.hidden     = this.selectedId === null;
        return;
      }
      if (st.state === "cancelled") {
        this.refs.hint.textContent = "Survey cancelled.";
        this.refs.setup.hidden     = this.selectedId === null;
        return;
      }
      if (st.state === "done") {
        if (st.path && st.path !== this.selectedId) this._adoptPath(st.path);
        if (this.renderedPath !== st.path) await this._loadResult(st.path);
      }
    };
    tick();
  }

  _adoptPath(path) {
    this.selectedId = path;
    this.refs.setupRun.textContent = path.split("/").pop();
    this._renderStrip();
  }

  async _loadResult(path) {
    const out = await Api.get("/api/survey/result");
    if (!out || !out.ok) {
      this.refs.hint.textContent = (out && out.error) || "Backend unreachable.";
      return;
    }

    this.renderedPath = path;
    this._renderResult(out.result);
  }

  _seconds(value) {
    const minutes = Math.floor(value / 60);
    return minutes ? `${minutes} min ${Math.round(value - minutes * 60)} s` : `${value.toFixed(1)} s`;
  }

  _renderResult(r) {
    this.resultData  = r;
    this.gridChannel = -1;

    this.refs.hint.textContent = `Surveyed ${r.backbone} over the ${r.split} split.`;
    this.refs.cards.hidden     = false;

    this.refs.badge.textContent = r.backbone;
    this.refs.meta.innerHTML = [
      `<span class="ms-chip"><b>split</b> ${SharedCharts.esc(r.split)}</span>`,
      `<span class="ms-chip"><b>region</b> ${r.region[0]}&times;${r.region[1]} px</span>`,
      `<span class="ms-chip"><b>surveyed</b> ${r.coverage.tiles} of ${r.coverage.total_tiles} tiles</span>`,
      `<span class="ms-chip"><b>pixels</b> ${r.coverage.pixels.toLocaleString("en-US")}</span>`,
      `<span class="ms-chip"><b>window</b> ${r.patch[0]}&times;${r.patch[1]} px</span>`,
      `<span class="ms-chip"><b>K</b> ${r.n_gaussians}</span>`,
      `<span class="ms-chip"><b>in</b> ${r.channels.length} ch</span>`,
      `<span class="ms-chip"><b>took</b> ${this._seconds(r.seconds)}</span>`,
    ].join("");

    this._renderFit(r);
    this._renderAttribution(r);
    this._renderAblation(r);
    this._renderFlips(r);
    this._renderNoise(r);
    this._renderOcclusion(r);

    SharedCharts.vitalsTable(this.refs.vitalsSum, this.refs.vitals, r.vitals, {});
  }

  _statRow(label, stat) {
    return `
      <tr>
        <td class="ms-slots__key">${label}</td>
        <td>${stat.mean.toExponential(2)}</td>
        <td>${stat.median.toExponential(2)}</td>
        <td>${stat.p90.toExponential(2)}</td>
      </tr>`;
  }

  _renderFit(r) {
    const head = `<thead><tr><th class="ms-slots__key">CURVE MSE</th><th>MEAN</th><th>MEDIAN</th><th>P90</th></tr></thead>`;
    const rows = [];
    if (r.fit.curve_mse_gt) rows.push(this._statRow("vs ground truth", r.fit.curve_mse_gt));
    rows.push(this._statRow("vs raw cube", r.fit.curve_mse_raw));
    this.refs.fit.innerHTML = `${head}<tbody>${rows.join("")}</tbody>`;

    if (!r.detection) {
      this.refs.detection.innerHTML  = "no ground truth on this split &mdash; detection and matched errors need GT parameters";
      this.refs.matchNote.hidden     = true;
      this.refs.match.innerHTML      = "";
      return;
    }

    this.refs.detection.innerHTML = [
      `mean active slots <b>${r.detection.mean_pred_active.toFixed(2)}</b> predicted vs <b>${r.detection.mean_gt_active.toFixed(2)}</b> in the ground truth`,
      `exact slot-count match at <b>${(r.detection.count_match_frac * 100).toFixed(1)}%</b> of pixels`,
    ].join(" &middot; ");

    this.refs.matchNote.hidden = false;
    this.refs.match.innerHTML = `
      <thead><tr><th>AMP MAE</th><th>&mu; MAE [m]</th><th>&sigma; MAE [m]</th><th>MATCHED PX&middot;SLOTS</th></tr></thead>
      <tbody><tr>
        <td>${r.matched.amp.toFixed(4)}</td>
        <td>${r.matched.mu.toFixed(3)}</td>
        <td>${r.matched.sigma.toFixed(3)}</td>
        <td>${r.matched.n_matched.toLocaleString("en-US")}</td>
      </tr></tbody>`;
  }

  _renderAttribution(r) {
    SharedCharts.familyBars(this.refs.bars, this.refs.barsLegend, r.channels, r.attribution.families, "no dependence anywhere in the region");
    this._renderGradients(r);

    const live = r.erf.families.filter((f) => !f.dead);
    this.refs.erfNote.innerHTML = `cumulative gradient energy vs radius, averaged over ${r.erf.samples} sampled pixels`;

    if (r.erf.samples === 0) {
      this.refs.erf.hidden = true;
      this.refs.erfFacts.innerHTML = "receptive-field sampling was turned off for this survey";
      return;
    }
    if (!live.length) {
      this.refs.erf.hidden = true;
      this.refs.erfFacts.innerHTML = "no gradient energy at any sampled pixel";
      return;
    }

    this.refs.erf.hidden = false;
    SharedCharts.lineChart(
      this.refs.erf,
      live[0].radii,
      live.map((f) => ({ values: f.cumulative, color: SharedCharts.FAMILY_COLORS[f.family], width: 1.5, label: f.family })),
      { xUnit: "px" },
    );

    this.refs.erfFacts.innerHTML = r.erf.families.map((f) => f.dead
      ? `${f.family} &mdash; no dependence`
      : `${f.family} r<sub>50</sub> <b>${f.r50} px</b> / r<sub>90</sub> <b>${f.r90} px</b>`
    ).join(" &middot; ");
  }

  _renderGradients(r) {
    const g = r.gradients;

    if (g.samples === 0) {
      this.refs.gridNote.innerHTML  = "gradient sampling was turned off for this survey (0 ERF samples)";
      this.refs.gridCtl.hidden      = true;
      this.refs.gridChips.innerHTML = "";
      this.refs.grid.innerHTML      = "";
      this.refs.grid.className      = "ms-grid";
      return;
    }

    this.refs.gridNote.innerHTML = `|&part;output/&part;input| windows averaged over ${g.samples} sampled pixels, each centred on its pixel`;
    this.refs.gridCtl.hidden     = false;
    this._renderGridChips();
    SharedCharts.gradientGrid(this.refs.grid, { channels: r.channels, families: g.families, center: g.center, patch: g.patch }, this.gridChannel);
  }

  _renderGridChips() {
    this.refs.gridChips.innerHTML = "";

    const pick = (index) => () => {
      this.gridChannel = index;
      this._renderGradients(this.resultData);
    };

    this.refs.gridChips.appendChild(SharedCharts.chip("all channels", this.gridChannel === -1, pick(-1)));
    this.resultData.channels.forEach((label, index) => {
      this.refs.gridChips.appendChild(SharedCharts.chip(label, this.gridChannel === index, pick(index)));
    });
  }

  _renderAblation(r) {
    SharedCharts.ablationRows(this.refs.ablation, r.ablation.channels, r.fit.base_power, (c) =>
      c.flips_frac > 0 ? ` &middot; slots flip at ${(c.flips_frac * 100).toFixed(1)}% of px` : ""
    );
  }

  _renderFlips(r) {
    const power = r.fit.base_power;
    const head  = `<thead><tr><th class="ms-slots__key">FLIP</th><th>MEAN &Delta; CURVE MSE</th><th>% OF POWER</th></tr></thead>`;

    const rows = r.flips.filter((e) => e.flip !== "none").map((e) => `
      <tr>
        <td class="ms-slots__key">${SurveyView.FLIP_LABELS[e.flip]}</td>
        <td>${e.delta_mse.toExponential(2)}</td>
        <td>${power > 0 ? ((e.delta_mse / power) * 100).toFixed(2) : "&ndash;"}</td>
      </tr>`);

    this.refs.flips.innerHTML = `${head}<tbody>${rows.join("")}</tbody>`;
  }

  _renderNoise(r) {
    SharedCharts.lineChart(
      this.refs.noise,
      r.noise.map((p) => p.value),
      [{ values: r.noise.map((p) => p.delta_mse), color: SharedCharts.SERIES_COLORS.perturbed, width: 1.6, label: "Δ curve MSE" }],
      { xUnit: "" },
    );

    const last = r.noise[r.noise.length - 1];
    const rel  = r.fit.base_power > 0 ? ` (${((last.delta_mse / r.fit.base_power) * 100).toFixed(1)}% of power)` : "";
    this.refs.noiseFacts.innerHTML = `mean &Delta; curve MSE at &sigma; ${last.value.toFixed(1)}: <b>${last.delta_mse.toExponential(2)}</b>${rel}`;
  }

  _renderOcclusion(r) {
    SharedCharts.lineChart(
      this.refs.occl,
      r.occlusion.distance,
      [{ values: r.occlusion.delta, color: SharedCharts.SERIES_COLORS.pred, width: 1.6, label: "Δ curve MSE" }],
      { xUnit: "px" },
    );

    const first = r.occlusion.delta[0];
    const half  = r.occlusion.delta.findIndex((v) => v < first * 0.5);
    const reach = half < 0 ? null : r.occlusion.distance[half];
    const tail  = reach === null
      ? "and stays high across the whole window"
      : `and halves within ${reach.toFixed(0)} px of the pixel`;
    this.refs.occlFacts.innerHTML = `occluder ${r.occlusion.occluder[0]}&times;${r.occlusion.occluder[1]} px &middot; damage peaks at <b>${first.toExponential(2)}</b> under the occluder ${tail}`;
  }
}

window.SurveyView = SurveyView;
