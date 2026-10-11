"use strict";

class CubeSweep {
  constructor(refs, host) {
    this.host = host;
    this.axis = refs.axis;
    this.grid = refs.grid;
    this.atLabel = refs.at;
    this.fill = refs.fill;
    this.input = refs.input;
    this.rangeLabel = refs.range;
    this.playBtn = refs.play;
    this.panels = refs.panels || [];

    this.idx = 0;
    this.steps = 1;
    this.playing = false;
    this.token = 0;
    this.frameMs = 70;

    if (this.grid)    this.grid.addEventListener("wheel", (ev) => this._onWheel(ev), { passive: false });
    if (this.input)   this.input.addEventListener("change", () => this._onManual());
    if (this.playBtn) this.playBtn.addEventListener("click", () => this.toggle());
  }

  configure() {
    const meta = this.host.meta;
    this.steps = this._axisSteps(meta);
    this.idx   = Math.floor((this.steps - 1) / 2);

    if (this.input) {
      this.input.min = 1;
      this.input.max = this.steps;
      this.input.value = this.idx + 1;
    }
    if (this.rangeLabel) this.rangeLabel.textContent = this._rangeText(meta);

    this.panels.forEach((panel) => {
      panel.root.hidden = !meta.sources.includes(panel.source);
      panel.bitmap = null;
    });

    this._syncRows();
    this.stop();
  }

  holds(bitmap) {
    return this.panels.some((panel) => panel.bitmap === bitmap);
  }

  applyVisibility() {
    this.panels.forEach((panel) => {
      panel.root.hidden = !this.host.visible.has(panel.source);
    });

    this._syncRows();
    if (this.host.meta && !this.playing && this.host.view === this.axis) this._renderFrame();
  }

  _syncRows() {
    if (!this.grid) return;
    const shown = this.panels.filter((panel) => !panel.root.hidden).length;
    this.grid.style.setProperty("--cube-rows", String(Math.max(1, shown)));
  }

  syncSpace() {
    if (this.host.meta && !this.playing) this._renderFrame();
  }

  render() {
    if (this.host.meta) this._renderFrame();
  }

  play() {
    if (!this.host.meta || this.playing) return;
    if (this.steps < 2) { this._renderFrame(); return; }
    this.playing = true;
    this._syncPlayBtn();
    this._loop();
  }

  stop() {
    this.playing = false;
    this._syncPlayBtn();
  }

  toggle() {
    if (this.playing) this.stop();
    else this.play();
  }

  async _loop() {
    while (this.playing && this.host.meta && this.host.selectedId) {
      const frame = this._renderFrame();
      const next  = (this.idx + 1) % this.steps;

      this._prefetch(next);
      await Promise.all([frame, this._sleep(this.frameMs)]);
      if (!this.playing) break;

      this.idx = next;
    }
  }

  _prefetch(idx) {
    this.panels.filter((panel) => !panel.root.hidden).forEach((panel) => this.host.cacheBitmap(this._url(panel.source, idx)));
  }

  _onWheel(ev) {
    if (!this.host.meta) return;
    ev.preventDefault();
    this.stop();

    const step = ev.deltaY > 0 ? 1 : -1;
    const next = Math.min(this.steps - 1, Math.max(0, this.idx + step));
    if (next === this.idx) return;

    this.idx = next;
    this._renderFrame();
  }

  _onManual() {
    if (!this.host.meta || !this.input) return;
    this.stop();
    this.idx = this.host._clampInt(Number(this.input.value) - 1, this.steps);
    this._renderFrame();
  }

  _renderFrame() {
    if (!this.host.meta) return Promise.resolve();

    this.token += 1;
    const token = this.token;
    this._updateLabels();

    const jobs = this.panels.filter((panel) => !panel.root.hidden).map((panel) => this._fetch(panel, this.idx, token));
    return Promise.all(jobs);
  }

  _updateLabels() {
    const frac = this.steps > 1 ? this.idx / (this.steps - 1) : 0;

    if (this.fill) this.fill.style.width = `${frac * 100}%`;
    if (this.input && document.activeElement !== this.input) this.input.value = this.idx + 1;
    if (this.atLabel) this.atLabel.textContent = this._atText(frac);
  }

  async _fetch(panel, idx, token) {
    const url = this._url(panel.source, idx);
    const skeletonTimer = panel.bitmap ? null : setTimeout(() => panel.root.classList.add("is-loading"), 120);

    try {
      const bitmap = await this.host.cacheBitmap(url);
      if (!bitmap || token !== this.token) return;

      panel.bitmap = bitmap;
      this._paint(panel);
    } finally {
      if (skeletonTimer) clearTimeout(skeletonTimer);
      panel.root.classList.remove("is-loading");
    }
  }

  _paint(panel) {
    const bitmap = panel.bitmap;
    const canvas = panel.canvas;

    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
    }

    canvas.getContext("2d").drawImage(bitmap, 0, 0);
  }

  _url(source, idx) {
    const id    = encodeURIComponent(this.host.selectedId);
    const space = this.host.space;

    const cmap = this.host.cmap;

    if (this.axis === "z") {
      const frac = this.steps > 1 ? idx / (this.steps - 1) : 0;
      return `/api/cubes/plane?id=${id}&source=${source}&frac=${frac}&space=${space}&cmap=${cmap}`;
    }
    if (this.axis === "row") {
      return `/api/cubes/slice?id=${id}&source=${source}&axis=row&row=${idx}&col=0&space=${space}&cmap=${cmap}`;
    }
    return `/api/cubes/slice?id=${id}&source=${source}&axis=col&row=0&col=${idx}&space=${space}&cmap=${cmap}`;
  }

  _axisSteps(meta) {
    if (this.axis === "z") {
      const primary = meta.sources.includes("pred") ? "pred" : meta.sources[0];
      return Math.max(1, meta.n_z[primary] || 1);
    }
    if (this.axis === "row") return Math.max(1, meta.n_rows);
    return Math.max(1, meta.n_cols);
  }

  _rangeText(meta) {
    if (this.axis === "z") return `1–${this.steps} · ${this.host._fmt(meta.x_min)} … ${this.host._fmt(meta.x_max)}`;
    return `1–${this.steps} · index 0–${this.steps - 1}`;
  }

  _atText(frac) {
    if (this.axis === "z") {
      const zValue = this.host.meta.x_min + frac * (this.host.meta.x_max - this.host.meta.x_min);
      return `z ≈ ${this.host._fmt(zValue)} · bin ${this.idx + 1} / ${this.steps} · scroll or play to sweep`;
    }
    return `${this.axis} index ${this.idx} · bin ${this.idx + 1} / ${this.steps} · scroll or play to sweep`;
  }

  _syncPlayBtn() {
    if (!this.playBtn) return;
    this.playBtn.classList.toggle("is-playing", this.playing);
    this.playBtn.textContent = this.playing ? "Pause" : "Play";
  }

  _sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

class CubeParams {
  static SOURCE_LABELS = { pred: "pred", gt: "gt", error: "|pred − gt|" };
  static FIELD_LABELS  = { amp: "amplitude", mu: "mu", sigma: "sigma", count: "active slots" };

  constructor(refs, host) {
    this.host = host;
    this.sourceEl = refs.source;
    this.fieldEl = refs.field;
    this.slotEl = refs.slot;
    this.atEl = refs.at;
    this.canvas = refs.canvas;
    this.cross = refs.cross;
    this.cbar = refs.cbar;
    this.minEl = refs.min;
    this.maxEl = refs.max;
    this.coordsEl = refs.coords;
    this.tableEl = refs.table;
    this.openBtn = refs.open;

    this.meta = null;
    this.source = "pred";
    this.field = "amp";
    this.slot = 0;
    this.token = 0;
    this.pixelSeq = 0;
    this.picked = null;
    this.adopted = 0;

    this.fieldEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.addEventListener("click", () => this._setField(btn.dataset.field));
    });

    this.canvas.addEventListener("click", (ev) => this._onClick(ev));
    this.openBtn.addEventListener("click", () => this._openCuts());
  }

  configure(meta) {
    this.meta = meta.params || null;
    this.picked = null;
    this.adopted = 0;
    this.cross.hidden = true;
    this.openBtn.hidden = true;
    this.tableEl.innerHTML = "";
    this.coordsEl.textContent = "Click the map to read every Gaussian slot at that pixel.";

    if (!this.meta) return;

    const sources = [...this.meta.sources, ...(this.meta.error ? ["error"] : [])];
    if (!sources.includes(this.source)) this.source = sources[0];
    this.slot = Math.min(this.slot, this.meta.n_slots - 1);

    this._renderSourceBtns(sources);
    this._renderSlotBtns();
  }

  render() {
    if (!this.meta) return;
    this._adoptFocus();
    this._syncBtns();
    this._fetchMap();
    this._updateLegend();
    if (this.picked) this._fetchPixel(this.picked.row, this.picked.col);
  }

  _adoptFocus() {
    const host = this.host;
    if (!host.focus || this.adopted === host.focusSeq) return;

    this.adopted = host.focusSeq;
    const point  = host.focusPoint();
    if (!point) return;

    this.picked           = point;
    this.cross.hidden     = false;
    this.cross.style.left = `${point.fx * 100}%`;
    this.cross.style.top  = `${point.fy * 100}%`;
  }

  _setSource(source) {
    if (source === this.source) return;
    this.source = source;
    this.render();
  }

  _setField(field) {
    if (field === this.field) return;
    this.field = field;
    this.render();
  }

  _setSlot(slot) {
    if (slot === this.slot) return;
    this.slot = slot;
    this.render();
  }

  _renderSourceBtns(sources) {
    this.sourceEl.innerHTML = "";
    sources.forEach((source) => {
      const btn          = document.createElement("button");
      btn.type           = "button";
      btn.className      = "cube-space";
      btn.dataset.source = source;
      btn.textContent    = CubeParams.SOURCE_LABELS[source] || source;
      btn.addEventListener("click", () => this._setSource(source));
      this.sourceEl.appendChild(btn);
    });
  }

  _renderSlotBtns() {
    this.slotEl.innerHTML = "";
    for (let k = 0; k < this.meta.n_slots; k++) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "cube-space";
      btn.dataset.slot = String(k);
      btn.textContent = `slot ${k}`;
      btn.addEventListener("click", () => this._setSlot(k));
      this.slotEl.appendChild(btn);
    }
  }

  _syncBtns() {
    this.sourceEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.source === this.source);
    });
    this.fieldEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.field === this.field);
    });
    this.slotEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.classList.toggle("is-active", Number(btn.dataset.slot) === this.slot);
      btn.disabled = this.field === "count";
    });

    const slotText        = this.field === "count" ? "all slots" : `slot ${this.slot}`;
    this.atEl.textContent = `${CubeParams.SOURCE_LABELS[this.source]} · ${CubeParams.FIELD_LABELS[this.field]} · ${slotText}`;
  }

  async _fetchMap() {
    this.token += 1;
    const token = this.token;
    const url = `/api/cubes/param_map?id=${encodeURIComponent(this.host.selectedId)}&source=${this.source}&field=${this.field}&slot=${this.slot}`;

    try {
      const res = await fetch(url);
      if (!res.ok || token !== this.token) return;

      const bitmap = await createImageBitmap(await res.blob());
      if (token !== this.token) { if (bitmap.close) bitmap.close(); return; }

      if (this.canvas.width !== bitmap.width || this.canvas.height !== bitmap.height) {
        this.canvas.width = bitmap.width;
        this.canvas.height = bitmap.height;
      }
      this.canvas.getContext("2d").drawImage(bitmap, 0, 0);
    } catch (e) {
    }
  }

  _updateLegend() {
    const key = this.source === "error" ? `error_${this.field}` : this.field;
    const range = this.meta.ranges[key];
    if (!range) return;

    this.cbar.src = `/api/cubes/param_cbar?id=${encodeURIComponent(this.host.selectedId)}&source=${this.source}&field=${this.field}`;
    this.minEl.textContent = this.host._fmt(range[0]);
    this.maxEl.textContent = this.host._fmt(range[1]);
  }

  _onClick(ev) {
    if (!this.meta || !this.host.meta) return;

    const rect = this.canvas.getBoundingClientRect();
    const fx   = (ev.clientX - rect.left) / rect.width;
    const fy   = (ev.clientY - rect.top) / rect.height;
    const row  = Math.min(this.host.meta.n_rows - 1, Math.max(0, Math.floor(fy * this.host.meta.n_rows)));
    const col  = Math.min(this.host.meta.n_cols - 1, Math.max(0, Math.floor(fx * this.host.meta.n_cols)));

    this.picked           = { row, col, fx, fy };
    this.cross.hidden     = false;
    this.cross.style.left = `${fx * 100}%`;
    this.cross.style.top  = `${fy * 100}%`;

    this.host.setFocus(row, col);
    this.adopted = this.host.focusSeq;
    this._fetchPixel(row, col);
  }

  async _fetchPixel(row, col) {
    const seq  = ++this.pixelSeq;
    const data = await Api.get(`/api/cubes/params_at?id=${encodeURIComponent(this.host.selectedId)}&row=${row}&col=${col}`);
    if (seq !== this.pixelSeq || !data || !data.ok) return;

    this.coordsEl.textContent = `row = ${data.row} · col = ${data.col} · threshold ${data.threshold}`;
    this.openBtn.hidden       = false;
    this._renderTable(data);
  }

  _renderTable(data) {
    const sources = Object.keys(data.sources);

    let html = `<table class="cube-metrics cube-metrics--params"><thead><tr><th scope="col">slot</th>`;
    sources.forEach((source) => { html += `<th scope="col" colspan="3">${SharedCharts.esc(source)}</th>`; });
    html += `</tr><tr><th></th>`;
    sources.forEach(() => { html += `<th>amp</th><th>mu</th><th>sigma</th>`; });
    html += `</tr></thead><tbody>`;

    for (let k = 0; k < data.n_slots; k++) {
      html += `<tr><th scope="row">${k}</th>`;
      sources.forEach((source) => {
        const slot = data.sources[source][k];
        const cls = slot.active ? "" : ` class="is-inactive"`;
        const amp = slot.amp === null ? "–" : this.host._fmt(slot.amp);
        const mu = slot.active && slot.mu !== null ? this.host._fmt(slot.mu) : "–";
        const sigma = slot.active && slot.sigma !== null ? this.host._fmt(slot.sigma) : "–";
        html += `<td${cls}>${amp}</td><td${cls}>${mu}</td><td${cls}>${sigma}</td>`;
      });
      html += `</tr>`;
    }

    html += `</tbody></table>`;
    this.tableEl.innerHTML = html;
  }

  _openCuts() {
    if (!this.picked) return;
    this.host._setView("explorer");
    this.host._follow(this.picked, true);
    this.host._enterSlices(this.picked);
  }
}

class CubeMetrics {
  static COVERAGE_STEP    = 5;
  static COVERAGE_MIN     = 5;
  static COVERAGE_STEP_MS = 420;

  constructor(refs, host) {
    this.host = host;
    this.layerEl = refs.layer;
    this.rangeWrapEl = refs.rangeWrap;
    this.classesEl = refs.classes;
    this.vminEl = refs.vmin;
    this.vmaxEl = refs.vmax;
    this.resetEl = refs.reset;
    this.modeEl = refs.mode;
    this.thrEl = refs.thr;
    this.thrValEl = refs.thrVal;
    this.alphaEl = refs.alpha;
    this.img = refs.img;
    this.cross = refs.cross;
    this.cbar = refs.cbar;
    this.minEl = refs.min;
    this.maxEl = refs.max;
    this.coordsEl = refs.coords;
    this.readoutEl = refs.readout;
    this.openBtn = refs.open;
    this.selCovEl = refs.selCov;
    this.selValEl = refs.selVal;
    this.selTabEl = refs.selTab;
    this.selMaskEl = refs.selMask;
    this.selPlayEl = refs.selPlay;
    this.histEl = refs.hist;
    this.histNoteEl = refs.histNote;
    this.rectEl = refs.rect;
    this.regionHintEl = refs.regionHint;
    this.regionTabEl = refs.regionTab;
    this.regionWrapEl = refs.regionWrap;
    this.regionProfEl = refs.regionProf;

    this.layers = [];
    this.layer = null;
    this.mode = "all";
    this.alpha = 0.75;
    this.hist = null;
    this.histKey = null;
    this.picked = null;
    this.adopted = 0;
    this.debounceTimer = null;
    this.hoverQueued = null;
    this.hoverFetching = false;
    this.selectiveTimer = null;
    this.maskWindow = null;
    this.selPlayTimer = null;
    this.selPlayDir = -1;
    this.rectDrag = null;
    this.suppressClick = false;
    this.regionSeq = 0;

    if (this.selCovEl) this.selCovEl.addEventListener("input", () => this._onCoverage());
    this.selMaskEl.addEventListener("change", () => this._onMaskToggle());
    this.selPlayEl.addEventListener("click", () => this._toggleCoveragePlay());

    this.vminEl.addEventListener("change", () => this._refresh());
    this.vmaxEl.addEventListener("change", () => this._refresh());
    this.resetEl.addEventListener("click", () => this._resetRange());
    this.modeEl.addEventListener("change", () => this._setMode(this.modeEl.value));
    this.thrEl.addEventListener("input", () => this._onThreshold());
    this.alphaEl.addEventListener("input", () => this._onAlpha());
    this.img.addEventListener("mousemove", (ev) => this._onHover(ev));
    this.img.addEventListener("click", (ev) => this._onClick(ev));
    this.img.addEventListener("mousedown", (ev) => this._onRectStart(ev));
    window.addEventListener("mousemove", (ev) => this._onRectMove(ev));
    window.addEventListener("mouseup", () => this._onRectEnd());
    this.openBtn.addEventListener("click", () => this._openCuts());
    refs.download.addEventListener("click", () => this._downloadOverlay());
  }

  configure(meta) {
    this.layers = meta.metric_maps || [];
    this.hist = null;
    this.histKey = null;
    this.picked = null;
    this.adopted = 0;
    this.maskWindow = null;
    this._stopCoveragePlay();
    this.selMaskEl.checked = false;
    this.cross.hidden = true;
    this.openBtn.hidden = true;
    this.readoutEl.innerHTML = "";
    this.coordsEl.textContent = "Hover the map to read the value under the cursor; click to lock a pixel and read every layer.";
    this.rectDrag = null;
    this.rectEl.hidden = true;
    this.regionTabEl.innerHTML = "";
    this.regionWrapEl.hidden = true;
    this.regionHintEl.textContent = "Shift-drag a rectangle on the map to compare it against the scene.";

    if (!this.layers.length) {
      this.layer = null;
      return;
    }

    if (!this.layer || !this.layers.some((l) => l.key === this.layer.key)) this.layer = this.layers[0];
    else this.layer = this.layers.find((l) => l.key === this.layer.key);

    this._renderLayerBtns();
    this._resetControls();
  }

  render() {
    if (!this.layer) return;
    this._adoptFocus();
    this._syncBtns();
    this._refresh();
    this._fetchHist();
    this._onCoverage();
  }

  _adoptFocus() {
    const host = this.host;
    if (!host.focus || this.adopted === host.focusSeq || !this.layer) return;

    this.adopted = host.focusSeq;
    const point  = host.focusPoint();
    if (!point) return;

    this.picked               = point;
    this.cross.hidden         = false;
    this.cross.style.left     = `${point.fx * 100}%`;
    this.cross.style.top      = `${point.fy * 100}%`;
    this.coordsEl.textContent = `locked at row = ${point.row} · col = ${point.col} · click again to unlock`;
    this.openBtn.hidden       = false;

    this._fetchReadout(point.row, point.col);
  }

  async _fetchHist() {
    if (this.histKey === this.layer.key && this.hist) {
      this._drawHist();
      return;
    }

    const cubeId = this.host.selectedId;
    const key    = this.layer.key;
    const data   = await Api.get(`/api/cubes/metric_hist?id=${encodeURIComponent(cubeId)}&key=${encodeURIComponent(key)}`);
    if (!this.layer || this.layer.key !== key || this.host.selectedId !== cubeId) return;

    this.hist    = data && data.ok ? data : null;
    this.histKey = key;
    this._drawHist();
  }

  _drawHist() {
    const canvas = this.histEl;
    const dpr    = window.devicePixelRatio || 1;
    const w      = canvas.clientWidth || 240;
    const h      = canvas.clientHeight || 92;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width  = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    this.histNoteEl.hidden = true;
    if (!this.hist || this.histKey !== this.layer.key) return;

    const counts = this.hist.counts;
    const peak   = Math.max(...counts, 1);
    const pad    = 3;
    const innerH = h - 2 * pad;

    if (this.hist.kind === "classes") {
      const classes = this.layer.classes;
      const bw      = (w - 2 * pad) / counts.length;
      counts.forEach((count, i) => {
        const bh = (count / peak) * innerH;
        ctx.fillStyle = classes[i].color;
        ctx.fillRect(pad + i * bw + 1, h - pad - bh, Math.max(1, bw - 2), bh);
      });
      return;
    }

    const edges = this.hist.edges;
    const lo    = edges[0];
    const span  = edges[edges.length - 1] - lo || 1;
    const xAt   = (value) => pad + ((value - lo) / span) * (w - 2 * pad);

    const vmin = Number(this.vminEl.value);
    const vmax = Number(this.vmaxEl.value);
    if (Number.isFinite(vmin) && Number.isFinite(vmax) && vmax > vmin) {
      const x0 = Math.max(pad, xAt(vmin));
      const x1 = Math.min(w - pad, xAt(vmax));
      ctx.fillStyle = CubeView.tint(this.host.colors.accent, 0.10);
      if (x1 > x0) ctx.fillRect(x0, pad, x1 - x0, innerH);
    }

    ctx.fillStyle = CubeView.tint(this.host.colors.accent, 0.72);
    counts.forEach((count, i) => {
      const bh = (count / peak) * innerH;
      const x0 = xAt(edges[i]);
      const x1 = xAt(edges[i + 1]);
      ctx.fillRect(x0, h - pad - bh, Math.max(1, x1 - x0 - 0.6), bh);
    });

    if (this.mode !== "all") {
      const x = xAt(this._threshold());
      if (x >= pad && x <= w - pad) {
        ctx.strokeStyle = CubeView.tint(this.host.colors.warn, 0.95);
        ctx.lineWidth   = 1.6;
        ctx.beginPath();
        ctx.moveTo(x, pad);
        ctx.lineTo(x, h - pad);
        ctx.stroke();
      }
    }

    if (this.hist.n_low || this.hist.n_high) {
      this.histNoteEl.hidden      = false;
      this.histNoteEl.textContent = `${this.hist.n_low.toLocaleString()} px below and ${this.hist.n_high.toLocaleString()} px above the shown range`;
    }
  }

  _setLayer(key) {
    if (this.layer && key === this.layer.key) return;
    this.layer = this.layers.find((l) => l.key === key);
    this._resetControls();
    if (this.selTabEl) this.selTabEl.innerHTML = "";
    this.render();
    this._onCoverage();
  }

  _onCoverage() {
    if (!this.selCovEl || !this.layer) return;
    const pct = Number(this.selCovEl.value);
    this.selValEl.textContent = `${pct}%`;

    clearTimeout(this.selectiveTimer);
    this.selectiveTimer = setTimeout(() => this._fetchSelective(pct / 100), 220);
  }

  _onMaskToggle() {
    if (!this.selMaskEl.checked) {
      this.maskWindow = null;
      this._stopCoveragePlay();
      this._refresh();
      return;
    }
    this._onCoverage();
  }

  _toggleCoveragePlay() {
    if (this.selPlayTimer !== null) {
      this._stopCoveragePlay();
      return;
    }
    if (!this.layer) return;

    this.selMaskEl.checked = true;
    this.selPlayDir = -1;
    this.selPlayEl.textContent = "Pause";
    this.selPlayTimer = setInterval(() => this._coverageStep(), CubeMetrics.COVERAGE_STEP_MS);
  }

  _coverageStep() {
    if (this.host.view !== "metrics") {
      this._stopCoveragePlay();
      return;
    }

    let value = Number(this.selCovEl.value) + this.selPlayDir * CubeMetrics.COVERAGE_STEP;
    if (value <= CubeMetrics.COVERAGE_MIN) {
      value = CubeMetrics.COVERAGE_MIN;
      this.selPlayDir = 1;
    }
    if (value >= 100) {
      value = 100;
      this.selPlayDir = -1;
    }
    this.selCovEl.value = String(value);
    this._onCoverage();
  }

  _stopCoveragePlay() {
    if (this.selPlayTimer !== null) clearInterval(this.selPlayTimer);
    this.selPlayTimer = null;
    this.selPlayEl.textContent = "Play";
  }

  leave() {
    this._stopCoveragePlay();
  }

  async _fetchSelective(coverage) {
    if (!this.layer) return;
    try {
      const data = await Api.get(
        `/api/cubes/selective?id=${encodeURIComponent(this.host.selectedId)}&key=${encodeURIComponent(this.layer.key)}&coverage=${coverage}`
      );
      if (data.ok && this.selMaskEl.checked) {
        this.maskWindow = data.direction === "high"
          ? { keep_min: data.threshold, keep_max: "inf" }
          : { keep_min: "-inf", keep_max: data.threshold };
        this._refresh();
      }
      if (!data.ok) {
        this.selTabEl.innerHTML = `<p class="cube-hint">${data.error || "selective metrics unavailable"}</p>`;
        return;
      }

      let html = `<table class="cube-metrics cube-metrics--params"><thead><tr>` +
        `<th scope="col">metric</th><th scope="col">kept ${(data.coverage * 100).toFixed(0)}%</th><th scope="col">full</th></tr></thead><tbody>`;
      data.rows.forEach((row) => {
        const kept = row.kept === null ? "–" : this.host._fmt(row.kept);
        html += `<tr><td>${row.label}</td><td>${kept}</td><td>${this.host._fmt(row.full)}</td></tr>`;
      });
      html += `</tbody></table>`;
      const cmp = data.direction === "high" ? "≥" : "≤";
      html += `<p class="cube-hint">${data.n_kept.toLocaleString()} of ${data.n_total.toLocaleString()} pixels kept (layer ${cmp} ${this.host._fmt(data.threshold)})</p>`;
      this.selTabEl.innerHTML = html;
    } catch (err) {
      this.selTabEl.innerHTML = `<p class="cube-hint">selective metrics failed</p>`;
    }
  }

  _setMode(mode) {
    this.mode = mode;
    this.thrEl.disabled = mode === "all";
    this._syncThresholdLabel();
    this._refresh();
  }

  _resetControls() {
    this.vminEl.value = this._round(this.layer.vmin);
    this.vmaxEl.value = this._round(this.layer.vmax);
    this.mode = "all";
    this.modeEl.value = "all";
    this.thrEl.disabled = true;
    this.thrEl.value = 500;
    this._syncThresholdLabel();
  }

  _resetRange() {
    this.vminEl.value = this._round(this.layer.vmin);
    this.vmaxEl.value = this._round(this.layer.vmax);
    this._refresh();
  }

  _onThreshold() {
    this._syncThresholdLabel();
    this._refresh();
  }

  _onAlpha() {
    this.alpha = Number(this.alphaEl.value) / 100;
    this._refresh();
  }

  _threshold() {
    const frac = Number(this.thrEl.value) / 1000;
    return this.layer.vmin + frac * (this.layer.vmax - this.layer.vmin);
  }

  _syncThresholdLabel() {
    if (this.mode === "all") {
      this.thrValEl.textContent = "";
      return;
    }
    this.thrValEl.textContent = `${this.mode} ${this.host._fmt(this._threshold())}`;
  }

  _keepWindow() {
    if (this.maskWindow) return this.maskWindow;
    if (this.mode === "below") return { keep_min: "-inf", keep_max: this._threshold() };
    if (this.mode === "above") return { keep_min: this._threshold(), keep_max: "inf" };
    return { keep_min: "-inf", keep_max: "inf" };
  }

  _renderLayerBtns() {
    this.layerEl.innerHTML = "";
    this.layers.forEach((layer) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "cube-space";
      btn.dataset.key = layer.key;
      btn.textContent = layer.label;
      btn.title = layer.key;
      btn.addEventListener("click", () => this._setLayer(layer.key));
      this.layerEl.appendChild(btn);
    });
  }

  _syncBtns() {
    this.layerEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.classList.toggle("is-active", this.layer && btn.dataset.key === this.layer.key);
    });
  }

  _refresh() {
    clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => this._draw(), 160);
  }

  _draw() {
    if (!this.layer) return;

    const { keep_min, keep_max } = this._keepWindow();
    const vmin = Number(this.vminEl.value);
    const vmax = Number(this.vmaxEl.value);

    const query = `id=${encodeURIComponent(this.host.selectedId)}&key=${encodeURIComponent(this.layer.key)}` +
      `&vmin=${vmin}&vmax=${vmax}&keep_min=${keep_min}&keep_max=${keep_max}&alpha=${this.alpha}`;
    this.img.src = `/api/cubes/metric_map?${query}`;

    const classes = this.layer.classes;
    this.rangeWrapEl.hidden = !!classes;
    this.cbar.hidden        = !!classes;
    this.minEl.hidden       = !!classes;
    this.maxEl.hidden       = !!classes;
    this.classesEl.hidden   = !classes;
    this.classesEl.innerHTML = classes
      ? classes.map((c) => `<span class="cube-cat"><i style="background:${c.color}"></i>${SharedCharts.esc(c.name)}</span>`).join("")
      : "";

    this.cbar.src = `/api/cubes/cbar?cmap=${this.layer.diverging ? "coolwarm" : "viridis"}`;
    this.minEl.textContent = this.host._fmt(vmin);
    this.maxEl.textContent = this.host._fmt(vmax);

    this._drawHist();
  }

  _pointFromEvent(ev) {
    const rect = this.img.getBoundingClientRect();
    const fx = (ev.clientX - rect.left) / rect.width;
    const fy = (ev.clientY - rect.top) / rect.height;
    return {
      row: Math.min(this.host.meta.n_rows - 1, Math.max(0, Math.floor(fy * this.host.meta.n_rows))),
      col: Math.min(this.host.meta.n_cols - 1, Math.max(0, Math.floor(fx * this.host.meta.n_cols))),
      fx,
      fy,
    };
  }

  _onHover(ev) {
    if (!this.layer || !this.host.meta || this.picked) return;
    const point = this._pointFromEvent(ev);
    this.hoverQueued = point;
    this._hoverPump();
  }

  async _hoverPump() {
    if (this.hoverFetching) return;
    this.hoverFetching = true;

    while (this.hoverQueued) {
      const { row, col } = this.hoverQueued;
      this.hoverQueued = null;
      const data       = await Api.get(`/api/cubes/metric_at?id=${encodeURIComponent(this.host.selectedId)}&key=${encodeURIComponent(this.layer.key)}&row=${row}&col=${col}`);
      if (data && data.ok && !this.picked) {
        const shown = data.value === null ? "–" : (data.name || this.host._fmt(data.value));
        this.coordsEl.textContent = `row = ${data.row} · col = ${data.col} · ${this.layer.label} = ${shown}`;
      }
    }

    this.hoverFetching = false;
  }

  _onClick(ev) {
    if (this.suppressClick) {
      this.suppressClick = false;
      return;
    }
    if (!this.layer || !this.host.meta || ev.shiftKey) return;

    if (this.picked) {
      this.picked = null;
      this.cross.hidden = true;
      this.openBtn.hidden = true;
      this.readoutEl.innerHTML = "";
      this.coordsEl.textContent = "Hover the map to read the value under the cursor; click to lock a pixel and read every layer.";
      return;
    }

    const point               = this._pointFromEvent(ev);
    this.picked               = point;
    this.cross.hidden         = false;
    this.cross.style.left     = `${point.fx * 100}%`;
    this.cross.style.top      = `${point.fy * 100}%`;
    this.coordsEl.textContent = `locked at row = ${point.row} · col = ${point.col} · click again to unlock`;
    this.openBtn.hidden       = false;

    this.host.setFocus(point.row, point.col);
    this.adopted = this.host.focusSeq;
    this._fetchReadout(point.row, point.col);
  }

  async _fetchReadout(row, col) {
    const jobs = this.layers.map((layer) =>
      Api.get(`/api/cubes/metric_at?id=${encodeURIComponent(this.host.selectedId)}&key=${encodeURIComponent(layer.key)}&row=${row}&col=${col}`)
    );
    const results = await Promise.all(jobs);

    let html = `<table class="cube-metrics cube-metrics--params"><tbody>`;
    this.layers.forEach((layer, i) => {
      const data = results[i];
      const value = data && data.ok && data.value !== null ? (data.name ? SharedCharts.esc(data.name) : this.host._fmt(data.value)) : "–";
      html += `<tr><th scope="row">${SharedCharts.esc(layer.label)}</th><td>${value}</td></tr>`;
    });
    html += `</tbody></table>`;

    this.readoutEl.innerHTML = html;
  }

  _openCuts() {
    if (!this.picked) return;
    this.host._setView("explorer");
    this.host._follow(this.picked, true);
    this.host._enterSlices(this.picked);
  }

  _rectPoint(ev) {
    const rect = this.img.getBoundingClientRect();
    return {
      fx : Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width)),
      fy : Math.min(1, Math.max(0, (ev.clientY - rect.top) / rect.height)),
    };
  }

  _onRectStart(ev) {
    this.suppressClick = false;
    if (!ev.shiftKey || ev.button !== 0 || !this.layer || !this.host.meta) return;

    ev.preventDefault();
    this.rectDrag = { start: this._rectPoint(ev), end: this._rectPoint(ev) };
    this._paintRect();
  }

  _onRectMove(ev) {
    if (!this.rectDrag) return;
    this.rectDrag.end = this._rectPoint(ev);
    this._paintRect();
  }

  _paintRect() {
    const { start, end } = this.rectDrag;

    this.rectEl.hidden       = false;
    this.rectEl.style.left   = `${Math.min(start.fx, end.fx) * 100}%`;
    this.rectEl.style.top    = `${Math.min(start.fy, end.fy) * 100}%`;
    this.rectEl.style.width  = `${Math.abs(end.fx - start.fx) * 100}%`;
    this.rectEl.style.height = `${Math.abs(end.fy - start.fy) * 100}%`;
  }

  _onRectEnd() {
    if (!this.rectDrag) return;

    const { start, end } = this.rectDrag;
    this.rectDrag        = null;
    this.suppressClick   = true;

    if (Math.abs(end.fx - start.fx) < 0.01 || Math.abs(end.fy - start.fy) < 0.01) {
      this.rectEl.hidden = true;
      return;
    }

    const meta = this.host.meta;
    const row0 = Math.floor(Math.min(start.fy, end.fy) * meta.n_rows);
    const row1 = Math.min(meta.n_rows - 1, Math.floor(Math.max(start.fy, end.fy) * meta.n_rows));
    const col0 = Math.floor(Math.min(start.fx, end.fx) * meta.n_cols);
    const col1 = Math.min(meta.n_cols - 1, Math.floor(Math.max(start.fx, end.fx) * meta.n_cols));

    this._fetchRegion(row0, row1, col0, col1);
  }

  async _fetchRegion(row0, row1, col0, col1) {
    const seq    = ++this.regionSeq;
    const cubeId = this.host.selectedId;
    const data   = await Api.get(`/api/cubes/region?id=${encodeURIComponent(cubeId)}&row0=${row0}&row1=${row1}&col0=${col0}&col1=${col1}`);
    if (seq !== this.regionSeq || cubeId !== this.host.selectedId) return;

    if (!data || !data.ok) {
      Toast.show((data && data.error) || "Region statistics failed.", "error");
      return;
    }

    this.regionHintEl.textContent = `rows ${data.row0}–${data.row1} · cols ${data.col0}–${data.col1} · ${data.n_pixels.toLocaleString()} pixels · shift-drag again to replace`;

    let html = `<table class="cube-metrics cube-metrics--params"><thead><tr><th scope="col">metric</th><th scope="col">region</th><th scope="col">scene</th></tr></thead><tbody>`;
    data.rows.forEach((row) => {
      const region = row.region === null ? "–" : this.host._fmt(row.region);
      const scene  = row.scene === null ? "–" : this.host._fmt(row.scene);
      html += `<tr><td>${SharedCharts.esc(row.label)}</td><td>${region}</td><td>${scene}</td></tr>`;
    });
    html += `</tbody></table>`;
    this.regionTabEl.innerHTML = html;

    this._drawRegionProfiles(data.profiles);
  }

  _drawRegionProfiles(profiles) {
    const sources = Object.keys(profiles);
    if (!sources.length) {
      this.regionWrapEl.hidden = true;
      return;
    }
    this.regionWrapEl.hidden = false;

    const canvas  = this.regionProfEl;
    const dpr     = window.devicePixelRatio || 1;
    const w       = canvas.clientWidth || 240;
    const h       = canvas.clientHeight || 92;
    canvas.width  = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    let zMin = Infinity, zMax = -Infinity, vMax = 0;
    sources.forEach((source) => {
      const series = profiles[source];
      zMin = Math.min(zMin, series.z_values[0]);
      zMax = Math.max(zMax, series.z_values[series.z_values.length - 1]);
      vMax = Math.max(vMax, ...series.values);
    });
    if (!Number.isFinite(zMin) || !(zMax > zMin)) return;
    if (!(vMax > 0)) vMax = 1;

    const pad = 4;
    const xAt = (z) => pad + ((z - zMin) / (zMax - zMin)) * (w - 2 * pad);
    const yAt = (value) => h - pad - (value / vMax) * (h - 2 * pad);

    sources.forEach((source) => {
      const series    = profiles[source];
      ctx.strokeStyle = this.host.colors[source] || "#555";
      ctx.lineWidth   = 1.5;
      ctx.beginPath();
      series.z_values.forEach((z, i) => {
        const x = xAt(z);
        const y = yAt(series.values[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  }

  async _downloadOverlay() {
    if (!this.layer || !this.img.src) return;

    const res = await fetch(this.img.src).catch(() => null);
    if (!res || !res.ok) {
      Toast.show("Overlay download failed.", "error");
      return;
    }

    FigureExport.download(`${this.host.fileTag()}_metric_${this.layer.key}.png`, await res.blob());
  }

  _round(value) {
    return Number(Number(value).toPrecision(5));
  }
}

class CubeTransect {
  static HINT = "Click points on the map to trace a polyline; every segment is cut through the cubes.";

  constructor(refs, host) {
    this.host          = host;
    this.atEl          = refs.at;
    this.undoBtn       = refs.undo;
    this.clearBtn      = refs.clear;
    this.printBtn      = refs.print;
    this.downloadBtn   = refs.download;
    this.nameEl        = refs.name;
    this.saveBtn       = refs.save;
    this.savedSel      = refs.saved;
    this.removeBtn     = refs.remove;
    this.stripWrap     = refs.stripWrap;
    this.stripLayerSel = refs.stripLayer;
    this.stripValEl    = refs.stripVal;
    this.stripCanvas   = refs.strip;
    this.map           = refs.map;
    this.overlay       = refs.overlay;
    this.grid          = refs.grid;
    this.panels        = refs.panels || [];

    this.points    = [];
    this.token     = 0;
    this.saving    = false;
    this.stripData = null;

    this.map.addEventListener("click", (ev) => this._onClick(ev));
    this.undoBtn.addEventListener("click", () => this._undo());
    this.clearBtn.addEventListener("click", () => this._reset());
    this.printBtn.addEventListener("click", () => this._print());
    this.downloadBtn.addEventListener("click", () => this._download());
    this.saveBtn.addEventListener("click", () => this._saveLine());
    this.savedSel.addEventListener("change", () => this._loadLine());
    this.removeBtn.addEventListener("click", () => this._deleteLine());
    this.stripLayerSel.addEventListener("change", () => this._drawStrip());
  }

  configure() {
    this._reset();
    this.map.src = `/api/cubes/primary?id=${encodeURIComponent(this.host.selectedId)}`;
    this._renderSaved();
    this.renderStripLayers();
  }

  render() {
    this._applyVisibility();
    if (this.points.length >= 2) this._fetchAll();
  }

  syncSpace() {
    if (this.host.view === "transect" && this.points.length >= 2) this._fetchAll();
  }

  renderStripLayers() {
    const layers = (this.host.meta && this.host.meta.metric_maps) || [];
    const kept   = this.stripLayerSel.value;

    this.stripLayerSel.innerHTML = layers.map((layer) => `<option value="${SharedCharts.esc(layer.key)}">${SharedCharts.esc(layer.label)}</option>`).join("");
    if (layers.some((layer) => layer.key === kept)) this.stripLayerSel.value = kept;

    if (this.points.length >= 2) this._fetchStrip(this.token);
  }

  _applyVisibility() {
    const meta = this.host.meta;
    this.panels.forEach((panel) => {
      panel.root.hidden = !meta || !meta.sources.includes(panel.source) || !this.host.visible.has(panel.source);
    });
  }

  _reset() {
    this.token            += 1;
    this.points            = [];
    this.stripData         = null;
    this.overlay.innerHTML = "";
    this.grid.hidden       = true;
    this.stripWrap.hidden  = true;
    this.atEl.textContent  = CubeTransect.HINT;
    this._syncButtons();
  }

  _syncButtons() {
    const some = this.points.length >= 1;
    const line = this.points.length >= 2;

    this.undoBtn.hidden     = !some;
    this.clearBtn.hidden    = !some;
    this.printBtn.hidden    = !line;
    this.downloadBtn.hidden = !line;
  }

  _pointFromEvent(ev) {
    const rect = this.map.getBoundingClientRect();
    const fx = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
    const fy = Math.min(1, Math.max(0, (ev.clientY - rect.top) / rect.height));
    return {
      row: Math.min(this.host.meta.n_rows - 1, Math.floor(fy * this.host.meta.n_rows)),
      col: Math.min(this.host.meta.n_cols - 1, Math.floor(fx * this.host.meta.n_cols)),
      fx,
      fy,
    };
  }

  _onClick(ev) {
    if (!this.host.meta) return;
    const point = this._pointFromEvent(ev);

    const last = this.points[this.points.length - 1];
    if (last && last.row === point.row && last.col === point.col) return;

    this.points.push(point);
    this._afterChange();
  }

  _undo() {
    if (!this.points.length) return;
    this.points.pop();
    this._afterChange();
  }

  _afterChange() {
    this._drawOverlay();
    this._syncButtons();

    if (this.points.length < 2) {
      this.token += 1;
      this.stripData        = null;
      this.grid.hidden      = true;
      this.stripWrap.hidden = true;

      const tail = this.points[this.points.length - 1];
      this.atEl.textContent = tail ? `start row = ${tail.row} · col = ${tail.col} · click the next point` : CubeTransect.HINT;
      return;
    }

    const head = this.points[0];
    const tail = this.points[this.points.length - 1];
    this.atEl.textContent = `transect row ${head.row},${head.col} to row ${tail.row},${tail.col} · ${this.points.length} points · click to extend`;
    this._fetchAll();
  }

  _pts() {
    return this.points.map((p) => `${p.row},${p.col}`).join(";");
  }

  _drawOverlay() {
    let svg = "";
    this.points.forEach((point, index) => {
      if (index > 0) {
        const prev = this.points[index - 1];
        svg += `<line x1="${prev.fx * 100}" y1="${prev.fy * 100}" x2="${point.fx * 100}" y2="${point.fy * 100}" class="cube-tline" />`;
      }
      svg += `<circle cx="${point.fx * 100}" cy="${point.fy * 100}" r="0.8" class="cube-tdot" />`;
    });
    this.overlay.innerHTML = svg;
  }

  _fetchAll() {
    this.grid.hidden = false;
    this._applyVisibility();

    this.token += 1;
    const token = this.token;

    this.panels.forEach((panel) => {
      if (panel.root.hidden) return;
      this._fetch(panel, token);
    });

    this._fetchStrip(token);
  }

  async _fetch(panel, token) {
    const url = `/api/cubes/transect?id=${encodeURIComponent(this.host.selectedId)}&source=${panel.source}` +
      `&pts=${encodeURIComponent(this._pts())}&space=${this.host.space}&cmap=${this.host.cmap}`;

    const skeletonTimer = setTimeout(() => panel.root.classList.add("is-loading"), 120);
    try {
      const res = await fetch(url);
      if (!res.ok || token !== this.token) return;

      const bitmap = await createImageBitmap(await res.blob());
      if (token !== this.token) { if (bitmap.close) bitmap.close(); return; }

      const canvas = panel.canvas;
      if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
      }
      canvas.getContext("2d").drawImage(bitmap, 0, 0);
    } catch (e) {
    } finally {
      clearTimeout(skeletonTimer);
      panel.root.classList.remove("is-loading");
    }
  }

  async _fetchStrip(token) {
    if (!this.stripLayerSel.options.length) {
      this.stripData        = null;
      this.stripWrap.hidden = true;
      return;
    }

    const data = await Api.get(`/api/cubes/transect_metrics?id=${encodeURIComponent(this.host.selectedId)}&pts=${encodeURIComponent(this._pts())}`);
    if (token !== this.token || !data || !data.ok) return;

    this.stripData        = data;
    this.stripWrap.hidden = false;
    this._drawStrip();
  }

  _drawStrip() {
    if (!this.stripData) return;

    const key = this.stripLayerSel.value;
    const row = this.stripData.rows.find((entry) => entry.key === key) || this.stripData.rows[0];
    if (!row) return;

    const canvas = this.stripCanvas;
    const dpr    = window.devicePixelRatio || 1;
    const w      = canvas.clientWidth || 320;
    const h      = canvas.clientHeight || 70;
    canvas.width  = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const finite = row.values.filter((value) => value !== null && Number.isFinite(value));
    if (!finite.length) {
      this.stripValEl.textContent = "no finite values along the line";
      return;
    }

    const lo   = Math.min(...finite);
    const hi   = Math.max(...finite);
    const span = hi - lo || 1;
    const pad  = 3;
    const n    = row.values.length;
    const xAt  = (i) => pad + (i / Math.max(1, n - 1)) * (w - 2 * pad);
    const yAt  = (value) => h - pad - ((value - lo) / span) * (h - 2 * pad);

    ctx.strokeStyle = CubeView.tint(this.host.colors.accent, 0.9);
    ctx.lineWidth   = 1.5;
    ctx.beginPath();
    let open = false;
    row.values.forEach((value, i) => {
      if (value === null || !Number.isFinite(value)) { open = false; return; }
      const x = xAt(i);
      const y = yAt(value);
      if (!open) { ctx.moveTo(x, y); open = true; }
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    this.stripValEl.textContent = `${this.host._fmt(lo)} … ${this.host._fmt(hi)} over ${n} samples`;
  }

  _storeKey() {
    const meta = this.host.meta;
    return `cube-transects:${meta.n_rows}x${meta.n_cols}`;
  }

  _stored() {
    return JSON.parse(localStorage.getItem(this._storeKey()) || "[]");
  }

  _renderSaved() {
    const stored = this._stored();
    this.savedSel.innerHTML = `<option value="">saved lines…</option>` +
      stored.map((entry) => `<option value="${SharedCharts.esc(entry.name)}">${SharedCharts.esc(entry.name)} (${entry.points.length} pts)</option>`).join("");
    this.removeBtn.hidden = true;
  }

  _saveLine() {
    if (this.points.length < 2) {
      Toast.show("Trace a line with at least two points first.", "warn");
      return;
    }

    const name   = this.nameEl.value.trim() || `line ${this._stored().length + 1}`;
    const stored = this._stored().filter((entry) => entry.name !== name);
    stored.push({ name, points: this.points.map((p) => [p.row, p.col]) });

    localStorage.setItem(this._storeKey(), JSON.stringify(stored));
    this.nameEl.value = "";
    this._renderSaved();
    this.savedSel.value   = name;
    this.removeBtn.hidden = false;
    Toast.show(`Saved transect '${name}' for every cube with this footprint.`, "ok");
  }

  _loadLine() {
    const name = this.savedSel.value;
    this.removeBtn.hidden = !name;
    if (!name || !this.host.meta) return;

    const entry = this._stored().find((candidate) => candidate.name === name);
    if (!entry) return;

    const meta = this.host.meta;
    this.points = entry.points.map(([row, col]) => {
      const r = Math.min(meta.n_rows - 1, Math.max(0, row));
      const c = Math.min(meta.n_cols - 1, Math.max(0, col));
      return { row: r, col: c, fx: (c + 0.5) / meta.n_cols, fy: (r + 0.5) / meta.n_rows };
    });
    this._afterChange();
  }

  _deleteLine() {
    const name = this.savedSel.value;
    if (!name) return;

    localStorage.setItem(this._storeKey(), JSON.stringify(this._stored().filter((entry) => entry.name !== name)));
    this._renderSaved();
    Toast.show(`Deleted saved transect '${name}'.`, "ok");
  }

  async _print() {
    if (this.points.length < 2 || this.saving) return;

    this.saving = true;
    this.printBtn.disabled = true;

    const res = await Api.post("/api/cubes/save_transect", {
      id    : this.host.selectedId,
      pts   : this.points.map((p) => [p.row, p.col]),
      space : this.host.space,
      cmap  : this.host.cmap,
    });

    this.saving = false;
    this.printBtn.disabled = false;

    if (!res || !res.ok) {
      Toast.show((res && res.error) || "Transect figure save failed.", "error");
      return;
    }

    Toast.show(`Saved ${res.files.length} transect figures → ${res.rel}`, "ok");
  }

  _download() {
    if (this.points.length < 2) return;

    const head = this.points[0];
    const tail = this.points[this.points.length - 1];
    const via  = this.points.length > 2 ? `_via${this.points.length - 2}` : "";
    const tag  = `${this.host.fileTag()}_transect_row${head.row}_col${head.col}_to_row${tail.row}_col${tail.col}${via}`;
    let count = 0;

    this.panels.forEach((panel) => {
      if (panel.root.hidden || !panel.canvas.width) return;
      FigureExport.saveCanvas(panel.canvas, `${tag}_${panel.source}_${this.host.space}.png`);
      count += 1;
    });

    if (!count) Toast.show("No rendered transect panel to download.", "warn");
  }
}

class CubeCloud {
  static VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  static CURVE_SOURCES = ["full"];
  static SWEEP_MS = 9000;
  static NO_STD_RGB = [128, 128, 128];
  static MISSING = {
    gt      : "This run has no ground-truth parameter cube.",
    full    : "The reference cube is not available for this run.",
  };

  constructor(refs, host) {
    this.host     = host;
    this.sourceEl = refs.source;
    this.colorEl  = refs.color;
    this.thrEl    = refs.thr;
    this.thrLabel = refs.thrLabel;
    this.thrValEl = refs.thrVal;
    this.maxEl      = refs.max;
    this.cutAxisEl  = refs.cutAxis;
    this.cutPosEl   = refs.cutPos;
    this.cutSweepEl = refs.cutSweep;
    this.auxWrap    = refs.auxWrap;
    this.auxEl      = refs.aux;
    this.atEl       = refs.at;
    this.canvas     = refs.canvas;

    this.source        = "pred";
    this.colorBy       = "mu";
    this.points        = null;
    this.width         = 0;
    this.total         = 0;
    this.muBounds      = null;
    this.stdRange      = null;
    this.auxGrid       = null;
    this.auxBounds     = null;
    this.debounceTimer = null;
    this.sweepFrame    = null;
    this.sweepT0       = null;
    this.token         = 0;

    this.yaw = 0.7;
    this.pitch = 0.9;
    this.zoom = 1.0;
    this.dragging = null;

    this.sourceEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.addEventListener("click", () => this._setSource(btn.dataset.source));
    });
    this.colorEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.addEventListener("click", () => this._setColor(btn.dataset.color));
    });
    this.thrEl.addEventListener("input", () => this._onThreshold());
    this.maxEl.addEventListener("change", () => this._fetch());
    this.auxEl.addEventListener("change", () => this._onAux());
    this.cutAxisEl.addEventListener("change", () => this._onCutAxis());
    this.cutPosEl.addEventListener("input", () => this._onCutPos());
    this.cutSweepEl.addEventListener("change", () => this._onCutSweep());

    this.canvas.addEventListener("mousedown", (ev) => { this.dragging = { x: ev.clientX, y: ev.clientY }; });
    window.addEventListener("mousemove", (ev) => this._onDrag(ev));
    window.addEventListener("mouseup", () => { this.dragging = null; });
    this.canvas.addEventListener("wheel", (ev) => this._onWheel(ev), { passive: false });
    this.canvas.addEventListener("dblclick", () => this._resetView());
  }

  configure(meta) {
    const sources = this._sources(meta);
    this.available = sources.length > 0;
    if (!sources.includes(this.source)) this.source = sources[0] || "pred";
    this.token         += 1;
    this.points         = null;
    this.muBounds       = null;
    this.stdRange       = null;
    this.auxGrid        = null;
    this.auxBounds      = null;
    this.auxWrap.hidden = !meta.aux;
    this.auxEl.checked  = false;
    this._stopSweep();
    this._resetView(false);
    this._syncThresholdLabel();
  }

  _sources(meta = this.host.meta) {
    const out = meta.params ? meta.params.sources.slice() : [];
    CubeCloud.CURVE_SOURCES.forEach((s) => { if (meta.sources.includes(s)) out.push(s); });
    return out;
  }

  _isParam() {
    return this.source === "pred" || this.source === "gt";
  }

  render() {
    this._syncBtns();
    if (!this.points) this._fetch();
    else this._draw();
    if (this.cutSweepEl.checked && this.sweepFrame === null) this._startSweep();
  }

  _setSource(source) {
    if (source === this.source) return;
    if (!this._sources().includes(source)) {
      Toast.show(CubeCloud.MISSING[source] || "This source is not available.", "warn");
      return;
    }
    this.source = source;
    this._syncBtns();
    this._syncThresholdLabel();
    this._fetch();
  }

  _setColor(colorBy) {
    if (colorBy === this.colorBy) return;
    this.colorBy = colorBy;
    this._syncBtns();
    this._draw();
  }

  _onThreshold() {
    this._syncThresholdLabel();
    clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => this._fetch(), 250);
  }

  async _onAux() {
    if (this.auxEl.checked && !this.auxGrid) {
      this.auxGrid   = await this._fetchBinary(`/api/cubes/aux_grid?id=${encodeURIComponent(this.host.selectedId)}`);
      this.auxBounds = this.auxGrid ? CubeCloud.valueBounds(this.auxGrid.rows, 0, 1) : null;
    }
    this._draw();
  }

  _onCutAxis() {
    if (this.cutAxisEl.value === "off") {
      this.cutSweepEl.checked = false;
      this._stopSweep();
    }
    this._draw();
  }

  _onCutPos() {
    if (!this._cutRaf) {
      this._cutRaf = requestAnimationFrame(() => {
        this._cutRaf = null;
        this._draw();
      });
    }
  }

  _onCutSweep() {
    if (this.cutSweepEl.checked && this.cutAxisEl.value === "off") this.cutAxisEl.value = "row";
    if (this.cutSweepEl.checked) this._startSweep();
    else this._stopSweep();
  }

  _startSweep() {
    this._stopSweep();
    this.sweepFrame = requestAnimationFrame((now) => this._sweepStep(now));
  }

  _stopSweep() {
    if (this.sweepFrame !== null) cancelAnimationFrame(this.sweepFrame);
    this.sweepFrame = null;
    this.sweepT0    = null;
  }

  _sweepStep(now) {
    if (this.host.view !== "cloud" || !this.cutSweepEl.checked || this.cutAxisEl.value === "off") {
      this._stopSweep();
      return;
    }

    if (this.sweepT0 === null) this.sweepT0 = now - (Number(this.cutPosEl.value) / 100) * (CubeCloud.SWEEP_MS / 2);
    const phase = ((now - this.sweepT0) % CubeCloud.SWEEP_MS) / CubeCloud.SWEEP_MS;
    const tri   = phase < 0.5 ? phase * 2 : 2 - phase * 2;

    this.cutPosEl.value = String(Math.round(tri * 100));
    this._draw();
    this.sweepFrame = requestAnimationFrame((next) => this._sweepStep(next));
  }

  leave() {
    this._stopSweep();
  }

  _cutSpec() {
    const axis = this.cutAxisEl.value;
    if (axis === "off" || !this.host.meta || !this.muBounds) return null;

    const meta = this.host.meta;
    const frac = Number(this.cutPosEl.value) / 100;
    const lo   = axis === "z" ? this.muBounds[0] : 0;
    const hi   = axis === "row" ? meta.n_rows : axis === "col" ? meta.n_cols : this.muBounds[1];
    return { axis, frac, threshold: lo + frac * (hi - lo) };
  }

  _ampMin() {
    const frac = Number(this.thrEl.value) / 100;

    if (this._isParam()) {
      const params = this.host.meta.params;
      const top = Math.max(params.ranges.amp[1], params.threshold * 10);
      return params.threshold * Math.pow(top / params.threshold, frac);
    }

    const [lo, hi] = this.host.meta.intensity[this.source];
    return lo + frac * (hi - lo);
  }

  _syncThresholdLabel() {
    const meta = this.host.meta;
    if (!meta || (this._isParam() && !meta.params)) return;
    this.thrLabel.textContent = this._isParam() ? "amp ≥" : "int ≥";
    this.thrValEl.textContent = this.host._fmt(this._ampMin());
  }

  _syncBtns() {
    const sources = this._sources();
    this.sourceEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.source === this.source);
      btn.disabled = !sources.includes(btn.dataset.source);
    });
    const hasStd = this.host.hasMetricLayer("seed_std_profile");
    if (!hasStd && this.colorBy === "std") this.colorBy = "mu";
    this.colorEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.color === this.colorBy);
      if (btn.dataset.color === "std") btn.disabled = !hasStd;
    });

    const binAxis       = this.source === "full";
    this.auxEl.disabled = binAxis;
    this.auxWrap.title  = binAxis ? "The auxiliary drape needs a metric z axis" : "Drape the cloud on the auxiliary field: point height = aux + z";
  }

  async _fetchBinary(url) {
    try {
      const res = await fetch(url);
      if (!res.ok) return null;
      const raw = new Float32Array(await res.arrayBuffer());
      return { header: raw.subarray(0, 4), rows: raw.subarray(4) };
    } catch (e) {
      return null;
    }
  }

  async _fetch() {
    if (!this.available) return;

    const url = `/api/cubes/points?id=${encodeURIComponent(this.host.selectedId)}&source=${this.source}` +
      `&amp_min=${this._ampMin()}&max=${this.maxEl.value}`;

    this.token += 1;
    const token = this.token;

    const data = await this._fetchBinary(url);
    if (!data || token !== this.token) return;

    this.points   = data.rows;
    this.width    = data.header[2];
    this.total    = data.header[1];
    this.muRange  = this._sampleRange(data.rows, 2);
    this.muBounds = CubeCloud.valueBounds(data.rows, 2, this.width);
    this.stdRange = CubeCloud.finiteRange(data.rows, 4, this.width);
    this._draw();
  }

  _sampleRange(rows, offset) {
    const values = [];
    const stride = Math.max(1, Math.floor(rows.length / this.width / 4096)) * this.width;
    for (let i = offset; i < rows.length; i += stride) values.push(rows[i]);
    if (!values.length) return [0, 1];

    values.sort((a, b) => a - b);
    const lo = values[Math.floor(values.length * 0.02)];
    const hi = values[Math.floor(values.length * 0.98)];
    return hi > lo ? [lo, hi] : [lo, lo + 1];
  }

  static finiteRange(rows, offset, width) {
    const values = [];
    const stride = Math.max(1, Math.floor(rows.length / width / 4096)) * width;
    for (let i = offset; i < rows.length; i += stride) {
      if (Number.isFinite(rows[i])) values.push(rows[i]);
    }
    if (!values.length) return null;

    values.sort((a, b) => a - b);
    const lo = values[Math.floor(values.length * 0.02)];
    const hi = values[Math.floor(values.length * 0.98)];
    return hi > lo ? [lo, hi] : [lo, lo + 1];
  }

  static valueBounds(rows, offset, width) {
    let lo = Infinity, hi = -Infinity;
    for (let i = offset; i < rows.length; i += width) {
      const v = rows[i];
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    return hi >= lo ? [lo, hi] : null;
  }

  _resetView(draw = true) {
    this.yaw = 0.7;
    this.pitch = 0.9;
    this.zoom = 1.0;
    if (draw) this._draw();
  }

  _onDrag(ev) {
    if (!this.dragging) return;
    this.yaw += (ev.clientX - this.dragging.x) * 0.008;
    this.pitch = Math.min(1.55, Math.max(0.05, this.pitch + (ev.clientY - this.dragging.y) * 0.006));
    this.dragging = { x: ev.clientX, y: ev.clientY };
    if (!this._dragRaf) {
      this._dragRaf = requestAnimationFrame(() => {
        this._dragRaf = null;
        this._draw();
      });
    }
  }

  _onWheel(ev) {
    ev.preventDefault();
    this.zoom = Math.min(8, Math.max(0.3, this.zoom * (ev.deltaY > 0 ? 0.9 : 1.11)));
    this._draw();
  }

  _palette(t) {
    const stops = CubeCloud.VIRIDIS;
    const x     = Math.min(0.9999, Math.max(0, t)) * (stops.length - 1);
    const i     = Math.floor(x);
    const f     = x - i;
    return [
      Math.round(stops[i][0] + (stops[i + 1][0] - stops[i][0]) * f),
      Math.round(stops[i][1] + (stops[i + 1][1] - stops[i][1]) * f),
      Math.round(stops[i][2] + (stops[i + 1][2] - stops[i][2]) * f),
    ];
  }

  _draw() {
    if (!this.points || this.host.view !== "cloud") return;

    const meta = this.host.meta;
    const stage = this.canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(320, stage.clientWidth);
    const h = Math.max(320, Math.round(window.innerHeight * 0.62));

    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
      this.canvas.style.height = `${h}px`;
    }

    const W = this.canvas.width;
    const H = this.canvas.height;
    const ctx = this.canvas.getContext("2d");
    const image = ctx.createImageData(W, H);
    const buf = new Uint32Array(image.data.buffer);
    buf.fill(0xff1a1510);

    const cx = meta.n_cols / 2;
    const cy = meta.n_rows / 2;
    const [muLo, muHi] = this.muRange || [meta.x_min, meta.x_max];
    const zMid = (muLo + muHi) / 2;
    const zSpan = (muHi - muLo) || 1;

    const rowStep = 1;
    const colStep = 1;
    const zScale  = (Math.max(meta.n_rows, meta.n_cols) * 0.35) / (zSpan / 2);
    const extent  = Math.max(meta.n_rows * rowStep, meta.n_cols * colStep);

    const sinY = Math.sin(this.yaw), cosY = Math.cos(this.yaw);
    const sinP = Math.sin(this.pitch), cosP = Math.cos(this.pitch);
    const fit = (Math.min(W, H) / (extent * 1.9)) * this.zoom;

    const cut = this._cutSpec();

    const plot = (x, y, z, rgb) => {
      const rx = x * cosY - y * sinY;
      const ry = x * sinY + y * cosY;
      const sx = Math.round(W / 2 + rx * fit);
      const sy = Math.round(H / 2 + (ry * cosP - z * sinP) * fit);
      if (sx < 0 || sy < 0 || sx >= W - 1 || sy >= H - 1) return;
      const color = 0xff000000 | (rgb[2] << 16) | (rgb[1] << 8) | rgb[0];
      buf[sy * W + sx] = color;
      buf[sy * W + sx + 1] = color;
      buf[(sy + 1) * W + sx] = color;
      buf[(sy + 1) * W + sx + 1] = color;
    };

    const grid = this.auxEl.checked && !this.auxEl.disabled && this.auxGrid ? this.auxGrid.rows : null;

    if (grid) {
      for (let row = 0; row < meta.n_rows; row += 4) {
        if (cut && cut.axis === "row" && row > cut.threshold) continue;
        for (let col = 0; col < meta.n_cols; col += 4) {
          if (cut && cut.axis === "col" && col > cut.threshold) continue;
          const g = grid[row * meta.n_cols + col];
          if (!Number.isFinite(g)) continue;
          plot((col - cx) * colStep, (row - cy) * rowStep, (g - zMid) * zScale, [110, 116, 122]);
        }
      }
    }

    const rows = this.points;
    const logAmp = this._isParam();
    let ampLo, ampHi;
    if (logAmp) {
      const params = meta.params;
      ampLo = Math.log(Math.max(params.threshold, 1e-6));
      ampHi = Math.log(Math.max(params.ranges.amp[1], params.threshold * 10));
    } else {
      [ampLo, ampHi] = meta.intensity[this.source];
    }

    const cutOffset = cut ? { row: 0, col: 1, z: 2 }[cut.axis] : 0;
    let drawn = 0;

    const width          = this.width;
    const [stdLo, stdHi] = this.stdRange || [0, 1];

    for (let i = 0; i < rows.length; i += width) {
      if (cut && rows[i + cutOffset] > cut.threshold) continue;
      drawn += 1;

      const mu  = rows[i + 2];
      const amp = rows[i + 3];
      const std = rows[i + 4];

      let z = mu - zMid;
      if (grid) {
        const g = grid[rows[i] * meta.n_cols + rows[i + 1]];
        if (!Number.isFinite(g)) continue;
        z += g;
      }

      let rgb;
      if (this.colorBy === "std") {
        rgb = Number.isFinite(std) ? this._palette((std - stdLo) / Math.max(stdHi - stdLo, 1e-9)) : CubeCloud.NO_STD_RGB;
      } else {
        const t = this.colorBy === "amp"
          ? ((logAmp ? Math.log(Math.max(amp, 1e-6)) : amp) - ampLo) / Math.max(ampHi - ampLo, 1e-6)
          : (mu - muLo) / zSpan;
        rgb = this._palette(t);
      }
      plot((rows[i + 1] - cx) * colStep, (rows[i] - cy) * rowStep, z * zScale, rgb);
    }

    ctx.putImageData(image, 0, 0);
    if (cut) {
      const [gLo, gHi] = grid && this.auxBounds ? this.auxBounds : [0, 0];
      const zLo = (this.muBounds[0] - zMid + gLo) * zScale;
      const zHi = (this.muBounds[1] - zMid + gHi) * zScale;
      this._drawCutPlane(ctx, cut, { meta, cx, cy, rowStep, colStep, zMid, zScale, zLo, zHi, W, H, fit, sinY, cosY, sinP, cosP });
    }

    const shown   = cut ? drawn : rows.length / width;
    const unit    = logAmp ? "components" : "voxels";
    const cutNote = cut ? ` · cut ${cut.axis} @ ${Math.round(cut.frac * 100)}%` : "";
    this.atEl.textContent = `${shown.toLocaleString()} of ${Math.round(this.total).toLocaleString()} ${unit}${cutNote} · drag to orbit · wheel to zoom · double-click to reset`;
  }

  _drawCutPlane(ctx, cut, g) {
    const x0 = (0 - g.cx) * g.colStep, x1 = (g.meta.n_cols - g.cx) * g.colStep;
    const y0 = (0 - g.cy) * g.rowStep, y1 = (g.meta.n_rows - g.cy) * g.rowStep;

    let corners;
    if (cut.axis === "row") {
      const y = (cut.threshold - g.cy) * g.rowStep;
      corners = [[x0, y, g.zLo], [x1, y, g.zLo], [x1, y, g.zHi], [x0, y, g.zHi]];
    } else if (cut.axis === "col") {
      const x = (cut.threshold - g.cx) * g.colStep;
      corners = [[x, y0, g.zLo], [x, y1, g.zLo], [x, y1, g.zHi], [x, y0, g.zHi]];
    } else {
      const z = (cut.threshold - g.zMid) * g.zScale;
      corners = [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]];
    }

    const pts = corners.map(([x, y, z]) => {
      const rx = x * g.cosY - y * g.sinY;
      const ry = x * g.sinY + y * g.cosY;
      return [g.W / 2 + rx * g.fit, g.H / 2 + (ry * g.cosP - z * g.sinP) * g.fit];
    });

    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    pts.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
    ctx.closePath();
    ctx.fillStyle   = "rgba(255, 209, 102, 0.10)";
    ctx.strokeStyle = "rgba(255, 209, 102, 0.55)";
    ctx.lineWidth   = Math.max(1, window.devicePixelRatio || 1);
    ctx.fill();
    ctx.stroke();
  }
}

class CubeVolume {
  static VIRIDIS   = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  static COOLWARM  = [[59, 76, 192], [144, 178, 254], [221, 221, 221], [246, 153, 122], [180, 4, 38]];
  static MAX_DIM   = 192;
  static UNIFORMS  = ["uVolume", "uLut", "uEye", "uRight", "uUp", "uForward", "uHalf", "uClipMin", "uClipMax", "uAspect", "uTfLo", "uTfHi", "uOpacity", "uSteps"];
  static CLIP_AXES = { col: 0, row: 1, z: 2 };

  static VERTEX_SRC = `#version 300 es
    const vec2 corners[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    out vec2 vNdc;
    void main() {
      vNdc = corners[gl_VertexID];
      gl_Position = vec4(corners[gl_VertexID], 0.0, 1.0);
    }`;

  static FRAGMENT_SRC = `#version 300 es
    precision highp float;
    precision highp sampler3D;

    in vec2 vNdc;
    out vec4 outColor;

    uniform sampler3D uVolume;
    uniform sampler2D uLut;
    uniform vec3 uEye;
    uniform vec3 uRight;
    uniform vec3 uUp;
    uniform vec3 uForward;
    uniform vec3 uHalf;
    uniform vec3 uClipMin;
    uniform vec3 uClipMax;
    uniform float uAspect;
    uniform float uTfLo;
    uniform float uTfHi;
    uniform float uOpacity;
    uniform int uSteps;

    vec2 boxHit(vec3 origin, vec3 dir, vec3 lo, vec3 hi) {
      vec3 inv = 1.0 / dir;
      vec3 t0 = (lo - origin) * inv;
      vec3 t1 = (hi - origin) * inv;
      vec3 tmin = min(t0, t1);
      vec3 tmax = max(t0, t1);
      return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
    }

    void main() {
      vec3 dir = normalize(uForward + vNdc.x * uAspect * uRight + vNdc.y * uUp);

      vec3 lo = -uHalf + 2.0 * uHalf * uClipMin;
      vec3 hi = -uHalf + 2.0 * uHalf * uClipMax;

      vec2 hit = boxHit(uEye, dir, lo, hi);
      if (hit.y <= max(hit.x, 0.0)) { outColor = vec4(0.0); return; }

      float tNear = max(hit.x, 0.0);
      float span = hit.y - tNear;
      float dt = span / float(uSteps);

      vec3 color = vec3(0.0);
      float alpha = 0.0;

      for (int i = 0; i < 768; i++) {
        if (i >= uSteps) break;
        vec3 p = uEye + dir * (tNear + (float(i) + 0.5) * dt);
        vec3 tc = (p + uHalf) / (2.0 * uHalf);
        float v = texture(uVolume, tc).r;

        float a = smoothstep(uTfLo, uTfHi, v) * uOpacity * dt * 90.0;
        a = clamp(a, 0.0, 1.0);
        vec3 c = texture(uLut, vec2(v, 0.5)).rgb;

        color += (1.0 - alpha) * a * c;
        alpha += (1.0 - alpha) * a;
        if (alpha > 0.97) break;
      }

      outColor = vec4(color, alpha);
    }`;

  constructor(refs, host) {
    this.host       = host;
    this.sourceEl   = refs.source;
    this.loEl       = refs.lo;
    this.hiEl       = refs.hi;
    this.opacityEl  = refs.opacity;
    this.stepsEl    = refs.steps;
    this.clipAxisEl = refs.clipAxis;
    this.clipPosEl  = refs.clipPos;
    this.atEl       = refs.at;
    this.canvas     = refs.canvas;

    this.source    = "pred";
    this.volume    = null;
    this.volumeKey = null;
    this.token     = 0;
    this.failed    = false;

    this.yaw      = 0.7;
    this.pitch    = 0.55;
    this.zoom     = 1.0;
    this.dragging = null;

    this.gl           = null;
    this.program      = null;
    this.uniforms     = null;
    this.texture      = null;
    this.lut          = null;
    this.lutDiverging = null;

    this.sourceEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.addEventListener("click", () => this._setSource(btn.dataset.source));
    });
    [this.loEl, this.hiEl, this.opacityEl, this.clipPosEl].forEach((el) => {
      el.addEventListener("input", () => this._draw());
    });
    this.stepsEl.addEventListener("change", () => this._draw());
    this.clipAxisEl.addEventListener("change", () => this._draw());

    this.canvas.addEventListener("mousedown", (ev) => { this.dragging = { x: ev.clientX, y: ev.clientY }; });
    window.addEventListener("mousemove", (ev) => this._onDrag(ev));
    window.addEventListener("mouseup", () => { this.dragging = null; });
    this.canvas.addEventListener("wheel", (ev) => this._onWheel(ev), { passive: false });
    this.canvas.addEventListener("dblclick", () => this._resetView());

    this.canvas.addEventListener("webglcontextlost", (ev) => {
      ev.preventDefault();
      this.gl      = null;
      this.program = null;
    });
    this.canvas.addEventListener("webglcontextrestored", () => {
      if (!this._ensureGl()) return;
      if (this.volume) this._uploadVolume();
      this._draw();
    });
  }

  get available() {
    return !this.failed;
  }

  configure(meta) {
    this._resetView(false);
    this.refreshSources(meta);
  }

  refreshSources(meta) {
    if (!meta.sources.includes(this.source)) this.source = meta.sources[0];
    this.token    += 1;
    this.volume    = null;
    this.volumeKey = null;
    this._syncBtns(meta.sources);
  }

  render() {
    if (!this._ensureGl()) return;
    this._syncBtns(this.host.meta.sources);
    if (!this.volume || this.volumeKey !== this._key()) this._fetch();
    else this._draw();
  }

  _key() {
    return `${this.host.selectedId}|${this.source}`;
  }

  _syncBtns(sources) {
    this.sourceEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.source === this.source);
      btn.disabled = !sources.includes(btn.dataset.source);
    });
  }

  _setSource(source) {
    if (source === this.source || !this.host.meta || !this.host.meta.sources.includes(source)) return;
    this.source = source;
    this._syncBtns(this.host.meta.sources);
    this._fetch();
  }

  _ensureGl() {
    if (this.gl) return true;
    if (this.failed) return false;

    const gl = this.canvas.getContext("webgl2", { alpha: true, antialias: false, preserveDrawingBuffer: true });
    if (!gl) {
      this.failed = true;
      this.atEl.textContent = "WebGL2 is not available in this browser; the volume view is disabled.";
      return false;
    }

    const compile = (type, src) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, src);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };

    try {
      const program = gl.createProgram();
      gl.attachShader(program, compile(gl.VERTEX_SHADER, CubeVolume.VERTEX_SRC));
      gl.attachShader(program, compile(gl.FRAGMENT_SHADER, CubeVolume.FRAGMENT_SRC));
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));

      this.gl       = gl;
      this.program  = program;
      this.uniforms = {};
      CubeVolume.UNIFORMS.forEach((name) => { this.uniforms[name] = gl.getUniformLocation(program, name); });

      this.lut          = this._makeLut(gl, CubeVolume.VIRIDIS);
      this.lutDiverging = this._makeLut(gl, CubeVolume.COOLWARM);
      gl.bindVertexArray(gl.createVertexArray());
    } catch (error) {
      this.failed = true;
      this.atEl.textContent = `Volume renderer initialisation failed: ${error.message}`;
      return false;
    }

    return true;
  }

  _makeLut(gl, stops) {
    const data = new Uint8Array(256 * 3);
    for (let i = 0; i < 256; i++) {
      const x = (i / 255) * (stops.length - 1);
      const k = Math.min(stops.length - 2, Math.floor(x));
      const f = x - k;
      for (let c = 0; c < 3; c++) data[i * 3 + c] = Math.round(stops[k][c] + (stops[k + 1][c] - stops[k][c]) * f);
    }

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, 256, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    return texture;
  }

  async _fetch() {
    this.token += 1;
    const token = this.token;
    const key   = this._key();
    this.atEl.innerHTML = ThinkingOrb.busy("listening", "Streaming volume…");

    let buffer = null;
    try {
      const res = await fetch(`/api/cubes/volume?id=${encodeURIComponent(this.host.selectedId)}&source=${this.source}&max=${CubeVolume.MAX_DIM}`);
      if (res.ok) buffer = await res.arrayBuffer();
    } catch (error) {
      buffer = null;
    }

    if (token !== this.token) return;
    if (!buffer) {
      this.atEl.textContent = "Volume fetch failed.";
      return;
    }

    const header = new Float32Array(buffer, 0, 8);
    this.volume = {
      dims      : [header[0], header[1], header[2]].map((v) => Math.round(v)),
      diverging : header[3] > 0.5,
      vmin      : header[4],
      vmax      : header[5],
      voxels    : new Uint8Array(buffer, 32),
    };
    this.volumeKey = key;

    if (!this.gl) return;
    this._uploadVolume();
    this._draw();
  }

  _uploadVolume() {
    const gl                 = this.gl;
    const [nZ, nRows, nCols] = this.volume.dims;

    if (this.texture) gl.deleteTexture(this.texture);
    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, this.texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, nCols, nRows, nZ, 0, gl.RED, gl.UNSIGNED_BYTE, this.volume.voxels);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  }

  _onDrag(ev) {
    if (!this.dragging) return;
    this.yaw += (ev.clientX - this.dragging.x) * 0.008;
    this.pitch = Math.min(1.45, Math.max(-1.45, this.pitch + (ev.clientY - this.dragging.y) * 0.006));
    this.dragging = { x: ev.clientX, y: ev.clientY };
    this._draw();
  }

  _onWheel(ev) {
    ev.preventDefault();
    this.zoom = Math.min(6, Math.max(0.35, this.zoom * (ev.deltaY > 0 ? 0.92 : 1.09)));
    this._draw();
  }

  _resetView(draw = true) {
    this.yaw   = 0.7;
    this.pitch = 0.55;
    this.zoom  = 1.0;
    if (draw) this._draw();
  }

  _clip() {
    const axis = CubeVolume.CLIP_AXES[this.clipAxisEl.value];
    const min  = [0, 0, 0];
    const max  = [1, 1, 1];

    if (axis !== undefined) max[axis] = Number(this.clipPosEl.value) / 100;

    return { min, max };
  }

  _draw() {
    if (!this.gl || !this.volume || this.host.view !== "volume") return;

    const gl    = this.gl;
    const stage = this.canvas.parentElement;
    const dpr   = Math.min(window.devicePixelRatio || 1, 1.5);
    const w     = Math.max(320, stage.clientWidth);
    const h     = Math.max(320, Math.round(window.innerHeight * 0.62));

    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width        = Math.round(w * dpr);
      this.canvas.height       = Math.round(h * dpr);
      this.canvas.style.height = `${h}px`;
    }

    const [nZ, nRows, nCols] = this.volume.dims;
    const longest            = Math.max(nZ, nRows, nCols);
    const half               = [nCols / longest * 0.5, nRows / longest * 0.5, nZ / longest * 0.5];

    const distance = 1.9 / this.zoom;
    const cosP     = Math.cos(this.pitch);
    const eye      = [
      distance * cosP * Math.sin(this.yaw),
      distance * cosP * Math.cos(this.yaw),
      distance * Math.sin(this.pitch),
    ];

    const forward = eye.map((v) => -v / distance);
    const worldUp = [0, 0, 1];
    const right   = [
      forward[1] * worldUp[2] - forward[2] * worldUp[1],
      forward[2] * worldUp[0] - forward[0] * worldUp[2],
      forward[0] * worldUp[1] - forward[1] * worldUp[0],
    ];
    const rLen = Math.hypot(...right) || 1;
    right.forEach((v, i) => { right[i] = v / rLen; });
    const up = [
      right[1] * forward[2] - right[2] * forward[1],
      right[2] * forward[0] - right[0] * forward[2],
      right[0] * forward[1] - right[1] * forward[0],
    ];

    const clip  = this._clip();
    const fov   = Math.tan((38 * Math.PI) / 360);
    const steps = Number(this.stepsEl.value) || 256;
    const lo    = Number(this.loEl.value) / 100;

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.055, 0.07, 0.086, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, this.texture);
    gl.uniform1i(this.uniforms.uVolume, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.volume.diverging ? this.lutDiverging : this.lut);
    gl.uniform1i(this.uniforms.uLut, 1);

    gl.uniform3fv(this.uniforms.uEye, eye);
    gl.uniform3fv(this.uniforms.uRight, right.map((v) => v * fov));
    gl.uniform3fv(this.uniforms.uUp, up.map((v) => v * fov));
    gl.uniform3fv(this.uniforms.uForward, forward);
    gl.uniform3fv(this.uniforms.uHalf, half);
    gl.uniform3fv(this.uniforms.uClipMin, clip.min);
    gl.uniform3fv(this.uniforms.uClipMax, clip.max);
    gl.uniform1f(this.uniforms.uAspect, this.canvas.width / this.canvas.height);
    gl.uniform1f(this.uniforms.uTfLo, lo);
    gl.uniform1f(this.uniforms.uTfHi, Math.max(lo + 0.01, Number(this.hiEl.value) / 100));
    gl.uniform1f(this.uniforms.uOpacity, Number(this.opacityEl.value) / 100);
    gl.uniform1i(this.uniforms.uSteps, Math.min(768, steps));

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    const clipNote = this.clipAxisEl.value === "off" ? "" : ` · clip ${this.clipAxisEl.value} @ ${this.clipPosEl.value}%`;
    const range    = `${this.host._fmt(this.volume.vmin)} … ${this.host._fmt(this.volume.vmax)}`;
    this.atEl.textContent = `${nRows} × ${nCols} × ${nZ} voxels (row × col × z) · intensity ${range}${clipNote} · drag to orbit · wheel to zoom · double-click to reset`;
  }
}

class CubeTimeline {
  static FRAME_MS = 650;

  constructor(refs, host) {
    this.host     = host;
    this.sourceEl = refs.source;
    this.axisEl   = refs.axis;
    this.playBtn  = refs.play;
    this.posEl    = refs.pos;
    this.atEl     = refs.at;
    this.canvas   = refs.canvas;
    this.labelEl  = refs.label;

    this.stamps    = [];
    this.idx       = 0;
    this.source    = "pred";
    this.axis      = "col";
    this.token     = 0;
    this.playing   = false;
    this.loadedFor = null;

    this.sourceEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.addEventListener("click", () => this._setSource(btn.dataset.source));
    });
    this.axisEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.addEventListener("click", () => this._setAxis(btn.dataset.axis));
    });
    this.posEl.addEventListener("input", () => this._onScrub());
    this.playBtn.addEventListener("click", () => this._togglePlay());
  }

  configure(meta) {
    this.stamps    = [];
    this.loadedFor = null;
    this.idx       = 0;
    if (!meta.sources.includes(this.source)) this.source = "pred";
    this._stopPlay();
  }

  async render() {
    this._syncBtns();

    if (this.loadedFor !== this.host.selectedId) {
      const cubeId = this.host.selectedId;
      const data   = await Api.get(`/api/cubes/timeline?id=${encodeURIComponent(cubeId)}`);
      if (cubeId !== this.host.selectedId) return;

      if (!data || !data.ok) {
        this.atEl.textContent = (data && data.error) || "Timeline unavailable.";
        return;
      }

      this.loadedFor   = cubeId;
      this.stamps      = data.stamps;
      this.idx         = data.current;
      this.posEl.max   = String(Math.max(0, this.stamps.length - 1));
      this.posEl.value = String(this.idx);
    }

    await this._show();
  }

  leave() {
    this._stopPlay();
  }

  _setSource(source) {
    if (source === this.source || !this.host.meta.sources.includes(source)) return;
    this.source = source;
    this._syncBtns();
    this._show();
  }

  _setAxis(axis) {
    if (axis === this.axis) return;
    this.axis = axis;
    this._syncBtns();
    this._show();
  }

  _onScrub() {
    this._stopPlay();
    this.idx = Number(this.posEl.value);
    this._show();
  }

  _syncBtns() {
    const sources = this.host.meta.sources;
    this.sourceEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.source === this.source);
      btn.disabled = !sources.includes(btn.dataset.source);
    });
    this.axisEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.axis === this.axis);
    });
  }

  _position() {
    const point = this.host.focusPoint();
    const meta  = this.host.meta;
    return point ? { row: point.row, col: point.col } : { row: Math.floor(meta.n_rows / 2), col: Math.floor(meta.n_cols / 2) };
  }

  _url(index) {
    const stamp        = this.stamps[index];
    const { row, col } = this._position();
    return `/api/slices/slice?id=${encodeURIComponent(stamp.id)}&source=${this.source}&axis=${this.axis}` +
      `&row=${row}&col=${col}&space=${this.host.space}&cmap=${this.host.cmap}`;
  }

  async _show() {
    if (!this.stamps.length || this.host.view !== "timeline") return;

    this.token += 1;
    const token = this.token;
    this.idx    = Math.min(this.stamps.length - 1, Math.max(0, this.idx));

    const stamp        = this.stamps[this.idx];
    const { row, col } = this._position();
    const cut          = this.axis === "col" ? `col = ${col}` : `row = ${row}`;
    const single       = this.stamps.length < 2 ? " · this run holds a single inference stamp; run inference again to grow the timeline" : "";
    const at           = `${stamp.stamp} · ${this.idx + 1} / ${this.stamps.length} · ${cut}${single}`;

    this.atEl.textContent    = at;
    this.labelEl.textContent = stamp.mse === null ? stamp.stamp : `${stamp.stamp} · curve MSE ${this.host._fmt(stamp.mse)}`;
    this.posEl.value         = String(this.idx);

    if (this.idx + 1 < this.stamps.length) this.host.cacheBitmap(this._url(this.idx + 1));

    const bitmap = await this.host.cacheBitmap(this._url(this.idx));
    if (token !== this.token) return;

    if (!bitmap) {
      this.atEl.textContent = `${at} · no ${CubeView.LABELS[this.source]} cut for this stamp`;
      this.canvas.getContext("2d").clearRect(0, 0, this.canvas.width, this.canvas.height);
      return;
    }

    if (this.canvas.width !== bitmap.width || this.canvas.height !== bitmap.height) {
      this.canvas.width  = bitmap.width;
      this.canvas.height = bitmap.height;
    }
    this.canvas.getContext("2d").drawImage(bitmap, 0, 0);
  }

  _togglePlay() {
    if (this.playing) this._stopPlay();
    else this._startPlay();
  }

  _startPlay() {
    if (this.stamps.length < 2) return;
    this.playing             = true;
    this.playBtn.textContent = "Pause";
    this.playBtn.classList.add("is-playing");
    this._loop();
  }

  _stopPlay() {
    this.playing             = false;
    this.playBtn.textContent = "Play";
    this.playBtn.classList.remove("is-playing");
  }

  async _loop() {
    while (this.playing && this.host.view === "timeline" && this.stamps.length) {
      await this._show();
      await new Promise((resolve) => setTimeout(resolve, CubeTimeline.FRAME_MS));
      if (!this.playing) break;
      this.idx = (this.idx + 1) % this.stamps.length;
    }
    this._stopPlay();
  }
}

class CubeFigures {
  constructor(refs, host) {
    this.host = host;
    this.root = refs.root;
    this.loadedFor = null;
  }

  configure() {
    this.loadedFor = null;
    this.root.innerHTML = ThinkingOrb.panel("composing", "Loading figures…");
  }

  async render() {
    if (this.loadedFor === this.host.selectedId) return;

    const cubeId = this.host.selectedId;
    const data   = await Api.get(`/api/cubes/figures?id=${encodeURIComponent(cubeId)}`);
    if (cubeId !== this.host.selectedId) return;

    this.loadedFor = cubeId;

    if (!data || !data.ok) {
      this.root.innerHTML = `<p class="cube-hint">${SharedCharts.esc((data && data.error) || "Figure listing failed.")}</p>`;
      return;
    }
    if (!data.groups.length) {
      this.root.innerHTML = `<p class="cube-hint">This stamp holds no saved figures.</p>`;
      return;
    }

    const total = data.groups.reduce((sum, group) => sum + group.files.length, 0);
    let html    = `<p class="cube-hint">${total.toLocaleString()} saved figures in ${data.groups.length} folders · click a thumbnail to open it full size.</p>`;

    data.groups.forEach((group, index) => {
      const items = group.files.map((file) => {
        const rel = `${group.dir}/${file}`;
        const url = `/api/cubes/figure?id=${encodeURIComponent(cubeId)}&rel=${encodeURIComponent(rel)}`;
        return `<a class="cube-fig" href="${url}" target="_blank" rel="noopener" title="${SharedCharts.esc(rel)}">` +
          `<img src="${url}" loading="lazy" alt="${SharedCharts.esc(file)}" /><span>${SharedCharts.esc(file)}</span></a>`;
      }).join("");

      html += `<details class="cube-figgroup"${index === 0 ? " open" : ""}>` +
        `<summary>${SharedCharts.esc(group.dir)} <em>${group.files.length}</em></summary>` +
        `<div class="cube-figgrid">${items}</div></details>`;
    });

    this.root.innerHTML = html;
  }
}

class CubeView {
  static LABELS = { pred: "pred", predb: "pred B", diff: "A − B", gt: "gt", full: "reference" };
  static SLOT_COLORS = ["#e45756", "#4c78a8", "#54a24b", "#b279a2", "#eeca3b", "#72b7b2", "#f58518", "#9d755d"];
  static HOLD_SAVE_MS = 4000;
  static STATUS_TIMEOUT_MS = 15000;
  static STALL_AFTER_S = 5;
  static HOLD_HINT_MS = 800;
  static SWEEP_CACHE_BYTES = 192 * 1024 * 1024;
  static STORY_KEY = "cube-story";
  static STORY_STEP_MS = 2600;

  static tint(hex, alpha) {
    const raw  = hex.replace("#", "");
    const full = raw.length === 3 ? raw.split("").map((c) => c + c).join("") : raw;
    const r    = parseInt(full.slice(0, 2), 16);
    const g    = parseInt(full.slice(2, 4), 16);
    const b    = parseInt(full.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  constructor(refs) {
    this.strip         = refs.strip;
    this.stage         = refs.stage;
    this.page          = refs.stage.closest(".page");
    this.deck          = refs.deck;
    this.topdown       = refs.topdown;
    this.cross         = refs.cross;
    this.coords        = refs.coords;
    this.back          = refs.back;
    this.hint          = refs.hint;
    this.panels        = refs.panels;
    this.slicesEl      = refs.slices;
    this.slicesAt      = refs.slicesAt;
    this.sourcesEl     = refs.sources;
    this.profilesEl    = refs.profiles;
    this.profAt        = refs.profAt;
    this.profMetricsEl = refs.profMetrics;
    this.profModeBtns  = refs.profModeBtns;
    this.profMixWrap   = refs.profMixWrap;
    this.profMixChk    = refs.profMix;
    this.profPanels    = refs.profPanels;
    this.spaceBtns     = refs.spaceBtns || [];
    this.modeBtns      = refs.modeBtns || [];
    this.viewEls       = refs.views || [];
    this.jumpAz        = refs.jumpAz;
    this.jumpRg        = refs.jumpRg;
    this.jumpGo        = refs.jumpGo;
    this.jumpPrint     = refs.jumpPrint;
    this.jumpDownload  = refs.jumpDownload;
    this.jumpCsv       = refs.jumpCsv;
    this.jumpParams    = refs.jumpParams;
    this.jumpRowRange  = refs.jumpRowRange;
    this.jumpColRange  = refs.jumpColRange;
    this.progress      = refs.progress;
    this.progressFill  = refs.progressFill;
    this.progressLabel = refs.progressLabel;

    this.atLabels = {
      col : this.slicesEl.querySelector('.cube-cutgroup__at[data-axis="col"]'),
      row : this.slicesEl.querySelector('.cube-cutgroup__at[data-axis="row"]'),
    };

    this.cubes = [];
    this.selectedId = null;
    this.runStrip = null;
    this.meta = null;
    this.space = "physical";
    this.point = null;
    this.mode = "map";
    this.locked = null;
    this.focus = null;
    this.focusSeq = 0;
    this.cutsAdopted = 0;
    this.holdTimer = null;
    this.holdHintTimer = null;
    this.holdHintOn = false;
    this.holdFired = false;
    this.saving = false;
    this.entered = false;
    this.polling = false;
    this.profMode = "raw";
    this.profData = null;
    this.profParams = null;
    this.profMix = false;
    this.profQueued = null;
    this.profFetching = false;
    this.ssimQueued = null;
    this.ssimFetching = false;
    this.view = "explorer";
    this.colors = {};
    this.visible = new Set();
    this.cmap = localStorage.getItem("cube-cmap") || "jet";
    this.bitmapCache = new Map();
    this.bitmapBytes = 0;

    this.sweeps   = (refs.sweeps || []).map((sweep) => new CubeSweep(sweep, this));
    this.params   = refs.params ? new CubeParams(refs.params, this) : null;
    this.metrics  = refs.metrics ? new CubeMetrics(refs.metrics, this) : null;
    this.transect = refs.transect ? new CubeTransect(refs.transect, this) : null;
    this.cloud    = refs.cloud ? new CubeCloud(refs.cloud, this) : null;
    this.volume   = refs.volume ? new CubeVolume(refs.volume, this) : null;
    this.timeline = new CubeTimeline(refs.timeline, this);
    this.figures  = new CubeFigures(refs.figures, this);

    this.mapWrap = this.topdown.closest(".cube-map__wrap");

    this.topdown.addEventListener("mousemove", (ev) => this._onMove(ev));
    this.topdown.addEventListener("mousedown", (ev) => this._onHoldStart(ev));
    this.topdown.addEventListener("mouseup", () => this._cancelHold());
    this.topdown.addEventListener("mouseleave", () => this._cancelHold());
    this.topdown.addEventListener("click", (ev) => this._onClick(ev));
    this.topdown.addEventListener("load", () => this.mapWrap.classList.remove("is-loading"));
    this.topdown.addEventListener("error", () => this.mapWrap.classList.remove("is-loading"));

    this.back.addEventListener("click", () => this._exitSlices());
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && this.mode === "slices") this._exitSlices();
    });

    this.panels.forEach((panel) => {
      panel.canvas.addEventListener("mousemove", (ev) => this._onSliceMove(panel, ev));
    });

    this.spaceBtns.forEach((btn) => {
      btn.addEventListener("click", () => this._setSpace(btn.dataset.space));
    });
    this.profModeBtns.forEach((btn) => {
      btn.addEventListener("click", () => this._setProfMode(btn.dataset.mode));
    });
    this.profMixChk.addEventListener("change", () => this._setProfMix(this.profMixChk.checked));
    this.modeBtns.forEach((btn) => {
      btn.addEventListener("click", () => this._setView(btn.dataset.view));
    });

    this.cmapSel = refs.cmapSel || null;
    if (this.cmapSel) {
      this.cmapSel.value = this.cmap;
      this.cmapSel.addEventListener("change", () => this._setCmap(this.cmapSel.value));
    }

    if (this.jumpAz) this.jumpAz.addEventListener("change", () => this._setManualCut());
    if (this.jumpRg) this.jumpRg.addEventListener("change", () => this._setManualCut());
    if (this.jumpGo) this.jumpGo.addEventListener("click", () => this._setManualCut());
    if (this.jumpPrint) this.jumpPrint.addEventListener("click", () => this._printSlices());
    this.jumpDownload.addEventListener("click", () => this._downloadCuts());
    this.jumpCsv.addEventListener("click", () => this._downloadProfilesCsv());
    this.jumpParams.addEventListener("click", () => this._setView("params"));

    this.pendingState   = null;
    this.storySnapshots = {};
    this.storyTimer     = null;
    this.storyPanel     = refs.story.panel;
    this.storyName      = refs.story.name;
    this.storyPlayBtn   = refs.story.play;
    this.storyHint      = refs.story.hint;
    this.storyList      = refs.story.list;

    refs.linkBtn.addEventListener("click", () => this._copyLink());
    refs.story.toggle.addEventListener("click", () => this._toggleStoryPanel());
    refs.story.add.addEventListener("click", () => this._storyAdd());
    refs.story.export.addEventListener("click", () => this._storyExport());
    refs.story.clear.addEventListener("click", () => this._storyClear());
    this.storyPlayBtn.addEventListener("click", () => this._storyTogglePlay());
  }

  leave() {
    this._stopSweeps();
    this._storyStopPlay();
    if (this.metrics) this.metrics.leave();
    if (this.cloud) this.cloud.leave();
    this.timeline.leave();
  }

  async enter(param = null) {
    const state = this._parseState(param);
    if (state) this.pendingState = state;

    if (this.entered) {
      await this._refreshStrip();
      this._applyPendingState();
      return;
    }
    this.entered = true;
    await this.refresh();
    this._applyPendingState();
  }

  _parseState(param) {
    if (!param) return null;

    let query;
    try {
      query = new URLSearchParams(decodeURIComponent(param));
    } catch (e) {
      Toast.show("The cube link could not be read.", "warn");
      return null;
    }
    if (!query.get("id")) return null;

    return {
      id    : query.get("id"),
      view  : query.get("view"),
      row   : query.has("row") ? Number(query.get("row")) : null,
      col   : query.has("col") ? Number(query.get("col")) : null,
      layer : query.get("layer"),
      space : query.get("space"),
      cmap  : query.get("cmap"),
    };
  }

  stateParam() {
    const query = new URLSearchParams();
    query.set("id", this.selectedId);
    query.set("view", this.view);
    query.set("space", this.space);
    query.set("cmap", this.cmap);

    const point = this.locked || this.focus;
    if (point) {
      query.set("row", String(point.row));
      query.set("col", String(point.col));
    }
    if (this.metrics && this.metrics.layer) query.set("layer", this.metrics.layer.key);

    return encodeURIComponent(query.toString());
  }

  _applyPendingState() {
    const state = this.pendingState;
    if (!state || this.polling) return;

    if (state.id === this.selectedId) {
      this.pendingState = null;
      if (this.meta) this._applyState(state);
      return;
    }

    if (this.cubes.some((cube) => cube.id === state.id)) {
      this.select(state.id);
    } else if (this.cubes.length) {
      Toast.show("The linked cube is not in the current runs root.", "warn");
      this.pendingState = null;
    }
  }

  _applyState(state) {
    if (state.cmap && [...this.cmapSel.options].some((option) => option.value === state.cmap)) {
      this.cmapSel.value = state.cmap;
      this._setCmap(state.cmap);
    }
    if (state.space) this._setSpace(state.space);

    if (Number.isFinite(state.row) && Number.isFinite(state.col)) {
      const row   = Math.max(0, Math.min(this.meta.n_rows - 1, state.row));
      const col   = Math.max(0, Math.min(this.meta.n_cols - 1, state.col));
      const point = { row, col, fx: (col + 0.5) / this.meta.n_cols, fy: (row + 0.5) / this.meta.n_rows };
      if (this.view !== "explorer") this._setView("explorer");
      this._follow(point, true);
      this._enterSlices(point);
    }

    if (state.layer && this.metrics && this.metrics.layers.some((layer) => layer.key === state.layer)) this.metrics._setLayer(state.layer);
    if (state.view) {
      const btn = this.modeBtns.find((candidate) => candidate.dataset.view === state.view);
      if (btn && !btn.hidden) this._setView(state.view);
      else if (btn) Toast.show(`This cube has no ${btn.textContent.trim()} view; the link opened without it.`, "warn");
      else Toast.show(`The link names an unknown view '${state.view}'; it opened without it.`, "warn");
    }
  }

  _copyLink() {
    if (!this.selectedId || !this.meta) {
      Toast.show("Load a cube first.", "warn");
      return;
    }
    this._shareLink(this.stateParam(), "View link copied to the clipboard.");
  }

  _shareLink(param, copied) {
    const url = `${location.origin}${location.pathname}#/cube/${param}`;
    if (!navigator.clipboard) {
      Toast.show(url, "ok");
      return;
    }
    navigator.clipboard.writeText(url).then(() => Toast.show(copied, "ok")).catch(() => Toast.show(url, "ok"));
  }

  _storyStore() {
    return JSON.parse(localStorage.getItem(CubeView.STORY_KEY) || "[]");
  }

  _storySave(steps) {
    localStorage.setItem(CubeView.STORY_KEY, JSON.stringify(steps));
  }

  _toggleStoryPanel() {
    this.storyPanel.hidden = !this.storyPanel.hidden;
    if (!this.storyPanel.hidden) this._storyRender();
  }

  _storyAdd() {
    if (!this.selectedId || !this.meta) {
      Toast.show("Load a cube first.", "warn");
      return;
    }

    const steps = this._storyStore();
    const step  = {
      id    : `${Date.now()}_${Math.floor(Math.random() * 1e6)}`,
      name  : this.storyName.value.trim() || `step ${steps.length + 1}`,
      param : this.stateParam(),
    };

    const snapshot = this._snapshotDataUrl();
    if (snapshot) this.storySnapshots[step.id] = snapshot;

    steps.push(step);
    this._storySave(steps);
    this.storyName.value = "";
    this._storyRender();
    Toast.show(`Story step '${step.name}' recorded.`, "ok");
  }

  _snapshotDataUrl() {
    const active = this.viewEls.find((el) => !el.hidden);
    const canvas = [...active.querySelectorAll("canvas")].find((c) => c.width > 0 && c.offsetParent !== null);
    if (canvas) return canvas.toDataURL("image/png");

    const img = [...active.querySelectorAll("img")].find((candidate) => candidate.naturalWidth > 0 && candidate.offsetParent !== null);
    if (!img) return null;

    const copy  = document.createElement("canvas");
    copy.width  = img.naturalWidth;
    copy.height = img.naturalHeight;
    copy.getContext("2d").drawImage(img, 0, 0);
    return copy.toDataURL("image/png");
  }

  _storyRender() {
    const steps = this._storyStore();
    this.storyHint.textContent = steps.length ? `${steps.length} steps · snapshots last until the page reloads` : "No steps yet; set up a view and press Add step.";

    this.storyList.innerHTML = "";
    steps.forEach((step, index) => {
      const item  = document.createElement("li");
      const label = document.createElement("span");
      label.className   = "cube-story__name";
      label.textContent = `${index + 1}. ${step.name}`;
      item.appendChild(label);

      item.appendChild(this._storyButton("Go", () => this._storyGo(step)));
      item.appendChild(this._storyButton("Copy link", () => this._shareLink(step.param, "Step link copied.")));
      item.appendChild(this._storyButton("Remove", () => this._storyRemove(step)));
      this.storyList.appendChild(item);
    });
  }

  _storyButton(text, onClick) {
    const btn = document.createElement("button");
    btn.type        = "button";
    btn.className   = "btn btn--mini";
    btn.textContent = text;
    btn.addEventListener("click", onClick);
    return btn;
  }

  _storyRemove(step) {
    this._storySave(this._storyStore().filter((entry) => entry.id !== step.id));
    delete this.storySnapshots[step.id];
    this._storyRender();
  }

  _storyGo(step) {
    const state = this._parseState(step.param);
    if (!state) return;
    this.pendingState = state;
    this._applyPendingState();
  }

  _storyTogglePlay() {
    if (this.storyTimer !== null) {
      this._storyStopPlay();
      return;
    }

    const steps = this._storyStore();
    if (!steps.length) {
      Toast.show("The story has no steps yet.", "warn");
      return;
    }

    let index = 0;
    this.storyPlayBtn.textContent = "Stop";
    const advance = () => {
      this._storyGo(steps[index]);
      index += 1;
      if (index >= steps.length) {
        this._storyStopPlay();
        return;
      }
      this.storyTimer = setTimeout(advance, CubeView.STORY_STEP_MS);
    };
    this.storyTimer = setTimeout(advance, 0);
  }

  _storyStopPlay() {
    if (this.storyTimer !== null) clearTimeout(this.storyTimer);
    this.storyTimer = null;
    this.storyPlayBtn.textContent = "Play";
  }

  _storyExport() {
    const steps = this._storyStore();
    let count   = 0;

    steps.forEach((step, index) => {
      const snapshot = this.storySnapshots[step.id];
      if (!snapshot) return;
      FigureExport.download(`story_${String(index + 1).padStart(2, "0")}_${FigureExport.slug(step.name)}.png`, snapshot);
      count += 1;
    });

    if (!count) Toast.show("No snapshots in memory; a snapshot is taken when a step is added and lasts until the page reloads.", "warn");
  }

  _storyClear() {
    this._storySave([]);
    this.storySnapshots = {};
    this._storyRender();
  }

  fileTag() {
    const cube = this.cubes.find((c) => c.id === this.selectedId);
    const raw  = cube ? `${cube.run}_${cube.stamp}` : "cube";
    return raw.replace(/[^A-Za-z0-9._-]+/g, "_");
  }

  figureTag() {
    const parts = [this.fileTag(), this.space];
    if (this.point) parts.push(`row${String(this.point.row).padStart(4, "0")}_col${String(this.point.col).padStart(4, "0")}`);

    return parts.filter(Boolean).join("_");
  }

  async _refreshStrip() {
    const data = await Api.get(`/api/cubes?base=${encodeURIComponent(this._runsBase())}`);
    if (!data || data.error) return;
    this.cubes = data.cubes || [];
    this._renderStrip();
  }

  async refresh() {
    this.hint.hidden = false;
    this.hint.innerHTML = ThinkingOrb.busy("searching", "Loading saved cubes…");

    let data;
    try {
      data = await Api.get(`/api/cubes?base=${encodeURIComponent(this._runsBase())}`);
    } catch (e) {
      this.hint.classList.remove("is-loading");
      this.hint.textContent = "Backend unreachable.";
      return;
    }

    this.hint.classList.remove("is-loading");
    this.cubes = data.cubes || [];
    this._renderStrip();

    if (!this.cubes.length) {
      this.hint.textContent = data.error || "No saved cubes found. Run an inference with save_cubes=True first.";
      this.hint.hidden = false;
      this._showStage(false);
      return;
    }

    this.hint.textContent = "Select a cube directory to load it into memory.";
    this.hint.hidden = false;
  }

  _runsBase() {
    return ResultsSources.runs();
  }

  _renderStrip() {
    if (!this.runStrip) {
      this.runStrip = new RunStrip(this.strip, {
        stateFor : (cube) => cube.id === this.selectedId,
        onPick   : (cube) => this.select(cube.id),
        extras   : (cube) => this._stripExtras(cube),
      });
    }
    this.runStrip.render(this.cubes);
  }

  _stripExtras(cube) {
    if (!this.meta || cube.id === this.selectedId) return [];

    const attached = this.meta.attached && this.meta.attached.id === cube.id;
    const vs = document.createElement("span");
    vs.className = "cube-run__vs" + (attached ? " is-on" : "");
    vs.setAttribute("role", "button");
    vs.title = attached ? "Detach this comparison" : "Compare against the loaded cube";
    vs.textContent = attached ? "detach" : "vs";
    vs.addEventListener("click", (ev) => {
      ev.stopPropagation();
      this._toggleAttach(cube.id);
    });
    return [vs];
  }

  async select(cubeId) {
    if (this.polling) {
      Toast.show("A cube is still loading.", "warn");
      return;
    }
    if (cubeId === this.selectedId && this.meta) return;

    this._stopSweeps();
    this.selectedId = cubeId;
    if (this.runStrip) this.runStrip.close((this.cubes.find((c) => c.id === cubeId) || {}).group);
    this.meta = null;
    this.point = null;
    this.locked = null;
    this.focus = null;
    this.cutsAdopted = this.focusSeq;
    this.mode = "map";
    this.panels.forEach((panel) => {
      this._releasePanel(panel);
      panel.marker = 0;
      panel.queued = null;
      panel.fetching = false;
      if (panel.metric) panel.metric.textContent = "";
    });
    this.profQueued = null;
    this.profData = null;
    this.profParams = null;
    this.ssimQueued = null;
    this._clearProfMetrics();
    this.deck.dataset.mode = "map";
    this.back.hidden = true;
    this.cross.hidden = true;
    this._hideRefs();
    this.coords.textContent   = "Hover the image to cut every cube · click to lock the slices";
    this.slicesAt.textContent = "";
    this.profAt.textContent   = "Hover a slice to read the profiles at that position.";
    if (this.atLabels.col) this.atLabels.col.textContent = "";
    if (this.atLabels.row) this.atLabels.row.textContent = "";
    this._showStage(false);
    this.slicesEl.hidden = true;
    this.slicesEl.classList.remove("is-in");
    this.hint.hidden = true;
    this._renderStrip();

    const res = await Api.post("/api/cubes/load", { id: cubeId });
    if (!res.ok) {
      this.hint.textContent = res.error || "Cube load failed.";
      this.hint.hidden = false;
      return;
    }

    this._setProgress(0, "loading");
    this.progress.hidden = false;
    await this._poll();
  }

  async _poll() {
    this.polling = true;

    while (true) {
      const st = await Api.get("/api/cubes/status", CubeView.STATUS_TIMEOUT_MS);
      if (st.error) {
        this._failLoad(st.error);
        break;
      }

      if (st.id !== this.selectedId) {
        this.progress.hidden = true;
        break;
      }

      if (st.state === "loading") {
        this._setProgress(st.progress || 0, st.stage || "loading", st.reading || [], st.stalled_for || 0, st.where || []);
        await new Promise((r) => setTimeout(r, 400));
        continue;
      }

      if (st.state === "ready" && st.cube) {
        this._setProgress(1, "ready");
        this._display(st.cube);
        break;
      }

      this._failLoad(st.error || "Cube load failed.");
      break;
    }

    this.polling = false;
    this._applyPendingState();
  }

  _slowestStage(timings) {
    const [stage, seconds] = Object.entries(timings.stages).sort((a, b) => b[1] - a[1])[0];
    return `${stage} ${seconds.toFixed(1)} s`;
  }

  _failLoad(message) {
    this.progress.hidden = true;
    this.hint.textContent = message;
    this.hint.hidden = false;
  }

  _setProgress(frac, stage, reading = [], stalledFor = 0, where = []) {
    const pct                      = Math.max(0, Math.min(100, Math.round(frac * 100)));
    this.progressFill.style.width  = `${pct}%`;
    const label                    = CubeView.LABELS[stage] || stage;
    const mb                       = (bytes) => (bytes / 1048576).toFixed(0);
    const inFlight                 = reading.slice(0, 2).map((row) => `${row.file} ${mb(row.done)}/${mb(row.total)} MB`).join(", ");
    const tail                     = reading.length ? ` · ${reading.length} file${reading.length === 1 ? "" : "s"} in flight: ${inFlight}` : "";
    const stall                    = stalledFor >= CubeView.STALL_AFTER_S ? ` · no progress for ${Math.round(stalledFor)} s, worker at ${where.slice(-2).join(" < ") || "unknown"}` : "";
    this.progressLabel.textContent = `${label} — ${pct}%${tail}${stall}`;
  }

  _display(meta) {
    this.meta = meta;
    this.progress.hidden = true;
    this.coords.textContent = `loaded in ${meta.timings.total.toFixed(1)} s (${this._slowestStage(meta.timings)}) · hover the image to cut every cube · click to lock the slices`;
    this._clearBitmapCache();
    this.colors = this._readColors();

    this._syncSpaceBtns();
    this._syncProfModeBtns();

    const kept = meta.sources.filter((s) => this.visible.has(s));
    this.visible = new Set(kept.length ? kept : meta.sources);
    this._renderSourceToggles();
    this._applyVisibility();

    this.hint.hidden = true;
    this._showStage(true);
    this.mapWrap.classList.add("is-loading");
    this.topdown.src = `/api/cubes/primary?id=${encodeURIComponent(this.selectedId)}`;

    this._initCutBounds();
    this.sweeps.forEach((sweep) => sweep.configure());

    if (this.params) {
      this.params.configure(meta);
      const paramsBtn = this.modeBtns.find((btn) => btn.dataset.view === "params");
      if (paramsBtn) paramsBtn.hidden = !meta.params;
      this.jumpParams.hidden = !meta.params;
      this.profMixWrap.hidden = !meta.params;
      if (!meta.params && this.view === "params") this._setView("explorer");
    }

    if (this.metrics) {
      this.metrics.configure(meta);
      const hasMaps = (meta.metric_maps || []).length > 0;
      const metricsBtn = this.modeBtns.find((btn) => btn.dataset.view === "metrics");
      if (metricsBtn) metricsBtn.hidden = !hasMaps;
      if (!hasMaps && this.view === "metrics") this._setView("explorer");
    }

    if (this.transect) this.transect.configure();

    if (this.cloud) {
      this.cloud.configure(meta);
      const cloudBtn = this.modeBtns.find((btn) => btn.dataset.view === "cloud");
      if (cloudBtn) cloudBtn.hidden = !this.cloud.available;
      if (!this.cloud.available && this.view === "cloud") this._setView("explorer");
    }

    if (this.volume) {
      this.volume.configure(meta);
      const volumeBtn = this.modeBtns.find((btn) => btn.dataset.view === "volume");
      if (volumeBtn) volumeBtn.hidden = !this.volume.available;
      if (!this.volume.available && this.view === "volume") this._setView("explorer");
      if (this.view === "volume") this.volume.render();
    }

    this.timeline.configure(meta);
    if (this.view === "timeline") this.timeline.render();

    this.figures.configure();
    if (this.view === "figures") this.figures.render();

    this._follow({ row: Math.floor(meta.n_rows / 2), col: Math.floor(meta.n_cols / 2), fx: 0.5, fy: 0.5 }, true);
    this._consumeFocus();

    const sweep = this._sweepFor(this.view);
    if (sweep) sweep.play();
  }

  _showStage(shown) {
    this.stage.hidden = !shown;
    this._syncStageChrome();
  }

  _syncStageChrome() {
    this.page.classList.toggle("is-staged", !this.stage.hidden);
  }

  _readColors() {
    const css = getComputedStyle(this.stage);
    return {
      pred   : css.getPropertyValue("--src-pred").trim(),
      predb  : css.getPropertyValue("--src-predb").trim(),
      diff   : css.getPropertyValue("--src-diff").trim(),
      gt     : css.getPropertyValue("--src-gt").trim(),
      full   : css.getPropertyValue("--src-full").trim(),
      col    : css.getPropertyValue("--cut-col").trim(),
      row    : css.getPropertyValue("--cut-row").trim(),
      accent : css.getPropertyValue("--accent").trim(),
      warn   : css.getPropertyValue("--warn").trim(),
    };
  }

  openAt(cubeId, row, col) {
    this.pendingFocus = { row, col };
    if (cubeId === this.selectedId && this.meta) {
      this._consumeFocus();
      return;
    }
    this.select(cubeId);
  }

  _consumeFocus() {
    if (!this.pendingFocus || !this.meta) return;
    const { row, col } = this.pendingFocus;
    this.pendingFocus = null;

    const clampedRow = Math.max(0, Math.min(this.meta.n_rows - 1, row));
    const clampedCol = Math.max(0, Math.min(this.meta.n_cols - 1, col));

    this.setFocus(clampedRow, clampedCol);
    this.cutsAdopted = this.focusSeq;

    this._setView("explorer");
    this._follow({
      row : clampedRow,
      col : clampedCol,
      fx  : this.meta.n_cols > 1 ? clampedCol / (this.meta.n_cols - 1) : 0.5,
      fy  : this.meta.n_rows > 1 ? clampedRow / (this.meta.n_rows - 1) : 0.5,
    }, true);
  }

  async _toggleAttach(otherId) {
    if (!this.meta || !this.selectedId) return;

    const attached = this.meta.attached && this.meta.attached.id === otherId;
    const res = attached
      ? await Api.post("/api/cubes/detach", { id: this.selectedId })
      : await Api.post("/api/cubes/attach", { id: this.selectedId, other: otherId });

    if (!res || !res.ok) {
      Toast.show((res && res.error) || "Comparison attach failed.", "error");
      return;
    }

    this._refreshSources(res.cube);
    this._renderStrip();
    Toast.show(attached ? "Comparison detached." : "Comparison attached — pred B and A − B sources enabled.", "ok");
  }

  _refreshSources(meta) {
    this.meta = meta;
    this._clearBitmapCache();

    const kept = meta.sources.filter((s) => this.visible.has(s));
    this.visible = new Set(kept.length ? kept : meta.sources);
    ["predb", "diff"].forEach((source) => {
      if (meta.sources.includes(source)) this.visible.add(source);
    });

    this.panels.forEach((panel) => this._releasePanel(panel));

    this._renderSourceToggles();
    this._applyVisibility();
    this.sweeps.forEach((sweep) => sweep.configure());

    if (this.metrics) {
      this.metrics.configure(meta);
      const hasMaps    = (meta.metric_maps || []).length > 0;
      const metricsBtn = this.modeBtns.find((btn) => btn.dataset.view === "metrics");
      if (metricsBtn) metricsBtn.hidden = !hasMaps;
      if (this.view === "metrics") {
        if (hasMaps) this.metrics.render();
        else this._setView("explorer");
      }
    }

    if (this.transect) this.transect.renderStripLayers();

    if (this.volume) {
      this.volume.refreshSources(meta);
      if (this.view === "volume") this.volume.render();
    }

    const sweep = this._sweepFor(this.view);
    if (sweep) sweep.render();
    if (this.point) this._drawSlices(this.point.row, this.point.col);
    if (this.locked) this._queueProfiles(this.locked.row, this.locked.col);
  }

  _setCmap(cmap) {
    if (cmap === this.cmap) return;
    this.cmap = cmap;
    localStorage.setItem("cube-cmap", cmap);

    if (!this.meta) return;

    this.panels.forEach((panel) => this._releasePanel(panel));

    const sweep = this._sweepFor(this.view);
    if (sweep) sweep.syncSpace();
    if (this.transect) this.transect.syncSpace();
    if (this.view === "timeline") this.timeline.render();
    if (this.point) this._drawSlices(this.point.row, this.point.col);
  }

  _setSpace(space) {
    if (space === this.space || !["physical", "normalized"].includes(space)) return;
    this.space = space;
    this._syncSpaceBtns();

    if (!this.meta) return;
    const sweep = this._sweepFor(this.view);
    if (sweep) sweep.syncSpace();
    if (this.transect) this.transect.syncSpace();
    if (this.view === "timeline") this.timeline.render();
    if (this.point) this._drawSlices(this.point.row, this.point.col);
  }

  _setView(view) {
    if (!["explorer", "z", "row", "col", "params", "metrics", "transect", "cloud", "volume", "timeline", "figures"].includes(view) || view === this.view) return;

    this._stopSweeps();
    if (this.view === "timeline") this.timeline.leave();
    this.view = view;

    this.modeBtns.forEach((btn) => btn.classList.toggle("is-active", btn.dataset.view === view));
    this.viewEls.forEach((el) => { el.hidden = el.dataset.view !== view; });

    if (view === "params" && this.params && this.meta) {
      this.params.render();
      return;
    }
    if (view === "metrics" && this.metrics && this.meta) {
      this.metrics.render();
      return;
    }
    if (view === "transect" && this.transect && this.meta) {
      this.transect.render();
      return;
    }
    if (view === "cloud" && this.cloud && this.meta) {
      this.cloud.render();
      return;
    }
    if (view === "volume" && this.volume && this.meta) {
      this.volume.render();
      return;
    }
    if (view === "timeline" && this.meta) {
      this.timeline.render();
      return;
    }
    if (view === "figures" && this.meta) {
      this.figures.render();
      return;
    }

    if (view === "explorer" && this.meta && this.focus && this.cutsAdopted !== this.focusSeq) {
      const point = this.focusPoint();
      this._follow(point, true);
      this._enterSlices(point);
      return;
    }

    const sweep = this._sweepFor(view);
    if (sweep && this.meta) sweep.play();
  }

  _sweepFor(view) {
    return this.sweeps.find((sweep) => sweep.axis === view) || null;
  }

  _stopSweeps() {
    this.sweeps.forEach((sweep) => sweep.stop());
  }

  cacheBitmap(url) {
    const hit = this.bitmapCache.get(url);
    if (hit) {
      this.bitmapCache.delete(url);
      this.bitmapCache.set(url, hit);
      return hit.promise;
    }

    const entry = { promise: null, bytes: 0 };
    entry.promise = this._loadBitmap(url, entry);
    this.bitmapCache.set(url, entry);
    return entry.promise;
  }

  async _loadBitmap(url, entry) {
    try {
      const res = await fetch(url);
      if (!res.ok) { this.bitmapCache.delete(url); return null; }

      const bitmap = await createImageBitmap(await res.blob());
      entry.bytes = bitmap.width * bitmap.height * 4;
      this.bitmapBytes += entry.bytes;
      this._trimBitmapCache();
      return bitmap;
    } catch (e) {
      this.bitmapCache.delete(url);
      return null;
    }
  }

  _releasePanel(panel) {
    if (panel.bitmap && panel.bitmap.close) panel.bitmap.close();
    panel.bitmap = null;
    panel.key = null;
    panel.drawnSpace = null;
  }

  _trimBitmapCache() {
    while (this.bitmapBytes > CubeView.SWEEP_CACHE_BYTES && this.bitmapCache.size > 1) {
      const oldest = this.bitmapCache.keys().next().value;
      this.bitmapBytes -= this.bitmapCache.get(oldest).bytes;
      this._dropBitmap(this.bitmapCache.get(oldest));
      this.bitmapCache.delete(oldest);
    }
  }

  _clearBitmapCache() {
    this.bitmapCache.forEach((entry) => this._dropBitmap(entry));
    this.bitmapCache.clear();
    this.bitmapBytes = 0;
  }

  async _dropBitmap(entry) {
    const bitmap = await entry.promise;
    if (bitmap && bitmap.close && !this.sweeps.some((sweep) => sweep.holds(bitmap))) bitmap.close();
  }

  _setProfMode(mode) {
    if (mode === this.profMode || !["raw", "unit"].includes(mode)) return;
    this.profMode = mode;
    this._syncProfModeBtns();
    this._drawProfiles();
  }

  _renderSourceToggles() {
    this.sourcesEl.innerHTML = "";
    this.meta.sources.forEach((source) => {
      const btn          = document.createElement("button");
      btn.type           = "button";
      btn.className      = "cube-source";
      btn.dataset.source = source;
      btn.innerHTML      = `<i class="cube-dot" data-source="${source}"></i>${CubeView.LABELS[source]}`;
      btn.addEventListener("click", () => this._toggleSource(source));
      this.sourcesEl.appendChild(btn);
    });
  }

  _toggleSource(source) {
    if (this.visible.has(source)) {
      if (this.visible.size === 1) {
        Toast.show("At least one source must stay visible.", "warn");
        return;
      }
      this.visible.delete(source);
    } else {
      this.visible.add(source);
    }

    this._applyVisibility();
    if (this.point) this._drawSlices(this.point.row, this.point.col);
    this._drawProfiles();
  }

  _applyVisibility() {
    this.panels.forEach((panel) => {
      panel.root.hidden = !this.visible.has(panel.source);
    });
    this.profPanels.forEach((panel) => {
      panel.root.hidden = !this.visible.has(panel.source);
    });
    this.sourcesEl.querySelectorAll(".cube-source").forEach((btn) => {
      btn.classList.toggle("is-active", this.visible.has(btn.dataset.source));
    });
    if (this.profMetricsEl) {
      this.profMetricsEl.querySelectorAll("tr[data-source]").forEach((row) => {
        row.hidden = !this.visible.has(row.dataset.source);
      });
    }
    this.slicesEl.style.setProperty("--cube-rows", String(this.visible.size));

    this.sweeps.forEach((sweep) => sweep.applyVisibility());
  }

  _syncSpaceBtns() {
    this.spaceBtns.forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.space === this.space);
    });
  }

  _syncProfModeBtns() {
    this.profModeBtns.forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.mode === this.profMode);
    });
  }

  _pointFromEvent(ev) {
    if (!this.meta) return null;

    const rect = this.topdown.getBoundingClientRect();
    const fx = (ev.clientX - rect.left) / rect.width;
    const fy = (ev.clientY - rect.top) / rect.height;

    return {
      row: Math.min(this.meta.n_rows - 1, Math.max(0, Math.floor(fy * this.meta.n_rows))),
      col: Math.min(this.meta.n_cols - 1, Math.max(0, Math.floor(fx * this.meta.n_cols))),
      fx,
      fy,
    };
  }

  _onMove(ev) {
    if (this.mode !== "map") return;
    const point = this._pointFromEvent(ev);
    if (!point) return;
    this._follow(point);
  }

  _onClick(ev) {
    if (this.holdFired) {
      this.holdFired = false;
      return;
    }
    if (this.mode !== "map") return;
    const point = this._pointFromEvent(ev);
    if (!point) return;
    this._follow(point, true);
    this._enterSlices(point);
  }

  _onHoldStart(ev) {
    if (this.mode !== "map" || ev.button !== 0 || !this.meta) return;

    ev.preventDefault();
    this.holdFired = false;
    this._cancelHold();

    this.holdHintTimer = setTimeout(() => {
      this.holdHintOn = true;
      this.coords.textContent = "keep holding to save the slice figures…";
    }, CubeView.HOLD_HINT_MS);

    this.holdTimer = setTimeout(() => this._fireHoldSave(), CubeView.HOLD_SAVE_MS);
  }

  _cancelHold() {
    clearTimeout(this.holdHintTimer);
    clearTimeout(this.holdTimer);
    this.holdHintTimer = null;
    this.holdTimer = null;
    this.holdHintOn = false;
  }

  async _fireHoldSave() {
    this._cancelHold();
    if (this.mode !== "map" || !this.meta || !this.point) return;

    this.holdFired = true;
    this.holdHintOn = true;

    const { row, col } = this.point;
    this.coords.innerHTML = ThinkingOrb.busy("composing", `saving slice figures at row = ${row} · col = ${col}…`);

    await this._saveSlices(row, col);

    this.holdHintOn = false;
    if (this.point) this.coords.textContent = `row = ${this.point.row} · col = ${this.point.col} · click to lock`;
  }

  _printSlices() {
    if (!this.meta) return;

    const row = this._clampInt(this.jumpAz ? this.jumpAz.value : 0, this.meta.n_rows);
    const col = this._clampInt(this.jumpRg ? this.jumpRg.value : 0, this.meta.n_cols);

    this._syncCutInputs(row, col, true);
    this._saveSlices(row, col);
  }

  async _saveSlices(row, col) {
    if (!this.meta || !this.selectedId || this.saving) return;

    this.saving = true;
    if (this.jumpPrint) this.jumpPrint.disabled = true;

    const res = await Api.post("/api/cubes/save_slices", { id: this.selectedId, row, col, space: this.space, cmap: this.cmap });

    this.saving = false;
    if (this.jumpPrint) this.jumpPrint.disabled = false;

    if (!res || !res.ok) {
      Toast.show((res && res.error) || "Slice figure save failed.", "error");
      return;
    }

    Toast.show(`Saved ${res.files.length} slice figures → ${res.rel}`, "ok");
  }

  _downloadCuts() {
    if (!this.meta || !this.point) {
      Toast.show("Hover or lock a cut first.", "warn");
      return;
    }

    const row = String(this.point.row).padStart(4, "0");
    const col = String(this.point.col).padStart(4, "0");
    const tag = this.fileTag();

    let count = 0;
    this.panels.forEach((panel) => {
      if (panel.root.hidden || !panel.bitmap) return;
      FigureExport.saveBitmap(panel.bitmap, `${tag}_row${row}_col${col}_${panel.axis}_${panel.source}_${this.space}.png`);
      count += 1;
    });

    if (!count) Toast.show("No rendered cut to download.", "warn");
  }

  _downloadProfilesCsv() {
    if (!this.profData) {
      Toast.show("Lock a pixel first so the profiles are loaded.", "warn");
      return;
    }

    const row   = String(this.profData.row).padStart(4, "0");
    const col   = String(this.profData.col).padStart(4, "0");
    const lines = ["source,z,value"];

    Object.entries(this.profData.sources).forEach(([source, series]) => {
      series.z_values.forEach((z, i) => lines.push(`${source},${z},${series.values[i]}`));
    });

    FigureExport.download(`${this.fileTag()}_profiles_row${row}_col${col}.csv`, new Blob([lines.join("\n")], { type: "text/csv" }));
  }

  _setManualCut() {
    if (!this.meta) return;

    const row   = this._clampInt(this.jumpAz ? this.jumpAz.value : 0, this.meta.n_rows);
    const col   = this._clampInt(this.jumpRg ? this.jumpRg.value : 0, this.meta.n_cols);
    const point = { row, col, fx: (col + 0.5) / this.meta.n_cols, fy: (row + 0.5) / this.meta.n_rows };

    this._follow(point, true);
    this._enterSlices(point);
    this._syncCutInputs(row, col, true);
  }

  _syncCutInputs(row, col, force = false) {
    if (this.jumpAz && (force || document.activeElement !== this.jumpAz)) this.jumpAz.value = row;
    if (this.jumpRg && (force || document.activeElement !== this.jumpRg)) this.jumpRg.value = col;
  }

  _initCutBounds() {
    if (this.jumpAz) { this.jumpAz.min = 0; this.jumpAz.max = this.meta.n_rows - 1; }
    if (this.jumpRg) { this.jumpRg.min = 0; this.jumpRg.max = this.meta.n_cols - 1; }
    if (this.jumpRowRange) this.jumpRowRange.textContent = `0–${this.meta.n_rows - 1}`;
    if (this.jumpColRange) this.jumpColRange.textContent = `0–${this.meta.n_cols - 1}`;
  }

  setFocus(row, col) {
    this.focus     = { row, col };
    this.focusSeq += 1;
  }

  focusPoint() {
    if (!this.focus || !this.meta) return null;

    const row = Math.max(0, Math.min(this.meta.n_rows - 1, this.focus.row));
    const col = Math.max(0, Math.min(this.meta.n_cols - 1, this.focus.col));
    return { row, col, fx: (col + 0.5) / this.meta.n_cols, fy: (row + 0.5) / this.meta.n_rows };
  }

  _enterSlices(point) {
    this.mode                 = "slices";
    this.locked               = { row: point.row, col: point.col };
    this.deck.dataset.mode    = "slices";
    this.back.hidden          = false;
    this.slicesAt.textContent = `locked at row = ${point.row} · col = ${point.col} · hover a slice for profiles · Esc to go back`;

    this.setFocus(point.row, point.col);
    this.cutsAdopted = this.focusSeq;
    this._queueProfiles(point.row, point.col);
  }

  _exitSlices() {
    if (this.mode !== "slices") return;
    this.mode = "map";
    this.locked = null;
    this.deck.dataset.mode = "map";
    this.back.hidden = true;
    this.slicesAt.textContent = "";
    this._hideRefs();
  }

  _hideRefs() {
    this.panels.forEach((panel) => {
      if (panel.ref) panel.ref.hidden = true;
    });
  }

  _onSliceMove(panel, ev) {
    if (this.mode !== "slices" || !this.meta || !this.locked) return;

    const rect = panel.canvas.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
    const n    = panel.axis === "col" ? this.meta.n_rows : this.meta.n_cols;
    const idx  = Math.min(n - 1, Math.floor(frac * n));

    this.panels.forEach((p) => {
      if (!p.ref || p.root.hidden) return;
      if (p.axis === panel.axis) {
        p.ref.hidden = false;
        p.ref.style.left = `${frac * 100}%`;
      } else {
        p.ref.hidden = true;
      }
    });

    const row = panel.axis === "col" ? idx : this.locked.row;
    const col = panel.axis === "col" ? this.locked.col : idx;
    this._queueProfiles(row, col);
  }

  _follow(point, force = false) {
    if (!force && this.point && point.row === this.point.row && point.col === this.point.col) {
      this._moveCross(point);
      return;
    }

    this.point = point;
    this._moveCross(point);
    if (!this.holdHintOn) this.coords.textContent = `row = ${point.row} · col = ${point.col} · click to lock`;
    this._syncCutInputs(point.row, point.col);

    this._drawSlices(point.row, point.col);
  }

  _moveCross(point) {
    this.cross.hidden = false;
    this.cross.style.left = `${point.fx * 100}%`;
    this.cross.style.top = `${point.fy * 100}%`;
  }

  _revealSlices() {
    if (!this.slicesEl.hidden) return;
    this.slicesEl.hidden = false;
    requestAnimationFrame(() => requestAnimationFrame(() => this.slicesEl.classList.add("is-in")));
  }

  _drawSlices(row, col) {
    if (!this.meta) return;

    this._revealSlices();

    if (this.atLabels.col) this.atLabels.col.textContent = `col = ${col}`;
    if (this.atLabels.row) this.atLabels.row.textContent = `row = ${row}`;

    this.panels.forEach((panel) => {
      if (panel.root.hidden) return;
      this._updatePanel(panel, row, col);
    });

    this._queueSsim(row, col);
  }

  _updatePanel(panel, row, col) {
    const key    = panel.axis === "col" ? col : row;
    panel.marker = panel.axis === "col" ? row / this.meta.n_rows : col / this.meta.n_cols;

    if (panel.bitmap && panel.key === key && panel.drawnSpace === this.space) {
      this._paintSlice(panel);
      return;
    }

    panel.queued = { row, col, key };
    this._panelPump(panel);
  }

  async _panelPump(panel) {
    if (panel.fetching) return;
    panel.fetching = true;

    while (panel.queued) {
      const job = panel.queued;
      panel.queued = null;
      await this._fetchSlice(panel, job);
    }

    panel.fetching = false;
  }

  async _fetchSlice(panel, job) {
    const space         = this.space;
    const url           = `/api/cubes/slice?id=${encodeURIComponent(this.selectedId)}&source=${panel.source}&axis=${panel.axis}&row=${job.row}&col=${job.col}&space=${space}&cmap=${this.cmap}`;
    const skeletonTimer = panel.bitmap ? null : setTimeout(() => panel.root.classList.add("is-loading"), 120);

    try {
      const res = await fetch(url);
      if (!res.ok) return;

      const bitmap = await createImageBitmap(await res.blob());
      if (panel.bitmap && panel.bitmap.close) panel.bitmap.close();

      panel.bitmap = bitmap;
      panel.key = job.key;
      panel.drawnSpace = space;
      this._paintSlice(panel);
    } catch (e) {
    } finally {
      if (skeletonTimer) clearTimeout(skeletonTimer);
      panel.root.classList.remove("is-loading");
    }
  }

  _paintSlice(panel) {
    const bitmap = panel.bitmap;
    const canvas = panel.canvas;

    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
    }

    const ctx = canvas.getContext("2d");
    ctx.drawImage(bitmap, 0, 0);

    const x = panel.marker * canvas.width;

    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.lineWidth = 3.2;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();

    ctx.strokeStyle = this.colors[panel.axis] || "#fff";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  _queueSsim(row, col) {
    this.ssimQueued = { row, col };
    this._ssimPump();
  }

  async _ssimPump() {
    if (this.ssimFetching || !this.ssimQueued) return;
    this.ssimFetching = true;

    while (this.ssimQueued) {
      const { row, col } = this.ssimQueued;
      this.ssimQueued = null;
      await this._fetchSsim(row, col);
    }

    this.ssimFetching = false;
  }

  async _fetchSsim(row, col) {
    const space = this.space;

    let data;
    try {
      data = await Api.get(`/api/cubes/ssim?id=${encodeURIComponent(this.selectedId)}&row=${row}&col=${col}&space=${space}`);
    } catch (e) {
      return;
    }

    if (!data || !data.ok) return;
    this._applySsim(data);
  }

  _applySsim(data) {
    this.panels.forEach((panel) => {
      if (!panel.metric) return;

      if (panel.source === "gt") {
        panel.metric.textContent = "reference";
        return;
      }

      const group = data[panel.axis] || {};
      const value = group[panel.source];
      panel.metric.textContent = value === undefined || value === null ? "SSIM –" : `SSIM ${value.toFixed(3)}`;
    });
  }

  _queueProfiles(row, col) {
    this.profAt.textContent = `profiles at row = ${row} · col = ${col}`;
    this.profQueued         = { row, col };
    this._profPump();
  }

  async _profPump() {
    if (this.profFetching || !this.profQueued) return;
    this.profFetching = true;

    while (this.profQueued) {
      const { row, col } = this.profQueued;
      this.profQueued = null;
      await this._fetchProfiles(row, col);
    }

    this.profFetching = false;
  }

  _setProfMix(on) {
    this.profMix = !!on;
    if (this.profMix && this.profData && !this._mixtureMatches()) this._fetchProfParams(this.profData.row, this.profData.col);
    else this._drawProfiles();
  }

  _mixtureMatches() {
    return !!this.profParams && !!this.profData && this.profParams.row === this.profData.row && this.profParams.col === this.profData.col;
  }

  async _fetchProfParams(row, col) {
    const cubeId = this.selectedId;
    const data   = await Api.get(`/api/cubes/params_at?id=${encodeURIComponent(cubeId)}&row=${row}&col=${col}`);
    if (!data || !data.ok || cubeId !== this.selectedId || !this.profData || data.row !== this.profData.row || data.col !== this.profData.col) return;

    this.profParams = data;
    this._drawProfiles();
  }

  async _fetchProfiles(row, col) {
    const staleTimer = setTimeout(() => {
      this.profPanels.forEach((panel) => panel.root.classList.add("is-stale"));
    }, 180);

    let data;
    try {
      data = await Api.get(`/api/cubes/profiles?id=${encodeURIComponent(this.selectedId)}&row=${row}&col=${col}`);
    } catch (e) {
      data = null;
    }

    clearTimeout(staleTimer);
    this.profPanels.forEach((panel) => panel.root.classList.remove("is-stale"));

    if (!data || !data.ok) return;
    this.profData = data;
    this._drawProfiles();

    if (this.profMix && this.meta && this.meta.params && !this._mixtureMatches()) this._fetchProfParams(data.row, data.col);
    if (this.hasMetricLayer("seed_std_profile")) this._fetchProfStd(data.row, data.col);
  }

  async _fetchProfStd(row, col) {
    const cubeId = this.selectedId;
    const std    = await Api.get(`/api/cubes/metric_at?id=${encodeURIComponent(cubeId)}&key=seed_std_profile&row=${row}&col=${col}`);
    if (!std || !std.ok || cubeId !== this.selectedId || !this.profData || this.profData.row !== std.row || this.profData.col !== std.col) return;

    this.profAt.textContent = `profiles at row = ${std.row} · col = ${std.col} · seed std ${std.value === null ? "–" : this._fmt(std.value)}`;
  }

  hasMetricLayer(key) {
    return !!this.meta && this.meta.metric_maps.some((layer) => layer.key === key);
  }

  _drawProfiles() {
    if (!this.profData) return;

    let shared = null;
    if (this.profMode === "unit") {
      shared = 0;
      this.profPanels.forEach((panel) => {
        if (panel.root.hidden) return;
        const series = this.profData.sources[panel.source];
        if (series) shared = Math.max(shared, ...this._scaledValues(series));
      });
      if (!shared) shared = 1;
    }

    this.profPanels.forEach((panel) => {
      if (panel.root.hidden) return;
      const series = this.profData.sources[panel.source];
      if (series) this._drawProfile(panel, series, shared);
    });

    this._fillProfMetrics();
  }

  _fillProfMetrics() {
    if (!this.profMetricsEl) return;

    const gt = this.profData && this.profData.sources.gt;

    ["pred", "predb"].forEach((source) => {
      const row = this.profMetricsEl.querySelector(`tr[data-source="${source}"]`);
      if (!row) return;

      const series = this.profData && this.profData.sources[source];
      const scores = gt && series ? this._scoreCurve(series.values, gt.values) : null;

      ["mae", "mse", "r2"].forEach((key) => {
        const cell = row.querySelector(`td[data-k="${key}"]`);
        if (cell) cell.textContent = scores ? this._fmtMetric(scores[key], key) : "–";
      });
    });
  }

  _scoreCurve(values, ref) {
    const n = Math.min(values.length, ref.length);
    if (!n) return null;

    let sumAbs = 0, sumSq = 0, sumRef = 0;
    for (let i = 0; i < n; i++) {
      const d = values[i] - ref[i];
      sumAbs += Math.abs(d);
      sumSq  += d * d;
      sumRef += ref[i];
    }

    const mean = sumRef / n;
    let ssTot = 0;
    for (let i = 0; i < n; i++) ssTot += (ref[i] - mean) ** 2;

    return { mae: sumAbs / n, mse: sumSq / n, r2: 1 - sumSq / (ssTot + 1e-12) };
  }

  _fmtMetric(value, key) {
    if (!Number.isFinite(value)) return "–";
    if (key === "r2") return value.toFixed(3);
    return this._fmt(value);
  }

  _clearProfMetrics() {
    if (!this.profMetricsEl) return;
    this.profMetricsEl.querySelectorAll("td[data-k]").forEach((cell) => { cell.textContent = "–"; });
  }

  _drawProfile(panel, series, shared) {
    const canvas = panel.canvas;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 240;
    const h = canvas.clientHeight || 190;

    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const zValues = series.z_values;
    const values = this._scaledValues(series);
    const n = values.length;
    if (n < 2) return;

    const padL = 8, padR = 8, padT = 16, padB = 8;
    const innerW = w - padL - padR;
    const innerH = h - padT - padB;
    const hMin = zValues[0];
    const hMax = zValues[n - 1];
    const span = hMax - hMin || 1;
    const vMax = shared || Math.max(...values) || 1;

    const xAt = (value) => padL + (value / vMax) * innerW * 0.96;
    const yAt = (height) => padT + (1 - (height - hMin) / span) * innerH;

    ctx.strokeStyle = "rgba(220, 235, 245, 0.1)";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 3; i++) {
      const x = padL + (innerW * i) / 4;
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, h - padB);
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.moveTo(xAt(0), yAt(zValues[0]));
    for (let i = 0; i < n; i++) ctx.lineTo(xAt(values[i]), yAt(zValues[i]));
    ctx.lineTo(xAt(0), yAt(zValues[n - 1]));
    ctx.closePath();
    ctx.globalAlpha = 0.12;
    ctx.fillStyle = this.colors[panel.source] || "#555";
    ctx.fill();
    ctx.globalAlpha = 1;

    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      if (i === 0) ctx.moveTo(xAt(values[i]), yAt(zValues[i]));
      else ctx.lineTo(xAt(values[i]), yAt(zValues[i]));
    }
    ctx.strokeStyle = this.colors[panel.source] || "#555";
    ctx.lineWidth = 1.6;
    ctx.stroke();

    let peak = 0;
    for (let i = 1; i < n; i++) if (values[i] > values[peak]) peak = i;
    ctx.beginPath();
    ctx.arc(xAt(values[peak]), yAt(zValues[peak]), 2.6, 0, Math.PI * 2);
    ctx.fillStyle = this.colors[panel.source] || "#555";
    ctx.fill();

    ctx.fillStyle = "rgba(220, 235, 245, 0.86)";
    ctx.font = "600 9.5px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    ctx.fillText(this._fmt(hMax), padL + 2, padT - 5);
    ctx.fillText(this._fmt(hMin), padL + 2, h - padB - 3);
    ctx.textAlign = "right";
    ctx.fillText(`peak ${this._fmt(values[peak])} @ ${this._fmt(zValues[peak])}`, w - 4, padT - 5);

    this._drawMixture(panel, series, ctx, xAt, yAt);
  }

  _drawMixture(panel, series, ctx, xAt, yAt) {
    if (!this.profMix || !this._mixtureMatches()) return;

    const slotData = this.profParams.sources[panel.source];
    if (!slotData) return;

    const slots = slotData.filter((slot) => slot.active && Number.isFinite(slot.mu) && Number.isFinite(slot.amp));
    if (!slots.length) return;

    const zValues = series.z_values;
    const mixture = zValues.map((z) => slots.reduce((sum, slot) => {
      const sigma = Math.max(Math.abs(slot.sigma), 1e-6);
      return sum + slot.amp * Math.exp(-((z - slot.mu) ** 2) / (2 * sigma * sigma));
    }, 0));

    let factor = 1;
    if (this.profMode === "unit") {
      let area = 0;
      for (let i = 1; i < mixture.length; i++) area += 0.5 * (mixture[i] + mixture[i - 1]) * Math.abs(zValues[i] - zValues[i - 1]);
      factor = area > 0 ? 1 / area : 0;
    }

    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "rgba(220, 235, 245, 0.75)";
    ctx.lineWidth   = 1.2;
    ctx.beginPath();
    mixture.forEach((value, i) => {
      const x = xAt(value * factor);
      const y = yAt(zValues[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    slots.forEach((slot, index) => {
      const y     = yAt(slot.mu);
      const x     = xAt(slot.amp * factor);
      const color = CubeView.SLOT_COLORS[index % CubeView.SLOT_COLORS.length];

      ctx.strokeStyle = color;
      ctx.lineWidth   = 1.4;
      ctx.beginPath();
      ctx.moveTo(xAt(0), y);
      ctx.lineTo(x, y);
      ctx.stroke();

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  _scaledValues(series) {
    const values = series.values.map((v) => (Number.isFinite(v) ? Math.max(v, 0) : 0));
    if (this.profMode === "raw") return values;

    let area = 0;
    for (let i = 1; i < values.length; i++) {
      const dh = Math.abs(series.z_values[i] - series.z_values[i - 1]);
      area += 0.5 * (values[i] + values[i - 1]) * dh;
    }

    if (area <= 0) return values.map(() => 0);
    return values.map((v) => v / area);
  }

  _clampInt(value, count) {
    const n = Math.floor(Number(value));
    if (!Number.isFinite(n)) return 0;
    return Math.min(count - 1, Math.max(0, n));
  }

  _fmt(value) {
    const abs = Math.abs(value);
    if (abs >= 1000) return value.toFixed(0);
    if (abs >= 10)   return value.toFixed(1);
    if (abs >= 0.01 || abs === 0) return value.toFixed(2);
    return value.toExponential(1);
  }

}

window.CubeView = CubeView;
