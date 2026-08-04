"use strict";

const FT_DEFAULTS = {
  profile_autoencoder: { batch: "256,512,1024,2048,4096", workers: "0,2,4,6,8", prefetch: "2,4,8,16", timed: 80, warmup: 8, paths: true },
  image_autoencoder:   { batch: "4,8,16,32,64", workers: "0,4,8,12,16", prefetch: "2,4,8", timed: 40, warmup: 5, paths: true },
  backbone:            { batch: "4,8,16,32,64", workers: "0,4,8,12,16", prefetch: "2,4,8", timed: 40, warmup: 5, paths: true },
};

const FT_WORKER_COLORS = ["#1d4fd8", "#0f766e", "#0891b2", "#7c3aed", "#b45309", "#be123c"];
const FT_GRID = "rgba(20, 25, 30, 0.08)";
const FT_AXIS = "#7d858b";
const FT_CUSTOM = "__custom__";

class FtChart {
  constructor(canvas, legendEl) {
    this.canvas = canvas;
    this.legendEl = legendEl;
    this.ctx = canvas.getContext("2d");
    this.pad = { l: 50, r: 14, t: 14, b: 32 };
  }

  _resize() {
    const ratio = window.devicePixelRatio || 1;
    const width = this.canvas.clientWidth || 480;
    const height = this.canvas.clientHeight || 240;
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(height * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.w = width;
    this.h = height;
  }

  _frame(yMax, xLabels, rotateX) {
    const ctx = this.ctx;
    const { l, r, t } = this.pad;
    const b = rotateX ? 60 : this.pad.b;
    const plotW = this.w - l - r;
    const plotH = this.h - t - b;

    ctx.clearRect(0, 0, this.w, this.h);
    ctx.strokeStyle = FT_GRID;
    ctx.fillStyle = FT_AXIS;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.lineWidth = 1;

    const rows = 5;
    for (let i = 0; i <= rows; i++) {
      const y = t + (plotH * i) / rows;
      ctx.beginPath();
      ctx.moveTo(l, y);
      ctx.lineTo(l + plotW, y);
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.textBaseline = "alphabetic";
      ctx.fillText(this._fmt(yMax * (1 - i / rows)), l - 8, y + 3);
    }

    xLabels.forEach((label, i) => {
      const x = xLabels.length === 1 ? l + plotW / 2 : l + (plotW * i) / (xLabels.length - 1);
      if (rotateX) {
        ctx.save();
        ctx.translate(x, t + plotH + 8);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(label, 0, 0);
        ctx.restore();
      } else {
        ctx.textAlign = "center";
        ctx.textBaseline = "alphabetic";
        ctx.fillText(label, x, t + plotH + 18);
      }
    });

    return { l, t, plotW, plotH };
  }

  _fmt(value) {
    if (value >= 1000) return (value / 1000).toFixed(value >= 10000 ? 0 : 1) + "k";
    if (value >= 100) return value.toFixed(0);
    return value.toFixed(value >= 10 ? 0 : 1);
  }

  empty(message) {
    this._resize();
    this.ctx.clearRect(0, 0, this.w, this.h);
    this.ctx.fillStyle = "#9aa196";
    this.ctx.font = '12px "JetBrains Mono", monospace';
    this.ctx.textAlign = "center";
    this.ctx.fillText(message, this.w / 2, this.h / 2);
    this._renderLegend([]);
  }

  _renderLegend(items) {
    if (!this.legendEl) return;
    this.legendEl.innerHTML = items
      .map((item) => `<span class="ft-legend__item"><i style="background:${item.color}"></i>${item.label}</span>`)
      .join("");
  }

  throughput(rows) {
    this._resize();
    const main = rows.filter((row) => row.phase === "main" && row.status === "ok");
    if (!main.length) {
      this.empty("awaiting results");
      return;
    }

    const batches = [...new Set(main.map((row) => row.batch_size))].sort((a, b) => a - b);
    const workers = [...new Set(main.map((row) => row.num_workers))].sort((a, b) => a - b);
    const yMax = Math.max(...main.map((row) => row.end_to_end_samples_per_s)) * 1.1 || 1;

    const geo = this._frame(yMax, batches.map(String));
    const ctx = this.ctx;

    const xAt = (batch) => {
      const i = batches.indexOf(batch);
      return batches.length === 1 ? geo.l + geo.plotW / 2 : geo.l + (geo.plotW * i) / (batches.length - 1);
    };
    const yAt = (value) => geo.t + geo.plotH * (1 - value / yMax);

    workers.forEach((worker, wi) => {
      const color = FT_WORKER_COLORS[wi % FT_WORKER_COLORS.length];
      const series = main.filter((row) => row.num_workers === worker).sort((a, b) => a.batch_size - b.batch_size);
      if (!series.length) return;

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      series.forEach((row, i) => {
        const x = xAt(row.batch_size);
        const y = yAt(row.end_to_end_samples_per_s);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      ctx.fillStyle = color;
      series.forEach((row) => {
        ctx.beginPath();
        ctx.arc(xAt(row.batch_size), yAt(row.end_to_end_samples_per_s), 3, 0, Math.PI * 2);
        ctx.fill();
      });
    });

    this._renderLegend(workers.map((worker, wi) => ({ label: `${worker} workers`, color: FT_WORKER_COLORS[wi % FT_WORKER_COLORS.length] })));
  }

  utilization(rows) {
    this._resize();
    const main = rows.filter((row) => row.phase === "main" && row.status === "ok");
    if (!main.length) {
      this.empty("awaiting results");
      return;
    }

    const ordered = main.slice().sort((a, b) => a.batch_size - b.batch_size || a.num_workers - b.num_workers);
    const labels = ordered.map((row) => `${row.batch_size}/${row.num_workers}`);
    const geo = this._frame(100, labels, true);
    const ctx = this.ctx;

    const yAt = (value) => geo.t + geo.plotH * (1 - value / 100);
    const slot = geo.plotW / ordered.length;
    const barW = Math.max(3, Math.min(22, slot * 0.45));

    ordered.forEach((row, i) => {
      const cx = geo.l + slot * (i + 0.5);
      const top = yAt(row.gpu_util_mean);
      const gradient = ctx.createLinearGradient(0, top, 0, geo.t + geo.plotH);
      gradient.addColorStop(0, "#0f766e");
      gradient.addColorStop(1, "rgba(15, 118, 110, 0.12)");
      ctx.fillStyle = gradient;
      ctx.fillRect(cx - barW / 2, top, barW, geo.t + geo.plotH - top);
    });

    ctx.strokeStyle = "#b45309";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ordered.forEach((row, i) => {
      const cx = geo.l + slot * (i + 0.5);
      const y = yAt(row.data_wait_fraction * 100);
      if (i === 0) ctx.moveTo(cx, y);
      else ctx.lineTo(cx, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#b45309";
    ordered.forEach((row, i) => {
      const cx = geo.l + slot * (i + 0.5);
      ctx.beginPath();
      ctx.arc(cx, yAt(row.data_wait_fraction * 100), 2.4, 0, Math.PI * 2);
      ctx.fill();
    });

    this._renderLegend([
      { label: "GPU utilization", color: "#0f766e" },
      { label: "data wait", color: "#b45309" },
    ]);
  }
}

class FtGauge {
  constructor(canvas, colorFn) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.colorFn = colorFn;
  }

  draw(value) {
    const ratio = window.devicePixelRatio || 1;
    const width = this.canvas.clientWidth || 140;
    const height = this.canvas.clientHeight || 116;
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(height * ratio);
    const ctx = this.ctx;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const cx = width / 2;
    const cy = height * 0.66;
    const radius = Math.min(width / 2, height * 0.78) - 12;
    const start = Math.PI * 0.75;
    const sweep = Math.PI * 1.5;
    const frac = value == null || isNaN(value) ? 0 : Math.max(0, Math.min(1, value / 100));

    ctx.lineCap = "round";
    ctx.lineWidth = 9;

    ctx.strokeStyle = "#e4e7e2";
    ctx.beginPath();
    ctx.arc(cx, cy, radius, start, start + sweep);
    ctx.stroke();

    if (frac > 0) {
      ctx.strokeStyle = this.colorFn(value);
      ctx.beginPath();
      ctx.arc(cx, cy, radius, start, start + sweep * frac);
      ctx.stroke();
    }

    ctx.fillStyle = "#16191b";
    ctx.textAlign = "center";
    ctx.font = '600 22px "JetBrains Mono", monospace';
    ctx.fillText(value == null || isNaN(value) ? "–" : `${value.toFixed(0)}`, cx, cy + 4);
    ctx.fillStyle = "#7d858b";
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText("%", cx, cy + 18);
  }
}

class FeedTuner {
  constructor() {
    this.results = [];
    this.jobId = null;
    this.source = null;
    this.buffer = "";
    this.meta = null;
    this.expected = 0;
    this.ready = false;
    this.datasetPath = "";
    this.paramsPath = "";
    this.configPromise = null;
    this.datasetBase = "";
    this.gpuCards = null;
  }

  enter() {
    if (!this.ready) this._wire();
    this._loadGpus();
    if (FT_DEFAULTS[this.mode].paths) this._loadDatasets();
    requestAnimationFrame(() => this._draw());
  }

  _wire() {
    this.ready = true;

    this.modes = [...document.querySelectorAll("#ft-modes .ft-mode")];
    this.paths = document.getElementById("ft-paths");
    this.inputs = {
      datasetSelect: document.getElementById("ft-dataset-select"),
      datasetCustom: document.getElementById("ft-dataset-custom"),
      datasetNote: document.getElementById("ft-dataset-note"),
      paramsSelect: document.getElementById("ft-params-select"),
      paramsCustom: document.getElementById("ft-params-custom"),
      paramsNote: document.getElementById("ft-params-note"),
      gpu: document.getElementById("ft-gpu"),
      batch: document.getElementById("ft-batch"),
      workers: document.getElementById("ft-workers"),
      prefetch: document.getElementById("ft-prefetch"),
      timed: document.getElementById("ft-timed"),
      warmup: document.getElementById("ft-warmup"),
      amp: document.getElementById("ft-amp"),
    };
    this.runBtn = document.getElementById("ft-run");
    this.stopBtn = document.getElementById("ft-stop");
    this.hint = document.getElementById("ft-hint");

    this.idle = document.getElementById("ft-readout-idle");
    this.chips = document.getElementById("ft-chips");
    this.progress = document.getElementById("ft-progress");
    this.progressFill = document.getElementById("ft-progress-fill");
    this.progressLabel = document.getElementById("ft-progress-label");
    this.dials = document.getElementById("ft-dials");
    this.dialsAt = document.getElementById("ft-dials-at");
    this.reco = document.getElementById("ft-reco");
    this.tableBody = document.getElementById("ft-table-body");
    this.log = document.getElementById("ft-log");

    this.throughputChart = new FtChart(document.getElementById("ft-chart-throughput"), document.getElementById("ft-legend-throughput"));
    this.utilChart = new FtChart(document.getElementById("ft-chart-util"), document.getElementById("ft-legend-util"));
    this.gauges = {
      util: new FtGauge(document.getElementById("ft-dial-util"), () => "#0f766e"),
      eff: new FtGauge(document.getElementById("ft-dial-eff"), () => "#1d4fd8"),
      wait: new FtGauge(document.getElementById("ft-dial-wait"), (v) => (v <= 5 ? "#0f766e" : v <= 40 ? "#b45309" : "#b91c1c")),
    };

    this.mode = "profile_autoencoder";
    this.modes.forEach((btn) => btn.addEventListener("click", () => this._selectMode(btn.dataset.mode)));
    this.runBtn.addEventListener("click", () => this._run());
    this.stopBtn.addEventListener("click", () => this._stop());

    this.inputs.datasetSelect.addEventListener("change", () => this._onDatasetSelect());
    this.inputs.datasetCustom.addEventListener("input", () => { this.datasetPath = this.inputs.datasetCustom.value.trim(); });
    this.inputs.paramsSelect.addEventListener("change", () => this._onParamsSelect());
    this.inputs.paramsCustom.addEventListener("input", () => { this.paramsPath = this.inputs.paramsCustom.value.trim(); });
    window.addEventListener("resize", () => this._draw());

    this._applyDefaults("profile_autoencoder");
  }

  async _loadGpus() {
    const previous = this.gpuCards ? this.gpuCards.value() : [0];

    if (!this.gpuCards) {
      this.gpuCards = new window.GpuCardSelect(this.inputs.gpu, { multi: false, initial: previous });
    } else {
      this.gpuCards.set(previous);
    }

    await this.gpuCards.load();
  }

  _selectMode(mode) {
    this.mode = mode;
    this.modes.forEach((btn) => btn.classList.toggle("is-active", btn.dataset.mode === mode));
    this._applyDefaults(mode);
    if (FT_DEFAULTS[mode].paths) this._loadDatasets();
  }

  _applyDefaults(mode) {
    const preset = FT_DEFAULTS[mode];
    this.inputs.batch.value = preset.batch;
    this.inputs.workers.value = preset.workers;
    this.inputs.prefetch.value = preset.prefetch;
    this.inputs.timed.value = preset.timed;
    this.inputs.warmup.value = preset.warmup;
    this.paths.hidden = !preset.paths;
    this.hint.textContent = "Pick a dataset and parameters detected on this machine, then run the sweep.";
  }

  _config() {
    if (!this.configPromise) {
      this.configPromise = window.apiGet("/api/scripts/tune_dataloader/config").catch(() => ({ ok: false }));
    }
    return this.configPromise;
  }

  _parentDir(path) {
    const trimmed = String(path || "").replace(/\/+$/, "");
    const cut = trimmed.lastIndexOf("/");
    return cut > 0 ? trimmed.slice(0, cut) : trimmed;
  }

  async _loadDatasets() {
    const note = this.inputs.datasetNote;
    note.textContent = "loading…";

    const config = await this._config();
    const leaves = (config && config.leaves) || [];
    const datasetDefault = (leaves.find((leaf) => leaf.path === "paths.dataset_path") || {}).value || "";
    const paramsDefault = (leaves.find((leaf) => leaf.path === "paths.parameters_path") || {}).value || "";
    this.datasetBase = this._parentDir(datasetDefault);
    this.paramsDefault = paramsDefault;

    const res = await window.apiGet(`/api/fs/datasets?base=${encodeURIComponent(this.datasetBase)}`);
    const items = res && res.ok ? res.datasets || [] : [];

    if (!res || !res.ok) {
      note.textContent = (res && res.error) || "could not list datasets — use a custom path";
    } else {
      note.textContent = items.length ? `${items.length} in ${res.base}` : `no datasets in ${res.base}`;
    }

    const options = items.map((dataset) => ({ value: dataset.path, label: dataset.name + (dataset.has_params ? "" : "  (no params)") }));
    this._fillSelect(this.inputs.datasetSelect, options, datasetDefault);
    this._syncFromSelect(this.inputs.datasetSelect, this.inputs.datasetCustom, "datasetPath");
    await this._loadParams(this.datasetPath);
  }

  async _loadParams(dataset) {
    const note = this.inputs.paramsNote;
    if (!dataset) {
      note.textContent = "select a dataset first";
      this._fillSelect(this.inputs.paramsSelect, [], "");
      this._syncFromSelect(this.inputs.paramsSelect, this.inputs.paramsCustom, "paramsPath");
      return;
    }

    note.textContent = "loading…";
    const res = await window.apiGet(`/api/fs/params?dataset=${encodeURIComponent(dataset)}`);
    const items = res && res.ok ? res.files || [] : [];
    note.textContent = res && res.ok ? (items.length ? `${items.length} under ${res.params_root}` : "no extracted params yet") : (res && res.error) || "could not list params";

    const options = items.map((file) => ({ value: file.path, label: file.name }));
    const preferred = items.some((file) => file.path === this.paramsDefault) ? this.paramsDefault : "";
    this._fillSelect(this.inputs.paramsSelect, options, preferred);
    this._syncFromSelect(this.inputs.paramsSelect, this.inputs.paramsCustom, "paramsPath");
  }

  _fillSelect(select, options, preferred) {
    const known = new Set(options.map((option) => option.value));
    select.innerHTML = "";
    options.forEach((option) => select.appendChild(this._option(option.value, option.label)));
    select.appendChild(this._option(FT_CUSTOM, "Custom path…"));
    select.value = preferred && known.has(preferred) ? preferred : options.length ? options[0].value : FT_CUSTOM;
  }

  _option(value, label) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    return option;
  }

  _syncFromSelect(select, custom, field) {
    if (select.value === FT_CUSTOM) {
      custom.hidden = false;
      this[field] = custom.value.trim();
    } else {
      custom.hidden = true;
      this[field] = select.value;
    }
  }

  _onDatasetSelect() {
    this._syncFromSelect(this.inputs.datasetSelect, this.inputs.datasetCustom, "datasetPath");
    if (this.inputs.datasetSelect.value !== FT_CUSTOM) this._loadParams(this.datasetPath);
  }

  _onParamsSelect() {
    this._syncFromSelect(this.inputs.paramsSelect, this.inputs.paramsCustom, "paramsPath");
  }

  _overrides() {
    const overrides = {
      mode: this.mode,
      gpu: String((this.gpuCards ? this.gpuCards.value()[0] : 0) || 0),
      batch_sizes: this.inputs.batch.value,
      worker_counts: this.inputs.workers.value,
      prefetch_factors: this.inputs.prefetch.value,
      timed_batches: String(this.inputs.timed.value || "60"),
      warmup_batches: String(this.inputs.warmup.value || "8"),
      use_amp: this.inputs.amp.checked ? "true" : "false",
    };
    if (FT_DEFAULTS[this.mode].paths) {
      if (this.datasetPath) overrides["paths.dataset_path"] = this.datasetPath;
      if (this.paramsPath) overrides["paths.parameters_path"] = this.paramsPath;
    }
    return overrides;
  }

  async _run() {
    if (this.jobId) return;

    this._reset();
    const res = await window.apiPost("/api/run", { script_key: "tune_dataloader", overrides: this._overrides() });
    if (!res.ok) {
      window.toast(res.error || "Launch failed", "error");
      this._appendLog(`launch failed: ${res.error || "unknown error"}`);
      return;
    }

    this.jobId = res.job_id;
    this.runBtn.classList.add("is-busy");
    this.stopBtn.hidden = false;
    this.idle.hidden = true;
    this.progress.hidden = false;
    window.toast("Feed sweep launched", "ok");
    this._connect();
  }

  async _stop() {
    if (!this.jobId) return;
    await window.apiPost(`/api/jobs/${this.jobId}/stop`, {});
    window.toast("Stop signal sent", "ok");
  }

  _reset() {
    this.results = [];
    this.buffer = "";
    this.meta = null;
    this.expected = 0;
    this.tableBody.innerHTML = "";
    this.log.textContent = "";
    this.chips.hidden = true;
    this.chips.innerHTML = "";
    this.reco.hidden = true;
    this.dials.hidden = true;
    this.progressFill.style.width = "0%";
    this.progressLabel.textContent = "";
    this._draw();
  }

  _connect() {
    this.source = new EventSource(`/api/jobs/${this.jobId}/stream`);
    this.source.onmessage = (ev) => this._onEvent(ev);
    this.source.onerror = () => this._reconnect();
  }

  _reconnect() {
    if (!this.jobId) return;
    if (!this.source || this.source.readyState !== EventSource.CLOSED) return;

    this.source.close();
    this.source = null;
    this._appendLog("stream dropped, reconnecting");
    setTimeout(() => {
      if (this.jobId && !this.source) this._connect();
    }, 2000);
  }

  _disconnect() {
    if (this.source) {
      this.source.close();
      this.source = null;
    }
    this.jobId = null;
    this.runBtn.classList.remove("is-busy");
    this.stopBtn.hidden = true;
  }

  _onEvent(ev) {
    let data;
    try {
      data = JSON.parse(ev.data);
    } catch (e) {
      return;
    }

    if (data.type === "chunk") this._consume(data.data);
    else if (data.type === "status" && data.status && data.status !== "running") this._appendLog(`process ${data.status}${data.code != null ? ` (exit ${data.code})` : ""}`);
    else if (data.type === "end") this._disconnect();
  }

  _consume(text) {
    this.buffer += text;
    const lines = this.buffer.split("\n");
    this.buffer = lines.pop();
    lines.forEach((line) => this._line(line));
  }

  _line(line) {
    const clean = line.replace(/\x1b\[[0-9;]*m/g, "").replace(/[\r]/g, "");
    const marker = clean.indexOf("@TUNE ");
    if (marker < 0) {
      if (clean.trim()) this._appendLog(clean);
      return;
    }

    const rest = clean.slice(marker + 6);
    const space = rest.indexOf(" ");
    if (space < 0) return;

    const kind = rest.slice(0, space);
    let payload;
    try {
      payload = JSON.parse(rest.slice(space + 1));
    } catch (e) {
      return;
    }
    this._tune(kind, payload);
  }

  _tune(kind, payload) {
    if (kind === "meta") this._onMeta(payload);
    else if (kind === "result") this._onResult(payload);
    else if (kind === "recommendation") this._onRecommendation(payload);
    else if (kind === "done") this._onDone(payload);
  }

  _onMeta(meta) {
    this.meta = meta;
    this.expected = meta.n_specs || 0;
    this.idle.hidden = true;
    this.chips.hidden = false;
    this.dials.hidden = false;
    this.chips.innerHTML = [
      ["mode", meta.mode],
      ["device", meta.device],
      ["model", meta.model_name],
      ["params", this._compact(meta.parameters)],
      ["sample", meta.sample],
    ]
      .map(([key, value]) => `<span class="ft-chip"><b>${key}</b>${value}</span>`)
      .join("");
    this.dialsAt.textContent = "waiting for the first measurement…";
    this._appendLog(`target ready — ${meta.item_source}`);
    this._drawDials(null);
  }

  _onResult(record) {
    this.results.push(record);
    this._appendRow(record);
    this._tickProgress(record);
    if (record.status === "ok") this._updateDials(record);
    this._draw();
  }

  _tickProgress(record) {
    const done = this.results.length;
    const pct = this.expected ? Math.min(100, Math.round((done / this.expected) * 100)) : 0;
    this.progressFill.style.width = `${record.phase === "refine" ? 100 : pct}%`;
    this.progressLabel.textContent = record.phase === "refine"
      ? `refining prefetch & pin-memory · ${done} configs measured`
      : `main sweep · ${done}/${this.expected || "?"} configs`;
  }

  _updateDials(record) {
    this._drawDials(record);
    this.dialsAt.textContent = `latest · batch ${record.batch_size} · ${record.num_workers} workers`;
  }

  _drawDials(record) {
    if (!this.gauges) return;
    this.gauges.util.draw(record ? record.gpu_util_mean : null);
    this.gauges.eff.draw(record ? Math.min(100, record.compute_efficiency * 100) : null);
    this.gauges.wait.draw(record ? record.data_wait_fraction * 100 : null);
  }

  _onRecommendation(payload) {
    const rec = payload.recommendation || {};
    const final = payload.final || {};
    if (!rec.found) {
      this.reco.hidden = false;
      this.reco.innerHTML = `<div class="ft-reco__head"><span class="ft-reco__title">No successful configuration</span></div>`;
      return;
    }

    const best = this.results.find((row) => row.status === "ok" && row.phase === "main" && row.batch_size === rec.batch_size && row.num_workers === rec.num_workers);
    if (best) {
      this._drawDials(best);
      this.dialsAt.textContent = `recommended · batch ${best.batch_size} · ${best.num_workers} workers`;
    }

    const verdict = final.cpu_bound
      ? "CPU-bound — the input pipeline is the ceiling. Beyond these settings, lower the per-item cost."
      : "GPU saturated — this configuration keeps the device fed.";

    this.reco.hidden = false;
    this.reco.className = "ft-reco" + (final.cpu_bound ? " is-cpu" : " is-good");
    this.reco.innerHTML = `
      <div class="ft-reco__head">
        <span class="ft-reco__dot"></span>
        <span class="ft-reco__title">Recommended configuration</span>
        <span class="ft-reco__verdict">${verdict}</span>
      </div>
      <div class="ft-reco__grid">
        ${this._recoCell("batch size", final.batch_size)}
        ${this._recoCell("workers", final.num_workers)}
        ${this._recoCell("prefetch", final.prefetch_factor)}
        ${this._recoCell("pin memory", final.pin_memory)}
        ${this._recoCell("GPU util", `${(rec.gpu_util_mean || 0).toFixed(0)}%`)}
        ${this._recoCell("data wait", `${(100 * (rec.data_wait_fraction || 0)).toFixed(1)}%`)}
      </div>`;
  }

  _recoCell(label, value) {
    return `<div class="ft-reco__cell"><span class="ft-reco__k">${label}</span><span class="ft-reco__v">${value}</span></div>`;
  }

  _onDone(payload) {
    this._appendLog(`done — results at ${payload.results_path}`);
    this.progressLabel.textContent = "sweep complete";
    window.toast("Feed sweep complete", "ok");
  }

  _appendRow(record) {
    const row = document.createElement("tr");
    row.className = "ft-row ft-row--" + record.phase + (record.status !== "ok" ? " ft-row--bad" : "");
    if (record.status !== "ok") {
      row.innerHTML = `<td>${record.phase}</td><td>${record.batch_size}</td><td>${record.num_workers}</td><td colspan="8">${record.status}</td>`;
    } else {
      const wait = 100 * record.data_wait_fraction;
      const target = 100 * (this.meta ? this.meta.wait_target : 0.05);
      row.innerHTML =
        `<td>${record.phase}</td><td>${record.batch_size}</td><td>${record.num_workers}</td>` +
        `<td>${record.prefetch_factor}</td><td>${record.pin_memory ? "on" : "off"}</td>` +
        `<td>${this._compact(record.loader_only_samples_per_s)}</td>` +
        `<td>${this._compact(record.compute_ceiling_samples_per_s)}</td>` +
        `<td class="ft-num">${this._compact(record.end_to_end_samples_per_s)}</td>` +
        `<td>${record.feed_ratio.toFixed(2)}</td>` +
        `<td class="${wait > target ? "ft-warn" : "ft-ok"}">${wait.toFixed(1)}%</td>` +
        `<td>${record.gpu_util_mean.toFixed(0)}%</td>`;
    }
    this.tableBody.appendChild(row);
  }

  _appendLog(text) {
    this.log.textContent += text + "\n";
    this.log.scrollTop = this.log.scrollHeight;
  }

  _compact(value) {
    if (value == null || isNaN(value)) return "–";
    if (value >= 1000) return (value / 1000).toFixed(value >= 10000 ? 0 : 1) + "k";
    return Number(value).toFixed(value >= 10 ? 0 : 1);
  }

  _draw() {
    if (!this.throughputChart) return;
    this.throughputChart.throughput(this.results);
    this.utilChart.utilization(this.results);
  }
}

window.FeedTuner = FeedTuner;
