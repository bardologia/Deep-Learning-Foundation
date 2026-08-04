"use strict";

class FitLabChart {
  static setup(canvas, cssHeight) {
    const dpr  = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const w    = Math.max(220, rect.width);

    canvas.width        = Math.round(w * dpr);
    canvas.height       = Math.round(cssHeight * dpr);
    canvas.style.width  = `${w}px`;
    canvas.style.height = `${cssHeight}px`;

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h: cssHeight };
  }

  static profile(canvas, height, pixel, series, showComponents) {
    const { ctx, w, h } = FitLabChart.setup(canvas, 220);
    const padL = 34, padR = 18, padT = 8, padB = 20;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    const h0 = height[0];
    const h1 = height[height.length - 1];
    let vMax  = Math.max(...pixel.raw, ...pixel.target);
    series.forEach((s) => { vMax = Math.max(vMax, ...s.total); });
    vMax = vMax > 0 ? vMax * 1.08 : 1;

    const x = (hv) => padL + ((hv - h0) / (h1 - h0)) * plotW;
    const y = (v)  => padT + plotH - (Math.max(0, v) / vMax) * plotH;

    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "rgba(20, 26, 22, 0.12)";
    ctx.fillStyle   = "rgba(20, 26, 22, 0.55)";
    ctx.font        = "10px sans-serif";
    ctx.lineWidth   = 1;
    ctx.textAlign   = "center";
    const step = (h1 - h0) / 5;
    for (let i = 0; i <= 5; i += 1) {
      const hv = h0 + step * i;
      ctx.beginPath();
      ctx.moveTo(x(hv), padT);
      ctx.lineTo(x(hv), padT + plotH);
      ctx.stroke();
      ctx.fillText(`${hv.toFixed(0)} m`, x(hv), h - 6);
    }
    ctx.textAlign = "right";
    for (let i = 0; i <= 2; i += 1) {
      const v = (vMax / 1.08) * i / 2;
      ctx.fillText(v.toExponential(1), padL - 4, y(v) + 3);
    }

    const line = (values, color, width, dash) => {
      ctx.strokeStyle = color;
      ctx.lineWidth   = width;
      ctx.setLineDash(dash || []);
      ctx.beginPath();
      values.forEach((v, i) => {
        const px = x(height[i]);
        const py = y(v);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    };

    ctx.fillStyle = "rgba(20, 26, 22, 0.10)";
    ctx.beginPath();
    ctx.moveTo(x(height[0]), y(0));
    pixel.target.forEach((v, i) => ctx.lineTo(x(height[i]), y(v)));
    ctx.lineTo(x(height[height.length - 1]), y(0));
    ctx.closePath();
    ctx.fill();

    line(pixel.raw, "rgba(20, 26, 22, 0.30)", 1);
    line(pixel.target, "rgba(20, 26, 22, 0.75)", 1.4);

    series.forEach((s) => {
      if (showComponents) {
        s.components.forEach((comp) => line(comp, s.color, 1, [4, 3]));
      }
      line(s.total, s.color, 2);
    });
  }

  static ksweep(canvas, series) {
    const { ctx, w, h } = FitLabChart.setup(canvas, 84);
    const padL = 34, padR = 18, padT = 6, padB = 16;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    const kMax = Math.max(...series.map((s) => s.perK.length));
    let lo = Infinity, hi = -Infinity;
    series.forEach((s) => s.perK.forEach((row) => {
      lo = Math.min(lo, row.mse, row.penalised);
      hi = Math.max(hi, row.mse, row.penalised);
    }));
    if (!isFinite(lo) || lo <= 0) lo = 1e-8;
    if (!isFinite(hi) || hi <= lo) hi = lo * 10;

    const lLo = Math.log10(lo), lHi = Math.log10(hi);
    const x = (k) => padL + (kMax === 1 ? 0.5 : (k - 1) / (kMax - 1)) * plotW;
    const y = (v) => padT + plotH - ((Math.log10(Math.max(v, 1e-12)) - lLo) / (lHi - lLo || 1)) * plotH;

    ctx.clearRect(0, 0, w, h);
    ctx.font      = "10px sans-serif";
    ctx.fillStyle = "rgba(20, 26, 22, 0.55)";
    ctx.textAlign = "center";
    for (let k = 1; k <= kMax; k += 1) ctx.fillText(`K${k}`, x(k), h - 4);
    ctx.textAlign = "right";
    ctx.fillText(hi.toExponential(0), padL - 4, padT + 8);
    ctx.fillText(lo.toExponential(0), padL - 4, padT + plotH);

    const trace = (rows, pick, color, dash) => {
      ctx.strokeStyle = color;
      ctx.lineWidth   = 1.4;
      ctx.setLineDash(dash || []);
      ctx.beginPath();
      rows.forEach((row, i) => {
        const px = x(row.k), py = y(pick(row));
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    };

    series.forEach((s) => {
      trace(s.perK, (r) => r.mse, s.color, [3, 3]);
      trace(s.perK, (r) => r.penalised, s.color);
      const chosen = s.perK[s.kUsed - 1];
      if (chosen) {
        ctx.fillStyle = s.color;
        ctx.beginPath();
        ctx.arc(x(chosen.k), y(chosen.penalised), 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }
}

class FitLabView {
  static DEFAULT_BASE = "/ste/rnd/User/vice_vi/Dataset";
  static MAX_PIXELS   = 24;
  static COMPONENTS   = ["sigma", "amp", "mu"];
  static PALETTE      = ["#1d4fd8", "#b91c1c", "#0f766e", "#7c3aed", "#b45309", "#0e7490", "#be185d", "#4d7c0f"];

  constructor(refs) {
    this.refs   = refs;
    this.meta   = null;
    this.mapSrc = "slc";
    this.pixels = [];
    this.runs   = [];
    this.showComponents = true;

    this.entered   = false;
    this.loadToken = 0;
    this.fitToken  = 0;
    this.runSeq    = 0;

    refs.scanBtn.addEventListener("click", () => this._scan());
    refs.baseInput.addEventListener("keydown", (ev) => { if (ev.key === "Enter") this._scan(); });
    refs.mapImg.addEventListener("click", (ev) => this._onMapClick(ev));
    refs.mapImg.addEventListener("mousemove", (ev) => this._onMapMove(ev));
    refs.addBtn.addEventListener("click", () => this._onAddPixel());
    refs.clearBtn.addEventListener("click", () => { this.pixels = []; this._renderPixels(); });
    refs.runBtn.addEventListener("click", () => this._runFit());

    refs.mapSrcWrap.querySelectorAll(".cube-space").forEach((btn) => {
      btn.addEventListener("click", () => this._setMapSrc(btn));
    });
    refs.modeWrap.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => {
        const on = btn.classList.toggle("is-active");
        btn.setAttribute("aria-pressed", on ? "true" : "false");
      });
    });
    refs.compsWrap.querySelectorAll(".cube-space").forEach((btn) => {
      btn.addEventListener("click", () => {
        refs.compsWrap.querySelectorAll(".cube-space").forEach((b) => b.classList.toggle("is-active", b === btn));
        this.showComponents = btn.dataset.comps === "on";
        this._renderCards();
      });
    });

    window.addEventListener("resize", () => { if (this.meta && this.runs.length) this._renderCards(); });
  }

  enter() {
    if (this.entered) return;
    this.entered = true;
    this.refs.baseInput.value = this._defaultBase();
    this._scan();
  }

  _defaultBase() {
    const own = localStorage.getItem("fitlab-base");
    if (own) return own;
    try {
      const shared = JSON.parse(localStorage.getItem("results-sources") || "{}");
      if (shared.datasets) return shared.datasets;
    } catch (e) { /* fall through */ }
    return FitLabView.DEFAULT_BASE;
  }

  async _scan() {
    const base = this.refs.baseInput.value.trim();
    if (!base) return;
    localStorage.setItem("fitlab-base", base);

    this.refs.hint.textContent = "scanning…";
    const data = await window.apiGet(`/api/fitlab/datasets?base=${encodeURIComponent(base)}`);

    this.refs.strip.innerHTML = "";
    if (!data.ok) {
      this.refs.hint.textContent = data.error || "scan failed";
      return;
    }
    if (!data.datasets.length) {
      this.refs.hint.textContent = "no datasets with data/tomogram_full.npy found here";
      return;
    }

    this.refs.hint.textContent = "pick a dataset to load its tomogram";
    data.datasets.forEach((ds) => {
      const btn = document.createElement("button");
      btn.type        = "button";
      btn.className   = "fl-dataset";
      btn.textContent = ds.name;
      btn.title       = ds.path;
      btn.addEventListener("click", () => this._select(ds.path, btn));
      this.refs.strip.appendChild(btn);
    });
  }

  async _select(path, btn) {
    this.refs.strip.querySelectorAll(".fl-dataset").forEach((b) => b.classList.toggle("is-active", b === btn));

    const res = await window.apiPost("/api/fitlab/load", { path });
    if (!res.ok) {
      this.refs.hint.textContent = res.error || "load refused";
      return;
    }

    this.meta   = null;
    this.pixels = [];
    this.runs   = [];
    this.refs.stage.hidden    = true;
    this.refs.results.hidden  = true;
    this.refs.progress.hidden = false;
    this._renderRuns();

    const token = ++this.loadToken;
    while (token === this.loadToken) {
      const st = await window.apiGet("/api/fitlab/status");
      if (st.state === "error") {
        this.refs.progress.hidden  = true;
        this.refs.hint.textContent = st.error || "load failed";
        return;
      }
      if (st.state === "ready") {
        this.refs.progress.hidden = true;
        this._ready(st.meta);
        return;
      }
      this.refs.progressFill.style.width  = `${Math.round((st.progress || 0) * 100)}%`;
      this.refs.progressLabel.textContent = st.stage || "loading…";
      await new Promise((r) => setTimeout(r, 400));
    }
  }

  _ready(meta) {
    this.meta = meta;
    this.refs.hint.textContent = `${meta.name} · ${meta.az} × ${meta.rg} px · ${meta.h} height bins · heights ${meta.height_range[0]} to ${meta.height_range[1]} m`;
    this.refs.stage.hidden = false;

    this.refs.azInput.max = meta.az - 1;
    this.refs.rgInput.max = meta.rg - 1;
    this.refs.azRange.textContent = `0–${meta.az - 1}`;
    this.refs.rgRange.textContent = `0–${meta.rg - 1}`;
    this.refs.truncationInput.value = Math.min(Number(this.refs.truncationInput.value), meta.h) || meta.h;

    this._refreshMap();
    this._renderPixels();
    this._renderCards();
  }

  _refreshMap() {
    this.refs.mapImg.src = `/api/fitlab/map?src=${this.mapSrc}&t=${Date.now()}`;
  }

  _setMapSrc(btn) {
    this.refs.mapSrcWrap.querySelectorAll(".cube-space").forEach((b) => b.classList.toggle("is-active", b === btn));
    this.mapSrc = btn.dataset.src;
    if (this.meta) this._refreshMap();
  }

  _mapCoords(ev) {
    const rect = this.refs.mapImg.getBoundingClientRect();
    const fx   = (ev.clientX - rect.left) / rect.width;
    const fy   = (ev.clientY - rect.top) / rect.height;
    const az   = Math.min(this.meta.az - 1, Math.max(0, Math.round(fx * this.meta.az - 0.5)));
    const rg   = Math.min(this.meta.rg - 1, Math.max(0, Math.round(fy * this.meta.rg - 0.5)));
    return { az, rg };
  }

  _onMapMove(ev) {
    if (!this.meta) return;
    const { az, rg } = this._mapCoords(ev);
    this.refs.coords.textContent = `az ${az} · rg ${rg}`;
  }

  _onMapClick(ev) {
    if (!this.meta) return;
    const { az, rg } = this._mapCoords(ev);
    this._addPixel(az, rg);
  }

  _onAddPixel() {
    if (!this.meta) return;
    const az = Number(this.refs.azInput.value);
    const rg = Number(this.refs.rgInput.value);
    if (!Number.isInteger(az) || !Number.isInteger(rg)) return;
    if (az < 0 || az >= this.meta.az || rg < 0 || rg >= this.meta.rg) {
      this._flash("pixel out of range");
      return;
    }
    this._addPixel(az, rg);
  }

  _addPixel(az, rg) {
    const at = this.pixels.findIndex((p) => p.az === az && p.rg === rg);
    if (at >= 0) {
      this.pixels.splice(at, 1);
    } else {
      if (this.pixels.length >= FitLabView.MAX_PIXELS) {
        this._flash(`at most ${FitLabView.MAX_PIXELS} pixels`);
        return;
      }
      this.pixels.push({ az, rg });
    }
    this._renderPixels();
  }

  _renderPixels() {
    this.refs.marks.innerHTML  = "";
    this.refs.pixels.innerHTML = "";
    if (!this.meta) return;

    this.pixels.forEach((p, i) => {
      const mark = document.createElement("i");
      mark.className   = "fl-mark";
      mark.style.left  = `${((p.az + 0.5) / this.meta.az) * 100}%`;
      mark.style.top   = `${((p.rg + 0.5) / this.meta.rg) * 100}%`;
      mark.title       = `az ${p.az} · rg ${p.rg}`;
      mark.addEventListener("click", (ev) => { ev.stopPropagation(); this._addPixel(p.az, p.rg); });
      this.refs.marks.appendChild(mark);

      const chip = document.createElement("span");
      chip.className = "fl-pixel";
      chip.innerHTML = `<b>${i + 1}</b> az ${p.az} · rg ${p.rg}`;
      const rm = document.createElement("button");
      rm.type        = "button";
      rm.className   = "fl-pixel__rm";
      rm.textContent = "×";
      rm.addEventListener("click", () => this._addPixel(p.az, p.rg));
      chip.appendChild(rm);
      this.refs.pixels.appendChild(chip);
    });
  }

  _flash(message) {
    this.refs.runMsg.textContent = message;
    setTimeout(() => { if (this.refs.runMsg.textContent === message) this.refs.runMsg.textContent = ""; }, 4000);
  }

  _readMode() {
    const comps = FitLabView.COMPONENTS.filter((comp) => this.refs.modeWrap.querySelector(`button[data-comp="${comp}"]`).classList.contains("is-active"));
    if (!comps.length) throw new Error("free at least one parameter");
    return comps.join("_");
  }

  _readConfig() {
    const num = (input, name, integer) => {
      const v = Number(input.value);
      if (!isFinite(v) || (integer && !Number.isInteger(v))) throw new Error(`invalid ${name}`);
      return v;
    };

    return {
      k_max              : num(this.refs.kmaxInput, "K max", true),
      lambda_k           : num(this.refs.lambdaInput, "lambda"),
      mode               : this._readMode(),
      threshold_factor   : num(this.refs.thresholdInput, "threshold factor"),
      truncation_index   : num(this.refs.truncationInput, "truncation index", true),
      prominence_frac    : num(this.refs.prominenceInput, "prominence frac"),
      sigma_init_divisor : num(this.refs.sigdivInput, "sigma init divisor"),
      activity_threshold : num(this.refs.activityInput, "activity threshold"),
      adam_steps         : num(this.refs.stepsInput, "Adam steps", true),
      adam_lr            : num(this.refs.lrInput, "Adam lr"),
    };
  }

  _label(config) {
    const symbols = { sigma: "σ", amp: "A", mu: "μ" };
    const mode    = config.mode.split("_").map((comp) => symbols[comp]).join("+");
    return `K${config.k_max} λ${config.lambda_k} ${mode} thr${config.threshold_factor} σ÷${config.sigma_init_divisor} s${config.adam_steps} lr${config.adam_lr}`;
  }

  async _runFit() {
    if (!this.meta) return;
    if (!this.pixels.length) {
      this._flash("add at least one pixel");
      return;
    }

    let config;
    try {
      config = this._readConfig();
    } catch (err) {
      this._flash(err.message);
      return;
    }

    const res = await window.apiPost("/api/fitlab/fit", { pixels: this.pixels, config });
    if (!res.ok) {
      this._flash(res.error || "fit refused");
      return;
    }

    this.refs.runBtn.disabled     = true;
    this.refs.fitProgress.hidden  = false;

    const dataset = this.loadToken;
    const token   = ++this.fitToken;
    while (token === this.fitToken) {
      const st = await window.apiGet("/api/fitlab/fit_status");
      if (st.state === "error") {
        this.refs.runBtn.disabled    = false;
        this.refs.fitProgress.hidden = true;
        this._flash(st.error || "fit failed");
        return;
      }
      if (st.state === "done") break;
      this.refs.fitProgressFill.style.width  = `${Math.round((st.progress || 0) * 100)}%`;
      this.refs.fitProgressLabel.textContent = st.stage || "fitting…";
      await new Promise((r) => setTimeout(r, 350));
    }

    this.refs.runBtn.disabled    = false;
    this.refs.fitProgress.hidden = true;
    if (token !== this.fitToken || dataset !== this.loadToken) return;

    const result = await window.apiGet("/api/fitlab/fit_result");
    if (!result.ok) {
      this._flash(result.error || "no result");
      return;
    }

    this.runSeq += 1;
    this.runs.push({
      id      : this.runSeq,
      color   : FitLabView.PALETTE[(this.runSeq - 1) % FitLabView.PALETTE.length],
      label   : this._label(result.config),
      result,
      kSel    : "best",
      visible : true,
    });

    this._renderRuns();
    this._renderCards();
  }

  _renderRuns() {
    this.refs.runs.innerHTML = "";

    this.runs.forEach((run) => {
      const row = document.createElement("div");
      row.className = "fl-run" + (run.visible ? "" : " is-off");

      const dot = document.createElement("i");
      dot.className        = "fl-run__dot";
      dot.style.background = run.color;

      const label = document.createElement("span");
      label.className   = "fl-run__label";
      label.textContent = run.label;
      label.title       = "toggle visibility";
      label.addEventListener("click", () => { run.visible = !run.visible; this._renderRuns(); this._renderCards(); });

      const kSel = document.createElement("select");
      kSel.className = "fl-run__k";
      const kMax = run.result.config.k_max;
      [["best", "K best"], ...Array.from({ length: kMax }, (_, i) => [String(i + 1), `K ${i + 1}`])].forEach(([value, text]) => {
        const opt = document.createElement("option");
        opt.value       = value;
        opt.textContent = text;
        kSel.appendChild(opt);
      });
      kSel.value = run.kSel;
      kSel.addEventListener("change", () => { run.kSel = kSel.value; this._renderCards(); });

      const rm = document.createElement("button");
      rm.type        = "button";
      rm.className   = "fl-pixel__rm";
      rm.textContent = "×";
      rm.addEventListener("click", () => {
        this.runs = this.runs.filter((r) => r !== run);
        this._renderRuns();
        this._renderCards();
      });

      row.append(dot, label, kSel, rm);
      this.refs.runs.appendChild(row);
    });
  }

  _runSeries(run, key) {
    const pixel = run.result.pixels.find((p) => `${p.az},${p.rg}` === key);
    if (!pixel || !pixel.active || !run.visible) return null;

    const kUsed = run.kSel === "best" ? pixel.best_k : Math.min(Number(run.kSel), pixel.per_k.length);
    const row   = pixel.per_k[kUsed - 1];

    return {
      run,
      pixel,
      kUsed,
      row,
      color      : run.color,
      total      : row.total,
      components : row.components,
      perK       : pixel.per_k,
    };
  }

  _renderCards() {
    this.refs.cards.innerHTML = "";
    const shown = this.runs.filter((r) => r.visible);
    this.refs.results.hidden = shown.length === 0;
    if (!shown.length) return;

    const keys = [];
    shown.forEach((run) => run.result.pixels.forEach((p) => {
      const key = `${p.az},${p.rg}`;
      if (!keys.includes(key)) keys.push(key);
    }));

    keys.forEach((key) => {
      const [az, rg] = key.split(",").map(Number);
      const sample   = shown.map((run) => run.result.pixels.find((p) => `${p.az},${p.rg}` === key)).find(Boolean);
      const series   = shown.map((run) => this._runSeries(run, key)).filter(Boolean);

      const card = document.createElement("article");
      card.className = "fl-card";

      const head = document.createElement("header");
      head.className = "fl-card__head";
      head.innerHTML = `<b>az ${az} · rg ${rg}</b><span>${sample.active ? `scale ${sample.scale.toExponential(2)}` : "inactive pixel — below activity threshold, not fitted"}</span>`;
      card.appendChild(head);

      const media = document.createElement("div");
      media.className = "fl-card__media";
      const canvas = document.createElement("canvas");
      media.appendChild(canvas);
      card.appendChild(media);

      const legend = document.createElement("div");
      legend.className = "fl-card__legend";
      series.forEach((s) => {
        const row  = document.createElement("div");
        row.className = "fl-legend";
        const best = s.kUsed === s.pixel.best_k ? "*" : "";
        const params = s.row.params.map((p) => `(a ${p.amp.toPrecision(3)}, μ ${p.mu.toFixed(1)}, σ ${p.sigma.toFixed(2)})`).join(" ");
        row.innerHTML = `<i class="fl-run__dot" style="background:${s.color}"></i><span class="fl-legend__label">${s.run.label}</span><span class="fl-legend__k">K${s.kUsed}${best}</span><span class="fl-legend__mse">mse ${s.row.mse.toExponential(2)}</span><span class="fl-legend__params">${params}</span>`;
        legend.appendChild(row);
      });
      card.appendChild(legend);

      let sweep = null;
      if (series.length) {
        const strip = document.createElement("div");
        strip.className = "fl-card__sweep";
        sweep = document.createElement("canvas");
        strip.appendChild(sweep);
        card.appendChild(strip);
      }

      this.refs.cards.appendChild(card);

      const height = series.length ? series[0].run.result.height : shown[0].result.height;
      FitLabChart.profile(canvas, height, sample, series, this.showComponents);
      if (sweep) FitLabChart.ksweep(sweep, series);
    });
  }
}

window.FitLabView = FitLabView;
