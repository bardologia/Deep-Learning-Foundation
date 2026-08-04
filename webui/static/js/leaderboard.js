"use strict";

class LeaderboardView {
  static DIFF_TAGS = ["A", "B", "C", "D", "E", "F"];
  static AXIS_COLS = [
    { key: "model",    label: "model" },
    { key: "head",     label: "head" },
    { key: "matching", label: "matching" },
    { key: "k",        label: "K" },
    { key: "aug",      label: "aug" },
    { key: "presence", label: "presence" },
    { key: "loss",     label: "loss" },
  ];

  constructor(root) {
    this.root = root;

    this.data     = null;
    this.view     = "table";
    this.diffData = null;

    this.sortKey  = "mtime";
    this.sortDir  = -1;
    this.filter   = "";
    this.axisSel  = {};
    this.selected = [];

    this.diffFilter  = "";
    this.diffSeries  = false;
    this.diffChanged = false;

    this.storedCols = this._loadVisible();
    this.visible    = [];

    this.trialsData   = null;
    this.trialsMetric = localStorage.getItem("leaderboard-trials-metric") || "curve_mse_gt";

    this.curveRuns     = null;
    this.curveSelected = new Set();
    this.curveFilter   = "";
    this.curveTag      = localStorage.getItem("leaderboard-curve-tag") || "";
    this.curveLog      = false;
    this.curveData     = null;
  }

  enter() {
    if (this.view === "trials") this.loadTrials();
    else if (this.view === "curves") this.loadCurveRuns();
    else this.load();
  }

  async load() {
    this.root.innerHTML = `<div class="res-empty">Scanning inference results&hellip;</div>`;

    const base = this._runsBase();
    const data = await window.apiGet(`/api/leaderboard?base=${encodeURIComponent(base)}`);

    if (!data || !data.ok) {
      this.data = null;
      this.root.innerHTML = `<div class="res-empty">${this._esc((data && data.error) || "Could not scan the runs directory.")}</div>`;
      return;
    }

    this.data     = data;
    this.selected = this.selected.filter((id) => data.rows.some((row) => row.id === id));
    this.visible  = this._reconcileVisible(data.columns);

    this._render();
  }

  async loadTrials() {
    this.root.innerHTML = `<div class="res-empty">Aggregating seeded trials&hellip;</div>`;

    const base = this._runsBase();
    const data = await window.apiGet(`/api/leaderboard/trials?base=${encodeURIComponent(base)}`);

    if (!data || !data.ok) {
      this.trialsData = null;
      this.root.innerHTML = `<div class="res-empty">${this._esc((data && data.error) || "Could not scan the runs directory.")}</div>`;
      return;
    }

    this.trialsData = data;
    this.view       = "trials";
    this._render();
  }

  async loadCurveRuns() {
    this.root.innerHTML = `<div class="res-empty">Scanning training runs&hellip;</div>`;

    const base = this._runsBase();
    const data = await window.apiGet(`/api/curves/runs?base=${encodeURIComponent(base)}`);

    if (!data || !data.ok) {
      this.curveRuns = null;
      this.root.innerHTML = `<div class="res-empty">${this._esc((data && data.error) || "Could not scan the runs directory.")}</div>`;
      return;
    }

    this.curveRuns = data.runs;
    this.curveSelected = new Set([...this.curveSelected].filter((id) => data.runs.some((run) => run.id === id)));
    this.view = "curves";
    this._render();

    if (this.curveSelected.size) this.loadCurves();
  }

  async loadCurves() {
    if (!this.curveSelected.size) {
      this.curveData = null;
      this._render();
      return;
    }

    const runsQuery = [...this.curveSelected].map((id) => `run=${encodeURIComponent(id)}`).join("&");
    const data = await window.apiGet(`/api/curves?${runsQuery}&tag=${encodeURIComponent(this.curveTag)}`);

    if (!data || !data.ok) {
      window.toast((data && data.error) || "Could not load the training curves.", "warn");
      return;
    }

    this.curveData = data;
    this.curveTag = data.tag;
    localStorage.setItem("leaderboard-curve-tag", this.curveTag);
    this._render();
  }

  async showDiff() {
    if (this.selected.length < 2) return;

    const runsQuery = this.selected.map((id) => `run=${encodeURIComponent(id)}`).join("&");
    const data      = await window.apiGet(`/api/leaderboard/diff?${runsQuery}`);

    if (!data || !data.ok) {
      window.toast((data && data.error) || "Could not load the comparison.", "warn");
      return;
    }

    this.diffData = data;
    this.view     = "diff";
    this._render();
  }

  _render() {
    if (this.view === "diff" && this.diffData) {
      this.root.innerHTML = this._diffHtml();
      this._bindDiff();
      return;
    }
    if (this.view === "trials" && this.trialsData) {
      this.root.innerHTML = this._trialsHtml();
      this._bindTrials();
      return;
    }
    if (this.view === "curves" && this.curveRuns) {
      this.root.innerHTML = this._curvesHtml();
      this._bindCurves();
      return;
    }

    this.root.innerHTML = this._tableHtml();
    this._bindTable();
  }

  _modeToggleHtml() {
    const mode = this.view === "trials" || this.view === "curves" ? this.view : "table";
    return (
      `<div class="res-views" role="group" aria-label="Leaderboard mode">` +
      `<button type="button" data-lbview="table" class="${mode === "table" ? "is-active" : ""}">Runs</button>` +
      `<button type="button" data-lbview="trials" class="${mode === "trials" ? "is-active" : ""}">Trials</button>` +
      `<button type="button" data-lbview="curves" class="${mode === "curves" ? "is-active" : ""}">Curves</button>` +
      `</div>`
    );
  }

  _bindModeToggle() {
    this.root.querySelectorAll("[data-lbview]").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.dataset.lbview === "trials") this.loadTrials();
        else if (btn.dataset.lbview === "curves") this.loadCurveRuns();
        else { this.view = "table"; this.load(); }
      });
    });
  }

  _rows() {
    let rows = this.data.rows;

    if (this.filter) {
      const needle = this.filter.toLowerCase();
      rows = rows.filter((row) => row.run.toLowerCase().includes(needle) || row.group.toLowerCase().includes(needle));
    }

    Object.entries(this.axisSel).forEach(([axis, value]) => {
      if (value === "") return;
      rows = rows.filter((row) => row.axes && String(row.axes[axis]) === value);
    });

    const dir = this.sortDir;
    const key = this.sortKey;

    return [...rows].sort((ra, rb) => {
      const va = this._sortValue(ra, key);
      const vb = this._sortValue(rb, key);
      if (va === null && vb === null) return 0;
      if (va === null) return 1;
      if (vb === null) return -1;
      if (typeof va === "string") return va.localeCompare(vb) * dir;
      return (va - vb) * dir;
    });
  }

  _sortValue(row, key) {
    if (key === "run")   return row.run;
    if (key === "mtime") return row.mtime;
    if (LeaderboardView.AXIS_COLS.some((c) => c.key === key)) {
      if (!row.axes || row.axes[key] === undefined) return null;
      return row.axes[key];
    }
    const value = row.metrics[key];
    return value === undefined ? null : value;
  }

  _tableHtml() {
    const rows    = this._rows();
    const columns = this.data.columns.filter((c) => this.visible.includes(c.key));
    const ranges  = this._ranges(rows, columns);

    let html = this._toolbarHtml(rows.length);

    if (this.data.errors && this.data.errors.length) {
      html += `<p class="lb-errors">${this.data.errors.length} unreadable metrics file${this.data.errors.length === 1 ? "" : "s"} skipped.</p>`;
    }

    if (!this.data.rows.length) {
      return html + `<div class="res-empty">No inference results found. Run an inference first, then revisit this tab.</div>`;
    }
    if (!rows.length) {
      return html + `<div class="res-empty">No run matches the current filters.</div>`;
    }

    html += `<div class="lb-scroll"><table class="lb-table"><thead><tr>`;
    html += this._headCell("run", "run");
    LeaderboardView.AXIS_COLS.forEach((col) => { html += this._headCell(col.key, col.label); });
    columns.forEach((col) => { html += this._headCell(col.key, col.label, true); });
    html += this._headCell("mtime", "when");
    html += `</tr></thead><tbody>`;

    rows.forEach((row) => {
      const selected = this.selected.includes(row.id);
      html += `<tr class="lb-row${row.n_seeds ? " lb-row--unit" : ""}${selected ? " is-selected" : ""}" data-id="${this._esc(row.id)}" title="${this._esc(row.id)}">`;
      html += `<td class="lb-run"><span class="lb-run__name">${this._esc(row.run)}</span><span class="lb-run__stamp">${this._esc(row.group !== "." ? row.group + " / " : "")}${this._esc(row.stamp)}</span></td>`;

      LeaderboardView.AXIS_COLS.forEach((col) => {
        const value = row.axes ? row.axes[col.key] : undefined;
        html += `<td class="lb-axis">${value === undefined || value === "" ? "&ndash;" : this._esc(value)}</td>`;
      });

      columns.forEach((col) => {
        const value = row.metrics[col.key];
        if (value === undefined) {
          html += `<td class="lb-val lb-val--none">&ndash;</td>`;
          return;
        }
        const good = this._goodness(value, ranges[col.key], col.direction);
        const best = good !== null && good > 0.999;
        html += `<td class="lb-val${best ? " is-best" : ""}">`;
        if (good !== null) html += `<i class="lb-bar" style="width:${Math.round(good * 100)}%"></i>`;
        html += `<span>${this._fmt(value)}</span></td>`;
      });

      html += `<td class="lb-when">${this._ago(row.mtime)}</td>`;
      html += `</tr>`;
    });

    html += `</tbody></table></div>`;
    return html;
  }

  _toolbarHtml(shown) {
    const total = this.data.rows.length;
    const count = shown === total ? `${total}` : `${shown} of ${total}`;

    let html = `<div class="lb-bar-top">`;
    html += this._modeToggleHtml();
    html += `<span class="lb-count">${count} inference result${total === 1 ? "" : "s"}</span>`;
    html += `<input type="search" class="res-filter" id="lb-filter" placeholder="Filter by run name" value="${this._esc(this.filter)}" spellcheck="false" />`;

    LeaderboardView.AXIS_COLS.forEach((col) => {
      if (col.key === "loss") return;
      const values = [...new Set(this.data.rows.filter((r) => r.axes).map((r) => String(r.axes[col.key])))].sort();
      if (values.length < 2) return;
      const active = this.axisSel[col.key] || "";
      html += `<select class="lb-select" data-axis="${col.key}" aria-label="Filter by ${col.label}">`;
      html += `<option value="">${col.label}: all</option>`;
      values.forEach((v) => { html += `<option value="${this._esc(v)}"${v === active ? " selected" : ""}>${this._esc(v)}</option>`; });
      html += `</select>`;
    });

    html += `<details class="lb-cols"><summary>Columns</summary><div class="lb-cols__menu">`;
    this.data.columns.forEach((col) => {
      const on = this.visible.includes(col.key);
      html += `<label><input type="checkbox" data-col="${col.key}"${on ? " checked" : ""} /> ${this._esc(col.label)}</label>`;
    });
    html += `</div></details>`;

    html += `<button type="button" class="btn btn--mini" id="lb-compare" ${this.selected.length >= 2 ? "" : "disabled"}>Compare (${this.selected.length})</button>`;
    html += `</div>`;
    return html;
  }

  _headCell(key, label, isMetric = false) {
    const active = this.sortKey === key;
    const arrow  = active ? (this.sortDir === 1 ? " &#9650;" : " &#9660;") : "";
    return `<th class="lb-th${isMetric ? " lb-th--metric" : ""}${active ? " is-sorted" : ""}" data-sort="${key}">${this._esc(label)}${arrow}</th>`;
  }

  _ranges(rows, columns) {
    const ranges = {};
    columns.forEach((col) => {
      const values = rows.map((row) => row.metrics[col.key]).filter((v) => v !== undefined && isFinite(v));
      if (values.length >= 2) ranges[col.key] = { min: Math.min(...values), max: Math.max(...values) };
    });
    return ranges;
  }

  _goodness(value, range, direction) {
    if (!range || !isFinite(value) || range.max === range.min || !direction) return null;
    const norm = (value - range.min) / (range.max - range.min);
    return direction > 0 ? norm : 1 - norm;
  }

  _trialsHtml() {
    const experiments = this.trialsData.experiments;
    const columns     = this.trialsData.columns;
    const metric      = columns.find((c) => c.key === this.trialsMetric) || columns[0];

    let html = `<div class="lb-bar-top">`;
    html += this._modeToggleHtml();
    html += `<span class="lb-count">${experiments.reduce((n, e) => n + e.units.length, 0)} seeded units in ${experiments.length} experiment${experiments.length === 1 ? "" : "s"}</span>`;
    html += `<select class="lb-select" id="lb-trials-metric" aria-label="Chart metric">`;
    columns.forEach((col) => { html += `<option value="${col.key}"${col.key === metric.key ? " selected" : ""}>${this._esc(col.label)}</option>`; });
    html += `</select>`;
    html += `</div>`;

    if (!experiments.length) {
      return html + `<div class="res-empty">No seeded trial runs found. Trials appear here once runs are laid out as <code>&lt;unit&gt;/seed&lt;N&gt;/</code> with saved inference metrics.</div>`;
    }

    experiments.forEach((experiment, index) => {
      html += `<section class="lb-diff__section">`;
      html += `<h4 class="res-section__cap">${this._esc(experiment.key === "." ? "runs root" : experiment.key)} <span>${experiment.units.length} units</span></h4>`;
      html += this._trialsChartHtml(experiment, metric);
      html += this._trialsTableHtml(experiment, columns, index);
      html += `</section>`;
    });

    return html;
  }

  _trialsChartHtml(experiment, metric) {
    const rows = experiment.units
      .map((unit) => ({ unit: unit.unit, agg: unit.metrics[metric.key] }))
      .filter((row) => row.agg);

    if (!rows.length) {
      return `<div class="res-empty res-empty--tight">No unit reports the metric "${this._esc(metric.label)}".</div>`;
    }

    rows.sort((a, b) => metric.direction >= 0 ? b.agg.mean - a.agg.mean : a.agg.mean - b.agg.mean);

    let lo = 0;
    let hi = 0;
    rows.forEach(({ agg }) => {
      lo = Math.min(lo, agg.mean - agg.std);
      hi = Math.max(hi, agg.mean + agg.std);
    });
    if (hi === lo) hi = lo + 1;

    const labelW = 260;
    const chartW = 640;
    const rowH = 26;
    const width = labelW + chartW + 90;
    const height = rows.length * rowH + 14;
    const xAt = (v) => labelW + ((v - lo) / (hi - lo)) * chartW;

    let svg = `<svg class="lb-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Seed-averaged ${this._esc(metric.label)} per unit">`;
    svg += `<line x1="${xAt(0)}" y1="4" x2="${xAt(0)}" y2="${height - 4}" class="lb-chart__zero" />`;

    rows.forEach((row, i) => {
      const y = i * rowH + 10;
      const mid = y + 7;
      const x0 = xAt(Math.min(0, row.agg.mean));
      const x1 = xAt(Math.max(0, row.agg.mean));
      const best = i === 0;

      svg += `<text x="${labelW - 10}" y="${mid + 4}" class="lb-chart__label" text-anchor="end">${this._esc(row.unit)}</text>`;
      svg += `<rect x="${x0}" y="${y}" width="${Math.max(1, x1 - x0)}" height="14" class="lb-chart__bar${best ? " is-best" : ""}" />`;
      if (row.agg.std > 0) {
        const e0 = xAt(row.agg.mean - row.agg.std);
        const e1 = xAt(row.agg.mean + row.agg.std);
        svg += `<line x1="${e0}" y1="${mid}" x2="${e1}" y2="${mid}" class="lb-chart__err" />`;
        svg += `<line x1="${e0}" y1="${mid - 4}" x2="${e0}" y2="${mid + 4}" class="lb-chart__err" />`;
        svg += `<line x1="${e1}" y1="${mid - 4}" x2="${e1}" y2="${mid + 4}" class="lb-chart__err" />`;
      }
      const sig = this._sigText(row.agg);
      svg += `<text x="${xAt(hi) + 8}" y="${mid + 4}" class="lb-chart__value">${this._fmt(row.agg.mean)} ± ${this._fmt(row.agg.std)} (n=${row.agg.n})${sig}</text>`;
    });

    svg += `</svg>`;
    return `<div class="lb-chart__wrap">${svg}</div>`;
  }

  _sigText(agg) {
    if (!agg || agg.is_best) return "";
    if (agg.p_vs_best === null || agg.p_vs_best === undefined) return "";
    const star = agg.p_vs_best < 0.05 ? "*" : "";
    return `, p=${agg.p_vs_best.toPrecision(2)}${star}`;
  }

  _trialsTableHtml(experiment, columns, index) {
    const present = columns.filter((col) => experiment.units.some((unit) => unit.metrics[col.key]));

    let html = `<details class="lb-trials-table"><summary>mean ± std across all headline metrics</summary>`;
    html += `<div class="lb-scroll"><table class="lb-table lb-table--diff"><thead><tr><th class="lb-th">unit</th><th class="lb-th">seeds</th>`;
    present.forEach((col) => { html += `<th class="lb-th">${this._esc(col.label)}</th>`; });
    html += `</tr></thead><tbody>`;

    experiment.units.forEach((unit) => {
      html += `<tr><td class="lb-key">${this._esc(unit.unit)}</td><td>${unit.seeds.join(", ")}</td>`;
      present.forEach((col) => {
        const agg = unit.metrics[col.key];
        html += `<td>${agg ? `${this._fmt(agg.mean)} ± ${this._fmt(agg.std)}${this._sigText(agg)}` : "&ndash;"}</td>`;
      });
      html += `</tr>`;
    });

    html += `</tbody></table></div></details>`;
    return html;
  }

  _bindTrials() {
    this._bindModeToggle();

    const metric = this.root.querySelector("#lb-trials-metric");
    if (metric) metric.addEventListener("change", () => {
      this.trialsMetric = metric.value;
      localStorage.setItem("leaderboard-trials-metric", this.trialsMetric);
      this._render();
    });
  }

  static CURVE_COLORS = ["#1d4fd8", "#0f766e", "#b45309", "#7c3aed", "#db2777", "#0891b2", "#4d7c0f", "#b91c1c"];

  _curvesHtml() {
    const needle = this.curveFilter.toLowerCase();
    const runs   = needle ? this.curveRuns.filter((run) => run.name.toLowerCase().includes(needle)) : this.curveRuns;

    let html = `<div class="lb-bar-top">`;
    html += this._modeToggleHtml();
    html += `<span class="lb-count">${this.curveRuns.length} training run${this.curveRuns.length === 1 ? "" : "s"} · ${this.curveSelected.size} selected</span>`;

    if (this.curveData && this.curveData.tags.length) {
      html += `<select class="lb-select" id="lb-curve-tag" aria-label="Scalar tag">`;
      this.curveData.tags.forEach((tag) => {
        html += `<option value="${this._esc(tag)}"${tag === this.curveTag ? " selected" : ""}>${this._esc(tag)}</option>`;
      });
      html += `</select>`;
      html += `<label class="lb-check"><input type="checkbox" id="lb-curve-log"${this.curveLog ? " checked" : ""} /> log scale</label>`;
    }
    html += `</div>`;

    html += `<div class="lb-curves">`;
    html += `<aside class="lb-curves__list">`;
    html += `<input type="search" class="res-filter" id="lb-curve-filter" placeholder="Filter runs" value="${this._esc(this.curveFilter)}" spellcheck="false" />`;

    if (!runs.length) {
      html += `<div class="res-empty res-empty--tight">${this.curveRuns.length ? "No run matches the filter." : "No training runs with tensorboard logs found."}</div>`;
    }
    runs.forEach((run, index) => {
      const on = this.curveSelected.has(run.id);
      const colorAt = [...this.curveSelected].indexOf(run.id);
      const dot = on ? `<i class="lb-curves__dot" style="background:${LeaderboardView.CURVE_COLORS[colorAt % 8]}"></i>` : `<i class="lb-curves__dot"></i>`;
      html += `<label class="lb-curves__run${on ? " is-on" : ""}" title="${this._esc(run.id)}">`;
      html += `<input type="checkbox" data-run="${this._esc(run.id)}"${on ? " checked" : ""} />${dot}<span>${this._esc(run.name)}</span>`;
      html += `</label>`;
    });
    html += `</aside>`;

    html += `<div class="lb-curves__chart">`;
    if (!this.curveSelected.size) {
      html += `<div class="res-empty">Tick training runs on the left to overlay their curves.</div>`;
    } else if (this.curveData && this.curveData.series.length) {
      html += this._curveChartSvg();
      html += `<div class="lb-curves__legend">`;
      this.curveData.series.forEach((series, index) => {
        html += `<span class="lb-curves__key"><i class="lb-curves__dot" style="background:${LeaderboardView.CURVE_COLORS[index % 8]}"></i>${this._esc(series.name)}</span>`;
      });
      html += `</div>`;
    } else if (this.curveData) {
      html += `<div class="res-empty">None of the selected runs logs the tag "${this._esc(this.curveTag)}".</div>`;
    } else {
      html += `<div class="res-empty">Loading curves&hellip;</div>`;
    }
    html += `</div></div>`;
    return html;
  }

  _curveChartSvg() {
    const series = this.curveData.series;
    const log    = this.curveLog;

    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    series.forEach((s) => {
      s.steps.forEach((step) => { xMin = Math.min(xMin, step); xMax = Math.max(xMax, step); });
      s.values.forEach((v) => {
        if (v === null || !isFinite(v) || (log && v <= 0)) return;
        yMin = Math.min(yMin, v);
        yMax = Math.max(yMax, v);
      });
    });
    if (!isFinite(yMin)) { yMin = 0; yMax = 1; }
    if (xMax === xMin) xMax = xMin + 1;
    if (yMax === yMin) yMax = yMin + (Math.abs(yMin) || 1) * 0.1;

    const ty = (v) => (log ? Math.log10(v) : v);
    const yLo = ty(yMin), yHi = ty(yMax);

    const W = 920, H = 420, mL = 68, mR = 16, mT = 14, mB = 34;
    const xAt = (step) => mL + ((step - xMin) / (xMax - xMin)) * (W - mL - mR);
    const yAt = (v) => mT + (1 - (ty(v) - yLo) / (yHi - yLo || 1)) * (H - mT - mB);

    let svg = `<svg class="lb-linechart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Training curves">`;

    for (let i = 0; i <= 4; i++) {
      const vt = yLo + (i / 4) * (yHi - yLo);
      const v = log ? Math.pow(10, vt) : vt;
      const y = mT + (1 - i / 4) * (H - mT - mB);
      svg += `<line x1="${mL}" y1="${y}" x2="${W - mR}" y2="${y}" class="lb-linechart__grid" />`;
      svg += `<text x="${mL - 8}" y="${y + 4}" class="lb-linechart__tick" text-anchor="end">${this._fmt(v)}</text>`;
    }
    for (let i = 0; i <= 4; i++) {
      const step = xMin + (i / 4) * (xMax - xMin);
      const x = mL + (i / 4) * (W - mL - mR);
      svg += `<text x="${x}" y="${H - 12}" class="lb-linechart__tick" text-anchor="middle">${Math.round(step)}</text>`;
    }

    series.forEach((s, index) => {
      let d = "";
      let pen = false;
      for (let i = 0; i < s.steps.length; i++) {
        const v = s.values[i];
        if (v === null || !isFinite(v) || (log && v <= 0)) { pen = false; continue; }
        d += `${pen ? "L" : "M"}${xAt(s.steps[i]).toFixed(1)} ${yAt(v).toFixed(1)} `;
        pen = true;
      }
      svg += `<path d="${d.trim()}" class="lb-linechart__line" style="stroke:${LeaderboardView.CURVE_COLORS[index % 8]}" />`;
    });

    svg += `</svg>`;
    return `<div class="lb-chart__wrap">${svg}</div>`;
  }

  _bindCurves() {
    this._bindModeToggle();

    const filter = this.root.querySelector("#lb-curve-filter");
    if (filter) filter.addEventListener("input", () => {
      this.curveFilter = filter.value.trim();
      const at = filter.selectionStart;
      this._render();
      const next = this.root.querySelector("#lb-curve-filter");
      if (next) { next.focus(); next.setSelectionRange(at, at); }
    });

    this.root.querySelectorAll(".lb-curves__run input[type=checkbox]").forEach((box) => {
      box.addEventListener("change", () => {
        if (box.checked) this.curveSelected.add(box.dataset.run);
        else this.curveSelected.delete(box.dataset.run);
        this.loadCurves();
      });
    });

    const tag = this.root.querySelector("#lb-curve-tag");
    if (tag) tag.addEventListener("change", () => {
      this.curveTag = tag.value;
      this.loadCurves();
    });

    const log = this.root.querySelector("#lb-curve-log");
    if (log) log.addEventListener("change", () => {
      this.curveLog = log.checked;
      this._render();
    });
  }

  _bindTable() {
    this._bindModeToggle();

    const filter = this.root.querySelector("#lb-filter");
    if (filter) filter.addEventListener("input", () => {
      this.filter = filter.value.trim();
      this._rerenderKeepFocus(filter);
    });

    this.root.querySelectorAll(".lb-select").forEach((sel) => {
      sel.addEventListener("change", () => {
        this.axisSel[sel.dataset.axis] = sel.value;
        this._render();
      });
    });

    this.root.querySelectorAll(".lb-cols input[type=checkbox]").forEach((box) => {
      box.addEventListener("change", () => {
        if (box.checked) this.visible = [...this.visible, box.dataset.col];
        else             this.visible = this.visible.filter((k) => k !== box.dataset.col);
        this._saveVisible();
        this._render();
        const menu = this.root.querySelector(".lb-cols");
        if (menu) menu.open = true;
      });
    });

    this.root.querySelectorAll(".lb-th").forEach((th) => {
      th.addEventListener("click", () => {
        const key = th.dataset.sort;
        if (this.sortKey === key) this.sortDir = -this.sortDir;
        else { this.sortKey = key; this.sortDir = key === "run" ? 1 : -1; }
        this._render();
      });
    });

    this.root.querySelectorAll(".lb-row").forEach((tr) => {
      tr.addEventListener("click", () => this._toggleSelect(tr.dataset.id));
    });

    const compare = this.root.querySelector("#lb-compare");
    if (compare) compare.addEventListener("click", () => this.showDiff());
  }

  _rerenderKeepFocus(input) {
    const at = input.selectionStart;
    this._render();
    const next = this.root.querySelector("#lb-filter");
    if (next) { next.focus(); next.setSelectionRange(at, at); }
  }

  _toggleSelect(id) {
    if (this.selected.includes(id)) this.selected = this.selected.filter((s) => s !== id);
    else this.selected = [...this.selected, id].slice(-LeaderboardView.DIFF_TAGS.length);
    this._render();
  }

  _diffHtml() {
    const { sides, directions, sections } = this.diffData;
    const tags = LeaderboardView.DIFF_TAGS;
    const pair = sides.length === 2;

    let html = `<div class="lb-bar-top">`;
    html += `<button type="button" class="btn btn--mini" id="lb-back">&larr; Back to table</button>`;
    html += `<span class="lb-count">Comparing ${sides.length} inference results</span>`;
    if (pair) html += `<button type="button" class="btn btn--mini" id="lb-autopsy">Autopsy A vs B</button>`;
    html += `</div>`;

    html += `<div class="lb-diff__heads">`;
    sides.forEach((side, i) => {
      html += `<div class="lb-diff__head"><span class="lb-diff__tag">${tags[i]}</span><span class="lb-run__name">${this._esc(side.run)}</span><span class="lb-run__stamp">${this._esc(side.stamp)}</span></div>`;
    });
    html += `</div>`;

    const all    = [...new Set(sides.flatMap((side) => Object.keys(side.metrics)))];
    const needle = this.diffFilter.toLowerCase();

    const stems = {};
    all.forEach((key) => {
      const stem = key.replace(/\d+$/, "");
      if (stem !== key) stems[stem] = (stems[stem] || 0) + 1;
    });
    const isSeries = (key) => {
      const stem = key.replace(/\d+$/, "");
      return stem !== key && stems[stem] >= 10;
    };
    const passes = (key) => {
      if (!this.diffSeries && isSeries(key)) return false;
      if (needle && !key.toLowerCase().includes(needle)) return false;
      if (this.diffChanged && new Set(sides.map((side) => side.metrics[key])).size < 2) return false;
      return true;
    };

    const kept   = all.filter(passes);
    const hotCut = this._hotCutoff(kept, sides);
    const index  = [];

    html += `<div class="lb-diff__layout"><div class="lb-diff__main">`;

    html += `<section class="lb-diff__section"><h4 class="res-section__cap">Metrics <span>${kept.length} of ${all.length}</span></h4>`;
    html += `<div class="lb-diff__controls">`;
    html += `<input type="search" class="res-filter" id="lb-diff-filter" placeholder="Filter metrics by name" value="${this._esc(this.diffFilter)}" spellcheck="false" />`;
    html += `<label class="lb-check"><input type="checkbox" id="lb-diff-changed"${this.diffChanged ? " checked" : ""} /> differing only</label>`;
    html += `<label class="lb-check"><input type="checkbox" id="lb-diff-series"${this.diffSeries ? " checked" : ""} /> per-index series</label>`;
    html += `</div>`;

    html += `<article class="res-md lb-diff__report">`;
    sections.forEach((section, si) => {
      const keys = section.keys.filter(passes);
      if (!keys.length) return;

      index.push({ id: `lb-diff-sec-${si}`, title: section.title, count: keys.length });
      html += `<h2 id="lb-diff-sec-${si}">${this._esc(section.title)} <span>${keys.length}</span></h2>`;
      html += `<table><thead><tr><th>metric</th>`;
      sides.forEach((_, i) => { html += `<th>${tags[i]}</th>`; });
      html += pair ? `<th>&Delta; (B&minus;A)</th><th>&Delta;%</th>` : `<th>spread%</th>`;
      html += `</tr></thead><tbody>`;
      keys.forEach((key) => { html += this._diffRowHtml(key, sides, directions[key], hotCut); });
      html += `</tbody></table>`;
    });
    html += `</article>`;

    if (!kept.length) html += `<div class="res-empty res-empty--tight">No metric matches the current filters.</div>`;
    html += `</section>`;

    html += this._configDiffHtml(sides, tags, index);

    html += `</div>`;
    html += `<nav class="res-index lb-diff__index" aria-label="Comparison index">`;
    index.forEach((entry) => {
      html += `<button type="button" class="res-index__link" data-target="${entry.id}">${this._esc(entry.title)}<span>${entry.count}</span></button>`;
    });
    html += `</nav></div>`;
    return html;
  }

  _rowSpread(values) {
    const nums = values.filter((v) => v !== undefined && isFinite(v));
    if (nums.length < 2) return null;

    const max = Math.max(...nums);
    const min = Math.min(...nums);
    if (max === min) return 0;

    const scale = Math.max(Math.abs(max), Math.abs(min));
    return scale > 0 ? ((max - min) / scale) * 100 : null;
  }

  _hotCutoff(keys, sides) {
    const mags = keys.map((key) => this._rowSpread(sides.map((side) => side.metrics[key]))).filter((m) => m !== null && m > 0).sort((x, y) => x - y);
    if (!mags.length) return null;
    return Math.max(mags[Math.floor(mags.length * 0.9)], 1);
  }

  _diffRowHtml(key, sides, direction, hotCut) {
    const values = sides.map((side) => side.metrics[key]);
    const nums   = values.filter((v) => v !== undefined && isFinite(v));
    const spread = this._rowSpread(values);
    const hot    = hotCut !== null && spread !== null && spread >= hotCut;

    const best  = direction && nums.length >= 2 ? (direction > 0 ? Math.max(...nums) : Math.min(...nums)) : null;
    const worst = direction && nums.length >= 2 ? (direction > 0 ? Math.min(...nums) : Math.max(...nums)) : null;

    let cells = "";
    values.forEach((v) => {
      let cls = "";
      if (sides.length > 2 && best !== null && best !== worst && v !== undefined) {
        if (v === best) cls = ` class="lb-delta lb-delta--good"`;
        else if (v === worst) cls = ` class="lb-delta lb-delta--bad"`;
      }
      cells += `<td${cls}>${v === undefined ? "&ndash;" : this._fmt(v)}</td>`;
    });

    if (sides.length === 2) {
      const [va, vb] = values;
      if (va === undefined || vb === undefined) {
        cells += `<td>&ndash;</td><td>&ndash;</td>`;
      } else {
        const delta = vb - va;
        const pct   = va !== 0 ? (delta / Math.abs(va)) * 100 : null;
        const cls   = this._deltaClass(delta, direction);
        cells += `<td class="${cls}">${delta === 0 ? "=" : this._fmt(delta)}</td><td class="${cls}">${pct === null || delta === 0 ? "&ndash;" : (pct > 0 ? "+" : "") + pct.toFixed(1) + "%"}</td>`;
      }
    } else {
      cells += `<td>${spread === null ? "&ndash;" : spread === 0 ? "=" : spread.toFixed(1) + "%"}</td>`;
    }

    return `<tr${hot ? ` class="lb-hot"` : ""}><td><code>${this._esc(key)}</code></td>${cells}</tr>`;
  }

  _configDiffHtml(sides, tags, index) {
    const keys    = [...new Set(sides.flatMap((side) => Object.keys(side.config)))].sort();
    const differs = keys.filter((key) => new Set(sides.map((side) => String(side.config[key]))).size > 1);

    let html = `<section class="lb-diff__section" id="lb-diff-config"><h4 class="res-section__cap">Config differences <span>${differs.length} of ${keys.length}</span></h4>`;
    index.push({ id: "lb-diff-config", title: "Config differences", count: differs.length });

    const notes = sides.flatMap((side, i) => side.config_notes.map((note) => `${tags[i]}: ${note}`));
    if (notes.length) {
      html += `<div class="res-empty res-empty--tight">${notes.map((note) => this._esc(note)).join("<br>")}</div>`;
    }

    if (!keys.length) {
      return html + `<div class="res-empty res-empty--tight">No resolved config files were found for these runs.</div></section>`;
    }
    if (!differs.length) {
      return html + `<div class="res-empty res-empty--tight">The resolved configs are identical.</div></section>`;
    }

    html += `<article class="res-md lb-diff__report lb-diff__report--config">`;
    html += `<table><thead><tr><th>field</th>`;
    sides.forEach((_, i) => { html += `<th>${tags[i]}</th>`; });
    html += `</tr></thead><tbody>`;
    differs.forEach((key) => {
      html += `<tr><td><code>${this._esc(key)}</code></td>`;
      sides.forEach((side) => {
        html += `<td>${side.config[key] === undefined ? "&ndash;" : this._esc(String(side.config[key]))}</td>`;
      });
      html += `</tr>`;
    });
    html += `</tbody></table></article></section>`;
    return html;
  }

  _deltaClass(delta, direction) {
    if (!direction || delta === 0) return "lb-delta";
    const improved = direction > 0 ? delta > 0 : delta < 0;
    return improved ? "lb-delta lb-delta--good" : "lb-delta lb-delta--bad";
  }

  _bindDiff() {
    const back = this.root.querySelector("#lb-back");
    if (back) back.addEventListener("click", () => {
      this.view = "table";
      this._render();
    });

    const autopsy = this.root.querySelector("#lb-autopsy");
    if (autopsy) autopsy.addEventListener("click", () => {
      const sides = this.diffData.sides;
      window.abAutopsyPair = { a: sides[0].id, b: sides[1].id };
      window.location.hash = "#/autopsy";
    });

    const filter = this.root.querySelector("#lb-diff-filter");
    if (filter) filter.addEventListener("input", () => {
      this.diffFilter = filter.value.trim();
      const at = filter.selectionStart;
      this._render();
      const next = this.root.querySelector("#lb-diff-filter");
      if (next) { next.focus(); next.setSelectionRange(at, at); }
    });

    const changed = this.root.querySelector("#lb-diff-changed");
    if (changed) changed.addEventListener("change", () => {
      this.diffChanged = changed.checked;
      this._render();
    });

    const series = this.root.querySelector("#lb-diff-series");
    if (series) series.addEventListener("change", () => {
      this.diffSeries = series.checked;
      this._render();
    });

    this.root.querySelectorAll(".lb-diff__index [data-target]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = this.root.querySelector(`#${btn.dataset.target}`);
        if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });
  }

  _loadVisible() {
    try {
      const raw = localStorage.getItem("leaderboard-cols-v2");
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;
    }
  }

  _reconcileVisible(columns) {
    const stored = this.storedCols;
    if (!stored) return columns.filter((col) => col.default).map((col) => col.key);

    const known = new Set(stored.known || []);
    const vis   = new Set(stored.visible || []);
    columns.forEach((col) => { if (!known.has(col.key) && col.default) vis.add(col.key); });
    return columns.map((col) => col.key).filter((key) => vis.has(key));
  }

  _saveVisible() {
    this.storedCols = { visible: this.visible, known: this.data.columns.map((col) => col.key) };
    localStorage.setItem("leaderboard-cols-v2", JSON.stringify(this.storedCols));
    localStorage.removeItem("leaderboard-cols");
  }

  _runsBase() {
    try {
      const raw = JSON.parse(localStorage.getItem("results-sources") || "{}");
      return raw.logs || ResultsView.DEFAULT_RUNS;
    } catch (e) {
      return ResultsView.DEFAULT_RUNS;
    }
  }

  _fmt(value) {
    if (value === 0) return "0";
    const magnitude = Math.abs(value);
    if (magnitude >= 10000 || magnitude < 0.001) return value.toExponential(2);
    return String(Number(value.toPrecision(4)));
  }

  _ago(mtime) {
    const seconds = Date.now() / 1000 - mtime;
    if (seconds < 3600)   return `${Math.max(1, Math.round(seconds / 60))}m ago`;
    if (seconds < 86400)  return `${Math.round(seconds / 3600)}h ago`;
    return `${Math.round(seconds / 86400)}d ago`;
  }

  _esc(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
}

window.LeaderboardView = LeaderboardView;
