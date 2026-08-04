"use strict";

class TriageView {
  static DEFAULT_BASE = "/ste/rnd/User/vice_vi/DLR-TomoSAR/runs";
  static VERDICTS     = ["label problem", "model problem", "interesting"];

  constructor(root) {
    this.root     = root;
    this.cubes    = [];
    this.cubeId   = null;
    this.cases    = [];
    this.payload  = null;
    this.built    = false;
    this.runStrip = null;
  }

  enter() {
    if (!this.built) {
      this._build();
      this.built = true;
    }
  }

  _build() {
    this.root.innerHTML = `
      <div class="cube-pick">
        <div class="fl-src" role="group" aria-label="Runs directory">
          <label for="triage-base">Runs dir</label>
          <input id="triage-base" type="text" spellcheck="false" autocomplete="off" placeholder="${TriageView.DEFAULT_BASE}" />
          <button type="button" class="btn btn--mini" id="triage-scan">Scan</button>
        </div>
        <div class="run-strip" id="triage-strip" aria-label="Saved inferences"></div>
        <p class="cube-hint" id="triage-hint">Scan a runs directory for saved inference cubes, then pick one to triage.</p>
      </div>
      <div id="triage-cases"></div>`;

    this.refs = {
      base  : this.root.querySelector("#triage-base"),
      scan  : this.root.querySelector("#triage-scan"),
      strip : this.root.querySelector("#triage-strip"),
      hint  : this.root.querySelector("#triage-hint"),
      cases : this.root.querySelector("#triage-cases"),
    };

    this.refs.base.value = TriageView.DEFAULT_BASE;
    this.refs.scan.addEventListener("click", () => this._scan());
    this.refs.base.addEventListener("keydown", (ev) => { if (ev.key === "Enter") this._scan(); });
  }

  async _scan() {
    const base = this.refs.base.value.trim();
    if (!base) return;

    const data = await window.apiGet(`/api/cubes?base=${encodeURIComponent(base)}`);
    if (!data.ok) {
      this.refs.hint.textContent = data.error || "scan failed";
      return;
    }

    this.cubes = data.cubes || [];
    this.refs.hint.textContent = this.cubes.length ? `${this.cubes.length} saved inference(s); pick one.` : "no saved inference cubes under this directory";

    if (!this.runStrip) {
      this.runStrip = new RunStrip(this.refs.strip, {
        stateFor : (cube) => cube.id === this.cubeId,
        onPick   : (cube) => this._pick(cube.id),
      });
    }
    this.runStrip.render(this.cubes);
  }

  async _pick(cubeId) {
    this.cubeId = cubeId;
    this.runStrip.repaint();

    const data = await window.apiGet(`/api/triage/cases?id=${encodeURIComponent(cubeId)}&n=40`);
    if (!data.ok) {
      this.refs.cases.innerHTML = `<p class="cube-hint">${this._esc(data.error || "failed to build cases")}</p>`;
      return;
    }

    this.cases   = data.cases;
    this.payload = data;
    this._renderCases();
  }

  _renderCases() {
    if (!this.cases.length) {
      this.refs.cases.innerHTML = `<p class="cube-hint">no triage cases; every block is below the error floor</p>`;
      return;
    }

    const cards = this.cases.map((row, index) => this._card(row, index)).join("");
    this.refs.cases.innerHTML = `<div class="script-grid">${cards}</div>`;
    this._wire(this.refs.cases);
    this.cases.forEach((row, index) => this._loadProfile(row, index));
  }

  static VERDICT_TONES = {
    "label problem" : "triage-verdict--label",
    "model problem" : "triage-verdict--model",
    "interesting"   : "triage-verdict--interesting",
  };

  _card(row, index) {
    const data       = this.payload || {};
    const annotation = row.annotation || {};
    const modeBadge  = data.has_modes && row.mode ? `<span class="cube-coords">mode: ${this._esc(row.mode)} (${(row.fail_frac * 100).toFixed(0)}% failing)</span><br>` : "";
    const aux        = (data.aux || [])
      .filter((name) => row[name] !== null && row[name] !== undefined)
      .map((name) => `${this._esc(name)}: ${Number(row[name]).toPrecision(3)}`)
      .join(" · ");

    const thumb = `/api/triage/thumb?id=${encodeURIComponent(this.cubeId)}&az0=${row.az0}&rg0=${row.rg0}`;

    const verdictBtns = TriageView.VERDICTS.map((verdict) => {
      const active = annotation.verdict === verdict;
      const tone   = TriageView.VERDICT_TONES[verdict] || "";
      return `<button type="button" class="triage-verdict ${tone}${active ? " is-active" : ""}" data-case="${index}" data-verdict="${active ? "" : verdict}">${verdict}</button>`;
    }).join("");

    return `
      <article class="fl-card">
        <h3 class="cube-panel-title">#${index + 1} · az ${row.az0}–${row.az1}, rg ${row.rg0}–${row.rg1}</h3>
        <div class="triage-media">
          <img class="triage-thumb" src="${thumb}" loading="lazy" alt="pixel MSE around the block" title="log MSE, white = this block, red = worst pixel; click to open cuts" data-open="${index}" />
          <canvas class="triage-profile" id="triage-profile-${index}" width="420" height="132" title="GT vs prediction at the worst pixel"></canvas>
        </div>
        <p class="cube-hint">
          block MSE ${row.mse_mean.toExponential(2)} (peak ${row.mse_max.toExponential(2)} at az ${row.worst_az}, rg ${row.worst_rg})<br>
          ${modeBadge}${aux ? `${aux}<br>` : ""}
        </p>
        <div class="triage-verdicts" role="group" aria-label="Verdict">${verdictBtns}</div>
        <div class="fl-pixrow">
          <input type="text" class="cube-jump__input cube-jump__input--wide" placeholder="note" value="${this._esc(annotation.note || "")}" data-note="${index}" />
          <button type="button" class="btn btn--mini" data-open="${index}">Open cuts</button>
        </div>
        ${annotation.updated ? `<p class="cube-hint">last touched ${this._esc(annotation.updated)}</p>` : ""}
      </article>`;
  }

  async _loadProfile(row, index) {
    const data = await window.apiGet(
      `/api/triage/profile?id=${encodeURIComponent(this.cubeId)}&az=${row.worst_az}&rg=${row.worst_rg}`
    );
    if (!data.ok) return;

    const canvas = this.refs.cases.querySelector(`#triage-profile-${index}`);
    if (!canvas) return;

    window.drawLineChart(canvas, data.x_axis, [
      { values: data.gt,   color: "#111827", width: 1.2, label: "GT" },
      { values: data.pred, color: "#0f766e", width: 1.7, label: "pred" },
    ]);
  }

  async _annotate(index, verdict, noteOverride) {
    const row        = this.cases[index];
    const annotation = row.annotation || {};
    const body       = {
      id      : this.cubeId,
      case    : row.case,
      verdict : verdict === null ? (annotation.verdict || "") : verdict,
      note    : noteOverride !== undefined ? noteOverride : (annotation.note || ""),
    };

    const out = await window.apiPost("/api/triage/annotate", body);

    if (out.ok) {
      row.annotation = out.annotation || null;
      this._repaintCard(index);
    }
  }

  _repaintCard(index) {
    const grid = this.refs.cases.querySelector(".script-grid");
    const old  = grid ? grid.children[index] : null;
    if (!old) return;

    const holder = document.createElement("div");
    holder.innerHTML = this._card(this.cases[index], index);
    const card = holder.firstElementChild;

    old.replaceWith(card);
    this._wire(card);
    this._loadProfile(this.cases[index], index);
  }

  _wire(scope) {
    scope.querySelectorAll("[data-open]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const row = this.cases[Number(btn.dataset.open)];
        if (window.tomogramView) {
          window.tomogramView.openAt(this.cubeId, row.worst_az, row.worst_rg);
          window.location.hash = "#/cube";
        }
      });
    });

    scope.querySelectorAll("[data-verdict]").forEach((btn) => {
      btn.addEventListener("click", () => this._annotate(Number(btn.dataset.case), btn.dataset.verdict));
    });

    scope.querySelectorAll("[data-note]").forEach((input) => {
      input.addEventListener("change", () => this._annotate(Number(input.dataset.note), null, input.value));
    });
  }

  _esc(text) {
    return String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
  }
}

window.TriageView = TriageView;
