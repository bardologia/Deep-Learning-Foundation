"use strict";

class AutopsyView {
  constructor(root) {
    this.root       = root;
    this.built      = false;
    this.pair       = null;
    this.runs       = [];
    this.sideA      = null;
    this.sideB      = null;
    this.runStrip   = null;
  }

  enter() {
    if (!this.built) {
      this._build();
      this.built = true;
    }

    const pending = window.abAutopsyPair;
    if (pending) {
      window.abAutopsyPair = null;
      this.sideA = pending.a;
      this.sideB = pending.b;
      this._compare();
    }

    this._refreshRuns();
  }

  _build() {
    this.root.innerHTML = `
      <div class="cube-pick">
        <div class="run-strip" id="autopsy-strip" aria-label="Saved inferences"></div>
        <div class="fl-src" role="group" aria-label="Chosen pair">
          <span class="autopsy-slot" id="autopsy-slot-a"><b>A</b> <span>none</span></span>
          <span class="autopsy-slot" id="autopsy-slot-b"><b>B</b> <span>none</span></span>
          <button type="button" class="btn btn--mini" id="autopsy-go" disabled>Compare</button>
        </div>
        <p class="cube-hint" id="autopsy-hint">Assign one saved inference to A and one to B (same region), or send a pair over from the leaderboard diff.</p>
      </div>
      <div id="autopsy-out"></div>`;

    this.refs = {
      strip : this.root.querySelector("#autopsy-strip"),
      slotA : this.root.querySelector("#autopsy-slot-a"),
      slotB : this.root.querySelector("#autopsy-slot-b"),
      goBtn : this.root.querySelector("#autopsy-go"),
      hint  : this.root.querySelector("#autopsy-hint"),
      out   : this.root.querySelector("#autopsy-out"),
    };

    this.refs.goBtn.addEventListener("click", () => this._compare());
  }

  _runsBase() {
    try {
      const raw = JSON.parse(localStorage.getItem("results-sources") || "{}");
      return raw.logs || ResultsView.DEFAULT_RUNS;
    } catch (e) {
      return ResultsView.DEFAULT_RUNS;
    }
  }

  async _refreshRuns() {
    const data = await window.apiGet(`/api/autopsy/runs?base=${encodeURIComponent(this._runsBase())}`);

    if (!data || !data.ok) {
      this.refs.hint.textContent = (data && data.error) || "Backend unreachable.";
      return;
    }

    this.runs = data.runs || [];
    this._renderStrip();

    if (!this.runs.length) {
      this.refs.hint.textContent = `No comparable inferences under ${data.root} (each needs metrics.json and cubes/pixel_mse.npy). Run inference with save_cubes first, or point the runs directory in the Results tab at the right place.`;
    }
  }

  _label(id) {
    const run = this.runs.find((r) => r.id === id);
    return run ? `${run.run} · ${run.stamp}` : (id ? id.split("/").slice(-3).join("/") : "none");
  }

  _syncSlots() {
    this.refs.slotA.querySelector("span").textContent = this._label(this.sideA);
    this.refs.slotB.querySelector("span").textContent = this._label(this.sideB);
    this.refs.slotA.classList.toggle("is-set", Boolean(this.sideA));
    this.refs.slotB.classList.toggle("is-set", Boolean(this.sideB));
    this.refs.goBtn.disabled = !(this.sideA && this.sideB && this.sideA !== this.sideB);
  }

  _assign(side, id) {
    if (side === "A") this.sideA = this.sideA === id ? null : id;
    else this.sideB = this.sideB === id ? null : id;
    this._renderStrip();

    if (this.sideA && this.sideB && this.sideA !== this.sideB) this._compare();
  }

  _renderStrip() {
    this._syncSlots();

    if (!this.runStrip) {
      this.runStrip = new RunStrip(this.refs.strip, {
        rowClass : "autopsy-run",
        stateFor : (run) => run.id === this.sideA || run.id === this.sideB,
        extras   : (run) => this._stripExtras(run),
      });
    }
    this.runStrip.render(this.runs);
  }

  _stripExtras(run) {
    return ["A", "B"].map((side) => {
      const on  = run.id === (side === "A" ? this.sideA : this.sideB);
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `autopsy-side autopsy-side--${side.toLowerCase()}` + (on ? " is-on" : "");
      btn.textContent = side;
      btn.title = on ? `Unassign ${side}` : `Use as ${side}`;
      btn.addEventListener("click", () => this._assign(side, run.id));
      return btn;
    });
  }

  async _compare() {
    const a = this.sideA;
    const b = this.sideB;
    if (!a || !b || a === b) return;

    this._syncSlots();
    this.refs.hint.textContent = "comparing…";
    const data = await window.apiGet(`/api/autopsy/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);

    if (!data.ok) {
      this.refs.hint.textContent = data.error || "comparison failed";
      return;
    }

    this.pair = { a, b };
    this.refs.hint.textContent = `A = ${data.run_a} · B = ${data.run_b} · region ${data.region[0]}×${data.region[1]} px`;
    this._render(data);
    data.hotspots.forEach((spot, index) => this._loadProfile(spot, index));
  }

  _render(data) {
    let html = `<div class="fl-card"><h3 class="cube-panel-title">Largest metric gaps</h3>`;
    html += `<div class="lb-scroll"><table class="lb-table lb-table--diff"><thead><tr>` +
      `<th class="lb-th">metric</th><th class="lb-th">A</th><th class="lb-th">B</th><th class="lb-th">Δ rel</th><th class="lb-th">winner</th></tr></thead><tbody>`;

    data.metrics.slice(0, 25).forEach((row) => {
      const tone = row.winner === "A" ? "#0f766e" : row.winner === "B" ? "#b91c1c" : "inherit";
      html += `<tr><td class="lb-key">${this._esc(row.key)}</td>` +
        `<td>${this._fmt(row.a)}</td><td>${this._fmt(row.b)}</td>` +
        `<td>${(row.rel * 100).toFixed(1)}%</td>` +
        `<td style="color:${tone};font-weight:600">${row.winner}</td></tr>`;
    });
    html += `</tbody></table></div></div>`;

    if (data.hotspots.length) {
      html += `<div class="script-grid">`;
      data.hotspots.forEach((spot, index) => {
        html += `
          <article class="fl-card">
            <h3 class="cube-panel-title">${spot.winner} wins · az ${spot.az}, rg ${spot.rg}</h3>
            <p class="cube-hint">block Δ pixel MSE (A−B): ${spot.mean_delta.toExponential(2)}</p>
            <canvas id="autopsy-spot-${index}" width="560" height="220"></canvas>
          </article>`;
      });
      html += `</div>`;
    } else {
      html += `<p class="cube-hint">no disagreement hotspots; the runs behave almost identically pixelwise</p>`;
    }

    this.refs.out.innerHTML = html;
  }

  async _loadProfile(spot, index) {
    const data = await window.apiGet(
      `/api/autopsy/profile?a=${encodeURIComponent(this.pair.a)}&b=${encodeURIComponent(this.pair.b)}&az=${spot.az}&rg=${spot.rg}`
    );
    if (!data.ok) return;

    const canvas = this.root.querySelector(`#autopsy-spot-${index}`);
    if (!canvas) return;

    window.drawLineChart(canvas, data.x_axis, [
      { values: data.gt, color: "#111827", width: 1.2, label: "GT" },
      { values: data.a,  color: "#0f766e", width: 1.7, label: "A" },
      { values: data.b,  color: "#b91c1c", width: 1.7, label: "B" },
    ]);
  }

  _fmt(value) {
    if (!Number.isFinite(value)) return "–";
    const abs = Math.abs(value);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e4)) return value.toExponential(2);
    return value.toPrecision(4);
  }

  _esc(text) {
    return String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
  }
}

window.AutopsyView = AutopsyView;
