"use strict";

class AutopsyView {
  constructor(root) {
    this.root       = root;
    this.built      = false;
    this.runs       = [];
    this.sideA      = null;
    this.sideB      = null;
    this.runStrip   = null;
    this.cmpToken   = 0;
  }

  enter() {
    if (!this.built) {
      this._build();
      this.built = true;
    }

    this._refreshRuns();
  }

  figureTag() {
    const named = (id) => {
      const run = this.runs.find((entry) => entry.id === id);
      return run ? run.run : "";
    };

    return [named(this.sideA), named(this.sideB)].filter(Boolean).join("_vs_");
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
        <p class="cube-hint" id="autopsy-hint">Assign one saved inference to A and one to B (same region).</p>
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

  async _refreshRuns() {
    const data = await Api.get(`/api/autopsy/runs?base=${encodeURIComponent(ResultsSources.runs())}`);

    if (!data || !data.ok) {
      this.refs.hint.textContent = (data && data.error) || "Backend unreachable.";
      return;
    }

    this.runs = data.runs || [];
    this._renderStrip();

    if (!this.runs.length) {
      this.refs.hint.textContent = `No comparable inferences under ${data.root} (each needs metrics.json and maps/predictions.npy with maps/targets.npy). Run inference with save_maps first, or point the runs directory in the Results tab at the right place.`;
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

    const token = ++this.cmpToken;

    this._syncSlots();
    this.refs.hint.innerHTML = SharedCharts.waitLine("comparing…");
    const data = await Api.get(`/api/autopsy/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);

    if (token !== this.cmpToken) return;

    if (!data.ok) {
      this.refs.hint.textContent = data.error || "comparison failed";
      return;
    }

    this.refs.hint.textContent = `A = ${data.run_a} · B = ${data.run_b} · region ${data.region[0]}×${data.region[1]} px`;
    this._render(data);
    data.hotspots.forEach((spot, index) => this._loadPixel(spot, index, a, b, token));
  }

  _render(data) {
    let html = `<div class="fl-card"><h3 class="cube-panel-title">Largest metric gaps</h3>`;
    html += `<div class="tbl-scroll"><table class="tbl tbl--diff"><thead><tr>` +
      `<th class="tbl-th">metric</th><th class="tbl-th">A</th><th class="tbl-th">B</th><th class="tbl-th">Δ rel</th><th class="tbl-th">winner</th></tr></thead><tbody>`;

    data.metrics.slice(0, 25).forEach((row) => {
      const tone = row.winner === "A" ? "#2dd4bf" : row.winner === "B" ? "#f87171" : "inherit";
      html += `<tr><td class="tbl-key">${SharedCharts.esc(row.key)}</td>` +
        `<td>${this._fmt(row.a)}</td><td>${this._fmt(row.b)}</td>` +
        `<td>${(row.rel * 100).toFixed(1)}%</td>` +
        `<td style="color:${tone};font-weight:700">${row.winner}</td></tr>`;
    });
    html += `</tbody></table></div></div>`;

    if (data.hotspots.length) {
      html += `<div class="script-grid">`;
      data.hotspots.forEach((spot, index) => {
        html += `
          <article class="fl-card">
            <h3 class="cube-panel-title">${spot.winner} wins · row ${spot.row}, col ${spot.col}</h3>
            <p class="cube-hint">block Δ standardized squared error (A−B): ${spot.mean_delta.toExponential(2)}</p>
            <div class="tbl-scroll" id="autopsy-spot-${index}"></div>
          </article>`;
      });
      html += `</div>`;
    } else {
      html += `<p class="cube-hint">no disagreement hotspots; the runs behave almost identically pixelwise</p>`;
    }

    this.refs.out.innerHTML = html;
  }

  async _loadPixel(spot, index, a, b, token) {
    const data = await Api.get(
      `/api/autopsy/pixel?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}&row=${spot.row}&col=${spot.col}`
    );

    if (token !== this.cmpToken) return;

    if (!data.ok) {
      this.refs.hint.textContent = data.error || "pixel lookup failed";
      return;
    }

    const holder = this.root.querySelector(`#autopsy-spot-${index}`);
    if (!holder) return;

    let html = `<table class="tbl tbl--diff"><thead><tr><th class="tbl-th">target channel</th><th class="tbl-th">target</th><th class="tbl-th">A</th><th class="tbl-th">B</th></tr></thead><tbody>`;
    data.channels.forEach((name, channel) => {
      html += `<tr><td class="tbl-key">${SharedCharts.esc(name)}</td><td>${this._fmt(data.target[channel])}</td><td>${this._fmt(data.a[channel])}</td><td>${this._fmt(data.b[channel])}</td></tr>`;
    });
    holder.innerHTML = `${html}</tbody></table>`;
  }

  _fmt(value) {
    if (!Number.isFinite(value)) return "–";
    const abs = Math.abs(value);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e4)) return value.toExponential(2);
    return value.toPrecision(4);
  }
}

window.AutopsyView = AutopsyView;
