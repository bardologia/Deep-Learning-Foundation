"use strict";

class TemplatesView {
  constructor(launchView, gridEl) {
    this.launchView = launchView;
    this.gridEl = gridEl;
    this.entries = [];
  }

  enter() {
    this.load();
  }

  async load() {
    const data = await Api.get("/api/config-templates");

    if (data.error) {
      this.gridEl.innerHTML = `<p class="saved-empty">Could not load config templates: ${SharedCharts.esc(data.error)}</p>`;
      return;
    }

    this.entries = data.templates || [];
    this._renderGrid();
  }

  _renderGrid() {
    this.gridEl.innerHTML = "";

    if (!this.entries.length) {
      this.gridEl.innerHTML = `<p class="saved-empty">No config templates yet. Set the fields you reuse on a launch page and press &ldquo;Save as template&rdquo; in the rail.</p>`;
      return;
    }

    this.entries.forEach((entry, i) => {
      const card = this._card(entry);
      card.style.transitionDelay = `${i * 0.04}s`;
      this.gridEl.appendChild(card);
    });

    window.revealScan();
  }

  _card(entry) {
    const card = document.createElement("article");
    card.className = "saved-card reveal";

    const overrides = Object.entries(entry.overrides);
    const rows = overrides.map(([path, value]) =>
      `<div class="saved-card__override"><span class="saved-card__path">${SharedCharts.esc(path)}</span><b>${SharedCharts.esc(value)}</b></div>`
    ).join("");

    const facts = [`${overrides.length} override${overrides.length === 1 ? "" : "s"}`, entry.saved_at.replace("T", " ")];

    card.innerHTML =
      `<div class="saved-card__top"><span class="saved-card__script">${SharedCharts.esc(entry.title)}</span>` +
      `<button type="button" class="saved-card__delete" title="Delete this template">&times;</button></div>` +
      `<h3 class="saved-card__name">${SharedCharts.esc(entry.name)}</h3>` +
      `<div class="saved-card__overrides">${rows || `<p class="saved-card__none">All defaults, no overrides.</p>`}</div>` +
      `<div class="saved-card__facts">${facts.map((f) => `<span>${SharedCharts.esc(f)}</span>`).join("")}</div>` +
      `<div class="saved-card__actions">` +
      `<button type="button" class="btn btn--primary saved-card__open" title="Open the launch page of this entry point with the template already applied to the form">&#8599;&nbsp; Open &amp; import</button>` +
      `<button type="button" class="btn btn--ghost saved-card__rename">&#9998;&nbsp; Rename</button>` +
      `</div>`;

    card.querySelector(".saved-card__open").addEventListener("click", () => this._open(entry));
    card.querySelector(".saved-card__rename").addEventListener("click", () => this._rename(entry));
    card.querySelector(".saved-card__delete").addEventListener("click", () => this._delete(entry));
    return card;
  }

  _open(entry) {
    this.launchView.pendingImport = entry;
    window.router.go(`launch/${entry.script}`);
  }

  async _rename(entry) {
    const raw = window.prompt(`Rename template "${entry.name}":`, entry.name);
    if (raw === null) return;

    const res = await Api.post(`/api/config-templates/${entry.template_id}/rename`, { name: raw });
    if (!res.ok) {
      Toast.show(res.error || "Rename failed", "error");
      return;
    }

    Toast.show(`Renamed to "${res.entry.name}"`, "ok");
    await this.load();
  }

  async _delete(entry) {
    if (!window.confirm(`Delete template "${entry.name}"?`)) return;

    const res = await Api.post(`/api/config-templates/${entry.template_id}/delete`, {});
    if (!res.ok) {
      Toast.show(res.error || "Delete failed", "error");
      return;
    }

    Toast.show(`Deleted ${entry.name}`, "ok");
    await this.load();
  }

}

window.TemplatesView = TemplatesView;
