"use strict";

class TensorboardView {
  static POLL_MS = 5000;
  static SETTINGS_KEY = "_tb_global_settings";
  static AUTO_RELOAD_MS = 31000;
  static PAGINATION_SIZE = 500;

  constructor() {
    this._seedSettings();
    this.strip = document.getElementById("tb-strip");
    this.frame = document.getElementById("tb-frame");
    this.empty = document.getElementById("tb-empty");
    this.emptyTitle = document.getElementById("tb-empty-title");
    this.emptyHint = document.getElementById("tb-empty-hint");
    this.openBtn = document.getElementById("tb-open");
    this.stopBtn = document.getElementById("tb-stop");
    this.startBtn = document.getElementById("tb-start");
    this.browseBtn = document.getElementById("tb-browse");
    this.picker = document.getElementById("tb-picker");
    this.pickerRoot = document.getElementById("tb-picker-root");
    this.pickerList = document.getElementById("tb-picker-list");

    this.instances = [];
    this.selectedId = null;
    this.loadedUrl = null;
    this.timer = null;
    this.active = false;
    this.pickerOpen = false;

    this.openBtn.addEventListener("click", () => {
      const inst = this._selected();
      if (inst) window.open(inst.url, "_blank");
    });
    this.stopBtn.addEventListener("click", () => this._stop());
    this.startBtn.addEventListener("click", () => this._start());
    this.browseBtn.addEventListener("click", () => this._toggleBrowse());
  }

  _seedSettings() {
    let stored = {};
    try {
      stored = JSON.parse(localStorage.getItem(TensorboardView.SETTINGS_KEY) || "{}");
    } catch (e) {
      stored = {};
    }
    stored.autoReload = true;
    stored.autoReloadPeriodInMs = TensorboardView.AUTO_RELOAD_MS;
    stored.paginationSize = TensorboardView.PAGINATION_SIZE;
    localStorage.setItem(TensorboardView.SETTINGS_KEY, JSON.stringify(stored));
  }

  enter() {
    if (this.active) return;
    this.active = true;
    this.refresh();
    this.timer = setInterval(() => this.refresh(), TensorboardView.POLL_MS);
  }

  leave() {
    this.active = false;
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this._closeBrowse();
  }

  async refresh() {
    let data;
    try {
      data = await window.apiGet("/api/tensorboard");
    } catch (e) {
      return;
    }
    if (!this.active || !data || data.error) return;

    this.instances = (data.instances || []).filter((i) => i.status === "starting" || i.status === "running" || i.status === "failed");

    if (!this.instances.some((i) => i.id === this.selectedId)) this.selectedId = null;
    if (!this.selectedId && this.instances.length) this.selectedId = this.instances[0].id;

    this._render();
  }

  _selected() {
    return this.instances.find((i) => i.id === this.selectedId) || null;
  }

  _shortdir(logdir) {
    const parts = String(logdir).replace(/\/+$/, "").split("/");
    return parts.slice(-2).join("/") || logdir;
  }

  _render() {
    this.strip.innerHTML = "";
    this.instances.forEach((inst) => {
      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = "tb-pill" + (inst.id === this.selectedId ? " is-active" : "") + (inst.status === "running" ? " is-running" : inst.status === "failed" ? " is-failed" : " is-starting");
      pill.title = inst.logdir;
      pill.innerHTML =
        `<span class="tb-pill__dot" aria-hidden="true"></span>` +
        `<span class="tb-pill__name">${this._shortdir(inst.logdir)}</span>` +
        `<span class="tb-pill__state">${inst.status}</span>`;
      pill.addEventListener("click", () => {
        this.selectedId = inst.id;
        this._render();
      });
      this.strip.appendChild(pill);
    });

    const inst = this._selected();
    const ready = inst && inst.status === "running";

    this.openBtn.disabled = !ready;
    this.stopBtn.disabled = !inst;

    if (ready) {
      this.empty.hidden = true;
      this.frame.classList.add("is-live");
      if (this.loadedUrl !== inst.url) {
        this.loadedUrl = inst.url;
        this.frame.src = inst.url;
      }
      return;
    }

    this.frame.classList.remove("is-live");
    if (this.loadedUrl !== null) {
      this.loadedUrl = null;
      this.frame.src = "about:blank";
    }

    this.empty.hidden = false;
    if (inst && inst.status === "failed") {
      this.emptyTitle.textContent = "TensorBoard failed to start";
      this.emptyHint.textContent = `The instance for ${inst.logdir} exited during startup; see the console server log for its stderr tail.`;
      this.startBtn.hidden = false;
    } else if (inst) {
      this.emptyTitle.textContent = "TensorBoard is starting";
      this.emptyHint.textContent = `Indexing ${inst.logdir} — the dashboard appears here as soon as it responds.`;
      this.startBtn.hidden = true;
    } else {
      this.emptyTitle.textContent = "No TensorBoard running";
      this.emptyHint.textContent = "Launch a training job and an instance starts automatically over its log directory, or start one manually over the default training logs.";
      this.startBtn.hidden = false;
    }
  }

  async _start() {
    this.startBtn.disabled = true;
    try {
      const res = await window.apiPost("/api/tensorboard/start", { script_key: "train_backbone" });
      if (res && res.error) window.toast(res.error, "error");
      else window.toast("TensorBoard starting", "ok");
    } catch (e) {
      window.toast("Could not start TensorBoard", "error");
    }
    this.startBtn.disabled = false;
    this.refresh();
  }

  async _stop() {
    const inst = this._selected();
    if (!inst) return;
    try {
      const res = await window.apiPost(`/api/tensorboard/${inst.id}/stop`, {});
      if (res && res.error) window.toast(res.error, "error");
    } catch (e) {
      window.toast("Could not stop TensorBoard", "error");
    }
    this.selectedId = null;
    this.loadedUrl = null;
    this.frame.src = "about:blank";
    this.refresh();
  }

  _toggleBrowse() {
    if (this.pickerOpen) {
      this._closeBrowse();
      return;
    }
    this.pickerOpen = true;
    this.picker.hidden = false;
    this.browseBtn.classList.add("is-active");
    this._loadLogdirs();
  }

  _closeBrowse() {
    this.pickerOpen = false;
    this.picker.hidden = true;
    this.browseBtn.classList.remove("is-active");
  }

  async _loadLogdirs() {
    this.pickerList.innerHTML = `<p class="tb-picker__msg">Scanning run directories…</p>`;
    this.pickerRoot.textContent = "";
    let data;
    try {
      data = await window.apiGet("/api/tensorboard/logdirs");
    } catch (e) {
      this.pickerList.innerHTML = `<p class="tb-picker__msg">Could not read run directories.</p>`;
      return;
    }
    if (!data || data.error || data.ok === false) {
      this.pickerList.innerHTML = `<p class="tb-picker__msg">${(data && data.error) || "No runs root available."}</p>`;
      return;
    }
    this.pickerRoot.textContent = data.runs_root || "";
    this._renderPicker(data.logdirs || []);
  }

  _renderPicker(logdirs) {
    this.pickerList.innerHTML = "";
    const withRuns = logdirs.filter((d) => d.run_count > 0);
    if (!withRuns.length) {
      this.pickerList.innerHTML = `<p class="tb-picker__msg">No directories with TensorBoard event data found.</p>`;
      return;
    }
    withRuns.forEach((dir) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "tb-logdir" + (dir.running ? " is-running" : "");
      row.title = dir.path;
      const runs = dir.run_count === 1 ? "1 run" : `${dir.run_count} runs`;
      const state = dir.running ? "live" : "launch";
      row.innerHTML =
        `<span class="tb-logdir__name">${dir.name}</span>` +
        `<span class="tb-logdir__count">${runs}</span>` +
        `<span class="tb-logdir__state">${state}</span>`;
      row.addEventListener("click", () => this._launchLogdir(dir));
      this.pickerList.appendChild(row);
    });
  }

  async _launchLogdir(dir) {
    if (dir.running) {
      this.selectedId = dir.running;
      this._closeBrowse();
      this._render();
      return;
    }
    try {
      const res = await window.apiPost("/api/tensorboard/start", { logdir: dir.path });
      if (res && res.error) {
        window.toast(res.error, "error");
        return;
      }
      if (res && res.id) this.selectedId = res.id;
      window.toast(`TensorBoard starting over ${dir.name}`, "ok");
    } catch (e) {
      window.toast("Could not start TensorBoard", "error");
      return;
    }
    this._closeBrowse();
    this.refresh();
  }
}

window.TensorboardView = TensorboardView;
