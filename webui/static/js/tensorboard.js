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
    this.emptyOrb  = document.getElementById("tb-empty-orb");
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
    this.tree = [];
    this.expanded = new Set();

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
    const data = await Api.get("/api/tensorboard");
    if (!this.active || data.error) return;

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
        (inst.status === "starting" ? ThinkingOrb.html("listening", 20, "starting") : `<span class="tb-pill__dot" aria-hidden="true"></span>`) +
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
    this.emptyOrb.hidden = !!inst && inst.status === "failed";
    this.emptyOrb.setAttribute("state", inst ? "listening" : "shaping");
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
      const res = await Api.post("/api/tensorboard/start", { script_key: "train_backbone" });
      if (res && res.error) Toast.show(res.error, "error");
      else Toast.show("TensorBoard starting", "ok");
    } catch (e) {
      Toast.show("Could not start TensorBoard", "error");
    }
    this.startBtn.disabled = false;
    this.refresh();
  }

  async _stop() {
    const inst = this._selected();
    if (!inst) return;
    try {
      const res = await Api.post(`/api/tensorboard/${inst.id}/stop`, {});
      if (res && res.error) Toast.show(res.error, "error");
    } catch (e) {
      Toast.show("Could not stop TensorBoard", "error");
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
    this._loadTree();
  }

  _closeBrowse() {
    this.pickerOpen = false;
    this.picker.hidden = true;
    this.browseBtn.classList.remove("is-active");
  }

  async _loadTree() {
    this.pickerList.innerHTML = ThinkingOrb.panel("searching", "Scanning run directories…");
    this.pickerRoot.textContent = "";
    const data = await Api.get("/api/tensorboard/tree");
    if (data.error || data.ok === false) {
      this.pickerList.innerHTML = `<p class="tb-picker__msg">${SharedCharts.esc(data.error || "No runs root available.")}</p>`;
      return;
    }
    this.pickerRoot.textContent = data.runs_root;
    this.tree = data.tree;
    this._renderPicker();
  }

  _renderPicker() {
    const scroll = this.pickerList.scrollTop;
    this.pickerList.innerHTML = "";
    if (!this.tree.length) {
      this.pickerList.innerHTML = `<p class="tb-picker__msg">No directories with TensorBoard event data found.</p>`;
      return;
    }
    this._renderNodes(this.tree, 0);
    this.pickerList.scrollTop = scroll;
  }

  _renderNodes(nodes, depth) {
    nodes.forEach((node) => {
      const isGroup = node.kind === "group";
      this.pickerList.appendChild(isGroup ? this._groupRow(node, depth) : this._launchRow(node, depth));
      if (isGroup && this.expanded.has(node.path)) this._renderNodes(node.children, depth + 1);
    });
  }

  _groupRow(node, depth) {
    const open = this.expanded.has(node.path);
    const row = this._treeRow(depth);

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "tb-logdir tb-logdir--group" + (open ? " is-open" : "");
    toggle.title = node.path;
    toggle.setAttribute("aria-expanded", String(open));
    toggle.innerHTML = `<span class="tb-logdir__chevron" aria-hidden="true"></span>` + this._nodeLabel(node);
    toggle.addEventListener("click", () => this._toggleNode(node));

    const all = document.createElement("button");
    all.type = "button";
    all.className = "tb-logdir__all" + (node.running ? " is-running" : "");
    all.title = `Serve all ${node.run_count} runs under ${node.path}`;
    all.textContent = node.running ? "live" : "open all";
    all.addEventListener("click", () => this._launchLogdir(node));

    row.append(toggle, all);
    return row;
  }

  _launchRow(node, depth) {
    const row = this._treeRow(depth);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "tb-logdir" + (node.running ? " is-running" : "");
    button.title = node.path;
    button.innerHTML =
      `<span class="tb-logdir__chevron is-leaf" aria-hidden="true"></span>` +
      this._nodeLabel(node) +
      `<span class="tb-logdir__state">${node.running ? "live" : "launch"}</span>`;
    button.addEventListener("click", () => this._launchLogdir(node));
    row.appendChild(button);
    return row;
  }

  _treeRow(depth) {
    const row = document.createElement("div");
    row.className = "tb-node";
    row.style.setProperty("--depth", depth);
    return row;
  }

  _nodeLabel(node) {
    const runs = node.run_count === 1 ? "1 run" : `${node.run_count} runs`;
    return `<span class="tb-logdir__name">${SharedCharts.esc(node.name)}</span><span class="tb-logdir__count">${runs}</span>`;
  }

  _toggleNode(node) {
    if (this.expanded.has(node.path)) this.expanded.delete(node.path);
    else this.expanded.add(node.path);
    this._renderPicker();
  }

  async _launchLogdir(dir) {
    if (dir.running) {
      this.selectedId = dir.running;
      this._closeBrowse();
      this._render();
      return;
    }
    try {
      const res = await Api.post("/api/tensorboard/start", { logdir: dir.path });
      if (res && res.error) {
        Toast.show(res.error, "error");
        return;
      }
      if (res && res.id) this.selectedId = res.id;
      Toast.show(`TensorBoard starting over ${dir.name}`, "ok");
    } catch (e) {
      Toast.show("Could not start TensorBoard", "error");
      return;
    }
    this._closeBrowse();
    this.refresh();
  }
}

window.TensorboardView = TensorboardView;
