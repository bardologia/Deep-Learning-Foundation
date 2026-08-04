"use strict";

class ConsoleTile {
  static GPU_POLL_MS = 4000;

  constructor(job, manager, host) {
    this.job = job;
    this.manager = manager;
    this.source = null;
    this.opened = false;

    this.root = document.createElement("div");
    this.root.className = "console-tile";

    const bar = document.createElement("div");
    bar.className = "console-tile__bar";

    this.desc = job.description || "";

    this.nameEl = document.createElement("span");
    this.nameEl.className = "console-tile__name";
    this.nameEl.textContent = job.script;
    this.nameEl.title = job.command;

    this.metaEl = document.createElement("span");
    this.metaEl.className = "console-tile__meta";
    this.metaEl.textContent = this._meta(job.pid ? `pid ${job.pid}` : "queued");
    if (this.desc) this.metaEl.title = this.desc;

    this.badgeEl = document.createElement("span");
    this.badgeEl.className = `badge badge--${job.status}`;
    this.badgeEl.textContent = job.status;

    this.gpuBtn = document.createElement("button");
    this.gpuBtn.className = "btn btn--mini";
    this.gpuBtn.textContent = "GPUs";
    this.gpuBtn.hidden = true;
    this.gpuBtn.title = "Resize the GPU pool of this running fan-out";
    this.gpuBtn.addEventListener("click", () => this._toggleGpus());

    this.stopBtn = document.createElement("button");
    this.stopBtn.className = "btn btn--mini btn--danger";
    this.stopBtn.addEventListener("click", () => this.manager.stop(this.job.job_id));

    this.closeBtn = document.createElement("button");
    this.closeBtn.className = "btn btn--mini";
    this.closeBtn.textContent = "Close";
    this.closeBtn.addEventListener("click", () => this.manager.close(this.job.job_id));

    bar.append(this.nameEl, this.metaEl, this.badgeEl, this.gpuBtn, this.stopBtn, this.closeBtn);

    this.poolGpus = [];
    this.gpuCards = null;
    this.gpuTimer = null;

    this.gpuPanel = document.createElement("div");
    this.gpuPanel.className = "console-tile__gpus";
    this.gpuPanel.hidden = true;

    this.gpuBoard = document.createElement("div");

    this.gpuNote = document.createElement("span");
    this.gpuNote.className = "console-tile__gpunote";

    this.gpuApply = document.createElement("button");
    this.gpuApply.className = "btn btn--mini btn--primary";
    this.gpuApply.textContent = "Apply";
    this.gpuApply.addEventListener("click", () => this._applyGpus());

    const gpuActions = document.createElement("div");
    gpuActions.className = "console-tile__gpuactions";
    gpuActions.append(this.gpuNote, this.gpuApply);

    this.gpuPanel.append(this.gpuBoard, gpuActions);

    this.progWrap = document.createElement("div");
    this.progWrap.className = "console-tile__progress";
    this.progWrap.hidden = true;

    this.progFill = document.createElement("i");
    this.progFill.className = "console-tile__pfill";

    const progTrack = document.createElement("span");
    progTrack.className = "console-tile__ptrack";
    progTrack.appendChild(this.progFill);

    this.progText = document.createElement("span");
    this.progText.className = "console-tile__ptext";

    this.progWrap.append(progTrack, this.progText);

    this.outEl = document.createElement("div");
    this.outEl.className = "console-tile__out";

    this.root.append(bar, this.progWrap, this.gpuPanel, this.outEl);
    host.appendChild(this.root);

    this.term = new Terminal({
      cols: 120,
      rows: 24,
      convertEol: true,
      disableStdin: true,
      cursorBlink: false,
      cursorInactiveStyle: "none",
      scrollback: 10000,
      fontFamily: '"JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace',
      fontSize: 12,
      lineHeight: 1.25,
      theme: {
        background: "#0e1216",
        foreground: "#dde4e7",
        cursor: "#0e1216",
        selectionBackground: "#2a3a52",
      },
    });
    this.fitAddon = new FitAddon.FitAddon();
    this.term.loadAddon(this.fitAddon);

    this.setStatus(job.status);
    this._connect();
  }

  _connect() {
    this.source = new EventSource(`/api/jobs/${this.job.job_id}/stream`);
    this.source.onmessage = (ev) => this._onEvent(ev);
    this.source.onerror = () => this._disconnect();
  }

  _disconnect() {
    if (this.source) {
      this.source.close();
      this.source = null;
    }
  }

  _onEvent(ev) {
    let data;
    try {
      data = JSON.parse(ev.data);
    } catch (e) {
      return;
    }

    if (data.type === "chunk") {
      this.term.write(data.data);
    } else if (data.type === "line") {
      this.term.writeln(data.text);
    } else if (data.type === "status") {
      if (data.status === "running") {
        if (data.adopted) this._note(`adopted process already running under your user (pid ${data.pid}) — output is not captured, stop still kills it`, "33");
        else this._note(data.detached ? `running detached (pid ${data.pid}) — stop still kills it` : `process started (pid ${data.pid})`, "36");
        this.metaEl.textContent = this._meta(data.adopted ? `pid ${data.pid} · adopted` : data.detached ? `pid ${data.pid} · detached` : `pid ${data.pid}`);
        this.setStatus("running");
      } else if (data.status === "scheduled") {
        this._note(`scheduled to run after ${data.after || "the current job"}`, "33");
      } else if (data.status === "queued") {
        this._note(`queued at position ${data.position} — starts when the previous job ends`, "33");
        this.setStatus("queued");
      } else if (data.status === "cancelled") {
        this._note("cancelled before start", "33");
        this.setStatus("cancelled");
        this.manager.refresh();
      } else {
        this._note(`process ${data.status} (exit ${data.code})`, data.code === 0 ? "36" : "31");
        this.setStatus(data.status);
        this.manager.refresh();
      }
    } else if (data.type === "end") {
      this._disconnect();
    }
  }

  _meta(base) {
    return this.desc ? `${base} · ${this.desc}` : base;
  }

  _note(text, color) {
    this.term.write(`\r\n\x1b[2;${color}m── ${text} ──\x1b[0m\r\n`);
  }

  setStatus(status) {
    this.job.status = status;
    this.badgeEl.className = `badge badge--${status}`;
    this.badgeEl.textContent = status;
    this.stopBtn.textContent = status === "scheduled" || status === "queued" ? "Cancel" : "Stop";
    this.stopBtn.disabled = status !== "running" && status !== "scheduled" && status !== "queued";

    if (status === "running") this._watchGpus();
    else {
      const hadTimer = Boolean(this.gpuTimer);
      this._unwatchGpus();
      if (hadTimer) this._pollProgress();
    }
  }

  _watchGpus() {
    if (this.gpuTimer) return;
    this._pollGpus();
    this._pollProgress();
    this.gpuTimer = setInterval(() => {
      this._pollGpus();
      this._pollProgress();
    }, ConsoleTile.GPU_POLL_MS);
  }

  _unwatchGpus() {
    clearInterval(this.gpuTimer);
    this.gpuTimer = null;
    this.gpuBtn.hidden = true;
    this.gpuPanel.hidden = true;
  }

  async _pollGpus() {
    let data = null;
    try {
      data = await window.apiGet(`/api/jobs/${this.job.job_id}/gpus`);
    } catch (e) {
      return;
    }

    const live = Boolean(data && data.ok && data.live);
    this.gpuBtn.hidden = !live;
    if (!live) {
      this.gpuPanel.hidden = true;
      return;
    }

    this.poolGpus = data.gpus || [];
    this.gpuBtn.textContent = this.poolGpus.length ? `GPUs ${this.poolGpus.join(",")}` : "GPUs parked";
    if (this.gpuPanel.hidden) return;
    this._paintGpuNote();
  }

  async _pollProgress() {
    let data = null;
    try {
      data = await window.apiGet(`/api/jobs/${this.job.job_id}/progress`);
    } catch (e) {
      return;
    }

    const prog = data && data.ok && data.progress;
    if (!prog || !prog.total) return;
    this._renderProgress(prog);
  }

  _renderProgress(p) {
    const done = p.done + p.failed;
    const bits = [`${done}/${p.total} units`];
    if (p.eta_s != null) bits.push(`avg ${window.fmtDuration(p.average_s)}/unit`, `ETA ${window.fmtDuration(p.eta_s)}`, `finish ≈ ${p.finish_at.slice(11, 16)}`);
    else if (done < p.total) bits.push("estimating ETA");
    if (p.failed) bits.push(`${p.failed} FAILED`);
    if (p.running && p.running.length) bits.push(`${p.running.length} on GPU now`);

    this.progWrap.hidden = false;
    this.progFill.style.width = `${Math.round((100 * done) / p.total)}%`;
    this.progFill.classList.toggle("is-failing", p.failed > 0);
    this.progText.textContent = bits.join(" · ");
    this.progText.title = p.failed ? `failed: ${p.failed_units.join(", ")}` : (p.running || []).map((r) => `GPU ${r.gpu}: ${r.name}`).join("\n");
  }

  _toggleGpus() {
    if (!this.gpuPanel.hidden) {
      this.gpuPanel.hidden = true;
      return;
    }

    this.gpuPanel.hidden = false;

    if (!this.gpuCards) {
      this.gpuCards = new window.GpuCardSelect(this.gpuBoard, {
        multi: true,
        initial: this.poolGpus,
        onChange: () => this._paintGpuNote(),
      });
      this.gpuCards.load().then(() => this._paintGpuNote());
      return;
    }

    this.gpuCards.set(this.poolGpus);
    this._paintGpuNote();
  }

  _paintGpuNote() {
    const chosen = this.gpuCards ? this.gpuCards.value() : [];
    const added = chosen.filter((g) => !this.poolGpus.includes(g));
    const dropped = this.poolGpus.filter((g) => !chosen.includes(g));
    const parking = !chosen.length;

    this.gpuApply.disabled = !added.length && !dropped.length;
    this.gpuApply.textContent = parking ? "Park" : "Apply";
    this.gpuApply.classList.toggle("btn--danger", parking);
    this.gpuApply.classList.toggle("btn--primary", !parking);

    if (parking) {
      this.gpuNote.textContent = this.poolGpus.length
        ? `park: runs in flight on ${this.poolGpus.join(",")} finish, nothing new starts until you add a GPU back`
        : "parked — nothing new starts until you add a GPU back";
      return;
    }
    if (!added.length && !dropped.length) {
      this.gpuNote.textContent = `pool: ${this.poolGpus.join(", ") || "parked"}`;
      return;
    }

    const bits = [];
    if (added.length) bits.push(`+${added.join(",")} start queued runs within seconds`);
    if (dropped.length) bits.push(`-${dropped.join(",")} retire once the run in flight ends`);
    this.gpuNote.textContent = bits.join(" · ");
  }

  async _applyGpus() {
    const chosen = this.gpuCards ? this.gpuCards.value() : [];
    const parking = !chosen.length;

    if (parking && !window.confirm(`Park ${this.job.script}?\n\nThe runs in flight will finish, then the experiment holds without starting anything new. It stays parked until you add a GPU back.`)) return;

    const res = await window.apiPost(`/api/jobs/${this.job.job_id}/gpus`, { gpus: chosen, park: parking });

    if (!res.ok) {
      window.toast(res.error || "Could not resize the GPU pool", "error");
      return;
    }

    this.poolGpus = res.gpus || [];
    this.gpuPanel.hidden = true;
    window.toast(res.parked ? "Experiment parked — no new runs will start" : `GPU pool set to ${this.poolGpus.join(", ")}`, res.parked ? "warn" : "ok");
    this._pollGpus();
  }

  fit() {
    const w = this.outEl.clientWidth;
    const h = this.outEl.clientHeight;
    if (!w || !h) return;

    if (!this.opened) {
      this.term.open(this.outEl);
      this.opened = true;
    }

    let font = Math.max(8, Math.min(16, (w - 16) / 72));
    this.term.options.fontSize = font;

    let dims = this.fitAddon.proposeDimensions();
    if (dims && dims.cols && dims.cols < 120) {
      font = Math.max(7, (font * dims.cols) / 120);
      this.term.options.fontSize = font;
      dims = this.fitAddon.proposeDimensions();
    }
    if (!dims || !dims.cols || !dims.rows) return;

    this.term.resize(Math.min(120, dims.cols), Math.max(2, dims.rows));
    this.term.scrollToBottom();
  }

  dispose() {
    this._disconnect();
    this._unwatchGpus();
    this.term.dispose();
    this.root.remove();
  }
}

class RunConsole {
  static LIST_POLL_MS = 5000;

  constructor(refs) {
    this.listEl = refs.list;
    this.tilesEl = refs.tiles;
    this.hintEl = refs.hint;
    this.jobs = [];
    this.tiles = new Map();
    this.dismissed = new Set();
    this._fitTimer = null;

    window.addEventListener("resize", () => this._queueFit());
    setInterval(() => {
      if (!document.hidden && this.jobs.some((j) => j.status === "running")) this.refresh();
    }, RunConsole.LIST_POLL_MS);
  }

  async refresh() {
    const data = await window.apiGet("/api/jobs");
    this.jobs = data.jobs || [];
    this.jobs
      .filter((j) => j.status === "running" && !this.dismissed.has(j.job_id) && !this.tiles.has(j.job_id))
      .forEach((j) => this.open(j.job_id));
    this._renderList();
  }

  async launch(scriptKey, interpreter, label, overrides, followUp, detach, queue) {
    const res = await window.apiPost("/api/run", { script_key: scriptKey, interpreter, overrides: overrides || {}, follow_up: followUp || null, detach: !!detach, queue: !!queue });
    if (!res.ok) {
      window.toast(res.error || (queue ? "Schedule failed" : "Launch failed"), "error");
      return null;
    }
    window.toast(res.queued ? `Scheduled ${label || scriptKey} to run after the current job` : `Launched ${label || scriptKey}`, "ok");
    if (window.router) window.router.go("console");
    await this.refresh();
    this.open(res.job_id);
    return res.job_id;
  }

  open(jobId) {
    if (this.tiles.has(jobId)) return;
    const job = this.jobs.find((j) => j.job_id === jobId);
    if (!job) return;
    this.dismissed.delete(jobId);
    this.tiles.set(jobId, new ConsoleTile(job, this, this.tilesEl));
    this._layout();
    this._renderList();
  }

  close(jobId) {
    const tile = this.tiles.get(jobId);
    if (!tile) return;
    tile.dispose();
    this.tiles.delete(jobId);
    this.dismissed.add(jobId);
    this._layout();
    this._renderList();
  }

  toggle(jobId) {
    if (this.tiles.has(jobId)) this.close(jobId);
    else this.open(jobId);
  }

  async stop(jobId) {
    const res = await window.apiPost(`/api/jobs/${jobId}/stop`, {});
    if (res.ok) window.toast("Stop signal sent", "ok");
    else window.toast(res.error || "Could not stop", "error");
  }

  onShow() {
    this._queueFit();
  }

  _layout() {
    const n = this.tiles.size;
    this.hintEl.style.display = n ? "none" : "flex";
    const cols = n <= 1 ? 1 : n <= 4 ? 2 : 3;
    const rows = Math.max(1, Math.ceil(n / cols));
    this.tilesEl.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
    this.tilesEl.style.gridTemplateRows = `repeat(${rows}, minmax(0, 1fr))`;
    this._queueFit();
  }

  _queueFit() {
    clearTimeout(this._fitTimer);
    this._fitTimer = setTimeout(() => this.tiles.forEach((t) => t.fit()), 120);
  }

  _renderList() {
    this.listEl.innerHTML = "";
    if (!this.jobs.length) {
      const empty = document.createElement("li");
      empty.className = "job-list__empty";
      empty.textContent = "No jobs yet.";
      this.listEl.appendChild(empty);
      return;
    }

    const followers = new Map();
    this.jobs.forEach((j) => {
      if (j.follow_of) followers.set(j.follow_of, j);
    });

    const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    this.jobs.forEach((job) => {
      if (job.follow_of) return;
      const item = document.createElement("li");
      item.className = "job-item" + (this.tiles.has(job.job_id) ? " is-active" : "");
      const prog = job.progress && job.progress.total ? job.progress : null;
      const progBits = prog ? window.fmtProgressBits(prog).join(" · ") : "";
      const progLine = prog
        ? `<div class="job-item__prog" title="${esc(progBits)}"><span class="job-item__pbar"><i style="width:${Math.round((100 * (prog.done + prog.failed)) / prog.total)}%" class="${prog.failed ? "is-failing" : ""}"></i></span>${esc(progBits)}</div>`
        : "";
      item.innerHTML =
        `<div class="job-item__top"><span class="job-item__name">${job.script}</span>` +
        `<span class="badge badge--${job.status}">${job.status}</span></div>` +
        (job.description ? `<div class="job-item__desc" title="${esc(job.description)}">${esc(job.description)}</div>` : "") +
        progLine +
        `<div class="job-item__meta">${job.started.replace("T", " ")}${job.pid ? ` · pid ${job.pid}` : ""}</div>`;
      item.addEventListener("click", () => this.toggle(job.job_id));

      const next = followers.get(job.job_id);
      if (next) {
        const sub = document.createElement("div");
        sub.className = "job-item__follow" + (this.tiles.has(next.job_id) ? " is-active" : "");
        sub.title = next.description || "";
        sub.innerHTML =
          `<span class="job-item__arrow" aria-hidden="true">&#8627;</span>` +
          `<span class="job-item__name">${next.script}</span>` +
          `<span class="badge badge--${next.status}">${next.status}</span>`;
        sub.addEventListener("click", (ev) => {
          ev.stopPropagation();
          this.toggle(next.job_id);
        });
        item.appendChild(sub);
      }

      this.listEl.appendChild(item);
    });
  }
}

window.RunConsole = RunConsole;
