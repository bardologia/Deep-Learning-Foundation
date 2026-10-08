"use strict";

class SequenceBoard {
  constructor(host, callbacks) {
    this.host      = host;
    this.callbacks = callbacks;
    this.surface   = null;
    this.svg       = null;
    this.entry     = null;
    this.live      = null;
    this.selected  = "";
    this.linking   = "";
    this.drag      = null;
    this.dragUsed  = false;
  }

  render(entry, live, selected, linking) {
    this.entry    = entry;
    this.live     = live;
    this.selected = selected;
    this.linking  = linking;

    this.host.innerHTML = "";

    const surface     = document.createElement("div");
    surface.className = "seq-surface";
    this.surface      = surface;

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.classList.add("seq-edges");
    this.svg = svg;

    surface.appendChild(svg);
    surface.appendChild(this._startBlock());
    entry.blocks.forEach((block) => surface.appendChild(this._block(block)));

    surface.addEventListener("click", (ev) => {
      if (this.dragUsed) {
        this.dragUsed = false;
        return;
      }
      if (ev.target === surface && this.callbacks.onBackground) this.callbacks.onBackground();
    });

    this.host.appendChild(surface);
    this.redraw();
  }

  redraw() {
    if (!this.surface || !this.surface.isConnected) return;
    this._resize();
    this._drawEdges();
  }

  _resize() {
    const blocks = [...this.surface.querySelectorAll(".seq-block")];
    const right  = Math.max(...blocks.map((el) => el.offsetLeft + el.offsetWidth), 0);
    const bottom = Math.max(...blocks.map((el) => el.offsetTop + el.offsetHeight), 0);

    this.surface.style.width  = `${Math.max(right + SequenceBoard.MARGIN, this.host.clientWidth)}px`;
    this.surface.style.height = `${bottom + SequenceBoard.MARGIN}px`;
  }

  paintStates(live) {
    this.live = live;
    if (!this.surface) return;

    this.surface.querySelectorAll(".seq-block[data-block]").forEach((el) => {
      const state = this._state(el.dataset.block);
      SequenceBoard.STATUSES.forEach((status) => el.classList.toggle(`is-${status}`, state.status === status));
      const badge = el.querySelector(".seq-block__state");
      if (badge) badge.textContent = state.status;
      const release = el.querySelector(".seq-block__release");
      if (release) release.hidden = state.status !== "held";
      this._paintError(el, state);
      this._paintJob(el, state);
      this._paintOrb(el, state);
    });
    this._paintEdgeStates();
  }

  _paintError(el, state) {
    let error = el.querySelector(".seq-block__error");
    if (!state.error) {
      if (error) error.remove();
      return;
    }
    if (!error) {
      error           = document.createElement("p");
      error.className = "seq-block__error";
      el.insertBefore(error, el.querySelector(".seq-block__actions"));
    }
    error.textContent = state.error;
    error.title       = state.error;
  }

  paintStudy(blockId, result) {
    if (!this.surface) return;
    const el = this.surface.querySelector(`.seq-block[data-block="${blockId}"]`);
    if (!el) return;
    let flag = el.querySelector(".seq-block__flag--study");
    const lines = Object.entries(result.studies || {}).map(([model, study]) => SequenceBoard.describeStudy(model, study, Object.keys(result.studies).length > 1));
    if (!lines.length) {
      if (flag) flag.remove();
      return;
    }
    if (!flag) {
      flag           = document.createElement("p");
      flag.className = "seq-block__flag seq-block__flag--study";
      el.querySelector(".seq-block__actions").before(flag);
    }
    flag.textContent = lines.join(" | ");
    flag.title       = "study snapshot read from the workspace; refreshed every minute";
  }

  static describeStudy(model, study, named) {
    const counts  = study.trials || {};
    const done    = (counts.complete || 0) + (counts.pruned || 0) + (counts.fail || 0);
    const best    = study.best ? `best ${Number(study.best.value).toPrecision(4)} @ ${study.best.trial}` : "no completed trial yet";
    const stalled = study.stalled === null || study.stalled === undefined ? "" : ` \u00b7 stalled ${study.stalled}/${study.plateau_patience}`;
    const eta     = study.eta_target_h === null || study.eta_target_h === undefined ? "" : ` \u00b7 ETA ${Number(study.eta_target_h).toFixed(1)} h`;
    return `${named ? `${model}: ` : ""}\u25d4 ${done} of ${study.target} (${counts.pruned || 0} pruned, ${counts.fail || 0} failed, ${counts.running || 0} running) \u00b7 ${best}${stalled} \u00b7 ${Number(study.trials_per_hour || 0).toFixed(1)}/h${eta}`;
  }

  _paintJob(el, state) {
    let job = el.querySelector(".seq-block__job");
    if (!state.job_id) {
      if (job) job.remove();
      return;
    }
    if (!job) {
      job             = document.createElement("button");
      job.type        = "button";
      job.className   = "btn btn--mini seq-block__job";
      job.textContent = "job";
      job.addEventListener("click", (ev) => {
        ev.stopPropagation();
        this.callbacks.onOpenJob(this._state(el.dataset.block).job_id);
      });
      el.querySelector(".seq-block__actions").appendChild(job);
    }
    job.title = `Open job ${state.job_id} in the console`;
  }

  _paintOrb(el, state) {
    const orb  = el.querySelector(".seq-block__orb");
    const kind = SequenceBoard.ORB_STATES[state.status];
    orb.hidden = !kind;
    if (kind) orb.setAttribute("state", kind);
  }

  static STATUSES     = ["idle", "pending", "held", "launching", "running", "finalizing", "deciding", "done", "failed", "finalize_failed", "cancelled", "skipped", "bypassed"];
  static ACTIVE       = ["launching", "running", "finalizing", "deciding"];
  static ORB_STATES   = { launching: "connecting", running: "working", finalizing: "composing", deciding: "solving" };
  static STUDY_STATES = ["running", "deciding", "done"];
  static SETTLED      = ["done", "skipped", "bypassed"];
  static BROKEN       = ["failed", "finalize_failed", "cancelled"];
  static SVG_NS       = "http://www.w3.org/2000/svg";
  static MARGIN   = 48;
  static GRID     = 24;
  static PORT_Y   = 34;

  _state(blockId) {
    const state = this.live && this.live.blocks ? this.live.blocks[blockId] : null;
    return state || { status: "idle", job_id: null, error: "" };
  }

  _startBlock() {
    const el     = document.createElement("div");
    el.className = "seq-block seq-block--start" + (this.selected === "__dataset__" ? " is-selected" : "");
    el.style.left = "24px";
    el.style.top  = "24px";

    const dataset = this.entry.dataset.dataset_path;
    const label   = dataset ? dataset.replace(/\/+$/, "").split("/").pop() : "no dataset selected";
    const hint    = dataset ? "delivered to every block" : "blocks run with their own paths";

    el.innerHTML =
      `<div class="seq-block__head"><span class="seq-block__kind">start</span><span class="seq-block__title">Dataset</span></div>` +
      `<p class="seq-block__meta" title="${SharedCharts.esc(dataset || "")}">${SharedCharts.esc(label)}</p>` +
      `<p class="seq-block__hint">${hint}</p>`;

    el.addEventListener("click", (ev) => {
      ev.stopPropagation();
      this.callbacks.onSelectDataset();
    });
    return el;
  }

  _block(block) {
    const state = this._state(block.block_id);
    const el    = document.createElement("div");
    el.className     = `seq-block is-${state.status}` + (block.skip ? " is-skip" : "") + (block.checkpoint ? " is-checkpoint" : "") + (this.selected === block.block_id ? " is-selected" : "") + (this.linking === block.block_id ? " is-linking" : "");
    el.dataset.block = block.block_id;
    el.style.left    = `${block.x}px`;
    el.style.top     = `${block.y}px`;

    const experiment = block.sweep || block.overrides["trials_enabled"] === "True" || ["benchmark", "cross_validate", "tune"].includes(block.script);
    const overrides  = Object.keys(block.overrides).length;
    const kind       = experiment ? "experiment" : "run";
    const decisions  = block.decisions || [];
    const undecided  = decisions.filter((decision) => !decision.choice);
    const finalizeOn = ConfigForm.FINALIZE_STEPS.filter((step) => (block.finalize || {})[step.key]);

    el.innerHTML =
      `<div class="seq-block__head">` +
      `<span class="seq-block__kind">${kind}</span>` +
      `<span class="seq-block__live"><thinking-orb class="seq-block__orb" state="${SequenceBoard.ORB_STATES[state.status] || "working"}" size="20"${SequenceBoard.ORB_STATES[state.status] ? "" : " hidden"}></thinking-orb><span class="seq-block__state">${state.status}</span></span>` +
      `</div>` +
      `<h4 class="seq-block__title" title="${SharedCharts.esc(block.script)}">${SharedCharts.esc(block.name || block.title)}</h4>` +
      `<p class="seq-block__meta">${SharedCharts.esc(block.script)} &middot; ${overrides} override${overrides === 1 ? "" : "s"}</p>` +
      (decisions.length ? `<p class="seq-block__flag seq-block__flag--decide" title="${SharedCharts.esc(decisions.map((decision) => `${decision.path} = ${decision.choice || "undecided"}`).join("\n"))}">&#9671; decides ${decisions.length} ${decisions.length === 1 ? "axis" : "axes"}${undecided.length ? ` &middot; ${undecided.length} open` : ""}</p>` : "") +
      (block.checkpoint ? `<p class="seq-block__flag">&#9670; checkpoint: holds until released</p>` : "") +
      (block.skip ? `<p class="seq-block__flag seq-block__flag--skip">&#8631; skip: bypassed on run</p>` : "") +
      (finalizeOn.length ? `<p class="seq-block__flag seq-block__flag--finalize">&#9993; finalize: ${finalizeOn.map((step) => step.short).join(" + ")} at end</p>` : "") +
      (state.error ? `<p class="seq-block__error" title="${SharedCharts.esc(state.error)}">${SharedCharts.esc(state.error)}</p>` : "") +
      `<div class="seq-block__actions">` +
      `<button type="button" class="btn btn--mini seq-block__link" title="Link: click here, then click the block that should run after this one">&rarr; link</button>` +
      `<button type="button" class="btn btn--mini seq-block__skip${block.skip ? " is-active" : ""}" title="A skipped block is bypassed once its own predecessors finish: it does not launch and then counts as satisfied for the blocks after it. Use after a crash or stop so finished blocks are not redone.">${block.skip ? "unskip" : "skip"}</button>` +
      `<button type="button" class="btn btn--mini seq-block__release" ${state.status === "held" ? "" : "hidden"} title="Release this checkpoint so the sequence continues">release</button>` +
      (state.job_id ? `<button type="button" class="btn btn--mini seq-block__job" title="Open job ${SharedCharts.esc(state.job_id)} in the console">job</button>` : "") +
      `</div>`;

    el.addEventListener("click", (ev) => {
      ev.stopPropagation();
      if (this.dragUsed) {
        this.dragUsed = false;
        return;
      }
      this.callbacks.onSelect(block.block_id);
    });
    el.querySelector(".seq-block__link").addEventListener("click", (ev) => {
      ev.stopPropagation();
      this.callbacks.onLink(block.block_id);
    });
    el.querySelector(".seq-block__skip").addEventListener("click", (ev) => {
      ev.stopPropagation();
      this.callbacks.onToggleSkip(block.block_id);
    });
    el.querySelector(".seq-block__release").addEventListener("click", (ev) => {
      ev.stopPropagation();
      this.callbacks.onRelease(block.block_id);
    });
    const job = el.querySelector(".seq-block__job");
    if (job) job.addEventListener("click", (ev) => {
      ev.stopPropagation();
      this.callbacks.onOpenJob(this._state(block.block_id).job_id);
    });

    el.addEventListener("pointerdown", (ev) => {
      if (ev.button !== 0 || ev.target.closest("button")) return;
      this._dragStart(ev, block, el);
    });
    return el;
  }

  _dragStart(ev, block, el) {
    ev.preventDefault();
    this.dragUsed = false;
    this.drag     = { block, el, dx: ev.clientX - block.x, dy: ev.clientY - block.y, moved: false };

    const move = (e) => {
      const grid = SequenceBoard.GRID;
      const rawX = Math.max(0, e.clientX - this.drag.dx);
      const rawY = Math.max(0, e.clientY - this.drag.dy);
      if (!this.drag.moved && Math.abs(rawX - block.x) + Math.abs(rawY - block.y) < 4) return;
      const x = Math.round(rawX / grid) * grid;
      const y = Math.round(rawY / grid) * grid;
      this.drag.moved = true;
      block.x = x;
      block.y = y;
      el.style.left = `${x}px`;
      el.style.top  = `${y}px`;
      this._drawEdges();
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      const moved = this.drag && this.drag.moved;
      this.drag     = null;
      this.dragUsed = Boolean(moved);
      if (moved) {
        this.redraw();
        this.callbacks.onMoved();
      }
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  _anchor(el) {
    return { x: el.offsetLeft, y: el.offsetTop, w: el.offsetWidth, h: el.offsetHeight };
  }

  _drawEdges() {
    if (!this.svg) return;
    this.svg.innerHTML = "";

    const byId  = new Map([...this.surface.querySelectorAll(".seq-block[data-block]")].map((el) => [el.dataset.block, el]));
    const start = this.surface.querySelector(".seq-block--start");
    const paths = [];

    this.entry.blocks.forEach((block) => {
      const target = byId.get(block.block_id);
      if (!target) return;

      if (!block.predecessors.length && start) paths.push([this._anchor(start), this._anchor(target), true, "__start__", block.block_id]);
      block.predecessors.forEach((pred) => {
        const source = byId.get(pred);
        if (source) paths.push([this._anchor(source), this._anchor(target), false, pred, block.block_id]);
      });
    });

    paths.forEach(([from, to, dashed, sourceId, targetId], index) => {
      const group = document.createElementNS(SequenceBoard.SVG_NS, "g");
      group.classList.add("seq-edgegroup");
      group.dataset.source = sourceId;
      group.dataset.target = targetId;

      const path = document.createElementNS(SequenceBoard.SVG_NS, "path");
      path.setAttribute("d", this._edgePath(from, to));
      path.setAttribute("id", `seq-edge-${index}`);
      path.classList.add("seq-edge");
      if (dashed) path.classList.add("seq-edge--start");
      group.appendChild(path);

      const dot = document.createElementNS(SequenceBoard.SVG_NS, "circle");
      dot.setAttribute("r", "3.5");
      dot.classList.add("seq-edge__dot");
      const motion = document.createElementNS(SequenceBoard.SVG_NS, "animateMotion");
      motion.setAttribute("dur", "1.6s");
      motion.setAttribute("repeatCount", "indefinite");
      const mpath = document.createElementNS(SequenceBoard.SVG_NS, "mpath");
      mpath.setAttributeNS("http://www.w3.org/1999/xlink", "href", `#seq-edge-${index}`);
      motion.appendChild(mpath);
      dot.appendChild(motion);
      group.appendChild(dot);

      this.svg.appendChild(group);
    });

    this._paintEdgeStates();
  }

  _paintEdgeStates() {
    if (!this.svg) return;
    const flowing = this._flowingEdges();

    this.svg.querySelectorAll(".seq-edgegroup").forEach((group) => {
      const status = this._state(group.dataset.target).status;
      const flows  = flowing.has(`${group.dataset.source}->${group.dataset.target}`);
      group.classList.toggle("is-flowing", flows);
      group.classList.toggle("is-done", !flows && SequenceBoard.SETTLED.includes(status));
      group.classList.toggle("is-failed", SequenceBoard.BROKEN.includes(status));
    });
  }

  _flowingEdges() {
    const byId    = new Map(this.entry.blocks.map((block) => [block.block_id, block]));
    const flowing = new Set();

    const feed = (blockId) => {
      const block = byId.get(blockId);
      const preds = block && block.predecessors.length ? block.predecessors : ["__start__"];
      preds.forEach((pred) => {
        const key = `${pred}->${blockId}`;
        if (flowing.has(key)) return;
        flowing.add(key);
        if (pred !== "__start__" && this._state(pred).status === "bypassed") feed(pred);
      });
    };

    this.entry.blocks.forEach((block) => {
      if (SequenceBoard.ACTIVE.includes(this._state(block.block_id).status)) feed(block.block_id);
    });
    return flowing;
  }

  _edgePath(from, to) {
    const fromCx = from.x + from.w / 2;
    const fromCy = from.y + from.h / 2;
    const toCx   = to.x + to.w / 2;
    const toCy   = to.y + to.h / 2;

    if (Math.abs(toCx - fromCx) >= Math.abs(toCy - fromCy)) {
      const fromPort  = from.y + Math.min(SequenceBoard.PORT_Y, from.h / 2);
      const toPort    = to.y + Math.min(SequenceBoard.PORT_Y, to.h / 2);
      const rightward = toCx >= fromCx;
      const x1 = rightward ? from.x + from.w : from.x;
      const x2 = rightward ? to.x : to.x + to.w;
      const dx = Math.max(40, Math.abs(x2 - x1) / 2) * (rightward ? 1 : -1);
      return `M ${x1} ${fromPort} C ${x1 + dx} ${fromPort}, ${x2 - dx} ${toPort}, ${x2} ${toPort}`;
    }

    const downward = toCy >= fromCy;
    const y1 = downward ? from.y + from.h : from.y;
    const y2 = downward ? to.y : to.y + to.h;
    const dy = Math.max(40, Math.abs(y2 - y1) / 2) * (downward ? 1 : -1);
    return `M ${fromCx} ${y1} C ${fromCx} ${y1 + dy}, ${toCx} ${y2 - dy}, ${toCx} ${y2}`;
  }
}


class SequencesView extends ConfigForm {

  constructor(runConsole, project) {
    super();
    this.runConsole  = runConsole;
    this.project     = project || {};
    this.host        = document.getElementById("sequences-root");
    this.copyControl = new window.RunConfigCopy(this);
    this.templateControl = new window.ConfigTemplates(this);

    this.sequences  = [];
    this.scripts    = [];
    this.current    = null;
    this.live       = null;
    this.selected   = "";
    this.linking    = "";
    this.editorKey  = "";
    this.pollTimer  = null;
    this.loaded     = false;
    this.drawerOpen = false;
    this.escHandler = (ev) => {
      if (ev.key === "Escape" && this.selected) this._closeDrawer();
    };


    this.railEl    = null;
    this.mainEl    = null;
    this.boardHost = null;
    this.drawerEl  = null;
    this.statusEl  = null;
    this.dirEl     = null;
    this.runBtn    = null;
    this.resumeBtn = null;
    this.board     = null;

    this.busyAction = false;
  }

  static async quickAdd(script, title) {
    let entry = null;
    const active = localStorage.getItem("seq-active") || "";

    if (active) {
      const res = await Api.get(`/api/sequences/${active}`);
      if (res.ok) entry = res.entry;
    }
    if (!entry) {
      const created = await Api.post("/api/sequences", { name: "" });
      if (!created.ok) {
        Toast.show(created.error || "Could not create a sequence", "error");
        return;
      }
      entry = created.entry;
      localStorage.setItem("seq-active", entry.sequence_id);
    }

    const n = entry.blocks.length;
    entry.blocks.push({ script, title, overrides: {}, decisions: [], predecessors: [], checkpoint: false, x: 48 + n * 240, y: 144 });

    const saved = await Api.post(`/api/sequences/${entry.sequence_id}/save`, entry);
    if (!saved.ok) {
      Toast.show(saved.error || "Could not add the block", "error");
      return;
    }

    Toast.show(`Added ${title} to sequence "${entry.name}". Open the Sequences tab to wire it.`, "ok");
  }

  async enter() {
    await this._loadList();

    const active = this.current ? this.current.sequence_id : (localStorage.getItem("seq-active") || "");
    const pick   = this.sequences.find((s) => s.sequence_id === active) || this.sequences[0];
    if (pick) await this._open(pick.sequence_id);
    this.loaded = true;

    this._render();
    this._startPoll();
    document.removeEventListener("keydown", this.escHandler);
    document.addEventListener("keydown", this.escHandler);
  }

  leave() {
    this._stopPoll();
    document.removeEventListener("keydown", this.escHandler);
    this.drawerOpen = false;
  }

  async _loadList() {
    const data     = await Api.get("/api/sequences");
    this.sequences = data.sequences || [];

    if (!this.scripts.length) {
      const catalog = await Api.get("/api/scripts");
      this.scripts  = (catalog.scripts || []).filter((s) => s.config_class);
    }
  }

  async _open(sequenceId) {
    const res = await Api.get(`/api/sequences/${sequenceId}`);
    if (!res.ok) {
      Toast.show(res.error || "Could not open the sequence", "error");
      return;
    }
    this.current  = res.entry;
    this.live     = res.state;
    this.selected = "";
    this.linking  = "";
    localStorage.setItem("seq-active", sequenceId);
  }

  async _create() {
    const name    = window.prompt("Name of the new sequence:", "");
    if (name === null) return;
    const created = await Api.post("/api/sequences", { name });
    if (!created.ok) {
      Toast.show(created.error || "Create failed", "error");
      return;
    }
    await this._loadList();
    await this._open(created.entry.sequence_id);
    this._render();
  }

  async _deleteSequence(entry) {
    if (!window.confirm(`Delete sequence "${entry.name}"?`)) return;
    const res = await Api.post(`/api/sequences/${entry.sequence_id}/delete`, {});
    if (!res.ok) {
      Toast.show(res.error || "Delete failed", "error");
      return;
    }
    if (this.current && this.current.sequence_id === entry.sequence_id) this.current = null;
    await this._loadList();
    this._render();
  }

  async _persist() {
    if (!this.current) return false;
    const saved = await Api.post(`/api/sequences/${this.current.sequence_id}/save`, this.current);
    if (!saved.ok) {
      Toast.show(saved.error || "Save failed", "error");
      return false;
    }
    const byId = new Map(this.current.blocks.filter((block) => block.block_id).map((block) => [block.block_id, block]));
    saved.entry.blocks.forEach((clean, index) => {
      const block = byId.get(clean.block_id) || this.current.blocks[index];
      block.block_id    = clean.block_id;
      block.name        = clean.name;
      block.title       = clean.title;
      block.interpreter = clean.interpreter;
      block.target      = clean.target;
      block.sweep       = clean.sweep;
      block.skip        = clean.skip;
      block.finalize    = clean.finalize;
      block.resources   = clean.resources;
      block.overrides   = clean.overrides;
      block.field_notes = clean.field_notes;
      block.decisions   = clean.decisions;
    });
    return true;
  }

  async _start(resumeDir = "", bypass = []) {
    if (this.busyAction) return;
    this.busyAction = true;
    try {
      if (!(await this._persist())) return;
      const res = await Api.post(`/api/sequences/${this.current.sequence_id}/start`, resumeDir ? { resume_dir: resumeDir, bypass } : {});
      if (!res.ok) {
        Toast.show(res.error || "Start failed", "error");
        return;
      }
      this.live = res.state;
      Toast.show(resumeDir ? "Sequence resumed in the previous workspace" : "Sequence started", "ok");
      if (this.board) this.board.paintStates(this.live);
      this._paintStatus();
      this._pollStudies();
    } finally {
      this.busyAction = false;
    }
  }

  async _resume() {
    const previous = this.live && this.live.run_dir ? this.live.run_dir : "";
    const input    = window.prompt("Workspace directory of the execution to resume (runs/sequences/…). Blocks it records as finished are bypassed on this start only; the rest runs inside this workspace:", previous);
    if (input === null || !input.trim()) return;
    const dir = input.trim();

    const scan = await Api.post(`/api/sequences/${this.current.sequence_id}/resume-scan`, { resume_dir: dir });
    if (!scan.ok) {
      Toast.show(scan.error || "Resume scan failed", "error");
      return;
    }

    if (!scan.recorded) {
      if (!window.confirm("This workspace holds no recorded execution state, so finished blocks cannot be detected. Start inside it using the current skip marks?")) return;
      await this._start(dir);
      return;
    }

    const byId  = new Map(this.current.blocks.map((b) => [b.block_id, b]));
    const names = scan.done.map((id) => byId.get(id).name || byId.get(id).title);
    const ask   = names.length
      ? `Recorded as finished in this workspace, bypassed on this start only:\n\n${names.join("\n")}\n\nThe remaining blocks run inside the workspace. Continue?`
      : "No block is recorded as finished in this workspace. Start inside it from the beginning?";
    if (!window.confirm(ask)) return;

    await this._start(dir, scan.done);
  }

  async _stop() {
    if (this.busyAction) return;
    this.busyAction = true;
    try {
      const res = await Api.post(`/api/sequences/${this.current.sequence_id}/stop`, {});
      if (!res.ok) {
        Toast.show(res.error || "Stop failed", "error");
        return;
      }
      this.live = res.state;
      Toast.show("Sequence stopped", "warn");
      if (this.board) this.board.paintStates(this.live);
      this._paintStatus();
    } finally {
      this.busyAction = false;
    }
  }

  async _release(blockId) {
    if (this.busyAction) return;
    const pending = this._upstreamUndecided(blockId);
    if (pending.length && !window.confirm(`These earlier blocks have no verdict recorded yet:\n\n${pending.map((item) => item.label).join("\n")}\n\nReleasing now runs this block on the pre-verdict values. Continue?`)) {
      this.selected = pending[0].block_id;
      this._render();
      return;
    }

    this.busyAction = true;
    try {
      const res = await Api.post(`/api/sequences/${this.current.sequence_id}/release`, { block_id: blockId });
      if (!res.ok) {
        Toast.show(res.error || "Release failed", "error");
        return;
      }
      this.live = res.state;
      Toast.show("Checkpoint released", "ok");
      if (this.board) this.board.paintStates(this.live);
      this._paintStatus();
    } finally {
      this.busyAction = false;
    }
  }

  async _refreshBoard() {
    let cleared = false;
    const active = this.live && ["running", "held"].includes(this.live.status);
    if (this.current && this.live && !active) {
      const res = await Api.post(`/api/sequences/${this.current.sequence_id}/clear`, {});
      if (!res.ok) {
        Toast.show(res.error || "Clear failed", "error");
        return;
      }
      cleared = res.cleared;
    }

    await this._loadList();
    if (this.current) await this._open(this.current.sequence_id);
    this._render();
    this._startPoll();
    Toast.show(cleared ? "Execution state cleared, board reset to idle" : "Sequence view refreshed from the server", "ok");
  }

  _startPoll() {
    this._stopPoll();
    this.pollTimer  = setInterval(() => this._poll(), 3000);
    this.studyTimer = setInterval(() => this._pollStudies(), 60000);
    this._pollStudies();
  }

  _stopPoll() {
    clearInterval(this.pollTimer);
    clearInterval(this.studyTimer);
    this.pollTimer  = null;
    this.studyTimer = null;
  }

  async _pollStudies() {
    if (!this.current || !this.live || !this.board) return;
    const blocks = this.current.blocks.filter((block) => block.script === "tune" && SequenceBoard.STUDY_STATES.includes((this.live.blocks[block.block_id] || {}).status));
    for (const block of blocks) {
      const res = await Api.get(`/api/sequences/${this.current.sequence_id}/study?block=${block.block_id}`);
      if (res && res.ok) this.board.paintStudy(block.block_id, res);
    }
  }

  async _poll() {
    if (!this.current) return;
    const res = await Api.get(`/api/sequences/${this.current.sequence_id}`);
    if (!res.ok) return;

    if (!this.selected && res.entry.updated_at !== this.current.updated_at) {
      this.current = res.entry;
      this.live    = res.state;
      this._render();
      return;
    }

    const changed = JSON.stringify(res.state) !== JSON.stringify(this.live);
    this.live = res.state;
    if (!changed) return;

    if (this.board) this.board.paintStates(this.live);
    this._paintStatus();
  }

  _render() {
    this.host.innerHTML = "";

    const layout     = document.createElement("div");
    layout.className = "seq-layout";

    this.railEl = this._buildRail();
    this.mainEl = this._buildMain();
    layout.appendChild(this.railEl);
    layout.appendChild(this.mainEl);

    this.host.appendChild(layout);
    if (this.board) this.board.redraw();
    this._paintStatus();
    this._pollStudies();
  }

  _buildRail() {
    const rail     = document.createElement("aside");
    rail.className = "seq-rail";

    const create       = document.createElement("button");
    create.type        = "button";
    create.className   = "btn btn--primary seq-rail__new";
    create.textContent = "+ New sequence";
    create.addEventListener("click", () => this._create());
    rail.appendChild(create);

    if (!this.sequences.length) {
      const empty       = document.createElement("p");
      empty.className   = "seq-rail__empty";
      empty.textContent = "No sequences yet. Create one here, or press “+ sequence” on any script card.";
      rail.appendChild(empty);
    }

    this.sequences.forEach((entry) => {
      const active = this.current && entry.sequence_id === this.current.sequence_id;
      const status = entry.state ? entry.state.status : "idle";

      const card     = document.createElement("div");
      card.className = "seq-rail__card" + (active ? " is-active" : "");
      card.innerHTML =
        `<div class="seq-rail__row"><span class="seq-rail__name">${SharedCharts.esc(entry.name)}</span>` +
        `<button type="button" class="seq-rail__delete" title="Delete this sequence">&times;</button></div>` +
        `<div class="seq-rail__row"><span class="seq-rail__facts">${entry.blocks.length} block${entry.blocks.length === 1 ? "" : "s"}</span>` +
        `<span class="seq-status seq-status--${status}">${status}</span></div>`;

      card.addEventListener("click", async () => {
        await this._open(entry.sequence_id);
        this._render();
      });
      card.querySelector(".seq-rail__delete").addEventListener("click", async (ev) => {
        ev.stopPropagation();
        await this._deleteSequence(entry);
      });
      rail.appendChild(card);
    });

    return rail;
  }

  _buildMain() {
    const main     = document.createElement("div");
    main.className = "seq-main";

    if (!this.current) {
      main.innerHTML = `<p class="seq-empty">Select or create a sequence on the left.</p>`;
      return main;
    }

    main.appendChild(this._buildBar());

    const gpuRow = this._buildGpuAllRow();
    if (gpuRow) main.appendChild(gpuRow);

    const dir     = document.createElement("p");
    dir.className = "seq-bar__dir";
    dir.hidden    = true;
    this.dirEl    = dir;
    main.appendChild(dir);

    const boardHost     = document.createElement("div");
    boardHost.className = "seq-board";
    this.boardHost      = boardHost;
    main.appendChild(boardHost);

    const drawer     = document.createElement("div");
    drawer.className = "seq-drawer";
    drawer.hidden    = true;
    this.drawerEl    = drawer;
    main.appendChild(drawer);

    this.board = new SequenceBoard(boardHost, {
      onSelect        : (blockId) => this._selectBlock(blockId),
      onSelectDataset : () => this._selectDataset(),
      onLink          : (blockId) => this._toggleLinking(blockId),
      onToggleSkip    : (blockId) => this._toggleSkip(blockId),
      onMoved         : () => this._persist(),
      onRelease       : (blockId) => this._release(blockId),
      onOpenJob       : (jobId) => this._openJob(jobId),
      onBackground    : () => this._closeDrawer(),
    });
    this.board.render(this.current, this.live, this.selected, this.linking);

    if (this.selected === "__dataset__") this._renderDatasetEditor();
    else if (this.selected) this._renderBlockEditor();

    return main;
  }

  _buildBar() {
    const bar     = document.createElement("div");
    bar.className = "seq-bar";

    const name       = document.createElement("input");
    name.className   = "cfg-edit__input seq-bar__name";
    name.value       = this.current.name;
    name.spellcheck  = false;
    name.title       = "Sequence name";
    name.addEventListener("change", async () => {
      this.current.name = name.value;
      if (await this._persist()) await this._loadList();
      this._render();
    });
    bar.appendChild(name);

    const add     = document.createElement("select");
    add.className = "cfg-edit__input picker__select seq-bar__add";
    add.innerHTML = `<option value="">add a block&hellip;</option>` + this.scripts.map((s) => `<option value="${SharedCharts.esc(s.key)}">${SharedCharts.esc(s.title)}</option>`).join("");
    add.addEventListener("change", async () => {
      if (!add.value) return;
      const script = this.scripts.find((s) => s.key === add.value);
      const n      = this.current.blocks.length;
      this.current.blocks.push({ script: script.key, title: script.title, overrides: {}, decisions: [], predecessors: [], checkpoint: false, x: 48 + n * 240, y: 144 });
      add.value = "";
      if (await this._persist()) this._render();
    });
    bar.appendChild(add);

    const detachWrap     = document.createElement("div");
    detachWrap.className = "rail-detach seq-bar__detach";
    const detachToggle   = document.createElement("button");
    detachToggle.type      = "button";
    detachToggle.className = "switch" + (this.current.detach ? " is-on" : "");
    detachToggle.setAttribute("role", "switch");
    detachToggle.setAttribute("aria-checked", String(this.current.detach));
    detachToggle.innerHTML = `<span class="switch__knob"></span>`;
    detachToggle.title     = "Detach each block's run from the console server";
    detachToggle.addEventListener("click", async () => {
      this.current.detach = !this.current.detach;
      if (await this._persist()) this._render();
    });
    const detachLabel       = document.createElement("span");
    detachLabel.className   = "rail-detach__label";
    detachLabel.textContent = "Detach runs";
    detachWrap.appendChild(detachToggle);
    detachWrap.appendChild(detachLabel);
    bar.appendChild(detachWrap);

    const isolateWrap     = document.createElement("div");
    isolateWrap.className = "rail-detach seq-bar__isolate";
    const isolateToggle   = document.createElement("button");
    isolateToggle.type      = "button";
    isolateToggle.className = "switch" + (this.current.isolate ? " is-on" : "");
    isolateToggle.setAttribute("role", "switch");
    isolateToggle.setAttribute("aria-checked", String(this.current.isolate));
    isolateToggle.innerHTML = `<span class="switch__knob"></span>`;
    isolateToggle.title     = "Each start creates a fresh runs/sequences/<name>_<stamp> directory; every block's run output root (and runs-tree inputs) is redirected there, so repeated executions never collide";
    isolateToggle.addEventListener("click", async () => {
      this.current.isolate = !this.current.isolate;
      if (await this._persist()) this._render();
    });
    const isolateLabel       = document.createElement("span");
    isolateLabel.className   = "rail-detach__label";
    isolateLabel.textContent = "Isolated dir";
    isolateWrap.appendChild(isolateToggle);
    isolateWrap.appendChild(isolateLabel);
    bar.appendChild(isolateWrap);


    const status     = document.createElement("span");
    status.className = "seq-status";
    this.statusEl    = status;
    bar.appendChild(status);

    const refresh       = document.createElement("button");
    refresh.type        = "button";
    refresh.className   = "btn btn--ghost seq-bar__refresh";
    refresh.innerHTML   = "&#8635; Refresh";
    refresh.title       = "Reload this sequence from the server and redraw the board; a finished, failed or stopped execution state is cleared so the board returns to idle (the workspace state file stays for resume)";
    refresh.addEventListener("click", () => this._refreshBoard());
    bar.appendChild(refresh);

    const run     = document.createElement("button");
    run.type      = "button";
    run.className = "btn btn--primary seq-bar__run";
    this.runBtn   = run;
    run.addEventListener("click", () => {
      const active = this.live && ["running", "held"].includes(this.live.status);
      if (active) this._stop();
      else this._start();
    });
    bar.appendChild(run);

    const resume     = document.createElement("button");
    resume.type      = "button";
    resume.className = "btn btn--ghost seq-bar__resume";
    resume.textContent = "Resume…";
    resume.title     = "Start this sequence inside the workspace of a previous execution instead of a fresh directory; blocks that workspace records as finished are bypassed on that start only, so only the rest replays";
    this.resumeBtn   = resume;
    resume.addEventListener("click", () => this._resume());
    bar.appendChild(resume);

    return bar;
  }

  _buildGpuAllRow() {
    if (this._allTarget() !== "local") return null;

    const row     = document.createElement("div");
    row.className = "seq-bar__gpurow";
    row.innerHTML = `<span class="seq-bar__gpulabel" title="One GPU pool written into the gpus override of every block; each block fans its runs out across this pool">GPU pool, all blocks</span>`;

    const board = document.createElement("div");
    row.appendChild(board);

    const note       = document.createElement("span");
    note.className   = "picker__note seq-bar__gpunote";
    row.appendChild(note);

    const paint = (indices) => {
      note.textContent = indices.length ? `${indices.length} GPU${indices.length === 1 ? "" : "s"} for every block` : "select at least one GPU";
    };

    const initial = this._commonGpus();
    new GpuCardSelect(board, {
      multi    : true,
      initial  : initial,
      onChange : (indices) => {
        paint(indices);
        this._applyGpusToAll(indices);
      },
    }).load();
    paint(initial);

    return row;
  }

  _commonGpus() {
    try {
      const parsed = PythonLiteral.parse(this.current.blocks[0].overrides["gpus"] || "[]");
      return Array.isArray(parsed) ? parsed.map(Number).filter(Number.isFinite) : [];
    } catch (e) {
      return [];
    }
  }

  async _applyGpusToAll(indices) {
    const value = PythonLiteral.render(indices);
    this.current.blocks.forEach((block) => { block.overrides["gpus"] = value; });
    if (await this._persist()) Toast.show(`GPU pool ${value} written into all ${this.current.blocks.length} blocks`, "ok");
  }

  _paintStatus() {
    if (!this.statusEl || !this.current) return;
    const status = this.live ? this.live.status : "idle";
    const active = ["running", "held"].includes(status);

    this.statusEl.textContent = status;
    this.statusEl.className   = `seq-status seq-status--${status}`;
    this.runBtn.textContent   = active ? "Stop sequence" : "Start sequence";
    this.runBtn.classList.toggle("btn--primary", !active);
    this.runBtn.classList.toggle("btn--ghost", active);
    this.resumeBtn.hidden     = active || !this.current.isolate;

    if (this.dirEl) {
      const runDir = this.live ? this.live.run_dir : "";
      this.dirEl.hidden      = !runDir;
      this.dirEl.textContent = runDir ? `workspace: ${runDir}` : "";
    }
  }

  _openJob(jobId) {
    window.router.go("console");
    this.runConsole.refresh().then(() => this.runConsole.open(jobId));
  }

  _selectBlock(blockId) {
    if (this.linking && this.linking !== blockId) {
      this._toggleEdge(this.linking, blockId);
      return;
    }
    this.linking  = "";
    this.selected = blockId;
    this._render();
  }

  _selectDataset() {
    this.linking  = "";
    this.selected = "__dataset__";
    this._render();
  }

  async _toggleSkip(blockId) {
    const block = this.current.blocks.find((b) => b.block_id === blockId);
    block.skip  = !block.skip;
    if (!(await this._persist())) block.skip = !block.skip;
    this._render();
  }

  _toggleLinking(blockId) {
    this.linking = this.linking === blockId ? "" : blockId;
    if (this.linking) Toast.show("Now click the block that should run after this one (click link again to cancel)", "ok");
    this.board.render(this.current, this.live, this.selected, this.linking);
  }

  async _toggleEdge(sourceId, targetId) {
    const target = this.current.blocks.find((b) => b.block_id === targetId);
    const before = target.predecessors.slice();
    const index  = target.predecessors.indexOf(sourceId);

    if (index >= 0) target.predecessors.splice(index, 1);
    else target.predecessors.push(sourceId);

    this.linking = "";
    if (!(await this._persist())) target.predecessors = before;
    this._render();
  }

  _closeDrawer() {
    if (!this.selected) return;
    this.selected   = "";
    this.linking    = "";
    this.drawerOpen = false;
    this._render();
  }

  _drawerShell(title, hint) {
    this.drawerEl.hidden    = false;
    this.drawerEl.innerHTML = "";
    this.drawerEl.classList.toggle("seq-drawer--enter", !this.drawerOpen);
    this.drawerEl.setAttribute("role", "dialog");
    this.drawerEl.setAttribute("aria-label", title);
    this.drawerOpen = true;

    const head     = document.createElement("div");
    head.className = "seq-drawer__head";
    head.innerHTML = `<h3 class="seq-drawer__title">${SharedCharts.esc(title)}</h3><span class="seq-drawer__hint">${SharedCharts.esc(hint)}</span><button type="button" class="seq-drawer__close" title="Close">&times;</button>`;
    head.querySelector(".seq-drawer__close").addEventListener("click", () => this._closeDrawer());
    this.drawerEl.appendChild(head);

    const body     = document.createElement("div");
    body.className = "seq-drawer__body";
    this.drawerEl.appendChild(body);
    return body;
  }

  _renderDatasetEditor() {
    this.reset();
    const body = this._drawerShell("Sequence dataset", "the starting block delivers this dataset to every block at launch");

    const rootLeaf   = { path: "seq.dataset_base", value: ResultsSources.datasets(), type: "str", editable: true, section: "seq" };
    const dsLeaf     = { path: "seq.dataset_path", value: this.current.dataset.dataset_path, type: "str", editable: true, section: "seq" };
    const paramsLeaf = { path: "seq.parameters_path", value: this.current.dataset.parameters_path, type: "str", editable: true, section: "seq" };
    this.byPath = new Map([[rootLeaf.path, rootLeaf], [dsLeaf.path, dsLeaf], [paramsLeaf.path, paramsLeaf]]);

    const rows = document.createElement("div");
    rows.className = "seq-dataset";

    const addRow = (label, control) => {
      const row     = document.createElement("div");
      row.className = "cfg-edit__row cfg-edit__row--wide";
      row.innerHTML = `<div class="cfg-edit__name">${SharedCharts.esc(label)}</div>`;
      row.appendChild(control.el);
      rows.appendChild(row);
    };

    addRow("datasets root", this._textControl(rootLeaf));
    addRow("dataset", new window.DatasetPicker(this, dsLeaf, { kind: "dataset", mode: "datasets", baseFrom: "seq.dataset_base", validOnly: true }).build());
    addRow("parameters", new window.DatasetPicker(this, paramsLeaf, { kind: "dataset", mode: "params", datasetFrom: "seq.dataset_path" }).build());
    body.appendChild(rows);

    const note       = document.createElement("p");
    note.className   = "cfg-note";
    note.textContent = "At launch, blocks exposing paths.dataset_path and paths.parameters_path get both; multi-dataset entries get dataset_base_path plus a dataset_filter for this folder. Clear the dataset to let each block use its own paths.";
    body.appendChild(note);

    const actions     = document.createElement("div");
    actions.className = "seq-drawer__actions";

    const clear       = document.createElement("button");
    clear.type        = "button";
    clear.className   = "btn btn--ghost";
    clear.textContent = "Clear dataset";
    clear.addEventListener("click", async () => {
      this.current.dataset = { dataset_path: "", parameters_path: "" };
      if (await this._persist()) this._render();
    });

    const apply       = document.createElement("button");
    apply.type        = "button";
    apply.className   = "btn btn--primary";
    apply.textContent = "Apply dataset";
    apply.addEventListener("click", async () => {
      this.current.dataset = { dataset_path: String(this._effective(dsLeaf)), parameters_path: String(this._effective(paramsLeaf)) };
      if (await this._persist()) {
        Toast.show("Dataset applied to the sequence", "ok");
        this._render();
      }
    });

    actions.appendChild(clear);
    actions.appendChild(apply);
    body.appendChild(actions);
  }

  async _renderBlockEditor() {
    const block = this.current.blocks.find((b) => b.block_id === this.selected);
    if (!block) {
      this.selected = "";
      return;
    }

    this.reset();
    const body = this._drawerShell(block.name || block.title, `${block.script} · edit and apply; waiting blocks pick edits up at launch`);

    const controls     = document.createElement("div");
    controls.className = "seq-drawer__controls";

    const name       = document.createElement("input");
    name.className   = "cfg-edit__input";
    name.placeholder = "block name (optional)";
    name.value       = block.name;
    name.spellcheck  = false;
    name.addEventListener("change", async () => {
      block.name = name.value;
      if (await this._persist()) this.board.render(this.current, this.live, this.selected, this.linking);
    });
    controls.appendChild(name);

    const checkpoint       = document.createElement("button");
    checkpoint.type        = "button";
    checkpoint.className   = "btn btn--mini" + (block.checkpoint ? " is-active" : "");
    checkpoint.textContent = block.checkpoint ? "◆ checkpoint on" : "◇ checkpoint off";
    checkpoint.title       = "A checkpoint holds the line here: the block waits, released by hand, so it can be edited against the previous results first";
    checkpoint.addEventListener("click", async () => {
      block.checkpoint = !block.checkpoint;
      if (await this._persist()) this._render();
    });
    controls.appendChild(checkpoint);

    const skip       = document.createElement("button");
    skip.type        = "button";
    skip.className   = "btn btn--mini" + (block.skip ? " is-active" : "");
    skip.textContent = block.skip ? "↷ skip on" : "↷ skip off";
    skip.title       = "A skipped block is bypassed: it does not launch and counts as satisfied for the blocks after it. Use after a crash or stop so finished blocks are not redone.";
    skip.addEventListener("click", () => this._toggleSkip(block.block_id));
    controls.appendChild(skip);

    const preds = document.createElement("div");
    preds.className = "seq-drawer__preds";
    const byId = new Map(this.current.blocks.map((b) => [b.block_id, b]));
    preds.innerHTML = `<span class="seq-drawer__predlabel">after:</span>` + (block.predecessors.length
      ? block.predecessors.map((pred) => `<span class="seq-drawer__pred" data-pred="${SharedCharts.esc(pred)}">${SharedCharts.esc((byId.get(pred) || {}).title || pred)} <b>&times;</b></span>`).join("")
      : `<span class="seq-drawer__prednone">the start block</span>`);
    preds.querySelectorAll(".seq-drawer__pred").forEach((chip) => {
      chip.addEventListener("click", async () => {
        block.predecessors = block.predecessors.filter((pred) => pred !== chip.dataset.pred);
        if (await this._persist()) this._render();
      });
    });
    controls.appendChild(preds);

    body.appendChild(controls);

    const decideHost     = document.createElement("div");
    decideHost.className = "seq-decide";
    body.appendChild(decideHost);

    const configHost     = document.createElement("div");
    configHost.className = "seq-drawer__config";
    configHost.innerHTML = ThinkingOrb.panel("breathing", "Loading configuration…");
    body.appendChild(configHost);

    const actions     = document.createElement("div");
    actions.className = "seq-drawer__actions";

    const remove       = document.createElement("button");
    remove.type        = "button";
    remove.className   = "btn btn--mini btn--danger seq-drawer__remove";
    remove.textContent = "Remove block";
    remove.addEventListener("click", async () => {
      if (!window.confirm(`Remove block "${block.name || block.title}" from the sequence?`)) return;
      this.current.blocks = this.current.blocks.filter((b) => b.block_id !== block.block_id);
      this.current.blocks.forEach((b) => { b.predecessors = b.predecessors.filter((pred) => pred !== block.block_id); });
      this.selected = "";
      if (await this._persist()) this._render();
    });
    actions.appendChild(remove);

    const apply       = document.createElement("button");
    apply.type        = "button";
    apply.className   = "btn btn--primary";
    apply.textContent = "Apply to block";
    apply.disabled    = true;
    actions.appendChild(apply);
    body.appendChild(actions);

    const editorKey = `${block.block_id}:${block.script}`;
    this.editorKey  = editorKey;

    const cfg = await Api.get(`/api/scripts/${block.script}/config`);
    if (this.editorKey !== editorKey) return;
    if (!cfg || !cfg.ok) {
      configHost.innerHTML = `<p class="cfg-note">${SharedCharts.esc((cfg && cfg.error) || "Could not resolve configuration.")}</p>`;
      return;
    }

    const endpoint = this._modelEndpoint(block, cfg);
    if (this.modelEndpoint !== endpoint) {
      const models       = await Api.get(endpoint);
      this.modelEndpoint = endpoint;
      this.modelFamilies = (models && models.families) || [];
      this.modelHeads    = (models && models.heads) || [];
    }
    if (this.editorKey !== editorKey) return;

    this.config = cfg;
    this._buildDecisions(decideHost, block, cfg);
    configHost.innerHTML = "";

    const copyRow     = document.createElement("div");
    copyRow.className = "seq-copy";
    copyRow.title     = "Fill the form from a previous run's resolved config, then press Apply to block";
    const parts       = this.copyControl.build();
    copyRow.append(parts.dir, parts.select, parts.note, parts.apply);
    configHost.appendChild(copyRow);

    const templateRow     = document.createElement("div");
    templateRow.className = "seq-copy";
    templateRow.title     = "Import a stored config template into the form, or save this block's overrides as one, then press Apply to block";
    const templateParts   = this.templateControl.build();
    templateRow.append(templateParts.select, templateParts.note, templateParts.actions);
    configHost.appendChild(templateRow);

    configHost.appendChild(this._buildToolbar(cfg));
    this._renderLayout(configHost, cfg);
    this._blockFinalizeSection(block);
    this.applyRunConfig(block.overrides);
    this._paintProvenance(configHost, block);

    apply.disabled = false;
    apply.addEventListener("click", async () => {
      const steered = this._steerDecisions(block);
      const applied = { ...this.dirty };

      block.overrides = applied;
      if (!(await this._persist())) return;

      const reverted = this._revertedAxes(block, applied, steered);
      this.applyRunConfig(block.overrides);
      this._buildDecisions(decideHost, block, cfg);
      this._paintProvenance(configHost, block);
      this.board.render(this.current, this.live, this.selected, this.linking);

      const count = Object.keys(block.overrides).length;
      const parts = [`Block updated: ${count} override${count === 1 ? "" : "s"}`];
      if (steered.length) parts.push(`pre-verdict value updated for ${steered.join(", ")}`);
      reverted.forEach(({ path, decider }) => parts.push(`${path} reverted: decided by '${decider}'`));
      Toast.show(parts.join(" · "), reverted.length ? "warn" : "ok");
    });
  }

  _steerDecisions(block) {
    const steered = [];

    (block.decisions || []).forEach((decision) => {
      const leaf = this.byPath.get(decision.path);
      if (!leaf) return;

      const edited = this.dirty[decision.path] !== undefined ? this.dirty[decision.path] : String(leaf.value);
      if (edited === decision.waiting) return;

      decision.waiting = edited;
      steered.push(decision.path);
    });

    return steered;
  }

  _revertedAxes(block, applied, steered) {
    const notes    = block.field_notes || {};
    const paths    = new Set([...Object.keys(applied), ...Object.keys(block.overrides)]);
    const reverted = [];

    paths.forEach((path) => {
      if (!notes[path] || steered.includes(path)) return;
      if (applied[path] === block.overrides[path]) return;

      const decider = (this.current.blocks || []).find((other) => other.block_id !== block.block_id && (other.decisions || []).some((decision) => decision.path === path));
      reverted.push({ path, decider: decider ? (decider.name || decider.title) : "an earlier block" });
    });

    return reverted;
  }

  _blockFinalizeSection(block) {
    const note = "The chain needs an entry that writes a sweep: it runs inside the sequence's isolated workspace when there is one and on the entry's own sweep directory otherwise. The enabled steps run in the order listed, without holding back the blocks after this one.";

    this._appendFinalizeSection(block.finalize || {}, note, async (key, next) => {
      const previous = { ...(block.finalize || {}) };
      block.finalize = { ...previous, [key]: next };

      if (!(await this._persist())) {
        block.finalize = previous;
        return false;
      }

      this.board.render(this.current, this.live, this.selected, this.linking);
      return true;
    });
  }

  _paintProvenance(configHost, block) {
    const notes = block.field_notes || {};
    const titles = {
      frozen  : "frozen: this value is the verdict of an earlier block in the sequence",
      waiting : "waiting: a later block decides this field; until then it holds the pre-experiment value",
    };

    this.states.forEach(({ leaf, row }) => {
      Object.keys(titles).forEach((key) => row.classList.remove(`cfg-edit__row--${key}`));
      row.title = `--${leaf.path}`;

      const note = titles[notes[leaf.path]] ? notes[leaf.path] : null;
      if (!note) return;
      row.classList.add(`cfg-edit__row--${note}`);
      row.title = `${titles[note]} · --${leaf.path}`;
    });
  }

  _buildDecisions(host, block, cfg) {
    host.innerHTML = "";
    block.decisions = block.decisions || [];

    const head     = document.createElement("div");
    head.className = "seq-decide__head";
    head.innerHTML =
      `<span class="seq-decide__title">Decided here</span>` +
      `<span class="seq-decide__hint">Config axes this block puts under test. Blocks after this one run the verdict, blocks before it run the pre-verdict value; both are written on save, so a verdict is recorded once instead of edited into every later block. Editing a decided axis in the form below rewrites its pre-verdict value on apply.</span>`;
    host.appendChild(head);

    const list     = document.createElement("div");
    list.className = "seq-decide__list";
    block.decisions.forEach((decision) => list.appendChild(this._decisionRow(block, decision, cfg)));
    if (!block.decisions.length) {
      const empty       = document.createElement("p");
      empty.className   = "cfg-note";
      empty.textContent = "This block decides nothing yet. Add the config path the experiment compares, then record the winner here once the run is read.";
      list.appendChild(empty);
    }
    host.appendChild(list);

    const add     = document.createElement("div");
    add.className = "seq-decide__add";

    const picker       = document.createElement("input");
    picker.className   = "cfg-edit__input seq-decide__picker";
    picker.placeholder = "config path under test, e.g. training.use_ema";
    picker.spellcheck  = false;
    picker.setAttribute("list", "seq-decide-paths");

    const options = document.createElement("datalist");
    options.id    = "seq-decide-paths";
    options.innerHTML = cfg.leaves.filter((leaf) => leaf.editable).map((leaf) => `<option value="${SharedCharts.esc(leaf.path)}"></option>`).join("");

    const button       = document.createElement("button");
    button.type        = "button";
    button.className   = "btn btn--mini";
    button.textContent = "+ add axis";
    button.addEventListener("click", () => this._addDecision(block, picker.value.trim(), cfg));
    picker.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") this._addDecision(block, picker.value.trim(), cfg);
    });

    add.append(picker, options, button);
    host.appendChild(add);
  }

  _decisionRow(block, decision, cfg) {
    const leaf = cfg.leaves.find((item) => item.path === decision.path) || null;
    const row  = document.createElement("div");
    row.className = "seq-decide__item" + (decision.choice ? " is-decided" : "");

    const path       = document.createElement("code");
    path.className   = "seq-decide__path";
    path.textContent = decision.path;
    path.title       = leaf ? `${leaf.type} · default ${leaf.value}` : "this block's entry point does not expose the path; the axis is still frozen into the blocks that do";
    if (!leaf) path.classList.add("is-foreign");
    row.appendChild(path);

    const waiting       = document.createElement("input");
    waiting.className   = "cfg-edit__input seq-decide__waiting";
    waiting.value       = decision.waiting;
    waiting.spellcheck  = false;
    waiting.title       = "Value every block that is not yet after this one runs with while the question is open";
    waiting.addEventListener("change", async () => {
      decision.waiting = waiting.value;
      if (await this._persist()) this._render();
    });
    row.appendChild(this._decisionField("before", waiting));

    const candidates      = document.createElement("input");
    candidates.className  = "cfg-edit__input seq-decide__options";
    candidates.value      = decision.options.join(", ");
    candidates.spellcheck = false;
    candidates.title      = "Comma-separated values the experiment compares; they become the verdict choices. A comma inside brackets, braces or quotes belongs to the value, so a list candidate like [40, 40] stays one candidate";
    candidates.addEventListener("change", async () => {
      decision.options = PythonLiteral.split(candidates.value);
      if (await this._persist()) this._render();
    });
    row.appendChild(this._decisionField("candidates", candidates));

    const verdict     = document.createElement("select");
    verdict.className = "cfg-edit__input picker__select seq-decide__verdict";
    verdict.title     = "Winner of this experiment; picking it freezes the value into every block after this one";
    const values      = decision.options.slice();
    if (decision.choice && !values.includes(decision.choice)) values.unshift(decision.choice);
    verdict.innerHTML = `<option value="">undecided</option>` + values.map((value) => `<option value="${SharedCharts.esc(value)}"${value === decision.choice ? " selected" : ""}>${SharedCharts.esc(value)}</option>`).join("");
    verdict.addEventListener("change", () => this._recordVerdict(block, decision.path, verdict.value));
    row.appendChild(this._decisionField("verdict", verdict));

    const remove       = document.createElement("button");
    remove.type        = "button";
    remove.className   = "btn btn--mini seq-decide__remove";
    remove.textContent = "×";
    remove.title       = "Stop deciding this axis here; the blocks it stamped keep their current values as plain overrides";
    remove.addEventListener("click", async () => {
      block.decisions = block.decisions.filter((item) => item.path !== decision.path);
      if (await this._persist()) this._render();
    });
    row.appendChild(remove);

    return row;
  }

  _decisionField(label, control) {
    const field     = document.createElement("label");
    field.className = "seq-decide__field";
    field.innerHTML = `<span class="seq-decide__name">${SharedCharts.esc(label)}</span>`;
    field.appendChild(control);
    return field;
  }

  async _addDecision(block, path, cfg) {
    if (!path) return;
    if (block.decisions.some((decision) => decision.path === path)) {
      Toast.show(`This block already decides ${path}`, "warn");
      return;
    }

    const leaf = cfg.leaves.find((item) => item.path === path) || null;
    block.decisions.push({
      path    : path,
      options : leaf && leaf.choices ? leaf.choices.slice() : (leaf && leaf.type === "bool" ? ["False", "True"] : []),
      waiting : block.overrides[path] !== undefined ? block.overrides[path] : (leaf ? String(leaf.value) : ""),
      choice  : "",
    });

    if (await this._persist()) this._render();
  }

  async _recordVerdict(block, path, choice) {
    const res = await Api.post(`/api/sequences/${this.current.sequence_id}/decide`, { block_id: block.block_id, path, choice });
    if (!res.ok) {
      Toast.show(res.error || "Could not record the verdict", "error");
      this._render();
      return;
    }

    this.current = res.entry;
    Toast.show(choice ? `${path} frozen to ${choice} in every block after this one` : `${path} reopened; the blocks after this one went back to the pre-verdict value`, "ok");
    this._render();
  }

  _openDecisions(blockId) {
    return (this.current.blocks.find((block) => block.block_id === blockId) || { decisions: [] }).decisions.filter((decision) => !decision.choice);
  }

  _upstreamUndecided(blockId) {
    const byId    = new Map(this.current.blocks.map((block) => [block.block_id, block]));
    const seen    = new Set();
    const pending = [];
    const stack   = [...(byId.get(blockId) || { predecessors: [] }).predecessors];

    while (stack.length) {
      const id = stack.pop();
      if (seen.has(id) || !byId.has(id)) continue;
      seen.add(id);
      this._openDecisions(id).forEach((decision) => pending.push({ block_id: id, label: `${byId.get(id).name || byId.get(id).title}: ${decision.path}` }));
      stack.push(...byId.get(id).predecessors);
    }

    return pending;
  }

  _modelEndpoint(block, cfg) {
    const typeTab = cfg.layout.type_tab || null;
    const tabType = typeTab ? (block.overrides[typeTab.field] !== undefined ? block.overrides[typeTab.field] : (cfg.leaves.find((leaf) => leaf.path === typeTab.field) || {}).value) : null;
    const type    = LaunchView.MODEL_KEY_TYPE[block.script] || tabType || "backbone";

    if (type === "image_autoencoder")   return "/api/image-autoencoders";
    if (type === "profile_autoencoder") return "/api/profile-autoencoders";
    return "/api/backbones";
  }


  _refresh() {
    if (this.countEl) {
      const n = Object.keys(this.dirty).length;
      this.countEl.textContent = n ? `${n} override${n === 1 ? "" : "s"}` : "all defaults";
    }
    if (this.builder) this.builder.refreshFromView();
    this._refreshBadges();
    if (this.layoutEl) this._refreshGates();
  }
}

window.SequenceBoard = SequenceBoard;
window.SequencesView = SequencesView;
