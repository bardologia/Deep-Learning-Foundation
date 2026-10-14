"use strict";
const fs   = require("fs");
const path = require("path");
const vm   = require("vm");

class StubNode {
  constructor(id) {
    this.id          = id;
    this.innerHTML   = "";
    this.textContent = "";
    this.className   = "";
    this.hidden      = false;
    this.disabled    = false;
    this.value       = "";
    this.style       = {};
    this.state       = undefined;
    this.listeners   = {};
    this.classes     = new Set();
    this.classList   = {
      toggle  : (name, on) => { const add = on === undefined ? !this.classes.has(name) : !!on; if (add) this.classes.add(name); else this.classes.delete(name); return add; },
      add     : (name) => { this.classes.add(name); },
      remove  : (name) => { this.classes.delete(name); },
      contains: (name) => this.classes.has(name),
    };
  }

  addEventListener(type, handler) {
    (this.listeners[type] = this.listeners[type] || []).push(handler);
  }

  async fire(type) {
    await Promise.all((this.listeners[type] || []).map((handler) => handler({ type, target: this })));
  }
}

class ImpactChecks {
  constructor(staticDir) {
    this.staticDir = staticDir;
    this.nodes     = new Map();
    this.toasts    = [];
    this.posts     = [];
    this.answer    = null;
  }

  _node(id) {
    if (!this.nodes.has(id)) this.nodes.set(id, new StubNode(id));
    return this.nodes.get(id);
  }

  _context() {
    const sandbox = {
      console,
      window      : {},
      document    : { getElementById: (id) => this._node(id), visibilityState: "visible" },
      SharedCharts: { esc: (text) => String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;") },
      ThinkingOrb : { busy: (state, text) => `<orb ${state}>${text}</orb>` },
      Toast       : { show: (text, kind) => { this.toasts.push({ text, kind }); } },
      Api         : { post: async (url, body) => { this.posts.push({ url, body }); return this.answer(body); } },
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    const source = fs.readFileSync(path.join(this.staticDir, "home.js"), "utf8");
    vm.runInContext(`${source}\nglobalThis.classes = { UsersTable, StatusBoard };`, sandbox, { filename: "home.js" });
    return sandbox.classes;
  }

  _board() {
    const { StatusBoard } = this._context();
    const board           = new StatusBoard({});
    const meter           = () => ({ value: null, set(v) { this.value = v; } });
    board.impactMeters    = { mem: meter(), io: meter(), cpu: meter() };
    return board;
  }

  _signals() {
    return {
      swap  : { in_mbs: 0, out_mbs: 0 },
      mem   : { used_pct: 61, mine_gb: 12.5, mine_share: 0.62, dominant: true },
      io    : { util: 40, mine_mbs: 12, mine_share: 0.1, dominant: false },
      cpu   : { load_ratio: 0.4, mine_pct: 300, mine_share: 0.3, dominant: false },
      mine  : { nproc: 4, mem_gb: 12.5, stall: { cpu: 2, mem: 8, io: 1 } },
      leak  : null,
      top   : { user: "me", mem_gb: 12.5, nproc: 4, is_mine: true },
      users : [{ uid: 1000, user: "me", is_mine: true }, { uid: 1001, user: "alice", is_mine: false }, { uid: 1002, user: "bob", is_mine: false }],
      others: {
        mem: { stall: 33.4, full: 5, user: "alice", n: 2 },
        io : { stall: 0, full: 0, user: "alice", n: 0 },
        cpu: { stall: 12, full: 0, user: "bob", n: 1 },
      },
    };
  }

  _read(board) {
    const shares = {};
    ["mem", "io", "cpu"].forEach((kind) => {
      const node   = this._node(`sb-impact-${kind}-share`);
      shares[kind] = { html: node.innerHTML, dominant: node.classes.has("is-dominant") };
    });
    return {
      meters : { mem: board.impactMeters.mem.value, io: board.impactMeters.io.value, cpu: board.impactMeters.cpu.value },
      shares,
      stalled: this._node("sb-impact-stalled").innerHTML,
      verdict: this._node("sb-impact-verdict").textContent,
      lede   : this._node("sb-impact-lede").textContent,
      light  : this._node("sb-impact-light").className,
      figure : this._node("sb-impact-balance").state,
    };
  }

  users() {
    const { UsersTable } = this._context();
    const table          = new UsersTable();
    const users          = [
      { user: "me", uid: 1000, me: true, cpu: 10, mem: 1073741824, mem_share: 10, gpu_mem: 0, gpus: [], nproc: 3, sessions: 1 },
      { user: "alice", uid: 1001, me: false, cpu: 120, mem: 2147483648, mem_share: 20, gpu_mem: 0, gpus: [], nproc: 9, sessions: 2 },
      { user: "bob", uid: 1002, me: false, cpu: 30, mem: 1073741824, mem_share: 10, gpu_mem: 0, gpus: [], nproc: 2, sessions: 1 },
      { user: "root", uid: 0, me: false, cpu: 1, mem: 1048576, mem_share: 0.1, gpu_mem: 0, gpus: [], nproc: 40, sessions: 0 },
    ];
    const slices = [
      { uid: 1000, stall: { cpu: 1.2, mem: 0, io: 2.4 } },
      { uid: 1001, stall: { cpu: 3.4, mem: 25.2, io: 0 } },
      { uid: 1002, stall: { cpu: 6, mem: 0.4, io: 1 } },
    ];

    table.render(users, slices);
    const html  = this._node("sb-users").innerHTML;
    const rows  = html.split('<div class="utable__row').slice(1);
    const cells = rows.map((row) => /<span class="utable__stall( is-\w+)?">([^<]*)<\/span>/.exec(row));

    return {
      head   : [...table.markup().matchAll(/<span[^>]*>([^<]*)<\/span>/g)].map((match) => match[1]).slice(2),
      columns: rows.map((row) => (row.match(/<span class="utable__\w+[^"]*">|<span>/g) || []).length),
      stall  : cells.map((match) => ({ text: match[2], level: match[1] ? match[1].trim() : "" })),
    };
  }

  impact() {
    const board  = this._board();
    const ok     = { psi: true, cgroup: true, slice: true, device: "nvme0n1" };
    const blamed = { kind: "mem", level: "warn", mine: true, message: "memory: alice and 1 more stalled" };
    const others = { kind: "cpu", level: "warn", mine: false, message: "CPU: bob waited for a core" };
    const read   = (payload) => { board._renderImpact(payload); return this._read(board); };

    const mine     = read({ auto_nuke: false, sources: ok, signals: this._signals(), active: [blamed], events: [] });
    const missing  = read({ auto_nuke: false, sources: { psi: false, cgroup: false, slice: false, device: null }, signals: {}, active: [], events: [] });
    const outside  = read({ auto_nuke: false, sources: { ...ok, slice: false }, signals: { ...this._signals(), others: {} }, active: [], events: [] });
    const notMine  = read({ auto_nuke: false, sources: ok, signals: { ...this._signals(), top: { user: "alice", mem_gb: 40, nproc: 9, is_mine: false } }, active: [others], events: [] });

    return { mine, missing, outside, notMine };
  }

  async politeness() {
    const board  = this._board();
    const ok     = { psi: true, cgroup: true, slice: true, device: "nvme0n1" };
    const blamed = { kind: "mem", level: "warn", mine: true, message: "memory: alice and 1 more stalled" };
    const state  = { level: "polite", auto_throttle: true, nice: 10, label: "nice 10, ionice best-effort 7", prefix: "nice -n 10 ionice -c 2 -n 7", levels: ["off", "polite", "background"] };
    const select = this._node("sb-impact-politeness");
    const toggle = this._node("sb-impact-throttle");

    board._renderImpact({ auto_nuke: false, politeness: state, throttled: ["mem"], sources: ok, signals: this._signals(), active: [blamed], events: [] });
    const shown = { level: select.value, toggle: toggle.textContent, safe: toggle.classes.has("is-safe"), lede: this._node("sb-impact-lede").textContent };

    const labels = { off: "no nice, no ionice", polite: "nice 10, ionice best-effort 7", background: "nice 19, ionice idle" };
    const server = { ...state };
    board._wireImpactPoliteness();
    this.answer  = (body) => { Object.assign(server, body); return { ok: true, ...server, label: labels[server.level] }; };
    select.value = "background";
    await select.fire("change");
    await toggle.fire("click");

    this.answer = () => ({ ok: false, error: "unknown politeness level 'loud'" });
    select.value = "loud";
    await select.fire("change");

    board._renderImpact({ auto_nuke: false, politeness: { ...state, auto_throttle: false }, throttled: [], sources: ok, signals: this._signals(), active: [blamed], events: [] });
    const after = { toggle: toggle.textContent, safe: toggle.classes.has("is-safe"), lede: this._node("sb-impact-lede").textContent, disabled: [select.disabled, toggle.disabled] };

    return { shown, posts: this.posts, toasts: this.toasts, after };
  }

  async run(name) {
    const result = await this[name]();
    process.stdout.write(JSON.stringify(result));
  }
}

new ImpactChecks(path.resolve(__dirname, "..", "..", "static", "js")).run(process.argv[2]);
