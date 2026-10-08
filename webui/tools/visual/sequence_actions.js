"use strict";
const fs   = require("fs");
const path = require("path");
const vm   = require("vm");

class SequenceActionProbe {

  MODULES = ["config_form.js", "sequences.js"];

  constructor() {
    this.staticDir = path.resolve(__dirname, "..", "..", "static", "js");
    this.posted    = [];
    this.sandbox   = {
      console,
      setTimeout,
      clearTimeout,
      document : { getElementById: () => null, createElement: () => ({}) },
      window   : { RunConfigCopy: class {}, ConfigTemplates: class {}, confirm: () => true },
      Toast    : { show: () => {} },
      Api      : { post: (url, body) => this.post(url, body) },
    };
    this.sandbox.globalThis = this.sandbox;
  }

  load() {
    vm.createContext(this.sandbox);
    this.MODULES.forEach((name) => {
      vm.runInContext(fs.readFileSync(path.join(this.staticDir, name), "utf8"), this.sandbox, { filename: name });
    });
  }

  post(url, body) {
    this.posted.push(url);
    const reply = { ok: true, entry: JSON.parse(JSON.stringify(body)), state: { status: "running", run_dir: "", blocks: {} } };
    return new Promise((resolve) => setImmediate(() => resolve(reply)));
  }

  view() {
    const view   = new this.sandbox.window.SequencesView(null, {});
    const block  = { block_id: "b1", name: "", title: "probe", script: "train_backbone", predecessors: [], decisions: [], overrides: {}, field_notes: {} };
    view.current = { sequence_id: "probe", blocks: [block] };
    return view;
  }

  async press(action, presses) {
    const view  = this.view();
    const calls = { start: () => view._start(), stop: () => view._stop(), release: () => view._release("b1") };

    this.posted = [];
    const overlapping = await Promise.allSettled(Array.from({ length: presses }, () => calls[action]()));
    const concurrent  = this.count(action);

    this.posted = [];
    const later = await Promise.allSettled([calls[action]()]);
    const after = this.count(action);

    const errors = [...overlapping, ...later].filter((outcome) => outcome.status === "rejected").map((outcome) => String(outcome.reason));
    return { concurrent, after, busy: view.busyAction, errors };
  }

  count(action) {
    return this.posted.filter((url) => url.endsWith(`/${action}`)).length;
  }

  async orchestrate() {
    this.load();

    const out = {};
    for (const action of ["start", "stop", "release"]) out[action] = await this.press(action, 3);
    process.stdout.write(JSON.stringify(out));
  }
}

new SequenceActionProbe().orchestrate();
