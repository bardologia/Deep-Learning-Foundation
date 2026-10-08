"use strict";
const fs   = require("fs");
const path = require("path");
const vm   = require("vm");

class StageChipProbe {

  MODULES = ["process_anim.js", "pipelines.js"];

  constructor(payloadPath) {
    this.staticDir = path.resolve(__dirname, "..", "..", "static", "js");
    this.pipelines = JSON.parse(fs.readFileSync(payloadPath, "utf8"));
    this.stagesBy  = Object.fromEntries(this.pipelines.map((pipeline) => [pipeline.key, pipeline.stages]));
    this.sandbox   = { window: { revealScan: () => {} }, console, document: { getElementById: () => this.element("div"), createElement: (tag) => this.element(tag), addEventListener: () => {} } };
    this.sandbox.globalThis = this.sandbox;
  }

  element(tag) {
    const classes = new Set();
    const el      = {
      tag,
      children    : [],
      style       : {},
      className   : "",
      textContent : "",
      classList   : { toggle: (name, on) => (on ? classes.add(name) : classes.delete(name)), add: (name) => classes.add(name), remove: (name) => classes.delete(name), contains: (name) => classes.has(name) },
      setAttribute       : () => {},
      addEventListener   : (name, handler) => { el[`on${name}`] = handler; },
      appendChild        : (child) => { el.children.push(child); return child; },
      querySelectorAll   : () => [],
      insertAdjacentHTML : () => { el.children.push(this.element("span")); },
      get firstChild() { return el.children[0]; },
      get innerHTML() { return ""; },
      set innerHTML(value) { el.children = []; },
    };
    return el;
  }

  load() {
    vm.createContext(this.sandbox);
    this.MODULES.forEach((name) => {
      vm.runInContext(fs.readFileSync(path.join(this.staticDir, name), "utf8"), this.sandbox, { filename: name });
    });
    return vm.runInContext("PipelineTransport", this.sandbox);
  }

  animator(key) {
    const scene = this.sandbox.window.ProcessAnimator.SCENES[key];
    return {
      sought      : [],
      sceneDef    : () => scene,
      seekChapter : function (i) { this.sought.push(i); },
      chapterIndex: () => 0,
      position    : () => 0,
      duration    : () => scene.T,
      playing     : () => false,
    };
  }

  probe(Transport, key, stages) {
    const animator  = this.animator(key);
    const transport = new Transport(animator);
    transport.bind(stages);

    const clicks = transport.chips.map((chip, stage) => {
      chip.onclick();
      const chapter = animator.sought[animator.sought.length - 1];
      transport._update(0, chapter, false);
      return {
        stage,
        chapter,
        active : transport.chips.findIndex((other) => other.classList.contains("is-active")),
        done   : transport.chips.map((other, j) => (other.classList.contains("is-done") ? j : -1)).filter((j) => j >= 0),
      };
    });

    return { chips: transport.chips.map((chip) => chip.textContent), stageOf: transport.stageOf, clicks };
  }

  rows() {
    const host = this.element("div");
    const flow = new this.sandbox.window.PipelineFlow(host);
    flow.pipelines = this.pipelines;
    flow._render();

    return Object.fromEntries(this.pipelines.map((pipeline, i) => {
      const row = host.children[i];
      return [pipeline.key, { playable: row.className.split(" ").includes("flow-row--play"), clickable: typeof row.onclick === "function" }];
    }));
  }

  orchestrate() {
    const Transport = this.load();
    const scenes    = this.sandbox.window.ProcessAnimator.SCENES;
    const out       = {};

    Object.keys(scenes).filter((key) => this.stagesBy[key]).forEach((key) => { out[key] = this.probe(Transport, key, this.stagesBy[key]); });
    process.stdout.write(JSON.stringify({ scenes: Object.keys(scenes), pipelines: out, rows: this.rows() }));
  }
}

new StageChipProbe(process.argv[2]).orchestrate();
