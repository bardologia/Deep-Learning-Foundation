"use strict";
const fs   = require("fs");
const path = require("path");
const vm   = require("vm");

class FlowSketchProbe {

  MODULES = ["flow_sketches.js", "flow_view.js"];

  constructor() {
    this.staticDir = path.resolve(__dirname, "..", "..", "static", "js");
    this.sandbox   = { window: {}, document: {}, console };
    this.sandbox.globalThis = this.sandbox;
  }

  load() {
    vm.createContext(this.sandbox);
    this.MODULES.forEach((name) => {
      vm.runInContext(fs.readFileSync(path.join(this.staticDir, name), "utf8"), this.sandbox, { filename: name });
    });
    return vm.runInContext("FlowSketches.CATALOG", this.sandbox);
  }

  probe(sketch) {
    const svg  = { innerHTML: "", querySelectorAll: () => [svg] };
    const gsap = { to: (els, opts) => ({ els, opts }) };

    sketch.build(svg);
    const tween = sketch.anim ? sketch.anim(svg, gsap) : null;

    return { tip: sketch.tip, markup: svg.innerHTML.length, animated: tween !== null };
  }

  orchestrate() {
    const catalog = this.load();
    const out     = {};
    Object.keys(catalog).forEach((key) => { out[key] = this.probe(catalog[key]); });
    process.stdout.write(JSON.stringify(out));
  }
}

new FlowSketchProbe().orchestrate();
