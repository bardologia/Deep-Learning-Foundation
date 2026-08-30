const fs   = require("fs");
const path = require("path");
const vm   = require("vm");

const staticDir = path.resolve(__dirname, "..", "..", "static", "js");
const payload   = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));

const element = () => ({
  className   : "",
  innerHTML   : "",
  textContent : "",
  title       : "",
  hidden      : false,
  classList   : { toggle: () => {}, add: () => {} },
  appendChild : (child) => child,
  addEventListener : () => {},
  setAttribute     : () => {},
  querySelector    : () => element(),
});

const sandbox = {
  document : { createElement: element, activeElement: null },
  window   : {},
  console,
  LaunchWidgetDom : { mini: () => element() },
};
sandbox.globalThis = sandbox;

vm.createContext(sandbox);
["python_literal.js", "experiment_builder.js"].forEach((name) => {
  vm.runInContext(fs.readFileSync(path.join(staticDir, name), "utf8"), sandbox, { filename: name });
});

const ExperimentBuilder = sandbox.window.ExperimentBuilder;

const byPath = new Map();
payload.leaves.forEach((leaf) => byPath.set(leaf.path, leaf));

const view = {
  config   : { leaves: payload.leaves },
  controls : {},
  dirty    : {},
  key      : payload.key,
  _effective : (leaf) => leaf.value,
  _setValue  : () => {},
};

const builder = new ExperimentBuilder(view, byPath);
const model   = builder.modelLeaf ? view._effective(builder.modelLeaf) : "model";

const out = {};
ExperimentBuilder.MODES.filter((mode) => mode.available(builder)).forEach((mode) => {
  out[mode.key] = mode.names ? mode.names(builder, model) : null;
});

console.log(JSON.stringify(out));
