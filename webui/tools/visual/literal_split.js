const fs   = require("fs");
const path = require("path");
const vm   = require("vm");

const staticDir = path.resolve(__dirname, "..", "..", "static", "js");
const payload   = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));

const sandbox = { window: {}, console };
sandbox.globalThis = sandbox;

vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(staticDir, "python_literal.js"), "utf8"), sandbox, { filename: "python_literal.js" });

const PythonLiteral = sandbox.window.PythonLiteral;

process.stdout.write(JSON.stringify(payload.map((text) => PythonLiteral.split(text))));
