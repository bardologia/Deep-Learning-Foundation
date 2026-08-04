"use strict";
const fs   = require("fs");
const os   = require("os");
const path = require("path");

class ChromiumResolver {

  CACHE_ROOT = path.join(os.homedir(), ".cache", "ms-playwright");
  EXE_RELS   = ["chrome-headless-shell-linux64/chrome-headless-shell", "chrome-linux/chrome"];

  resolve() {
    if (process.env.PW_CHROMIUM) return process.env.PW_CHROMIUM;

    const dirs = fs.readdirSync(this.CACHE_ROOT).filter(d => d.startsWith("chromium")).sort().reverse();
    for (const d of dirs) {
      for (const rel of this.EXE_RELS) {
        const exe = path.join(this.CACHE_ROOT, d, rel);
        if (fs.existsSync(exe)) return exe;
      }
    }
    throw new Error("no chromium found under " + this.CACHE_ROOT + " (set PW_CHROMIUM)");
  }
}

module.exports = { ChromiumResolver };
