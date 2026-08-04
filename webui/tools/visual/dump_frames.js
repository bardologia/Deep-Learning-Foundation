"use strict";
const fs   = require("fs");
const os   = require("os");
const path = require("path");
const { ChromiumResolver } = require("./chromium");

class FrameDumper {
  constructor(animFile, key, duration, step, outDir) {
    this.animFile = path.resolve(animFile);
    this.key      = key;
    this.duration = duration;
    this.step     = step;
    this.outDir   = path.resolve(outDir);
    this.toolDir  = __dirname;
  }

  stage() {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "tomosar-frames-"));
    fs.copyFileSync(path.join(this.toolDir, "harness.html"), path.join(dir, "harness.html"));
    fs.copyFileSync(path.join(this.toolDir, "mathjax-tex-svg.js"), path.join(dir, "mathjax-tex-svg.js"));
    fs.copyFileSync(this.animFile, path.join(dir, "process_anim.js"));
    return dir;
  }

  async sweep(page, screenshot) {
    const cv = page.locator("#cv");
    for (let t = 0; t <= this.duration; t += this.step) {
      await page.evaluate(([k, tt]) => window.renderAt(k, tt), [this.key, t]);
      if (screenshot) {
        const name = `${this.key}_t${String(t.toFixed(1)).padStart(6, "0")}.png`;
        await cv.screenshot({ path: path.join(this.outDir, name) });
      }
    }
  }

  async orchestrate() {
    const { chromium } = require("playwright-core");
    fs.mkdirSync(this.outDir, { recursive: true });

    const stageDir = this.stage();
    const browser  = await chromium.launch({ executablePath: new ChromiumResolver().resolve(), args: ["--allow-file-access-from-files"] });
    const page     = await browser.newPage({ viewport: { width: 940, height: 480 } });

    await page.goto("file://" + path.join(stageDir, "harness.html"));
    await page.waitForFunction("typeof window.renderAt === 'function' && window.MathJax && window.MathJax.tex2svg");

    await this.sweep(page, false);
    await page.waitForTimeout(2000);
    await this.sweep(page, false);
    await page.waitForTimeout(500);

    const stats = await page.evaluate(() => window.eqStats());
    console.log(`eq sprites: ${stats.ready}/${stats.total} ready, ${stats.failed} failed`);

    await this.sweep(page, true);
    console.log(`wrote ${Math.floor(this.duration / this.step) + 1} frames to ${this.outDir}`);

    await browser.close();
    fs.rmSync(stageDir, { recursive: true, force: true });
  }
}

const [animFile, key, dur, step, outDir] = process.argv.slice(2);
if (!outDir) {
  console.error("usage: node dump_frames.js <process_anim.js> <scene-key> <duration-s> <step-s> <out-dir>");
  process.exit(1);
}
new FrameDumper(animFile, key, parseFloat(dur), parseFloat(step), outDir).orchestrate().catch(e => { console.error(e); process.exit(1); });
