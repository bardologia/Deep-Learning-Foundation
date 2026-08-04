"use strict";
const fs   = require("fs");
const path = require("path");
const { ChromiumResolver } = require("./chromium");

class RepoMapSnapper {

  constructor(argv) {
    this.port    = 8765;
    this.outDir  = "/tmp/tomosar-repomap";
    this.settle  = 900;
    this.width   = 1680;
    this.height  = 1000;
    this.labels  = false;
    this.targets = [];
    this.parse(argv);
  }

  parse(argv) {
    for (let i = 0; i < argv.length; i++) {
      const a = argv[i];
      if      (a === "--port")   this.port   = Number(argv[++i]);
      else if (a === "--out")    this.outDir = path.resolve(argv[++i]);
      else if (a === "--settle") this.settle = Number(argv[++i]);
      else if (a === "--width")  this.width  = Number(argv[++i]);
      else if (a === "--height") this.height = Number(argv[++i]);
      else if (a === "--labels") this.labels = true;
      else if (a.startsWith("--")) throw new Error("unknown flag: " + a);
      else this.targets.push(a);
    }
  }

  async selectDiagram(page, folder, dkey) {
    await page.evaluate(({ folder, dkey }) => {
      const view = window.__repomap || null;
      const clickBtn = (sel, key) => {
        const els = [...document.querySelectorAll(sel)];
        const b = els.find(e => e.dataset.key === key);
        if (b) b.click();
        return !!b;
      };
      clickBtn(".rm-folder", folder);
      clickBtn(".rm-sub", dkey);
    }, { folder, dkey });
  }

  async snapOne(page, folder, dkey) {
    await this.selectDiagram(page, folder, dkey);
    await page.waitForTimeout(this.settle);

    if (this.labels) {
      await page.evaluate(() => {
        const b = document.querySelector(".rm-tool--labels");
        if (b && !document.querySelector(".rm-canvas").classList.contains("rm-showlabels")) b.click();
      });
      await page.waitForTimeout(300);
    }

    const suffix = this.labels ? "-labels" : "";
    const name = `${folder}__${dkey}${suffix}.png`;
    const file = path.join(this.outDir, name);
    const graph = await page.$(".rm__panel");
    if (graph) await graph.screenshot({ path: file });
    else await page.screenshot({ path: file, fullPage: true });
    console.log(file);
  }

  async orchestrate() {
    const { chromium } = require("playwright-core");
    fs.mkdirSync(this.outDir, { recursive: true });

    const browser = await chromium.launch({ executablePath: new ChromiumResolver().resolve() });
    const page    = await browser.newPage({ viewport: { width: this.width, height: this.height } });

    const base = `http://127.0.0.1:${this.port}/`;
    await page.goto(base, { waitUntil: "domcontentloaded" });
    await page.waitForSelector("#view");
    await page.evaluate(() => { window.location.hash = "#/repomap"; });
    await page.waitForSelector(".rm-folder", { timeout: 10000 });
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(600);

    for (const t of this.targets) {
      const [folder, dkey] = t.split(":");
      await this.snapOne(page, folder, dkey);
    }

    await browser.close();
  }
}

new RepoMapSnapper(process.argv.slice(2)).orchestrate().catch((err) => {
  console.error(err.message || err);
  process.exit(1);
});
