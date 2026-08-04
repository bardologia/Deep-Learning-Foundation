"use strict";
const fs   = require("fs");
const path = require("path");
const { ChromiumResolver } = require("./chromium");

class DiagramRenderer {

  constructor(argv) {
    this.jsonFile = null;
    this.folder   = null;
    this.dkey     = null;
    this.out      = null;
    this.width    = 1680;
    this.settle   = 700;
    this.labels   = false;
    this.parse(argv);
  }

  parse(argv) {
    for (let i = 0; i < argv.length; i++) {
      const a = argv[i];
      if      (a === "--json")   this.jsonFile = path.resolve(argv[++i]);
      else if (a === "--folder") this.folder   = argv[++i];
      else if (a === "--dkey")   this.dkey     = argv[++i];
      else if (a === "--out")    this.out      = path.resolve(argv[++i]);
      else if (a === "--width")  this.width    = Number(argv[++i]);
      else if (a === "--settle") this.settle   = Number(argv[++i]);
      else if (a === "--labels") this.labels   = true;
      else if (a === "--fit")    this.fit      = true;
      else throw new Error("unknown/positional arg: " + a);
    }
    if (!this.jsonFile || !this.out) throw new Error("need --json <file> and --out <png>");
  }

  pick() {
    const data = JSON.parse(fs.readFileSync(this.jsonFile, "utf-8"));
    const folders = data.folders || (Array.isArray(data) ? data : [data]);
    let folderMeta = null, diagram = null;
    for (const f of folders) {
      if (this.folder && f.folder !== this.folder) continue;
      for (const dg of (f.diagrams || [])) {
        if (this.dkey && dg.key !== this.dkey) continue;
        folderMeta = { folder: f.folder, title: f.title, blurb: f.blurb };
        diagram = dg;
        break;
      }
      if (diagram) break;
    }
    if (!diagram) {
      // Allow the json file to be a single bare diagram object.
      if (data.nodes && data.edges) {
        folderMeta = { folder: this.folder || "solo", title: this.folder || "Diagram", blurb: "" };
        diagram = data;
      }
    }
    if (!diagram) throw new Error(`diagram not found (folder=${this.folder} dkey=${this.dkey})`);
    return { folderMeta, diagram };
  }

  async orchestrate() {
    const { chromium } = require("playwright-core");
    fs.mkdirSync(path.dirname(this.out), { recursive: true });
    const { folderMeta, diagram } = this.pick();

    const browser = await chromium.launch({ executablePath: new ChromiumResolver().resolve() });
    const page    = await browser.newPage({ viewport: { width: this.width, height: 1000 } });
    const errs = [];
    page.on("pageerror", (e) => errs.push(String(e)));
    page.on("console", (m) => { if (m.type() === "error") errs.push(m.text()); });

    const harness = "file://" + path.join(__dirname, "repomap_harness.html") + (this.fit ? "?fit=1" : "");
    await page.goto(harness, { waitUntil: "networkidle" });
    if (this.fit) await page.evaluate(() => document.body.classList.add("fit"));
    await page.evaluate(() => { const h = new URL(location.href); if (h.searchParams.get("fit")) document.body.classList.add("fit"); });
    await page.evaluate(async ({ folderMeta, diagram }) => { await window.renderDiagram(folderMeta, diagram); },
                        { folderMeta, diagram });
    await page.waitForSelector(".rm-node", { timeout: 8000 });
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(this.settle);
    if (this.labels) {
      await page.evaluate(() => { const b = document.querySelector(".rm-tool--labels"); if (b) b.click(); });
      await page.waitForTimeout(350);
    }
    // Re-trigger wire draw after settle so SVG matches final node boxes.
    await page.evaluate(() => { if (window.__RM_VIEW__) window.__RM_VIEW__._drawWires(); });
    await page.waitForTimeout(200);

    const target = await page.$(".rm__panel");
    if (target) await target.screenshot({ path: this.out });
    else await page.screenshot({ path: this.out, fullPage: true });

    if (errs.length) console.error("PAGE ERRORS:\n" + errs.join("\n"));
    console.log(this.out);
    await browser.close();
  }
}

new DiagramRenderer(process.argv.slice(2)).orchestrate().catch((err) => {
  console.error(err.message || err);
  process.exit(1);
});
