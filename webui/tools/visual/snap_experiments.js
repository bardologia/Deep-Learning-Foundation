"use strict";
const fs   = require("fs");
const path = require("path");
const { ChromiumResolver } = require("./chromium");

class ExperimentSnapper {

  constructor(argv) {
    this.port   = 8772;
    this.outDir = "/tmp/ml-template-experiments";
    this.settle = 1200;
    this.width  = 1600;
    this.height = 1100;
    this.routes = [];
    this.parse(argv);
  }

  parse(argv) {
    for (let i = 0; i < argv.length; i++) {
      const a = argv[i];
      if      (a === "--port")   this.port   = Number(argv[++i]);
      else if (a === "--out")    this.outDir = path.resolve(argv[++i]);
      else if (a === "--settle") this.settle = Number(argv[++i]);
      else if (a.startsWith("--")) throw new Error("unknown flag: " + a);
      else this.routes.push(a);
    }
  }

  async openRoute(page, route) {
    await page.evaluate((r) => { window.location.hash = "#/" + r; }, route);
    await page.waitForSelector(`.page[data-page="${route.split("/")[0]}"].is-active`);
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(this.settle);
  }

  async openExperiments(page) {
    return page.evaluate(() => {
      const tabs = Array.from(document.querySelectorAll("button, a"));
      const tab  = tabs.find((el) => (el.textContent || "").trim().toLowerCase() === "experiments");
      if (!tab) return false;
      tab.click();
      return true;
    });
  }

  async selectMode(page, label) {
    return page.evaluate((wanted) => {
      const buttons = Array.from(document.querySelectorAll(".exp-picker__btn"));
      const button  = buttons.find((el) => (el.textContent || "").trim() === wanted);
      if (!button) return false;
      button.click();
      return true;
    }, label);
  }

  async report(page) {
    return page.evaluate(() => {
      const summary = document.querySelector(".exp-builder__summary");
      const names   = document.querySelector(".exp-builder__preview .exp-builder__names");
      const visible = Array.from(document.querySelectorAll(".exp-builder__body > div")).filter((el) => !el.hidden).length;
      return {
        summary : summary ? summary.textContent.trim() : null,
        names   : names ? names.textContent.trim().slice(0, 220) : null,
        panels  : visible,
      };
    });
  }

  async orchestrate() {
    fs.mkdirSync(this.outDir, { recursive: true });

    const { chromium } = require("playwright-core");
    const browser      = await chromium.launch({ executablePath: new ChromiumResolver().resolve() });
    const page         = await browser.newPage({ viewport: { width: this.width, height: this.height } });

    const errors = [];
    page.on("pageerror", (error) => errors.push(String(error)));
    page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); });

    await page.goto(`http://127.0.0.1:${this.port}/`, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(this.settle);

    const findings = {};

    for (const route of this.routes) {
      await this.openRoute(page, route);

      const opened = await this.openExperiments(page);
      await page.waitForTimeout(this.settle);

      const labels  = await page.evaluate(() => Array.from(document.querySelectorAll(".exp-picker__btn")).map((el) => el.textContent.trim()));
      const perMode = {};

      for (const label of labels) {
        await this.selectMode(page, label);
        await page.waitForTimeout(250);
        perMode[label] = await this.report(page);

        const name = `${route.replace(/\//g, "_")}__${label.replace(/[^a-z0-9]+/gi, "-")}.png`;
        await page.screenshot({ path: path.join(this.outDir, name), fullPage: true });
      }

      findings[route] = { experimentsTab: opened, modes: perMode };
    }

    await browser.close();

    console.log(JSON.stringify({ findings, errors }, null, 2));
  }
}

new ExperimentSnapper(process.argv.slice(2)).orchestrate().catch((error) => {
  console.error(error);
  process.exit(1);
});
