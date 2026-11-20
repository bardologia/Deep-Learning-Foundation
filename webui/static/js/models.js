"use strict";

class ModelGallery {
  constructor(listEl, detailEl) {
    this.listEl = listEl;
    this.detailEl = detailEl;
    this.endpoint = "/api/backbones";
    this.families = [];
    this.flat = [];
    this.activeKey = null;
  }

  async load() {
    const data = await Api.get(this.endpoint);

    if (data.error) {
      this.families = [];
      this.flat = [];
      this.listEl.innerHTML = `<div class="clist__empty">Could not load the model zoo: ${SharedCharts.esc(data.error)}</div>`;
      this.detailEl.innerHTML = `<div class="cdetail__empty"><p>The backend did not answer ${SharedCharts.esc(this.endpoint)}.</p><button class="btn btn--mini" id="models-retry">Retry</button></div>`;
      const retry = document.getElementById("models-retry");
      if (retry) retry.addEventListener("click", () => this.load());
      return;
    }

    this.families = data.families || [];
    this.flat = this.families.flatMap((f) => f.models.map((m) => ({ ...m, family: f.family })));
    this._renderList();
    const initial = this.flat.find((m) => m.recommended) || this.flat[0];
    if (initial) this._select(initial.key);
  }

  _renderList() {
    this.listEl.innerHTML = "";
    this.families.forEach((fam) => {
      const group = document.createElement("div");
      group.className = "mlist__group";
      group.innerHTML = `<div class="mlist__label">${fam.family}</div>`;
      fam.models.forEach((m) => {
        const item = document.createElement("button");
        item.className = "mlist__item";
        item.dataset.key = m.key;
        item.innerHTML =
          `<span class="mlist__name">${m.name}${m.recommended ? '<span class="mlist__star">best</span>' : ""}</span>` +
          `<span class="mlist__params">${m.params}</span>`;
        item.addEventListener("click", () => this._select(m.key));
        group.appendChild(item);
      });
      this.listEl.appendChild(group);
    });
  }

  _select(key) {
    this.activeKey = key;
    this.listEl.querySelectorAll(".mlist__item").forEach((b) => b.classList.toggle("is-active", b.dataset.key === key));
    const model = this.flat.find((m) => m.key === key);
    if (model) this._renderDetail(model);
  }

  _renderDetail(m) {
    this._pops = this._pops || [];
    this._closeFrom(0);

    const built   = window.ModelDiagram.build(m);
    this.blockMap = built.blocks;
    const diagram = built.network;
    const legend  = this._legend(m);

    this.detailEl.innerHTML =
      `<div class="mdetail__head">` +
      `<div><span class="mdetail__family">${m.family}</span>` +
      `<h3 class="mdetail__name">${m.name}${m.recommended ? `<span class="mdetail__best">best in benchmark</span>` : ""}</h3></div>` +
      `<span class="mdetail__params">${m.params} params</span>` +
      `</div>` +
      `<div class="mdetail__diagram">${diagram}</div>` +
      legend +
      `<div class="mdetail__specs">` +
      `<div class="spec"><span class="spec__k">backbone key</span><span class="spec__v mono">${m.key}</span></div>` +
      `<div class="spec"><span class="spec__k">skip mechanism</span><span class="spec__v">${m.skip}</span></div>` +
      `<div class="spec"><span class="spec__k">output head</span><span class="spec__v">${m.head}</span></div>` +
      `</div>` +
      `<div class="mdetail__notes" data-note-host><span class="mdetail__when-label">Architecture notes</span>` +
      `<div class="mdetail__note-body"><span class="mdetail__note-loading">${SharedCharts.waitLine("loading notes…")}</span></div></div>`;

    this.detailEl.classList.remove("is-swap");
    void this.detailEl.offsetWidth;
    this.detailEl.classList.add("is-swap");
    this._wireSchema();
    this._loadNote(m);
  }

  async _loadNote(m) {
    const host = this.detailEl.querySelector("[data-note-host] .mdetail__note-body");
    if (!host) return;
    const data = await Api.get(`${this.endpoint}/${m.key}/note`);
    if (this.activeKey !== m.key) return;

    if (typeof data.markdown !== "string") {
      host.innerHTML = "";
      const span = document.createElement("span");
      span.className   = "mdetail__note-loading";
      span.textContent = data.error || "no vault note available";
      host.appendChild(span);
      return;
    }

    host.innerHTML = this._mdToHtml(data.markdown, data.links || {});
    host.querySelectorAll("[data-model-link]").forEach((el) => {
      el.addEventListener("click", () => this._select(el.getAttribute("data-model-link")));
    });
    this._reorderTables(host);
    this._tightenShortBlocks(host);
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([host]).then(() => this._tightenShortBlocks(host));
    }
  }

  _reorderTables(host) {
    const isBoundary = (el) => /^H[1-6]$/.test(el.tagName) || el.classList.contains("note-hr");
    host.querySelectorAll(".note-tablebox").forEach((box) => {
      let last = null;
      let cur = box.nextElementSibling;
      while (cur && !isBoundary(cur) && !cur.classList.contains("note-tablebox")) {
        last = cur;
        cur = cur.nextElementSibling;
      }
      if (last) last.after(box);
    });
  }

  _tightenShortBlocks(host) {
    const lineCount = (el) => {
      const lineHeight = parseFloat(getComputedStyle(el).lineHeight) || 27;
      let height = 0;
      for (const rect of el.getClientRects()) height += rect.height;
      return height > 0 ? Math.round(height / lineHeight) : 0;
    };
    host.querySelectorAll("p, ul").forEach((el) => {
      if (!el.classList.contains("note-keep") && lineCount(el) <= 3) el.classList.add("note-keep");
    });

    const isSpanner = (el) =>
      el.classList.contains("note-tablebox") ||
      el.classList.contains("note-span") ||
      el.classList.contains("note-hr") ||
      /^H[1-6]$/.test(el.tagName);

    let region = [];
    const flushRegion = () => {
      if (!region.length) return;
      let total = 0;
      for (const el of region) total += lineCount(el);
      if (total > 0 && total <= 8) region.forEach((el) => el.classList.add("note-span"));
      region = [];
    };
    for (const el of host.children) {
      if (isSpanner(el)) flushRegion();
      else region.push(el);
    }
    flushRegion();
  }

  _mdToHtml(md, links) {
    const math = [];
    let src = md.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => {
      math.push(`<div class="note-math">\\[${tex}\\]</div>`);
      return `${math.length - 1}`;
    });
    src = src.replace(/\$([^$\n]+?)\$/g, (_, tex) => {
      math.push(`\\(${tex}\\)`);
      return `${math.length - 1}`;
    });

    const inline = (s) => {
      s = s.replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`);
      s = s.replace(/\[\[([^\]]+)\]\]/g, (_, name) => {
        const key = links[name.trim()];
        return key
          ? `<span class="note-link" data-model-link="${key}">${name}</span>`
          : `<span class="note-ref">${name}</span>`;
      });
      s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      s = s.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
      return s;
    };

    const lines = SharedCharts.esc(src).split("\n");
    const out = [];
    let para = [], list = null, table = null, quote = null;

    const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; } };
    const flushList = () => { if (list) { out.push(`<ul>${list.map((i) => `<li>${inline(i)}</li>`).join("")}</ul>`); list = null; } };
    const flushTable = () => {
      if (!table) return;
      const head = table.shift().map((c) => `<th>${inline(c)}</th>`).join("");
      const body = table.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("");
      out.push(`<div class="note-tablebox"><table class="note-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`);
      table = null;
    };
    const flushQuote = () => { if (quote) { out.push(`<aside class="note-callout">${quote.map((q) => `<p>${inline(q)}</p>`).join("")}</aside>`); quote = null; } };
    const flushAll = () => { flushPara(); flushList(); flushTable(); flushQuote(); };

    let firstHeading = true;
    for (const raw of lines) {
      const line = raw.trimEnd();
      const t = line.trim();

      if (/^\|/.test(t)) {
        flushPara(); flushList(); flushQuote();
        if (/^\|[\s\-|:]+\|$/.test(t)) continue;
        table = table || [];
        table.push(t.replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
        continue;
      }
      flushTable();

      if (/^&gt;/.test(t)) {
        flushPara(); flushList();
        quote = quote || [];
        const q = t.replace(/^&gt;\s?/, "").replace(/^\[!\w+\]\s*/, "");
        if (q) quote.push(q);
        continue;
      }
      flushQuote();

      const h = t.match(/^(#{1,4})\s+(.*)$/);
      if (h) {
        flushAll();
        if (h[1].length === 1 && firstHeading) { firstHeading = false; continue; }
        firstHeading = false;
        const lvl = Math.min(h[1].length + 3, 6);
        out.push(`<h${lvl} class="note-h">${inline(h[2])}</h${lvl}>`);
        continue;
      }
      if (/^---+$/.test(t)) { flushAll(); out.push(`<hr class="note-hr">`); continue; }
      if (/^[-*]\s+/.test(t)) {
        flushPara(); flushQuote();
        list = list || [];
        list.push(t.replace(/^[-*]\s+/, ""));
        continue;
      }
      flushList();
      if (t === "") { flushAll(); continue; }
      para.push(t);
    }
    flushAll();

    return out.join("\n").replace(/(\d+)/g, (_, i) => math[+i]);
  }

  _wireSchema() {
    this._pops = this._pops || [];
    if (!this._docWired) {
      document.addEventListener("click", (e) => {
        if (this._pops.length && !e.target.closest(".dgm-pop") && !e.target.closest("[data-block]")) this._closeFrom(0);
      });
      document.addEventListener("keydown", (e) => { if (e.key === "Escape") this._closeFrom(0); });
      this._docWired = true;
    }
    this.detailEl.querySelectorAll(".mdetail__diagram [data-block]").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        const id = el.getAttribute("data-block");
        const side = id === "enc" ? "left" : id === "dec" ? "right" : "bottom";
        this._openBlock(id, el, 0, side, null);
      });
    });
  }

  _openBlock(id, sourceEl, level, side, parentPop, parentDef) {
    const base = this.blockMap[id];
    if (!base) return;
    let def = parentDef ? { ...base, upward: !!parentDef.upward, horizontal: !!parentDef.horizontal } : base;
    const flowAttr = level === 0 && sourceEl.getAttribute ? sourceEl.getAttribute("data-flow") : null;
    if (flowAttr) {
      const [fi, fo, fup] = flowAttr.split(";");
      const parse = (s) => {
        const sides = [], counts = {};
        (s || "").split(",").filter(Boolean).forEach((t) => {
          const [sd, n] = t.split(":");
          sides.push(sd);
          counts[sd] = +n;
        });
        return { sides, counts };
      };
      const fIn = parse(fi), fOut = parse(fo);
      def = { ...def };
      if (fIn.sides.length) { def.flowIn = fIn.sides; def.flowInN = fIn.counts; }
      if (fOut.sides.length) { def.flowOut = fOut.sides; def.flowOutN = fOut.counts; }
      if (!def.horizontal) def.upward = fup === "1";
    }
    this._closeFrom(level);

    const diagram = this.detailEl.querySelector(".mdetail__diagram");
    const pop = document.createElement("div");
    pop.className = "dgm-pop" + (def.horizontal ? " dgm-pop--wide" : "");
    pop.dataset.level = level;

    const svg = window.ModelDiagram.blockSvg(def);
    const natW = svg.match(/--natw:([\d.]+)px/);
    if (def.horizontal && natW) pop.style.width = Math.min(+natW[1] * 1.1 + 40, 1100) + "px";
    const hasSub = /data-subblock=/.test(svg);
    pop.innerHTML =
      `<div class="dgm-pop__head"><span class="dgm-pop__title">${def.title}</span>` +
      `<button class="dgm-pop__close" aria-label="Close">&times;</button></div>` +
      `<div class="dgm-pop__body">${svg}</div>` +
      (hasSub ? `<div class="dgm-pop__hint">click a highlighted block to expand it</div>` : "");

    diagram.appendChild(pop);
    sourceEl.classList.add("is-sel");
    this._positionPop(pop, parentPop || sourceEl, side, diagram, !!parentPop);
    requestAnimationFrame(() => pop.classList.add("is-in"));
    this._pops.push({ el: pop, sourceEl, level });

    pop.querySelector(".dgm-pop__close").addEventListener("click", (e) => { e.stopPropagation(); this._closeFrom(level); });
    pop.querySelectorAll("[data-subblock]").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        const sid = el.getAttribute("data-subblock");
        if (this.blockMap[sid]) this._openBlock(sid, el, level + 1, side, pop, def);
      });
    });
  }

  _positionPop(pop, anchorEl, side, diagram, alignTop) {
    const cR = diagram.getBoundingClientRect();
    const aR = anchorEl.getBoundingClientRect();
    const nR = document.getElementById("nav").getBoundingClientRect();
    const sidebar = nR.height >= window.innerHeight;
    const minLeft = (sidebar ? nR.right : 0) + 8;
    const minTop  = (sidebar ? 0 : nR.bottom) + 8;
    const pw = pop.offsetWidth, ph = pop.offsetHeight, gap = 16;
    let vpLeft, vpTop, ox, oy;

    if (side === "bottom") {
      vpTop = aR.bottom + gap;
      vpLeft = aR.left + aR.width / 2 - pw / 2;
      ox = "50%"; oy = "0%";
    } else {
      const goLeft = side === "left";
      let l = goLeft ? aR.left - gap - pw : aR.right + gap;
      if (goLeft && l < minLeft) { l = aR.right + gap; ox = "0%"; }
      else if (!goLeft && l + pw > window.innerWidth - 8) { l = aR.left - gap - pw; ox = "100%"; }
      else ox = goLeft ? "100%" : "0%";
      vpLeft = l;
      vpTop = alignTop ? aR.top : aR.top + aR.height / 2 - ph / 2;
      oy = "50%";
    }

    vpLeft = Math.max(minLeft, Math.min(vpLeft, window.innerWidth - pw - 8));
    vpTop = Math.max(minTop, Math.min(vpTop, window.innerHeight - ph - 8));
    pop.style.left = (vpLeft - cR.left) + "px";
    pop.style.top = (vpTop - cR.top) + "px";
    pop.style.setProperty("--ox", ox);
    pop.style.setProperty("--oy", oy);
  }

  _closeFrom(level) {
    if (!this._pops) { this._pops = []; return; }
    this._pops = this._pops.filter((p) => {
      if (p.level >= level) {
        p.el.remove();
        p.sourceEl.classList.remove("is-sel");
        return false;
      }
      return true;
    });
  }

  _legend(m) {
    const s = window.ModelDiagram.spec(m);
    const notes = {
      concat: "Skip features are concatenated into the decoder.",
      residual: "Encoder blocks carry residual connections.",
      resmaxpool: "Residual conv blocks with max-pool downsampling; skips concatenated.",
      attention: "A gate (diamond) filters skip features per region.",
      nested: "Dense intermediate nodes bridge the encoder-decoder gap.",
      additive: "Skips are summed into the decoder, not concatenated.",
      tokens: "Transformer tokens feed the CNN decoder.",
      convnext: "ConvNeXt blocks under the U topology; skips concatenated.",
      dense: "Densely-connected blocks reuse features; skips concatenated.",
      respath: "Each skip is refined through a residual ResPath before concat.",
      rsu: "Each stage is an RSU mini U-Net; skips concatenated.",
      pyramid: "All encoder scales project to an MLP decoder and fuse.",
      lowlevel: "ASPP context fuses with projected low-level detail.",
      branch: "Parallel-resolution branches exchange features by fusion.",
      lateral: "Lateral 1x1 connections add into a top-down pyramid.",
    };
    return (
      `<div class="mdetail__legend">` +
      `<span class="legend__dot"></span>${notes[s.skipKind] || notes.concat}` +
      `<span class="legend__sep">/</span>one output convolution emits every target channel.` +
      `</div>`
    );
  }
}

window.ModelGallery = ModelGallery;
