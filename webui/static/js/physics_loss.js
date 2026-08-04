"use strict";

class PhysicsLossView {
  constructor(root) {
    this.root    = root;
    this.data    = null;
    this.active  = 0;
    this.mjReady = null;
    this.loaded  = false;
  }

  async load() {
    if (this.loaded) return;
    this.loaded = true;
    this.data   = await window.apiGet("/api/physics-loss");
    if (!this.data || this.data.error) {
      this.loaded = false;
      return;
    }
    this._render();
  }

  _render() {
    this.root.innerHTML = "";
    this.root.appendChild(this._buildHead());
    this.root.appendChild(this._buildOperator());
    this.root.appendChild(this._buildDataset());
    this.root.appendChild(this._buildTermsSection());
    this.root.appendChild(this._buildComparison());
    this.root.appendChild(this._buildConfig());
    this._selectTerm(0);
    window.revealScan();
  }

  _buildHead() {
    const intro = this.data.intro || {};
    const head  = document.createElement("header");
    head.className = "phys-hero";

    const kicker       = document.createElement("p");
    kicker.className   = "phys-hero__kicker";
    kicker.textContent = intro.kicker || "";

    const title       = document.createElement("h2");
    title.className   = "phys-hero__title";
    title.textContent = intro.title || "";

    const lead       = document.createElement("p");
    lead.className   = "phys-hero__lead";
    lead.textContent = intro.lead || "";

    head.appendChild(kicker);
    head.appendChild(title);
    head.appendChild(lead);

    const points     = document.createElement("div");
    points.className = "phys-points";
    (intro.points || []).forEach((p) => {
      const card     = document.createElement("div");
      card.className = "phys-point reveal";
      const label       = document.createElement("h3");
      label.className   = "phys-point__label";
      label.textContent = p.label;
      const text       = document.createElement("p");
      text.className   = "phys-point__text";
      text.textContent = p.text;
      card.appendChild(label);
      card.appendChild(text);
      points.appendChild(card);
    });
    head.appendChild(points);

    return head;
  }

  _buildOperator() {
    const op      = this.data.operator || {};
    const section = document.createElement("section");
    section.className = "phys-block";

    section.appendChild(this._sectionHead("Foundation", op.title, op.blurb));

    const grid     = document.createElement("div");
    grid.className = "phys-op-grid";
    (op.items || []).forEach((item, i) => {
      const card     = document.createElement("div");
      card.className = "phys-op reveal";
      card.style.transitionDelay = `${Math.min(i * 0.05, 0.3)}s`;

      const title       = document.createElement("h4");
      title.className   = "phys-op__title";
      title.textContent = item.title;

      const tex       = document.createElement("div");
      tex.className   = "phys-op__tex";
      this._typeset(tex, item.tex, true);

      const note       = document.createElement("p");
      note.className   = "phys-op__note";
      note.textContent = item.note;

      card.appendChild(title);
      card.appendChild(tex);
      card.appendChild(note);
      card.appendChild(this._varList(item.vars));
      grid.appendChild(card);
    });
    section.appendChild(grid);

    return section;
  }

  _buildDataset() {
    const ds      = this.data.dataset || {};
    const section = document.createElement("section");
    section.className = "phys-block";
    section.appendChild(this._sectionHead("From the dataset", ds.title, ds.blurb));

    const srcLabel       = document.createElement("h4");
    srcLabel.className   = "phys-sub";
    srcLabel.textContent = "Artifacts read";
    section.appendChild(srcLabel);

    const srcGrid     = document.createElement("div");
    srcGrid.className = "phys-src-grid";
    (ds.sources || []).forEach((src) => {
      const card     = document.createElement("div");
      card.className = "phys-src reveal";

      const file       = document.createElement("code");
      file.className   = "phys-src__file";
      file.textContent = src.file;

      const container       = document.createElement("span");
      container.className   = "phys-src__container";
      container.textContent = src.container;

      const role       = document.createElement("p");
      role.className   = "phys-src__role";
      role.textContent = src.role;

      const head     = document.createElement("div");
      head.className = "phys-src__head";
      head.appendChild(file);
      head.appendChild(container);

      card.appendChild(head);
      card.appendChild(role);

      const list     = document.createElement("dl");
      list.className = "phys-src__fields";
      (src.fields || []).forEach((f) => {
        const row     = document.createElement("div");
        row.className = "phys-src__field";

        const name       = document.createElement("dt");
        name.className   = "phys-src__name";
        name.textContent = f.name;

        const shape       = document.createElement("span");
        shape.className   = "phys-src__shape";
        shape.textContent = f.shape;

        const desc       = document.createElement("dd");
        desc.className   = "phys-src__desc";
        desc.textContent = f.desc;

        name.appendChild(shape);
        row.appendChild(name);
        row.appendChild(desc);
        list.appendChild(row);
      });
      card.appendChild(list);
      srcGrid.appendChild(card);
    });
    section.appendChild(srcGrid);

    const pipeLabel       = document.createElement("h4");
    pipeLabel.className   = "phys-sub";
    pipeLabel.textContent = "Construction chain";
    section.appendChild(pipeLabel);

    const pipe     = document.createElement("ol");
    pipe.className = "phys-pipe";
    (ds.pipeline || []).forEach((stp) => {
      const item     = document.createElement("li");
      item.className = "phys-step reveal";

      const no       = document.createElement("span");
      no.className   = "phys-step__no";
      no.textContent = stp.step;

      const body     = document.createElement("div");
      body.className = "phys-step__body";

      const title       = document.createElement("h5");
      title.className   = "phys-step__title";
      title.textContent = stp.title;
      body.appendChild(title);

      if (stp.tex) {
        const tex     = document.createElement("div");
        tex.className = "phys-step__tex";
        this._typeset(tex, stp.tex, true);
        body.appendChild(tex);
      }

      const detail       = document.createElement("p");
      detail.className   = "phys-step__detail";
      detail.textContent = stp.detail;
      body.appendChild(detail);

      const out       = document.createElement("code");
      out.className   = "phys-step__out";
      out.textContent = stp.output;
      body.appendChild(out);

      item.appendChild(no);
      item.appendChild(body);
      pipe.appendChild(item);
    });
    section.appendChild(pipe);

    const example = ds.example || {};
    if ((example.rows || []).length) {
      const exWrap     = document.createElement("div");
      exWrap.className = "phys-example reveal";

      const exHead       = document.createElement("div");
      exHead.className   = "phys-example__head";
      const exTitle       = document.createElement("span");
      exTitle.className   = "phys-example__title";
      exTitle.textContent = example.title;
      const exBlurb       = document.createElement("p");
      exBlurb.className   = "phys-example__blurb";
      exBlurb.textContent = example.blurb;
      exHead.appendChild(exTitle);
      exHead.appendChild(exBlurb);
      exWrap.appendChild(exHead);

      const grid     = document.createElement("dl");
      grid.className = "phys-example__grid";
      example.rows.forEach((r) => {
        const row     = document.createElement("div");
        row.className = "phys-example__row";
        const k       = document.createElement("dt");
        k.className   = "phys-example__k";
        k.textContent = r.k;
        const v       = document.createElement("dd");
        v.className   = "phys-example__v";
        v.textContent = r.v;
        row.appendChild(k);
        row.appendChild(v);
        grid.appendChild(row);
      });
      exWrap.appendChild(grid);
      section.appendChild(exWrap);
    }

    return section;
  }

  _buildTermsSection() {
    const section = document.createElement("section");
    section.className = "phys-block";
    section.appendChild(this._sectionHead("The five terms", "One physical quantity each", "Select a term to read what it matches, why it behaves the way it does, and what it costs. Every term reuses the forward operator above and defaults off."));

    const pills     = document.createElement("div");
    pills.className = "phys-pills";
    this.pillEls    = [];
    (this.data.terms || []).forEach((term, i) => {
      const pill     = document.createElement("button");
      pill.className = "phys-pill";
      pill.innerHTML = `<span class="phys-pill__no">${String(term.index).padStart(2, "0")}</span>${this._esc(term.name)}`;
      pill.addEventListener("click", () => this._selectTerm(i));
      this.pillEls.push(pill);
      pills.appendChild(pill);
    });
    section.appendChild(pills);

    this.detailEl           = document.createElement("div");
    this.detailEl.className = "phys-detail";
    section.appendChild(this.detailEl);

    return section;
  }

  _selectTerm(index) {
    this.active = index;
    this.pillEls.forEach((p, i) => p.classList.toggle("is-active", i === index));
    this._renderTerm(this.data.terms[index]);
  }

  _renderTerm(term) {
    if (!term) return;
    this.detailEl.innerHTML = "";

    const card     = document.createElement("article");
    card.className = `phys-card phys-card--${term.role}`;

    const head     = document.createElement("div");
    head.className = "phys-card__head";

    const badge       = document.createElement("span");
    badge.className   = "phys-card__no";
    badge.textContent = String(term.index).padStart(2, "0");

    const titles       = document.createElement("div");
    titles.className   = "phys-card__titles";
    const name         = document.createElement("h3");
    name.className     = "phys-card__name";
    name.textContent   = term.name;
    const tagline       = document.createElement("p");
    tagline.className   = "phys-card__tagline";
    tagline.textContent = term.tagline;
    titles.appendChild(name);
    titles.appendChild(tagline);

    const role       = document.createElement("span");
    role.className   = `phys-role phys-role--${term.role}`;
    role.textContent = term.role_label;

    head.appendChild(badge);
    head.appendChild(titles);
    head.appendChild(role);
    card.appendChild(head);

    card.appendChild(this._metaRow(term));

    const tex       = document.createElement("div");
    tex.className   = "phys-card__tex";
    this._typeset(tex, term.tex, true);
    card.appendChild(tex);

    const story       = document.createElement("p");
    story.className   = "phys-card__story";
    story.textContent = term.story;
    card.appendChild(story);

    const caveat     = document.createElement("div");
    caveat.className = "phys-caveat";
    const caveatText       = document.createElement("p");
    caveatText.textContent = term.caveat;
    caveat.appendChild(caveatText);
    card.appendChild(caveat);

    card.appendChild(this._varList(term.vars));

    const code       = document.createElement("code");
    code.className   = "phys-card__code";
    code.textContent = term.code;
    card.appendChild(code);

    this.detailEl.appendChild(card);
    window.revealScan();
  }

  _metaRow(term) {
    const row     = document.createElement("div");
    row.className = "phys-meta";

    row.appendChild(this._metaItem("Matches", term.quantity));
    row.appendChild(this._metaItem("Invariant to", term.invariant));

    const cost       = document.createElement("div");
    cost.className   = "phys-meta__item";
    const label       = document.createElement("span");
    label.className   = "phys-meta__label";
    label.textContent = "Compute cost";
    const meter       = document.createElement("div");
    meter.className   = "phys-cost";
    for (let i = 1; i <= 5; i += 1) {
      const dot     = document.createElement("i");
      dot.className = "phys-cost__dot" + (i <= term.cost ? " is-on" : "");
      meter.appendChild(dot);
    }
    const costName       = document.createElement("span");
    costName.className   = "phys-cost__label";
    costName.textContent = term.cost_label;
    const costWrap       = document.createElement("div");
    costWrap.className   = "phys-cost__wrap";
    costWrap.appendChild(meter);
    costWrap.appendChild(costName);
    cost.appendChild(label);
    cost.appendChild(costWrap);
    row.appendChild(cost);

    return row;
  }

  _metaItem(label, value) {
    const item     = document.createElement("div");
    item.className = "phys-meta__item";
    const l       = document.createElement("span");
    l.className   = "phys-meta__label";
    l.textContent = label;
    const v       = document.createElement("span");
    v.className   = "phys-meta__value";
    v.textContent = value;
    item.appendChild(l);
    item.appendChild(v);
    return item;
  }

  _buildComparison() {
    const cmp     = this.data.comparison || {};
    const section = document.createElement("section");
    section.className = "phys-block";
    section.appendChild(this._sectionHead("At a glance", cmp.title, cmp.blurb));

    const wrap     = document.createElement("div");
    wrap.className = "phys-table-wrap reveal";
    const table     = document.createElement("table");
    table.className = "phys-table";

    const thead = document.createElement("thead");
    const htr   = document.createElement("tr");
    (cmp.columns || []).forEach((c) => {
      const th       = document.createElement("th");
      th.textContent = c;
      htr.appendChild(th);
    });
    thead.appendChild(htr);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    (cmp.rows || []).forEach((r) => {
      const tr = document.createElement("tr");
      [r.term, r.quantity, r.invariant, r.cost, r.role].forEach((value, i) => {
        const cell       = document.createElement(i === 0 ? "th" : "td");
        cell.textContent = value;
        if (i === 2) cell.className = "phys-inv phys-inv--" + value.toLowerCase().replace(/[^a-z]/g, "");
        tr.appendChild(cell);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    section.appendChild(wrap);

    return section;
  }

  _buildConfig() {
    const cfg     = this.data.config || {};
    const section = document.createElement("section");
    section.className = "phys-block";
    section.appendChild(this._sectionHead("Wiring", cfg.title, cfg.blurb));

    const grid     = document.createElement("div");
    grid.className = "phys-cfg-grid";
    (cfg.groups || []).forEach((group) => {
      const panel     = document.createElement("div");
      panel.className = "phys-cfg reveal";
      const name       = document.createElement("h4");
      name.className   = "phys-cfg__name";
      name.textContent = group.name;
      panel.appendChild(name);

      const list     = document.createElement("dl");
      list.className = "phys-cfg__list";
      (group.fields || []).forEach((f) => {
        const row     = document.createElement("div");
        row.className = "phys-cfg__row";
        const field       = document.createElement("dt");
        field.className   = "phys-cfg__field";
        field.textContent = f.field;
        const def       = document.createElement("span");
        def.className   = "phys-cfg__default";
        def.textContent = f.default;
        const meaning       = document.createElement("dd");
        meaning.className   = "phys-cfg__meaning";
        meaning.textContent = f.meaning;
        field.appendChild(def);
        row.appendChild(field);
        row.appendChild(meaning);
        list.appendChild(row);
      });
      panel.appendChild(list);
      grid.appendChild(panel);
    });
    section.appendChild(grid);

    if ((cfg.cli || []).length) {
      const cli     = document.createElement("div");
      cli.className = "phys-cli reveal";
      const cliHead       = document.createElement("span");
      cliHead.className   = "phys-cli__head";
      cliHead.textContent = "Single-run overrides";
      cli.appendChild(cliHead);
      cfg.cli.forEach((line) => {
        const pre       = document.createElement("pre");
        pre.className   = "phys-cli__line";
        pre.textContent = line;
        cli.appendChild(pre);
      });
      section.appendChild(cli);
    }

    return section;
  }

  _sectionHead(kicker, title, blurb) {
    const head     = document.createElement("div");
    head.className = "phys-shead";
    const k       = document.createElement("span");
    k.className   = "phys-shead__kicker";
    k.textContent = kicker;
    const t       = document.createElement("h3");
    t.className   = "phys-shead__title";
    t.textContent = title || "";
    const b       = document.createElement("p");
    b.className   = "phys-shead__blurb";
    b.textContent = blurb || "";
    head.appendChild(k);
    head.appendChild(t);
    head.appendChild(b);
    return head;
  }

  _varList(vars) {
    const list     = document.createElement("div");
    list.className = "phys-vars";
    (vars || []).forEach((v) => {
      const row     = document.createElement("div");
      row.className = "phys-var";
      const sym     = document.createElement("span");
      sym.className = "phys-var__sym";
      this._typeset(sym, v.sym, false);
      const desc       = document.createElement("span");
      desc.className   = "phys-var__desc";
      desc.textContent = v.desc;
      row.appendChild(sym);
      row.appendChild(desc);
      list.appendChild(row);
    });
    return list;
  }

  _whenMathJax() {
    if (this.mjReady) return this.mjReady;
    this.mjReady = new Promise((resolve) => {
      const check = () => {
        if (window.MathJax && window.MathJax.tex2svgPromise) resolve();
        else setTimeout(check, 120);
      };
      check();
    }).then(() => {
      if (!document.getElementById("MJX-SVG-styles")) {
        document.head.appendChild(window.MathJax.svgStylesheet());
      }
    });
    return this.mjReady;
  }

  _typeset(el, tex, display) {
    el.textContent = tex;
    this._whenMathJax().then(() => {
      return window.MathJax.tex2svgPromise(tex, { display: !!display });
    }).then((node) => {
      el.textContent = "";
      el.appendChild(node);
      if (display) {
        const svg = node.querySelector("svg");
        if (svg) { svg.style.maxWidth = "100%"; svg.style.height = "auto"; }
      }
    }).catch(() => {});
  }

  _esc(s) {
    return String(s == null ? "" : s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  }
}

window.PhysicsLossView = PhysicsLossView;
