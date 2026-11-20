"use strict";

class RevealObserver {
  constructor() {
    this.io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("is-in");
            this.io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.1, rootMargin: "0px 0px -6% 0px" }
    );
  }

  scan() {
    document.querySelectorAll(".reveal:not(.is-in)").forEach((el) => this.io.observe(el));
  }
}

class App {
  constructor() {
    this.project = null;
    this.scenes = [];
    this.reveal = new RevealObserver();
    window.revealScan = () => this.reveal.scan();
  }

  async init() {
    this._initScenes();
    this._initNav();
    await this._loadProject();
    this._initComponents();
    this._initRouter();
    this.reveal.scan();
  }

  _initNav() {
    const nav = document.getElementById("nav");
    if (!nav) return;
    const edge = document.querySelector(".nav-edge");
    nav.querySelectorAll("a[data-route]").forEach((a) => {
      a.addEventListener("click", () => {
        a.blur();
        nav.classList.add("is-dismissed");
      });
    });
    nav.querySelectorAll(".nav__grouphead").forEach((head) => {
      head.addEventListener("click", () => {
        const group = head.closest(".nav__group");
        const open  = group.classList.contains("is-open");
        nav.querySelectorAll(".nav__group").forEach((g) => g.classList.toggle("is-open", g === group && !open));
      });
    });
    if (edge) edge.addEventListener("mouseenter", () => nav.classList.remove("is-dismissed"));
  }

  _initScenes() {
    const radar = document.getElementById("radar");

    if (radar) this.scenes.push(new window.TrainingScene(radar));
    for (const mast of document.querySelectorAll(".mast--flanked")) this.scenes.push(new window.MastFigures(mast));
  }

  async _loadProject() {
    const project = await Api.get("/api/project");

    if (project.error) {
      this.project = { interpreters: [], counts: {} };
      this._setStatus(false);
      this._retryProject();
      return;
    }

    this.project = project;
    window.ResultsSources.configure(project.roots);
    this._setStatus(true);
    this._initStatus();
  }

  _retryProject() {
    setTimeout(async () => {
      const project = await Api.get("/api/project");
      if (project.error) {
        this._retryProject();
        return;
      }

      this.project = project;
      window.ResultsSources.configure(project.roots);
      if (this.launchView) this.launchView.project = project;
      this._setStatus(true);
      this._initStatus();
    }, 3000);
  }

  _setStatus(ok) {
    const wrap = document.getElementById("nav-status");
    const text = document.getElementById("status-text");
    wrap.classList.toggle("is-ok", ok);
    wrap.classList.toggle("is-down", !ok);
    text.textContent = ok ? "backend live" : "offline";
  }

  _initStatus() {
    const footRoot = document.getElementById("footer-root");
    if (footRoot) footRoot.textContent = this.project.repo_root || "";

    this.statusBoard = new window.StatusBoard({
      host: document.getElementById("status-host"),
      sum: document.getElementById("status-sum"),
      board: document.getElementById("status-board"),
    });
    this.statusBoard.start();
  }

  _initComponents() {
    this.runConsole = new window.RunConsole({
      list: document.getElementById("job-list"),
      tiles: document.getElementById("console-tiles"),
      hint: document.getElementById("console-hint"),
    });
    this.runConsole.refresh();

    this.scriptPanel = new window.ScriptPanel({
      grid: document.getElementById("script-grid"),
      filters: document.getElementById("script-filters"),
    });
    this.scriptPanel.load();
    window.scriptPanel = this.scriptPanel;

    this.launchView = new window.LaunchView(this.runConsole, this.project || {});
    this.runConsole.launchView = this.launchView;

    this.savedRunsView = new window.SavedRunsView(this.runConsole, document.getElementById("saved-grid"));

    this.templatesView = new window.TemplatesView(this.launchView, document.getElementById("templates-grid"));

    this.sequencesView = new window.SequencesView(this.runConsole, this.project || {});

    this.modelGallery = new window.ModelGallery(
      document.getElementById("model-list"),
      document.getElementById("model-detail")
    );
    this.modelGallery.load();

    this.configBrowser = new window.ConfigBrowser(
      document.getElementById("config-list"),
      document.getElementById("config-detail"),
      document.getElementById("config-search")
    );
    this.configBrowser.load();

    this.tensorboardView = new window.TensorboardView();

    this.resultsView = new window.ResultsView(
      document.getElementById("results-list"),
      document.getElementById("results-detail")
    );

    this.microscopeView    = new window.MicroscopeView(document.getElementById("probe-root"));
    this.surveyView        = new window.SurveyView(document.getElementById("survey-root"));
    this.autopsyView       = new window.AutopsyView(document.getElementById("autopsy-root"));

    this.figureDock = new window.FigureDock();
    window.figureDock = this.figureDock;
  }

  _figureTabs() {
    return {
      microscope  : this.microscopeView,
      survey      : this.surveyView,
      autopsy     : this.autopsyView,
      results     : this.resultsView,
    };
  }

  _mountFigures(route) {
    const view = this._figureTabs()[route];
    const page = document.querySelector(`.page[data-page="${route}"]`);

    if (!view || !page) {
      this.figureDock.unmount();
      return;
    }

    this.figureDock.mount(route, page, () => view.figureTag());
  }

  _initRouter() {
    this.router = new window.Router((route, param) => this._onRoute(route, param));
    window.router = this.router;
    this.router.start();
  }

  _onRoute(route, param) {
    if (route === "home") {
      requestAnimationFrame(() => this.scenes.forEach((s) => s.resize && s.resize()));
    }
    if (route === "launch") {
      if (param) this.launchView.show(param);
      else this.router.go("scripts");
    } else {
      this.launchView.leave();
    }
    if (route === "saved") this.savedRunsView.enter();
    if (route === "templates") this.templatesView.enter();
    if (route === "sequences") this.sequencesView.enter();
    else this.sequencesView.leave();
    if (route === "tensorboard") this.tensorboardView.enter();
    else this.tensorboardView.leave();
    if (route === "results") this.resultsView.enter();
    if (route === "microscope") this.microscopeView.enter();
    if (route === "survey") this.surveyView.enter();
    if (route === "autopsy") this.autopsyView.enter();
    if (route === "console") this.runConsole.onShow();
    this._mountFigures(route);
    setTimeout(() => this.reveal.scan(), 60);
  }
}

document.addEventListener("DOMContentLoaded", () => new App().init());
