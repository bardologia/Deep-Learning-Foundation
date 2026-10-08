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
    const orb   = document.getElementById("backdrop-orb");
    const radar = document.getElementById("radar");

    if (orb) this.scenes.push(new window.BackdropOrb(orb));
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
      if (this.ablationView) this.ablationView.project = project;
      this._setStatus(true);
      this._initStatus();
    }, 3000);
  }

  _setStatus(ok) {
    const wrap = document.getElementById("nav-status");
    const text = document.getElementById("status-text");
    wrap.classList.toggle("is-ok", ok);
    wrap.classList.toggle("is-down", !ok);
    wrap.querySelector("thinking-orb").setAttribute("state", ok ? "listening" : "connecting");
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

    this.ablationView = new window.AblationView(this.runConsole, this.project || {});

    this.savedRunsView = new window.SavedRunsView(this.runConsole, document.getElementById("saved-grid"));

    this.templatesView = new window.TemplatesView(this.launchView, document.getElementById("templates-grid"));

    this.sequencesView = new window.SequencesView(this.runConsole, this.project || {});

    this.equationView = new window.EquationView(
      document.getElementById("eq-tabs"),
      document.getElementById("eq-grid")
    );
    this.equationView.load();

    this.flowView = new window.FlowView(document.getElementById("flowx"));
    this._initModelMode();


    this.pipelineFlow = new window.PipelineFlow(document.getElementById("flow"));
    this.pipelineFlow.load();

    this.repoMap = new window.RepoMapView(document.getElementById("repomap"));

    this.modelGallery = new window.ModelGallery(
      document.getElementById("model-list"),
      document.getElementById("model-detail")
    );
    this.modelGallery.load();
    this._initModelZoo();

    this.configBrowser = new window.ConfigBrowser(
      document.getElementById("config-list"),
      document.getElementById("config-detail"),
      document.getElementById("config-search")
    );
    this.configBrowser.load();

    this.tensorboardView = new window.TensorboardView();

    this.feedTuner = new window.FeedTuner();
    window.feedTuner = this.feedTuner;

    this.resultsView = new window.ResultsView(
      document.getElementById("results-list"),
      document.getElementById("results-detail")
    );

    this.leaderboardView = new window.LeaderboardView(document.getElementById("leaderboard-root"));

    this.microscopeView    = new window.MicroscopeView(document.getElementById("probe-root"));
    this.surveyView        = new window.SurveyView(document.getElementById("survey-root"));
    this.triageView        = new window.TriageView(document.getElementById("triage-root"));
    this.autopsyView       = new window.AutopsyView(document.getElementById("autopsy-root"));

    this.fitLabView = new window.FitLabView({
      baseInput        : document.getElementById("fl-base"),
      scanBtn          : document.getElementById("fl-scan"),
      strip            : document.getElementById("fl-strip"),
      hint             : document.getElementById("fl-hint"),
      progress         : document.getElementById("fl-progress"),
      progressFill     : document.getElementById("fl-progress-fill"),
      progressLabel    : document.getElementById("fl-progress-label"),
      stage            : document.getElementById("fl-stage"),
      mapSrcWrap       : document.getElementById("fl-map-src"),
      mapImg           : document.getElementById("fl-map-img"),
      marks            : document.getElementById("fl-marks"),
      coords           : document.getElementById("fl-coords"),
      rowInput         : document.getElementById("fl-row"),
      colInput         : document.getElementById("fl-col"),
      rowRange         : document.getElementById("fl-row-range"),
      colRange         : document.getElementById("fl-col-range"),
      addBtn           : document.getElementById("fl-add"),
      clearBtn         : document.getElementById("fl-clear"),
      pixels           : document.getElementById("fl-pixels"),
      modeWrap         : document.getElementById("fl-mode"),
      kmaxInput        : document.getElementById("fl-kmax"),
      lambdaInput      : document.getElementById("fl-lambda"),
      thresholdInput   : document.getElementById("fl-threshold"),
      truncationInput  : document.getElementById("fl-truncation"),
      prominenceInput  : document.getElementById("fl-prominence"),
      sigdivInput      : document.getElementById("fl-sigdiv"),
      activityInput    : document.getElementById("fl-activity"),
      stepsInput       : document.getElementById("fl-steps"),
      lrInput          : document.getElementById("fl-lr"),
      runBtn           : document.getElementById("fl-run"),
      runMsg           : document.getElementById("fl-run-msg"),
      fitProgress      : document.getElementById("fl-fit-progress"),
      fitProgressFill  : document.getElementById("fl-fit-progress-fill"),
      fitProgressLabel : document.getElementById("fl-fit-progress-label"),
      runs             : document.getElementById("fl-runs"),
      results          : document.getElementById("fl-results"),
      compsWrap        : document.getElementById("fl-comps"),
      cards            : document.getElementById("fl-cards"),
    });

    this.cubeView = new window.CubeView({
      strip         : document.getElementById("cube-strip"),
      stage         : document.getElementById("cube-stage"),
      deck          : document.getElementById("cube-deck"),
      topdown       : document.getElementById("cube-topdown"),
      cross         : document.getElementById("cube-cross"),
      coords        : document.getElementById("cube-coords"),
      back          : document.getElementById("cube-back"),
      hint          : document.getElementById("cube-hint"),
      slices        : document.getElementById("cube-slices"),
      slicesAt      : document.getElementById("cube-slices-at"),
      sources       : document.getElementById("cube-sources"),
      profiles      : document.getElementById("cube-profiles"),
      profAt        : document.getElementById("cube-prof-at"),
      profMetrics   : document.getElementById("cube-prof-metrics"),
      profMixWrap   : document.getElementById("cube-prof-mix-wrap"),
      profMix       : document.getElementById("cube-prof-mix"),
      progress      : document.getElementById("cube-progress"),
      progressFill  : document.getElementById("cube-progress-fill"),
      progressLabel : document.getElementById("cube-progress-label"),
      spaceBtns     : [...document.querySelectorAll(".cube-space[data-space]")],
      profModeBtns  : [...document.querySelectorAll("#cube-prof-mode .cube-space")],
      modeBtns      : [...document.querySelectorAll(".cube-mode[data-view]")],
      views         : [...document.querySelectorAll(".cube-view[data-view]")],
      params        : {
        source : document.getElementById("cube-param-source"),
        field  : document.getElementById("cube-param-field"),
        slot   : document.getElementById("cube-param-slot"),
        at     : document.getElementById("cube-param-at"),
        canvas : document.getElementById("cube-param-canvas"),
        cross  : document.getElementById("cube-param-cross"),
        cbar   : document.getElementById("cube-param-cbar"),
        min    : document.getElementById("cube-param-min"),
        max    : document.getElementById("cube-param-max"),
        coords : document.getElementById("cube-param-coords"),
        table  : document.getElementById("cube-param-table"),
        open   : document.getElementById("cube-param-open"),
      },
      metrics       : {
        layer  : document.getElementById("cube-metric-layer"),
        rangeWrap : document.getElementById("cube-metric-range-wrap"),
        classes   : document.getElementById("cube-metric-classes"),
        hist      : document.getElementById("cube-metric-hist"),
        histNote  : document.getElementById("cube-metric-hist-note"),
        download  : document.getElementById("cube-metric-download"),
        vmin   : document.getElementById("cube-metric-vmin"),
        vmax   : document.getElementById("cube-metric-vmax"),
        reset  : document.getElementById("cube-metric-reset"),
        mode   : document.getElementById("cube-metric-mode"),
        thr    : document.getElementById("cube-metric-thr"),
        thrVal : document.getElementById("cube-metric-thr-val"),
        alpha  : document.getElementById("cube-metric-alpha"),
        img    : document.getElementById("cube-metric-img"),
        cross  : document.getElementById("cube-metric-cross"),
        cbar   : document.getElementById("cube-metric-cbar"),
        min    : document.getElementById("cube-metric-min"),
        max    : document.getElementById("cube-metric-max"),
        coords : document.getElementById("cube-metric-coords"),
        readout: document.getElementById("cube-metric-readout"),
        open   : document.getElementById("cube-metric-open"),
        selCov : document.getElementById("cube-selective-cov"),
        selVal : document.getElementById("cube-selective-val"),
        selMask : document.getElementById("cube-selective-mask"),
        selPlay : document.getElementById("cube-selective-play"),
        selTab : document.getElementById("cube-selective-table"),
        rect       : document.getElementById("cube-metric-rect"),
        regionHint : document.getElementById("cube-region-hint"),
        regionTab  : document.getElementById("cube-region-table"),
        regionWrap : document.getElementById("cube-region-prof-wrap"),
        regionProf : document.getElementById("cube-region-prof"),
      },
      transect      : {
        at      : document.getElementById("cube-transect-at"),
        undo    : document.getElementById("cube-transect-undo"),
        clear   : document.getElementById("cube-transect-clear"),
        print   : document.getElementById("cube-transect-print"),
        download: document.getElementById("cube-transect-download"),
        name       : document.getElementById("cube-transect-name"),
        save       : document.getElementById("cube-transect-save"),
        saved      : document.getElementById("cube-transect-saved"),
        remove     : document.getElementById("cube-transect-delete"),
        stripWrap  : document.getElementById("cube-transect-strip-wrap"),
        stripLayer : document.getElementById("cube-transect-strip-layer"),
        stripVal   : document.getElementById("cube-transect-strip-val"),
        strip      : document.getElementById("cube-transect-strip"),
        map     : document.getElementById("cube-transect-map"),
        overlay : document.getElementById("cube-transect-overlay"),
        grid    : document.getElementById("cube-transect-grid"),
        panels  : [...document.querySelectorAll("#cube-transect-grid .cube-eplane")].map((root) => ({
          root,
          source : root.dataset.source,
          canvas : root.querySelector("canvas"),
        })),
      },
      cloud         : {
        source    : document.getElementById("cube-cloud-source"),
        color     : document.getElementById("cube-cloud-color"),
        thr       : document.getElementById("cube-cloud-thr"),
        thrLabel  : document.getElementById("cube-cloud-thr-label"),
        thrVal    : document.getElementById("cube-cloud-thr-val"),
        max       : document.getElementById("cube-cloud-max"),
        cutAxis   : document.getElementById("cube-cloud-cut-axis"),
        cutPos    : document.getElementById("cube-cloud-cut-pos"),
        cutSweep  : document.getElementById("cube-cloud-cut-sweep"),
        auxWrap   : document.getElementById("cube-cloud-aux-wrap"),
        aux       : document.getElementById("cube-cloud-aux"),
        at        : document.getElementById("cube-cloud-at"),
        canvas    : document.getElementById("cube-cloud-canvas"),
      },
      cmapSel      : document.getElementById("cube-cmap"),
      jumpAz       : document.getElementById("cube-jump-row"),
      jumpRg       : document.getElementById("cube-jump-col"),
      jumpGo       : document.getElementById("cube-jump-go"),
      jumpPrint    : document.getElementById("cube-jump-print"),
      jumpRowRange : document.getElementById("cube-jump-row-range"),
      jumpColRange : document.getElementById("cube-jump-col-range"),
      sweeps       : ["z", "row", "col"].map((axis) => {
        const grid = document.getElementById(`cube-${axis}-grid`);
        return {
          axis,
          grid,
          at     : document.getElementById(`cube-${axis}-at`),
          fill   : document.getElementById(`cube-${axis}-fill`),
          input  : document.getElementById(`cube-${axis}-input`),
          range  : document.getElementById(`cube-${axis}-range`),
          play   : document.getElementById(`cube-${axis}-play`),
          panels : [...grid.querySelectorAll(".cube-eplane")].map((root) => ({
            root,
            source : root.dataset.source,
            canvas : root.querySelector("canvas"),
          })),
        };
      }),
      panels        : [...document.querySelectorAll(".cube-slice")].map((root) => ({
        root,
        axis   : root.dataset.axis,
        source : root.dataset.source,
        canvas : root.querySelector("canvas"),
        ref    : root.querySelector(".cube-ref"),
        metric : root.querySelector(".cube-slice__metric"),
      })),
      profPanels    : [...document.querySelectorAll(".cube-prof")].map((root) => ({
        root,
        source : root.dataset.source,
        canvas : root.querySelector("canvas"),
      })),
    });
    window.cubeView = this.cubeView;

    this.sliceCollectorView = new window.SliceCollectorView({
      strip       : document.getElementById("sc-strip"),
      hint        : document.getElementById("sc-hint"),
      stage       : document.getElementById("sc-stage"),
      rowInput    : document.getElementById("sc-row"),
      colInput    : document.getElementById("sc-col"),
      rowRange    : document.getElementById("sc-row-range"),
      colRange    : document.getElementById("sc-col-range"),
      addBtn      : document.getElementById("sc-add"),
      axesWrap    : document.getElementById("sc-axes"),
      sourcesWrap : document.getElementById("sc-sources"),
      spaceWrap   : document.getElementById("sc-space"),
      sharedWrap  : document.getElementById("sc-shared"),
      cmapSel     : document.getElementById("sc-cmap"),
      nameInput   : document.getElementById("sc-name"),
      collectBtn  : document.getElementById("sc-collect"),
      msg         : document.getElementById("sc-msg"),
      rows        : document.getElementById("sc-rows"),
      mapRunSel   : document.getElementById("sc-map-run"),
      mapWrap     : document.getElementById("sc-map-wrap"),
      mapImg      : document.getElementById("sc-map-img"),
      cross       : document.getElementById("sc-cross"),
      marks       : document.getElementById("sc-marks"),
      coords      : document.getElementById("sc-coords"),
      basketSum   : document.getElementById("sc-basket-sum"),
      basketBody  : document.getElementById("sc-basket-body"),
      basketTog   : document.getElementById("sc-basket-toggle"),
      basketClear : document.getElementById("sc-basket-clear"),
      basketCount : document.getElementById("sc-basket-count"),
      mailTo      : document.getElementById("sc-mail-to"),
      mailNote    : document.getElementById("sc-mail-note"),
      mailBtn     : document.getElementById("sc-mail"),
    });

    this.figureDock = new window.FigureDock();
    window.figureDock = this.figureDock;
  }

  _figureTabs() {
    return {
      microscope  : this.microscopeView,
      survey      : this.surveyView,
      triage      : this.triageView,
      autopsy     : this.autopsyView,
      fitlab      : this.fitLabView,
      cube        : this.cubeView,
      slices      : this.sliceCollectorView,
      results     : this.resultsView,
      leaderboard : this.leaderboardView,
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

  _initModelZoo() {
    const wrap = document.getElementById("model-zoo");
    if (!wrap) return;
    wrap.querySelectorAll(".model-zoo__btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.classList.contains("is-active")) return;
        wrap.querySelectorAll(".model-zoo__btn").forEach((b) => b.classList.toggle("is-active", b === btn));
        wrap.closest(".page__head").querySelector(".page__figure").setAttribute("name", btn.dataset.figure);
        this.modelGallery.reload(btn.dataset.endpoint, btn.dataset.zoo);
      });
    });
  }

  _initModelMode() {
    const wrap  = document.getElementById("model-mode");
    const views = [...document.querySelectorAll(".model-view")];
    if (!wrap) return;
    wrap.querySelectorAll(".model-mode__btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        wrap.querySelectorAll(".model-mode__btn").forEach((b) => b.classList.toggle("is-active", b === btn));
        wrap.closest(".page__head").querySelector(".page__figure").setAttribute("name", btn.dataset.figure);
        const mode = btn.dataset.mode;
        views.forEach((v) => v.classList.toggle("is-hidden", v.dataset.view !== mode));
        if (mode === "walkthrough") this.flowView.load();
      });
    });
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
    if (route === "ablation") this.ablationView.enter();
    if (route === "saved") this.savedRunsView.enter();
    if (route === "templates") this.templatesView.enter();
    if (route === "sequences") this.sequencesView.enter();
    else this.sequencesView.leave();
    if (route === "physics") this.physicsLossView.load();
    if (route === "model" && document.querySelector("#model-mode .model-mode__btn.is-active")?.dataset.mode === "walkthrough") this.flowView.load();
    if (route === "repomap") this.repoMap.enter();
    if (route === "tensorboard") this.tensorboardView.enter();
    else this.tensorboardView.leave();
    if (route === "results") this.resultsView.enter();
    if (route === "leaderboard") this.leaderboardView.enter();
    if (route === "feedtuner") this.feedTuner.enter();
    if (route === "cube") this.cubeView.enter(param);
    else this.cubeView.leave();
    if (route === "slices") this.sliceCollectorView.enter();
    if (route === "fitlab") this.fitLabView.enter();
    if (route === "microscope") this.microscopeView.enter();
    if (route === "survey") this.surveyView.enter();
    if (route === "triage") this.triageView.enter();
    if (route === "autopsy") this.autopsyView.enter();
    if (route === "console") this.runConsole.onShow();
    this._mountFigures(route);
    setTimeout(() => this.reveal.scan(), 60);
  }
}

document.addEventListener("DOMContentLoaded", () => new App().init());
