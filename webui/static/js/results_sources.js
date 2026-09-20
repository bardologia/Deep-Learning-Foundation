"use strict";

class ResultsSources {
  static KEY              = "results-sources";
  static DEFAULT_DATASETS = "";
  static DEFAULT_RUNS     = "";

  static configure(roots) {
    ResultsSources.DEFAULT_DATASETS = (roots && roots.datasets) || "";
    ResultsSources.DEFAULT_RUNS     = (roots && roots.runs) || "";
  }

  static stored() {
    try {
      const raw = JSON.parse(localStorage.getItem(ResultsSources.KEY) || "{}");
      return raw && typeof raw === "object" ? raw : {};
    } catch (e) {
      return {};
    }
  }

  static datasets() {
    return ResultsSources.stored().datasets || ResultsSources.DEFAULT_DATASETS;
  }

  static runs() {
    return ResultsSources.stored().logs || ResultsSources.DEFAULT_RUNS;
  }

  static all() {
    return { datasets: ResultsSources.datasets(), logs: ResultsSources.runs() };
  }

  static save(sources) {
    localStorage.setItem(ResultsSources.KEY, JSON.stringify(sources));
  }
}

window.ResultsSources = ResultsSources;
