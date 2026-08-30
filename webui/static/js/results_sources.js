"use strict";

class ResultsSources {
  static KEY              = "results-sources";
  static get DEFAULT_DATASETS() { return window.repoPath("datasets"); }
  static get DEFAULT_RUNS()     { return window.repoPath("runs"); }

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
