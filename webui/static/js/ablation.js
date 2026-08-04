"use strict";

class AblationView extends ConfigForm {

  static ENTRIES = [
    ["train_backbone", "Backbone"],
    ["train_dual",     "Dual"],
  ];

  constructor(runConsole, project) {
    super();
    this.runConsole = runConsole;
    this.project    = project || {};
    this.host       = document.getElementById("ablation-root");
    this.key        = "train_backbone";

    this.detail = null;
    this.config = null;
    this.byPath = new Map();
    this.builder = null;
    this.detach  = true;

    this.loaded  = false;
    this.loading = false;

    this.configHost = null;
    this.interpEl   = null;
    this.detachEl   = null;
    this.topCountEl = null;
    this.launchBtn  = null;
    this.cmdEl      = null;
  }

  async enter() {
    if (!this.host || this.loaded || this.loading) return;
    this.loading = true;
    await this._load();
    this.loading = false;
  }

  async _load() {
    this.host.innerHTML = `<p class="ablation__note">Loading training configuration…</p>`;

    const detail = await window.apiGet(`/api/scripts/${this.key}`);
    if (!detail || detail.error) {
      this.host.innerHTML = `<p class="ablation__note ablation__note--error">Could not load the training script.</p>`;
      return;
    }
    this.detail = detail;

    const cfg = await window.apiGet(`/api/scripts/${this.key}/config`);
    if (!cfg || !cfg.ok) {
      this.host.innerHTML = `<p class="ablation__note ablation__note--error">${(cfg && cfg.error) || "Could not resolve configuration."}</p>`;
      return;
    }
    this.config = cfg;
    this.byPath = new Map(cfg.leaves.map((leaf) => [leaf.path, leaf]));

    if (!this.byPath.get("ablation_features")) {
      this.host.innerHTML = `<p class="ablation__note ablation__note--error">This build exposes no ablation configuration.</p>`;
      return;
    }

    const models = await window.apiGet("/api/backbones");
    this.modelFamilies = (models && models.families) || [];

    this.loaded = true;
    this._render();
  }

  _render() {
    this.host.innerHTML = "";

    this.dirty         = {};
    this.controls      = {};
    this.dependents    = {};
    this.states        = [];
    this.gates         = [];
    this.sections      = [];
    this.pairs         = [];
    this.activeSection = null;
    this.query         = "";
    this.builder       = null;

    this.host.appendChild(this._buildBar());

    const configHost     = document.createElement("div");
    configHost.className  = "ablation__config";
    this.configHost      = configHost;
    this.host.appendChild(configHost);

    const cmd     = document.createElement("pre");
    cmd.className = "ablation__cmd";
    this.cmdEl    = cmd;
    this.host.appendChild(cmd);

    this._arm();
    this._renderConfig();
    this._refresh();
  }

  async _switchEntry(key) {
    if (this.key === key || this.loading) return;
    this.key     = key;
    this.loaded  = false;
    this.loading = true;
    await this._load();
    this.loading = false;
  }

  _buildBar() {
    const bar     = document.createElement("div");
    bar.className = "ablation__bar";

    const entry     = document.createElement("div");
    entry.className = "exp-mode ablation__entry";
    AblationView.ENTRIES.forEach(([key, label]) => {
      const btn       = document.createElement("button");
      btn.type        = "button";
      btn.className   = "exp-mode__btn" + (key === this.key ? " is-on" : "");
      btn.textContent = label;
      btn.title       = `Run the ablation ladder through ${key}`;
      btn.addEventListener("click", () => this._switchEntry(key));
      entry.appendChild(btn);
    });
    bar.appendChild(entry);

    const interp     = document.createElement("select");
    interp.className = "ablation__interp";
    (this.project.interpreters || []).forEach((it) => {
      const opt       = document.createElement("option");
      opt.value       = it.path;
      opt.textContent = it.label || it.path;
      if (it.path === this.detail.preferred) opt.selected = true;
      interp.appendChild(opt);
    });
    this.interpEl = interp;
    bar.appendChild(interp);

    const detachWrap     = document.createElement("div");
    detachWrap.className  = "rail-detach ablation__detach";

    const detachToggle    = document.createElement("button");
    detachToggle.type     = "button";
    detachToggle.className = "switch";
    detachToggle.setAttribute("role", "switch");
    detachToggle.innerHTML = `<span class="switch__knob"></span>`;

    const detachLabel     = document.createElement("span");
    detachLabel.className  = "rail-detach__label";
    detachLabel.textContent = "Detach from server";

    const paintDetach = () => {
      detachToggle.classList.toggle("is-on", this.detach);
      detachToggle.setAttribute("aria-checked", String(this.detach));
      detachToggle.title = this.detach
        ? "The run survives a lost connection or a console restart. Output goes to logs/<script>_<stamp>.out in the repo."
        : "Output streams to this console. The run dies if the console server goes down.";
    };

    detachToggle.addEventListener("click", () => {
      this.detach = !this.detach;
      paintDetach();
      this._refresh();
    });

    paintDetach();
    this.detachEl = detachToggle;
    detachWrap.appendChild(detachToggle);
    detachWrap.appendChild(detachLabel);
    bar.appendChild(detachWrap);

    const count     = document.createElement("span");
    count.className = "ablation__count";
    this.topCountEl = count;
    bar.appendChild(count);

    const launch     = document.createElement("button");
    launch.type      = "button";
    launch.className = "btn btn--primary ablation__launch";
    launch.addEventListener("click", () => this._launch());
    this.launchBtn = launch;
    bar.appendChild(launch);

    return bar;
  }

  _arm() {
    const trials = this.byPath.get("trials_enabled");
    const mode   = this.byPath.get("trials_mode");
    if (mode)   this._setValue(mode, "ablation");
    if (trials) this._setValue(trials, "True");
  }

  _buildSpecialPanel(panel) {
    if (panel.panel === "experiment_builder") {
      this.builder = new window.AblationBuilder(this, this.byPath);
      return this.builder.build();
    }
    return super._buildSpecialPanel(panel);
  }

  _renderConfig() {
    const host     = this.configHost;
    host.innerHTML = "";

    host.appendChild(this._buildToolbar(this.config));
    this._renderLayout(host, this.config);
  }

  _active() {
    const trials = this.byPath.get("trials_enabled");
    const mode   = this.byPath.get("trials_mode");
    return Boolean(trials && mode && this._effective(trials) === "True" && this._effective(mode) === "ablation");
  }

  _refresh() {
    const active = this._active();
    const n      = Object.keys(this.dirty).length;
    const label  = n ? `${n} override${n === 1 ? "" : "s"}` : "all defaults";

    if (this.topCountEl) this.topCountEl.textContent = label;
    if (this.countEl)    this.countEl.textContent    = label;
    if (this.cmdEl)      this.cmdEl.textContent       = this._commandText();

    if (this.launchBtn) {
      this.launchBtn.disabled    = !active;
      this.launchBtn.textContent = active ? "Launch ablation study" : "Enable the ablation switch to launch";
    }

    if (this.builder) this.builder.refreshFromView();
    this._refreshBadges();
    this._refreshGates();
  }

  _commandText() {
    let text = (this.detail && this.detail.command) || "python main/training/train_backbone.py";
    Object.entries(this.dirty).forEach(([path, value]) => {
      const raw      = String(value);
      const rendered = /^[\w@%+=:,./-]+$/.test(raw) ? raw : `'${raw.replace(/'/g, `'\\''`)}'`;
      text += ` \\\n  --${path} ${rendered}`;
    });
    if (this.detach) text += ` \\\n  --detach`;
    return text;
  }

  _launch() {
    if (!this._active()) return;
    const interp = this.interpEl ? this.interpEl.value : this.detail.preferred;
    this.runConsole.launch(this.key, interp, "Ablation study", { ...this.dirty }, null, this.detach);
    window.router.go("console");
  }
}

window.AblationView = AblationView;
