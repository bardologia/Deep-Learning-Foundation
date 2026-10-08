"""Guided per-machine setup: the questions ``install.sh`` cannot answer by itself.

Probes the host, asks for the few locations and settings that differ between
installations, writes the machine profile and its shell twin, verifies the
environment through the recorded interpreter, and reports how to start the
console. Every question carries a detected or previously saved answer as its
default, so pressing Enter through the wizard is a valid setup, and ``--yes`` does
exactly that without a terminal. Preset answers can be supplied as JSON with
``--answers`` for unattended installs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rich.text import Text

from configuration.reference_layout import ReferenceLayout
from tools.monitoring.logger        import Logger
from tools.runtime.machine_probe    import MachineProbe
from tools.runtime.machine_profile  import MachineProfile


class Prompter:
    """Asks questions on the terminal with defaults, validation and an unattended mode.

    Attributes:
        answers : Preset answers keyed by question key, consulted before the terminal.
        assume  : Whether every unanswered question takes its default without asking.
        logger  : Logger the help lines and rejections are written through.
    """

    def __init__(self, answers: dict, assume: bool, logger: Logger) -> None:
        """Stores the preset answers, the unattended flag and the logger.

        Args:
            answers : Mapping of question key to preset answer.
            assume  : Whether to take defaults without asking.
            logger  : Logger for help text and validation messages.
        """
        self.answers = dict(answers)
        self.assume  = assume
        self.logger  = logger

    def ask(self, key: str, question: str, default: str = "", help: str = "", validate=None) -> str:
        """Returns the answer to one free-text question.

        A preset answer wins, then the default in unattended mode, then the terminal.
        A validator returning an error message rejects the input and asks again.

        Args:
            key      : Question key used for preset answers.
            question : Text shown to the user.
            default  : Value taken when the user presses Enter.
            help     : One-line explanation shown above the question.
            validate : Callable returning an error string for a bad value or None.

        Returns:
            The accepted answer as text.

        Raises:
            ValueError : If a preset or unattended default fails validation.
        """
        if key in self.answers:
            value = str(self.answers[key])
            error = validate(value) if validate else None
            if error:
                raise ValueError(f"preset answer {key}={value!r} rejected: {error}")
            return value

        if self.assume:
            error = validate(default) if validate else None
            if error:
                raise ValueError(f"default for {key} ({default!r}) rejected in unattended mode: {error}")
            return default

        if help:
            self.logger.render(Text(f"  {help}", style="dim"))

        while True:
            raw   = input(f"  {question} [{default}]: ").strip()
            value = raw or default
            error = validate(value) if validate else None

            if error is None:
                return value

            self.logger.warning(error)

    def confirm(self, key: str, question: str, default: bool, help: str = "") -> bool:
        """Returns the answer to a yes/no question.

        Args:
            key      : Question key used for preset answers.
            question : Text shown to the user.
            default  : Value taken when the user presses Enter.
            help     : One-line explanation shown above the question.

        Returns:
            True for yes.
        """
        shown  = "Y/n" if default else "y/N"
        answer = self.ask(key, f"{question} ({shown})", "yes" if default else "no", help, self._yes_no)

        return answer.strip().lower() in ("y", "yes", "true", "1", "on")

    @staticmethod
    def _yes_no(value: str) -> str | None:
        """Accepts the usual yes and no spellings."""
        if value.strip().lower() in ("y", "yes", "true", "1", "on", "n", "no", "false", "0", "off"):
            return None

        return f"please answer yes or no, not {value!r}"

    def choose(self, key: str, question: str, options: list[str], default: str, help: str = "") -> str:
        """Returns one of the options, chosen by number or by typing it.

        Args:
            key      : Question key used for preset answers.
            question : Text shown to the user.
            options  : Values offered, numbered from 1 in the terminal.
            default  : Value taken when the user presses Enter.
            help     : One-line explanation shown above the question.

        Returns:
            The chosen option.
        """
        if key not in self.answers and not self.assume:
            for number, option in enumerate(options, start=1):
                self.logger.render(Text(f"    {number}. {option}", style="dim"))

        def validate(value: str) -> str | None:
            """Accepts an option number, an option text, or any value when no options were detected."""
            if value in options or not options:
                return None
            if value.isdigit() and 1 <= int(value) <= len(options):
                return None
            return f"choose a number between 1 and {len(options)} or type a value"

        answer = self.ask(key, question, default, help, validate)

        if answer.isdigit() and options and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]

        return answer


class Validators:
    """Validation rules shared by the wizard's questions; each returns an error string or None."""

    EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    HOST  = re.compile(r"^[A-Za-z0-9.\-]+$")
    TOPIC = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

    @staticmethod
    def existing_dir(value: str) -> str | None:
        """Requires an existing directory."""
        return None if value and Path(value).expanduser().is_dir() else f"{value!r} is not an existing directory"

    @staticmethod
    def existing_file(value: str) -> str | None:
        """Requires an existing file."""
        return None if value and Path(value).expanduser().is_file() else f"{value!r} is not an existing file"

    @staticmethod
    def creatable_dir(value: str) -> str | None:
        """Requires an absolute path whose nearest existing ancestor is a directory."""
        if not value:
            return "a path is required"

        path = Path(value).expanduser()

        if not path.is_absolute():
            return f"{value!r} must be an absolute path"

        ancestor = path
        while not ancestor.exists():
            ancestor = ancestor.parent

        return None if ancestor.is_dir() else f"{ancestor} exists and is not a directory"

    @staticmethod
    def optional_dir(value: str) -> str | None:
        """Accepts an empty value or an existing directory."""
        return None if not value else Validators.existing_dir(value)

    @staticmethod
    def optional_file(value: str) -> str | None:
        """Accepts an empty value or an existing file."""
        return None if not value else Validators.existing_file(value)

    @staticmethod
    def port(value: str) -> str | None:
        """Requires a TCP port between 1024 and 65535."""
        return None if value.isdigit() and 1024 <= int(value) <= 65535 else f"{value!r} is not a port between 1024 and 65535"

    @staticmethod
    def host(value: str) -> str | None:
        """Requires a host name or address."""
        return None if Validators.HOST.match(value or "") else f"{value!r} is not a host name or address"

    @staticmethod
    def email(value: str) -> str | None:
        """Requires an email address."""
        return None if Validators.EMAIL.match(value or "") else f"{value!r} is not an email address"

    @staticmethod
    def optional_email(value: str) -> str | None:
        """Accepts an empty value or an email address."""
        return None if not value else Validators.email(value)

    @staticmethod
    def topic(value: str) -> str | None:
        """Requires an ntfy topic name."""
        return None if Validators.TOPIC.match(value or "") else f"{value!r} may only hold letters, digits, '-' and '_' (max 64 chars)"

    @staticmethod
    def gpu_list(indices: list):
        """Returns a validator accepting a comma-separated subset of the detected GPU indices."""
        def validate(value: str) -> str | None:
            """Rejects an empty list or an index that was not detected."""
            tokens = [token.strip() for token in value.split(",") if token.strip()]
            if not tokens:
                return "name at least one GPU index"
            bad = [token for token in tokens if not token.isdigit() or int(token) not in indices]
            return None if not bad else f"unknown GPU index(es) {bad}; detected {indices}"
        return validate

    @staticmethod
    def positive_int(value: str) -> str | None:
        """Requires a positive integer."""
        return None if value.isdigit() and int(value) > 0 else f"{value!r} is not a positive integer"

    @staticmethod
    def positive_float(value: str) -> str | None:
        """Requires a positive number."""
        try:
            return None if float(value) > 0 else f"{value!r} must be positive"
        except ValueError:
            return f"{value!r} is not a number"


class MachineSetup:
    """Walks a new user through the machine-dependent settings and writes the profile.

    Attributes:
        repo_root   : Checkout the profile describes.
        prompter    : Question helper honouring presets and the unattended mode.
        probe       : Host fact detector.
        profile     : Profile being filled; starts from the saved one on a rerun.
        logger      : Logger for progress and the final report.
        env_name    : Conda environment name handed over by the installer.
        interpreter : Interpreter handed over by the installer, or None to detect it.
        facts       : Host facts detected by ``probe_host``.
    """

    DESKTOP_NAME = "ml-template-console.desktop"
    STEPS        = 8

    def __init__(self, repo_root: Path, prompter: Prompter, probe: MachineProbe, profile: MachineProfile, logger: Logger, env_name: str | None, interpreter: Path | None, verify: bool, desktop: bool) -> None:
        """Stores the collaborators and the installer's hand-over values.

        Args:
            repo_root   : Repository checkout being set up.
            prompter    : Question helper.
            probe       : Host fact detector.
            profile     : Profile object bound to the location the answers are saved to.
            logger      : Logger for progress and the report.
            env_name    : Environment name chosen by the installer, or None to ask.
            interpreter : Interpreter chosen by the installer, or None to detect.
            verify      : Whether to run the environment verification step.
            desktop     : Whether to offer the desktop launcher.
        """
        self.repo_root   = repo_root
        self.prompter    = prompter
        self.probe       = probe
        self.profile     = profile
        self.logger      = logger
        self.env_name    = env_name
        self.interpreter = interpreter
        self.verify_env  = verify
        self.desktop     = desktop
        self.facts       = {}

    @property
    def saved(self) -> dict:
        """The sections of the previously saved profile, empty on a first run."""
        return self.profile.data

    def _previous(self, section: str, key: str, fallback):
        """Returns a value from the saved profile section, or the fallback when absent."""
        return self.saved.get(section, {}).get(key, fallback)

    def _step(self, number: int, title: str) -> None:
        """Opens one numbered wizard section."""
        self.logger.section(f"Step {number} of {self.STEPS}: {title}")

    def probe_host(self) -> None:
        """Detects the host facts and shows them."""
        self._step(1, "this machine")

        self.facts = self.probe.facts()
        gpus       = self.facts["gpus"]
        failure    = self.facts["driver_error"]
        note       = self.facts["torch_note"]
        table      = {
            "host"    : self.facts["hostname"],
            "CPUs"    : str(self.facts["cpu_count"]),
            "RAM"     : f"{self.facts['ram_gb']} GB",
            "GPUs"    : ", ".join(f"{gpu['index']}: {gpu['name']} ({gpu['memory_gb']} GB)" for gpu in gpus) or "none detected",
            "driver"  : self.facts["driver"] or ("nvidia-smi failed" if failure else "no NVIDIA driver"),
            "torch"   : self.facts["torch_index"] + (f" ({note})" if note else ""),
            "conda"   : self.facts["conda_root"] or "not found",
            "profile" : str(self.profile.path),
        }

        self.logger.kv_table(table, title="Detected")

        if failure:
            self.logger.warning(f"{failure}: the GPU list and the torch wheel choice are unreliable until the driver answers; after a kernel update install the matching linux-modules-nvidia package, then rerun")
        elif not gpus:
            self.logger.warning("no GPU detected: the console and the tests work, training and inference need an NVIDIA GPU")

    def choose_environment(self) -> None:
        """Records the environment name and the interpreter."""
        self._step(2, "Python environment")

        conda_root = self.facts["conda_root"] or self._previous("machine", "conda_root", "")
        env_name   = self.env_name or self.prompter.ask("env_name", "Name of the conda environment holding the project", self._previous("machine", "env_name", "Dune"), "install.sh created it; keep the name unless you made your own.")
        handed     = self.interpreter is not None and Path(self.interpreter).is_file() and "interpreter" not in self.prompter.answers

        if handed:
            interp = str(self.interpreter)
            self.logger.info(f"interpreter chosen by install.sh: {interp}")
        else:
            detected = self.probe.env_interpreter(Path(conda_root) if conda_root else None, env_name)
            default  = str(detected or self._previous("machine", "interpreter", "") or sys.executable)
            interp   = self.prompter.ask("interpreter", f"Python executable inside the '{env_name}' environment", default, "Detected automatically; press Enter to keep it. Every launch from the console runs with this file.", Validators.existing_file)

        self.profile.data["machine"] = {
            "hostname"    : self.facts["hostname"],
            "conda_root"  : conda_root,
            "env_name"    : env_name,
            "interpreter" : interp,
            "torch_index" : self.facts["torch_index"],
            "driver"      : self.facts["driver"],
            "cpu_count"   : self.facts["cpu_count"],
            "ram_gb"      : self.facts["ram_gb"],
            "gpus"        : self.facts["gpus"],
        }

    def choose_console(self) -> None:
        """Records the console bind address and port."""
        self._step(3, "web console")

        host = self.prompter.ask("console_host", "Address the console listens on", self._previous("console", "host", ReferenceLayout.CONSOLE_HOST), "127.0.0.1 keeps it private to this machine; 0.0.0.0 exposes it on the network.", Validators.host)
        port = self.prompter.ask("console_port", "Port", str(self._previous("console", "port", ReferenceLayout.CONSOLE_PORT)), "", Validators.port)

        self.profile.data["console"] = {"host": host, "port": int(port)}

    def choose_data(self) -> None:
        """Records the dataset root, the default dataset and its parameter file."""
        self._step(4, "data")

        data_root = self.prompter.ask("data_root", "Directory holding the datasets", self._previous("roots", "data", str(self.repo_root / ReferenceLayout.DATA_ROOT)), "One subdirectory per dataset; it is created if missing.", Validators.creatable_dir)
        data_root = str(Path(data_root).expanduser())
        Path(data_root).mkdir(parents=True, exist_ok=True)

        datasets  = self._datasets_under(Path(data_root))
        dataset   = self.prompter.choose("dataset_path", "Default dataset for training and analysis", datasets, self._previous("dataset", "path", datasets[0] if datasets else ""), "Pick a detected dataset, type a path, or leave empty until you have generated one.")
        params    = self._parameter_files(Path(dataset)) if dataset else []
        parameter = self.prompter.choose("parameters_path", "Default Gaussian parameter file for that dataset", params, self._previous("dataset", "parameters_path", params[0] if params else ""), "Written by extract_params; leave empty if you have not run it yet.")

        self.profile.data["dataset"] = {"path": dataset, "parameters_path": parameter}
        self.profile.data.setdefault("roots", {})["data"] = data_root

    @staticmethod
    def _datasets_under(root: Path) -> list[str]:
        """Returns the dataset directories directly below a data root."""
        found = []

        for child in sorted(root.iterdir()) if root.is_dir() else []:
            if (child / "data" / "dataset.json").is_file():
                found.append(str(child))

        return found

    @staticmethod
    def _parameter_files(dataset: Path) -> list[str]:
        """Returns the parameter arrays of every extraction run inside a dataset."""
        return sorted(str(path) for path in dataset.glob("params/*/parameters.npy"))

    def choose_runs(self) -> None:
        """Records the run and results roots and creates them."""
        self._step(5, "outputs")

        runs    = self.prompter.ask("runs_root", "Directory training and inference runs are written to", self._previous("roots", "runs", str(self.repo_root / ReferenceLayout.RUNS_ROOT)), "Runs can be large; a scratch disk is a good choice.", Validators.creatable_dir)
        results = self.prompter.ask("results_root", "Directory analysis outputs are written to", self._previous("roots", "results", str(self.repo_root / ReferenceLayout.RESULTS_ROOT)), "", Validators.creatable_dir)

        for path in (runs, results):
            Path(path).expanduser().mkdir(parents=True, exist_ok=True)

        roots = self.profile.data.setdefault("roots", {})
        roots.update({"repo": str(self.repo_root), "runs": str(Path(runs).expanduser()), "results": str(Path(results).expanduser())})

    def choose_hardware(self) -> None:
        """Records the GPU pool, the VRAM budget and the worker counts."""
        self._step(6, "hardware limits")

        gpus     = self.facts["gpus"]
        indices  = [gpu["index"] for gpu in gpus]
        cpus     = self.facts["cpu_count"]
        smallest = min((gpu["memory_gb"] for gpu in gpus), default=0.0)
        previous = self.saved.get("hardware", {})

        if indices:
            pool_text = self.prompter.ask("gpu_pool", "GPUs the entry points may use, comma separated", ",".join(str(index) for index in previous.get("gpu_pool", indices)), "Multi-GPU entries fan out over this pool; single-GPU entries take the first one.", Validators.gpu_list(indices))
            pool      = [int(token) for token in pool_text.split(",") if token.strip()]
            budget    = self.prompter.ask("vram_budget_gb", "VRAM budget in GB the batch-size probe may fill", str(previous.get("vram_budget_gb") or round(smallest * 0.9, 1)), f"Smallest GPU holds {smallest} GB; leave headroom for the display and other users.", Validators.positive_float)
        else:
            pool, budget = [], "0"

        cpu_workers = self.prompter.ask("cpu_workers", "CPU workers for inference post-processing", str(previous.get("cpu_workers") or cpus), f"{cpus} CPUs detected.", Validators.positive_int)
        gif_workers = self.prompter.ask("gif_workers", "CPU workers for animation rendering", str(previous.get("gif_workers") or max(1, cpus // 2)), "", Validators.positive_int)
        max_workers = self.prompter.ask("max_workers", "Upper limit for data-loader workers", str(previous.get("max_workers") or cpus), "Defaults above this are lowered to it.", Validators.positive_int)

        self.profile.data["hardware"] = {"gpu_pool": pool, "vram_budget_gb": float(budget), "cpu_workers": int(cpu_workers), "gif_workers": int(gif_workers), "max_workers": int(max_workers)}

    def choose_notifications(self) -> None:
        """Records the optional ntfy topic and report-mail identity."""
        self._step(7, "notifications (optional)")

        ntfy_prev = self.saved.get("ntfy", {})
        mail_prev = self.saved.get("mail", {})

        ntfy = self.prompter.confirm("ntfy_enabled", "Send job notifications to your phone through ntfy?", bool(ntfy_prev.get("enabled")), "Install the ntfy app, subscribe to a secret topic name, and enter it here.")
        if ntfy:
            topic  = self.prompter.ask("ntfy_topic", "ntfy topic", ntfy_prev.get("topic", ""), "", Validators.topic)
            server = self.prompter.ask("ntfy_server", "ntfy server", ntfy_prev.get("server", "https://ntfy.sh"), "")
            self.profile.data["ntfy"] = {"enabled": True, "topic": topic, "server": server.rstrip("/")}
        else:
            self.profile.data["ntfy"] = {"enabled": False, "topic": ntfy_prev.get("topic", ""), "server": ntfy_prev.get("server", "https://ntfy.sh")}

        mail = self.prompter.confirm("mail_enabled", "Mail experiment reports from this machine?", bool(mail_prev.get("recipient")), "Uses a Gmail app password read from REPORT_MAIL_PASSWORD at send time; the password is never stored here.")
        if mail:
            sender    = self.prompter.ask("mail_sender", "Sending address (the SMTP login)", mail_prev.get("sender", ""), "", Validators.email)
            recipient = self.prompter.ask("mail_recipient", "Recipient address", mail_prev.get("recipient", sender), "", Validators.email)
            host      = self.prompter.ask("mail_smtp_host", "SMTP host", mail_prev.get("smtp_host", "smtp.gmail.com"), "", Validators.host)
            port      = self.prompter.ask("mail_smtp_port", "SMTP port", str(mail_prev.get("smtp_port", 465)), "", Validators.positive_int)
            self.profile.data["mail"] = {"sender": sender, "recipient": recipient, "smtp_host": host, "smtp_port": int(port)}
        else:
            self.profile.data["mail"] = {"sender": "", "recipient": "", "smtp_host": mail_prev.get("smtp_host", "smtp.gmail.com"), "smtp_port": mail_prev.get("smtp_port", 465)}

    def write_profile(self) -> None:
        """Saves the profile and its shell twin."""
        self._step(8, "writing the profile")

        path = self.profile.save()

        self.logger.ok(f"profile written to {path}")
        self.logger.info(f"shell twin written to {self.profile.env_path}")

    def write_desktop_entry(self) -> None:
        """Writes an application launcher that starts the console and opens the browser."""
        if not self.desktop or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return

        if not self.prompter.confirm("desktop_entry", "Add an 'ML-Template Console' entry to the application menu?", True, "Starts the console in a terminal and opens it in the browser."):
            return

        console = self.profile.console
        url     = f"http://{console['host'] if console['host'] != '0.0.0.0' else '127.0.0.1'}:{console['port']}"
        run     = self.repo_root / "webui" / "run.sh"
        icon    = self.repo_root / "webui" / "static" / "favicon.svg"
        command = f"bash -c {shlex.quote(f'(sleep 4; xdg-open {url}) & exec {shlex.quote(str(run))}')}"
        target  = Path.home() / ".local" / "share" / "applications" / self.DESKTOP_NAME
        entry   = "\n".join(["[Desktop Entry]", "Type=Application", "Name=ML-Template Console", "Comment=Start the ML-Template control console", f"Exec={command}", f"Icon={icon}", "Terminal=true", "Categories=Science;Development;", ""])

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(entry, encoding="utf-8")
        target.chmod(0o755)

        self.logger.ok(f"application launcher written to {target}")

    def verify(self) -> None:
        """Checks torch and CUDA under the recorded interpreter and that the profile reaches an entry point."""
        if not self.verify_env:
            return

        self.logger.section("Verification")

        interpreter = self.profile.interpreter
        environ     = {**os.environ, MachineProfile.ENV_VAR: str(self.profile.path)}
        torch_probe = "import json, torch; print(json.dumps({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'devices': torch.cuda.device_count()}))"
        result      = subprocess.run([interpreter, "-c", torch_probe], capture_output=True, text=True, cwd=str(self.repo_root), env=environ)

        if result.returncode != 0:
            raise RuntimeError(f"torch does not import under {interpreter}:\n{result.stderr.strip()[-2000:]}")

        facts   = json.loads(result.stdout.strip().splitlines()[-1])
        reasons = " | ".join(line.strip() for line in result.stderr.strip().splitlines() if line.strip())[-600:]
        self.logger.kv_table({"torch": facts["torch"], "CUDA available": str(facts["cuda"]), "CUDA devices": str(facts["devices"])}, title="Interpreter")

        if self.facts["gpus"] and not facts["cuda"]:
            self.logger.warning(f"a GPU was detected but torch cannot use it: {reasons or 'torch gave no reason on stderr'}; check the torch build against the driver (see README section 1)")
        elif reasons:
            self.logger.debug(f"torch probe stderr: {reasons}")

        entry = subprocess.run([interpreter, str(self.repo_root / "main" / "training" / "train_backbone.py"), "--help-config"], capture_output=True, text=True, cwd=str(self.repo_root), env=environ)

        if entry.returncode != 0:
            raise RuntimeError(f"train_backbone.py --help-config failed under the new profile:\n{entry.stderr.strip()[-2000:]}")

        rewritten = entry.stdout.count("[machine profile]")
        self.logger.ok(f"train_backbone.py sees the profile: {rewritten} default(s) rewritten for this machine")

    def report(self) -> None:
        """Prints what was configured and how to start working."""
        console = self.profile.console
        url     = f"http://{console['host'] if console['host'] != '0.0.0.0' else '127.0.0.1'}:{console['port']}"
        summary = {
            "profile"     : str(self.profile.path),
            "interpreter" : self.profile.interpreter,
            "datasets"    : self.profile.roots["data"],
            "dataset"     : self.profile.dataset["path"] or "none yet",
            "runs"        : self.profile.roots["runs"],
            "GPU pool"    : ", ".join(str(index) for index in self.profile.gpu_pool) or "none",
            "ntfy"        : self.profile.ntfy["topic"] if self.profile.ntfy["enabled"] else "off",
            "report mail" : self.profile.mail["recipient"] or "off",
        }

        self.logger.section("Done")
        self.logger.kv_table(summary, title="This machine")
        self.logger.info(f"Start the console with:  {self.repo_root / 'webui' / 'run.sh'}   then open {url}")
        self.logger.info("Change any answer later by running ./install.sh again; every question defaults to the current value.")

    def run(self) -> None:
        """Runs the wizard steps in order."""
        self.probe_host()
        self.choose_environment()
        self.choose_console()
        self.choose_data()
        self.choose_runs()
        self.choose_hardware()
        self.choose_notifications()
        self.write_profile()
        self.write_desktop_entry()
        self.verify()
        self.report()


def main() -> None:
    """Parses the installer hand-over flags and runs the wizard."""
    parser = argparse.ArgumentParser(description="Guided per-machine setup of ML-Template. Detects the host, asks for the locations and settings that differ between installations, writes ~/.config/ml-template/machine.json (or the file MLTEMPLATE_MACHINE_PROFILE names) and verifies the environment. Normally started by ./install.sh.")
    parser.add_argument("--yes",         action="store_true", help="Take the default answer to every question (unattended).")
    parser.add_argument("--answers",     type=str, default=None, help="JSON file of preset answers keyed by question (see the Prompter keys in this script).")
    parser.add_argument("--env",         type=str, default=None, help="Conda environment name chosen by the installer.")
    parser.add_argument("--interpreter", type=str, default=None, help="Interpreter of that environment, as chosen by the installer.")
    parser.add_argument("--profile",     type=str, default=None, help="Profile file to write; defaults to MLTEMPLATE_MACHINE_PROFILE or ~/.config/ml-template/machine.json.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip the torch and entry-point verification.")
    parser.add_argument("--no-desktop",  action="store_true", help="Do not offer the application-menu launcher.")
    args = parser.parse_args()

    logger  = Logger(log_dir="logs", name="setup_machine")
    answers = json.loads(Path(args.answers).read_text(encoding="utf-8")) if args.answers else {}
    path    = Path(args.profile).expanduser() if args.profile else MachineProfile.location()

    if path is None:
        raise SystemExit(f"{MachineProfile.ENV_VAR} is set to an empty value, which disables profiles; unset it or pass --profile")

    profile = MachineProfile.load(path) if path.is_file() else MachineProfile(path)
    setup   = MachineSetup(REPO_ROOT, Prompter(answers, args.yes, logger), MachineProbe(), profile, logger, args.env, Path(args.interpreter) if args.interpreter else None, not args.skip_verify, not args.no_desktop)

    setup.run()


if __name__ == "__main__":
    main()
