from __future__ import annotations

import logging

import pytest

from tools.monitoring.logger import Logger, NullLogger, LiveMonitor, get_console


def _read_log(log_dir, name="experiment"):
    return (log_dir / f"{name}.log").read_text(encoding="utf-8")


def test_get_console_is_singleton():
    a = get_console()
    b = get_console()

    assert a is b


def test_init_creates_log_dir_and_file(tmp_path):
    log_dir = tmp_path / "logs"
    logger  = Logger(log_dir=str(log_dir), name="exp_init")

    assert log_dir.is_dir()
    assert (log_dir / "exp_init.log").is_file()

    logger.close()


def test_init_without_log_dir_has_no_file_handler():
    logger = Logger(log_dir="", name="exp_nofile")

    assert logger._file_handler is None

    logger.close()


def test_level_parsing_known(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_lvl", level="DEBUG")

    assert logger.logger.level == logging.DEBUG

    logger.close()


def test_level_parsing_case_insensitive(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_lvl2", level="warning")

    assert logger.logger.level == logging.WARNING

    logger.close()


def test_level_parsing_unknown_defaults_info(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_lvl3", level="BOGUS")

    assert logger.logger.level == logging.INFO

    logger.close()


def test_info_written_to_file(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_info")
    logger.info("hello world")
    logger.close()

    contents = _read_log(tmp_path, "exp_info")

    assert "hello world" in contents
    assert "INFO" in contents


def test_warning_and_error_written_to_file(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_we")
    logger.warning("be careful")
    logger.error("it broke")
    logger.close()

    contents = _read_log(tmp_path, "exp_we")

    assert "be careful" in contents
    assert "it broke" in contents
    assert "WARNING" in contents
    assert "ERROR" in contents


def test_debug_suppressed_below_level(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_dbg", level="INFO")
    logger.debug("invisible debug")
    logger.close()

    contents = _read_log(tmp_path, "exp_dbg")

    assert "invisible debug" not in contents


def test_debug_emitted_when_level_debug(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_dbg2", level="DEBUG")
    logger.debug("visible debug")
    logger.close()

    contents = _read_log(tmp_path, "exp_dbg2")

    assert "visible debug" in contents


def test_section_writes_uppercase_marker(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_sec")
    logger.section("data loading")
    logger.close()

    contents = _read_log(tmp_path, "exp_sec")

    assert ">>> DATA LOADING" in contents


def test_section_writes_banner_rules(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_sec_bar")
    logger.section("data loading")
    logger.close()

    contents = _read_log(tmp_path, "exp_sec_bar")
    bar      = "=" * Logger.FILE_RULE_WIDTH
    lines    = contents.splitlines()
    index    = next(i for i, line in enumerate(lines) if ">>> DATA LOADING" in line)

    assert lines[index - 1] == bar
    assert lines[index + 1] == bar
    assert "[INFO]" in lines[index]


def test_second_section_includes_elapsed(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_sec_dt")
    logger.section("first")
    logger.section("second")
    logger.close()

    contents = _read_log(tmp_path, "exp_sec_dt")

    assert ">>> FIRST\n" in contents
    assert ">>> SECOND  (+" in contents


def test_header_banner_written(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_head")
    logger.close()

    contents = _read_log(tmp_path, "exp_head")

    assert ">>> RUN exp_head" in contents
    assert ">>> END exp_head" in contents


def test_fmt_duration_scales():
    assert Logger._fmt_duration(12.34)   == "12.3s"
    assert Logger._fmt_duration(75.0)    == "1m15s"
    assert Logger._fmt_duration(3725.0)  == "1h02m05s"


def test_subsection_writes_marker(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_sub")
    logger.subsection("step one")
    logger.close()

    contents = _read_log(tmp_path, "exp_sub")
    line     = next(line for line in contents.splitlines() if "> step one" in line)

    assert "[INFO]" in line


def test_ok_writes_plus_marker(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_ok")
    logger.ok("done")
    logger.close()

    contents = _read_log(tmp_path, "exp_ok")

    assert "+ done" in contents


def test_fmt_float_uses_six_sig():
    assert Logger._fmt(3.14159265) == "3.14159"
    assert Logger._fmt(1000000.0)  == "1e+06"


def test_fmt_non_float_is_str():
    assert Logger._fmt(42)    == "42"
    assert Logger._fmt("abc") == "abc"
    assert Logger._fmt(None)  == "None"


def test_kv_table_does_not_crash(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_kv")
    logger.kv_table({"alpha": 1.5, "beta": "x", "gamma": None}, title="Params")
    logger.close()


def test_kv_table_empty_does_not_crash(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_kv2")
    logger.kv_table({})
    logger.close()


def test_metrics_table_does_not_crash(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_mt")
    rows   = [{"epoch": 1, "loss": 0.5}, {"epoch": 2, "loss": 0.25}]
    logger.metrics_table(rows, columns=["epoch", "loss", "missing"], title="Train")
    logger.close()


def test_panel_and_rule_and_render_do_not_crash(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_panel")
    logger.panel("body text", title="Title")
    logger.rule("a rule")
    logger.render("rendered text")
    logger.close()


def test_timer_logs_completion(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_timer")
    with logger.timer("phase"):
        pass
    logger.close()

    contents = _read_log(tmp_path, "exp_timer")

    assert "phase completed in" in contents


def test_timer_logs_failure_and_reraises(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_timer_fail")
    with pytest.raises(ValueError):
        with logger.timer("phase"):
            raise ValueError("boom")
    logger.close()

    contents = _read_log(tmp_path, "exp_timer_fail")

    assert "phase failed after" in contents
    assert "phase completed in" not in contents


def test_track_yields_progress(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_track")
    with logger.track(transient=True) as progress:
        task = progress.add_task("work", total=3)
        progress.advance(task, 3)
    logger.close()


def test_progress_bar_aliases_track():
    assert Logger.progress_bar is Logger.track


def test_live_monitor_updates(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_live")
    with logger.live_monitor(title="Run") as monitor:
        monitor.update(loss=0.1, step=5)
        monitor.update(loss=0.05)

    assert monitor._metrics["loss"] == 0.05
    assert monitor._metrics["step"] == 5

    logger.close()


def test_context_manager_closes(tmp_path):
    with Logger(log_dir=str(tmp_path), name="exp_ctx") as logger:
        logger.info("inside")

    contents = _read_log(tmp_path, "exp_ctx")

    assert "inside" in contents
    assert "[End] Duration:" in contents


def test_context_manager_logs_exception(tmp_path):
    with pytest.raises(RuntimeError):
        with Logger(log_dir=str(tmp_path), name="exp_ctx_exc") as logger:
            raise RuntimeError("model diverged")

    contents = _read_log(tmp_path, "exp_ctx_exc")

    assert "Aborted by RuntimeError: model diverged" in contents
    assert "[End] Duration:" in contents


def test_close_removes_all_handlers(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_close")
    logger.close()

    assert logger.logger.handlers == []
    assert logger._file_handler is None


def test_reinit_same_name_resets_handlers(tmp_path):
    a = Logger(log_dir=str(tmp_path), name="dup")
    b = Logger(log_dir=str(tmp_path), name="dup")

    assert len(b.logger.handlers) == 2

    b.close()


def test_propagate_disabled(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="exp_prop")

    assert logger.logger.propagate is False

    logger.close()


def test_live_monitor_render_returns_panel():
    monitor = LiveMonitor(get_console(), title="T")
    monitor.update(a=1.234567, b="text", big=5000.0)

    panel = monitor._render()

    assert panel is not None


def test_null_logger_swallows_everything():
    nl = NullLogger()

    assert nl.info("x")            is None
    assert nl.section("y")         is None
    assert nl.anything("a", b=2)   is None
    assert nl.kv_table({"k": "v"}) is None
