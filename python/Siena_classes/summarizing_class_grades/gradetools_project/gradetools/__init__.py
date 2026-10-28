"""gradetools: Canvas gradebook -> category averages, plots, and student emails."""

from .canvas import load_canvas, Gradebook
from .config import load_config, ConfigError
from .classify import classify
from .diagnostics import diagnose, assignment_table
from .scoring import compute, canvas_check, Results
from .plots import class_overview, student_figure
from .emailer import compose, build_outbox, preview, send_emails


def run(cfg_dict, gradebook=None, verbose=True):
    """Load, classify, check and compute in one call. Returns (cfg, gb, classified, results, issues)."""
    cfg = load_config(cfg_dict)
    gb = load_canvas(gradebook or cfg["gradebook"], skip_students=cfg["skip_students"])
    cls = classify(gb, cfg)
    issues = diagnose(gb, cls, cfg, verbose=verbose)
    res = compute(gb, cls, cfg)
    return cfg, gb, cls, res, issues
