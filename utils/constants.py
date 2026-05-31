"""Shared constants for benchmark target routing."""

from __future__ import annotations

BENCHMARK_SCRIPT_MAP = {
    "nextqa": "nextqa_pipeline.py",
    "ego": "egoschema_pipeline.py",
    "mlvu": "mlvu_pipeline.py",
}

BENCHMARK_ORDER = ("nextqa", "ego", "mlvu")

BENCHMARK_ALIASES = {
    "next": "nextqa",
    "nextvideo": "nextqa",
    "egoschema": "ego",
    "ego_schema": "ego",
}
