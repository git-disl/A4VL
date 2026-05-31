"""Target resolution and subprocess launcher for A4VL scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .constants import BENCHMARK_ALIASES, BENCHMARK_ORDER, BENCHMARK_SCRIPT_MAP


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_target(target: str) -> str:
    name = target.strip().lower()
    name = name.replace("-", "_")
    name = BENCHMARK_ALIASES.get(name, name)
    if name == "all":
        return name
    if name not in BENCHMARK_SCRIPT_MAP:
        known = ", ".join(available_targets())
        raise ValueError(f"Unknown target '{target}'. Available: {known}")
    return name


def available_targets() -> list[str]:
    return list(BENCHMARK_ORDER) + ["all"]


def _launch_script(script_name: str, script_args: Iterable[str]) -> int:
    root = _project_root()
    script_path = root / script_name
    cmd = [sys.executable, str(script_path), *list(script_args)]
    completed = subprocess.run(cmd, cwd=root)
    return completed.returncode


def launch_target(target: str, script_args: Iterable[str]) -> int:
    name = _normalize_target(target)
    if name == "all":
        for benchmark in BENCHMARK_ORDER:
            rc = _launch_script(BENCHMARK_SCRIPT_MAP[benchmark], script_args)
            if rc != 0:
                return rc
        return 0
    return _launch_script(BENCHMARK_SCRIPT_MAP[name], script_args)
