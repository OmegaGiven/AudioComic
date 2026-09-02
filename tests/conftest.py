"""Shared test fixtures.

The four pipeline stages live in ``scripts/`` as files whose names start with
a digit, so they can't be imported with a normal ``import`` statement. The
``script`` fixture loads them by path instead. Importing them is side-effect
free -- each guards its work behind ``if __name__ == "__main__"``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

# make ``import panelspeak`` work without an install
sys.path.insert(0, str(REPO_ROOT))

_SCRIPT_FILES = {
    "extract": "01_extract_and_segment.py",
    "vision": "02_vision_analyze.py",
    "narrative": "03_narrative.py",
    "tts": "04_tts_render.py",
    "tts_kokoro": "04_tts_render_kokoro.py",
}


def _load(name: str) -> ModuleType:
    path = SCRIPTS_DIR / _SCRIPT_FILES[name]
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def script():
    """``script("narrative")`` -> the loaded 03_narrative module."""
    cache: dict[str, ModuleType] = {}

    def _get(name: str) -> ModuleType:
        if name not in cache:
            cache[name] = _load(name)
        return cache[name]

    return _get


@pytest.fixture
def make_image(tmp_path):
    """``make_image(w, h)`` -> Path to a real jpg of that size."""
    PIL = pytest.importorskip("PIL.Image")
    counter = {"n": 0}

    def _make(w: int, h: int, name: str | None = None) -> Path:
        counter["n"] += 1
        fname = name or f"img-{counter['n']:03d}.jpg"
        p = tmp_path / fname
        PIL.new("RGB", (w, h), (128, 128, 128)).save(p, quality=90)
        return p

    return _make
