"""Shared pytest bootstrap for the suite.

``ingest_lib`` is an installed package (see pyproject's hatch config) but
``mcp_server`` is not — it lives at the repo root and is only importable
when the root is on ``sys.path``. Stage-A test files carry their own
per-file shims so they run standalone; this conftest makes the same
guarantee for the whole suite regardless of collection order.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
