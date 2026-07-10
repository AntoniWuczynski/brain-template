"""Connector SDK: pull external sources into the vault as archivable snapshots.

A connector is a thin plugin — a ``name`` and a ``pull()`` that yields
:class:`Snapshot`s — over a shared, deterministic runner (idempotent
snapshotting into ``inbox/``, per-connector state, atomic writes). The rest is
the existing ingest pipeline. See ``base.py`` for the contract and
``scripts/pull.py`` for the CLI.

New connectors register here so ``scripts/pull.py <name>`` can find them, and
register their extractor by source-class prefix in
``extractors._SOURCE_CLASS_REGISTRY``. The registry is empty until the first
concrete connector ships.
"""
from __future__ import annotations

from .base import Connector, Snapshot
from .runner import PullStats, run_connector
from .state import ConnectorState, load_state, save_state

# name -> factory. Concrete connectors append themselves here (kept a factory,
# not an instance, so import stays side-effect-free and env is read at run time).
from collections.abc import Callable

CONNECTORS: dict[str, Callable[[], Connector]] = {}

__all__ = [
    "Connector",
    "Snapshot",
    "ConnectorState",
    "PullStats",
    "run_connector",
    "load_state",
    "save_state",
    "CONNECTORS",
]
