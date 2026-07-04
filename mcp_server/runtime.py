"""Long-lived server collaborators, bundled for the tool layer.

``ServerConfig`` is pure data read from the environment; the objects
here hold THREADS and open resources (audit appender, async push
worker, background index refresher). Keeping them in one frozen bundle
means every tool function takes ``(cfg, runtime, ...)`` instead of
growing a parameter per collaborator, and tests can assemble a Runtime
over a throwaway vault with the workers disabled.
"""
from __future__ import annotations

from dataclasses import dataclass

from .audit import AuditLog
from .config import ServerConfig
from .push_queue import PushWorker
from .reindex import IndexRefresher


@dataclass(frozen=True)
class Runtime:
    """Everything stateful a tool call needs beyond config."""

    audit: AuditLog
    push_worker: PushWorker
    refresher: IndexRefresher


def build_runtime(cfg: ServerConfig) -> Runtime:
    """Wire the production runtime. The refresher pushes through the push
    worker so its derived-notes commits ride the same coalescing queue
    as the write tools' commits."""
    audit = AuditLog(cfg.vault_root)
    push_worker = PushWorker(
        cfg.vault_root,
        remote=cfg.git_remote,
        branch=cfg.git_branch,
        enabled=cfg.git_push_on_write,
    )
    refresher = IndexRefresher(
        cfg.vault_root,
        audit=audit,
        request_push=push_worker.request_push,
    )
    return Runtime(audit=audit, push_worker=push_worker, refresher=refresher)
