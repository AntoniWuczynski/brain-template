"""Run the embedding model in a child process that exits when idle.

A long-running process (the MCP server) that loads sentence-transformers
keeps torch and the weights resident for its whole life: dropping the model
and running ``gc.collect()`` returns nothing to the OS (measured: 769 MB
before and after on CPU). The only way to give that memory back is for the
process holding it to exit. ``SubprocessEmbedder`` keeps the model in a
spawned child, forwards ``encode`` calls over a pipe, and stops the child
after ``idle_s`` seconds without a call. The next call respawns it, paying
one cold load (about 5 s on CPU for bge-small).

The parent never imports torch, transformers or sentence-transformers on
this path. The optional reranker (``BRAIN_RANK_RERANK=1``, off by default)
still loads a CrossEncoder in-process, which keeps torch resident.

Trap: the child is *spawned*, so it re-imports the parent's main script.
A script run by path that triggers an encode at import time (no
``if __name__ == "__main__":`` guard) fails to start the child. The server
(``python -m mcp_server``) is not affected: spawn skips ``__main__`` modules.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import multiprocessing as mp
import os
import signal
import threading
import time
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from multiprocessing.process import BaseProcess

log = logging.getLogger(__name__)

DEFAULT_FACTORY = "sentence_transformers:SentenceTransformer"
# How long a cold start may take before the parent gives up: weights load
# plus torch init measured 4.5-5.9 s on CPU; a first-ever run also downloads
# the weights.
_READY_TIMEOUT_S = 300.0
_STOP_GRACE_S = 5.0
_MAX_REAP_INTERVAL_S = 30.0
# Sentences per locked round trip. A full rebuild encodes thousands of chunks
# in one call (minutes on CPU); slicing releases the lock between slices so
# a search can get in. The lock is not fair, so a search can still wait a few
# seconds behind a rebuild (4.5 s measured during a 10 s one), not minutes.
# sentence-transformers length-sorts within a call, so slicing changes padding
# and the floats move in the last digits only.
_SLICE = 256


def _resolve(factory: str) -> Any:
    module_name, _, attr = factory.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def factory_available(factory: str) -> bool:
    """Is the factory's module installed? Checked in the parent without
    importing it, so a deploy without the ML extras fails fast with an
    ImportError instead of spawning a child per search to find out."""
    return importlib.util.find_spec(factory.partition(":")[0]) is not None


def _worker_main(conn: Connection, factory: str, model_name: str, device: str | None) -> None:
    """Child entry point: load the model, then answer encode requests until
    told to stop or the parent's end of the pipe closes."""
    import numpy as np

    from .semantic import _autodetect_device  # imports torch: child only

    # Ctrl-C reaches the whole process group; the parent owns shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        resolved = device or _autodetect_device()
        model = _resolve(factory)(model_name, device=resolved)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent, then exit
        conn.send(("err", f"model load failed: {exc!r}"))
        return
    conn.send(("ready", resolved))
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        sentences, kwargs = msg
        try:
            out = np.asarray(model.encode(sentences, **kwargs), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - the parent raises it
            conn.send(("err", repr(exc)))
        else:
            conn.send(("ok", out))


class SubprocessEmbedder:
    """``Embedder`` whose model lives in a child process stopped when idle."""

    def __init__(
        self,
        model_name: str,
        device: str | None,
        idle_s: float,
        factory: str | None = None,
    ) -> None:
        if idle_s <= 0:
            raise ValueError("idle_s must be positive")
        self._model_name = model_name
        self._requested_device = device
        self._idle_s = idle_s
        self._factory = factory or DEFAULT_FACTORY  # "module:attr", imported in the child
        self._ctx = mp.get_context("spawn")  # torch and threads are not fork-safe
        self._lock = threading.Lock()
        self._proc: BaseProcess | None = None
        self._conn: Connection | None = None
        self._last_used = time.monotonic()
        self.device = device or "auto"
        self._reaper: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def alive(self) -> bool:
        """Is the model process running right now?"""
        proc = self._proc
        return proc is not None and proc.is_alive()

    def ensure_started(self) -> None:
        """Start the child if it is not running and wait until the model is
        loaded. Raises RuntimeError when the load fails, so callers that
        already catch a failed ``_load_embedder`` keep working."""
        with self._lock:
            self._start_locked()

    def close(self) -> None:
        """Stop the model process now; the next call starts a new one."""
        with self._lock:
            self._stop_locked()

    def _start_locked(self) -> None:
        if self.alive and self._conn is not None:
            # Counts as use: the reaper must not stop the child between
            # _load_embedder's ensure_started() and the caller's encode.
            self._last_used = time.monotonic()
            return
        self._stop_locked()
        parent_conn, child_conn = self._ctx.Pipe()
        proc = self._ctx.Process(
            target=_worker_main,
            args=(child_conn, self._factory, self._model_name, self._requested_device),
            name="brain-embedder",
            daemon=True,  # never outlive the server
        )
        try:
            proc.start()
        except BaseException:
            parent_conn.close()
            child_conn.close()
            raise
        child_conn.close()
        self._proc, self._conn = proc, parent_conn
        try:
            if not parent_conn.poll(_READY_TIMEOUT_S):
                raise RuntimeError(f"embedder did not start within {_READY_TIMEOUT_S:.0f} s")
            status, payload = parent_conn.recv()
        except (EOFError, OSError) as exc:
            self._stop_locked()
            raise RuntimeError(f"embedder process died while loading: {exc!r}") from exc
        except BaseException:
            # Includes KeyboardInterrupt mid-handshake: an unread "ready" left
            # in the pipe would shift every later reply by one call.
            self._stop_locked()
            raise
        if status != "ready":
            self._stop_locked()
            raise RuntimeError(str(payload))
        self.device = str(payload)
        self._last_used = time.monotonic()
        log.info("embedder: model %s loaded in child pid %s on %s",
                 self._model_name, proc.pid, self.device)
        self._start_reaper_locked()

    def _stop_locked(self) -> None:
        proc, conn = self._proc, self._conn
        self._proc = self._conn = None
        if conn is not None:
            try:
                conn.send(None)
            except (OSError, ValueError):
                pass
            conn.close()
        if proc is not None:
            proc.join(_STOP_GRACE_S)
            if proc.is_alive():
                proc.terminate()
                proc.join(_STOP_GRACE_S)
            if proc.is_alive():
                proc.kill()
                proc.join(_STOP_GRACE_S)

    # -- idle reaping --------------------------------------------------------

    def _start_reaper_locked(self) -> None:
        if self._reaper is not None and self._reaper.is_alive():
            return
        self._reaper = threading.Thread(
            target=self._reap_loop, name="brain-embedder-reaper", daemon=True
        )
        self._reaper.start()

    def _reap_loop(self) -> None:
        while True:
            # Wake when the idle window would end, so a child lives about
            # idle_s after its last call (a fixed interval allowed up to 2x).
            remaining = self._idle_s - (time.monotonic() - self._last_used)
            time.sleep(min(max(remaining, 0.05), _MAX_REAP_INTERVAL_S))
            # Non-blocking: an encode in progress is not idle.
            if not self._lock.acquire(blocking=False):
                # Busy means in use; look again a full window later.
                time.sleep(min(self._idle_s, _MAX_REAP_INTERVAL_S))
                continue
            try:
                if self._proc is None:
                    self._reaper = None
                    return
                if time.monotonic() - self._last_used >= self._idle_s:
                    pid = self._proc.pid
                    self._stop_locked()
                    log.info("embedder: stopped idle child pid %s after %.0f s",
                             pid, self._idle_s)
                    self._reaper = None
                    return
            finally:
                self._lock.release()

    # -- Embedder ------------------------------------------------------------

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        """``SentenceTransformer.encode`` in the child, one locked round trip
        per ``_SLICE`` sentences, rows returned in input order."""
        import numpy as np

        items = list(sentences)
        parts = [self._encode_slice(items[i:i + _SLICE], kwargs)
                 for i in range(0, len(items), _SLICE)] or [self._encode_slice([], kwargs)]
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def _encode_slice(self, sentences: list[str], kwargs: dict[str, Any]) -> np.ndarray:
        with self._lock:
            self._start_locked()
            conn = self._conn
            assert conn is not None
            try:
                conn.send((sentences, kwargs))
                status, payload = conn.recv()
            except (EOFError, OSError) as exc:
                self._stop_locked()
                raise RuntimeError(f"embedder process died: {exc!r}") from exc
            except BaseException:
                # Interrupted between send and recv: the reply may still be in
                # the pipe, and the next caller would read it as its own.
                self._stop_locked()
                raise
            finally:
                self._last_used = time.monotonic()
        if status != "ok":
            raise RuntimeError(f"embedder encode failed: {payload}")
        return payload  # type: ignore[no-any-return]


def idle_unload_seconds(env: dict[str, str] | None = None) -> float:
    """``BRAIN_EMBED_IDLE_UNLOAD_S`` as a float; 0 (off) when unset or invalid."""
    raw = (os.environ if env is None else env).get("BRAIN_EMBED_IDLE_UNLOAD_S", "")
    try:
        value = float(raw) if raw.strip() else 0.0
    except ValueError:
        log.warning("BRAIN_EMBED_IDLE_UNLOAD_S=%r is not a number; idle unload off", raw)
        return 0.0
    return max(value, 0.0)
