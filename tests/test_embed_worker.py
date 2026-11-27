"""SubprocessEmbedder: the model lives in a child process that exits when idle."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ingest_lib import semantic
from ingest_lib.embed_worker import SubprocessEmbedder, idle_unload_seconds

FAKE = "embed_fakes:FakeModel"


@pytest.fixture
def embedder() -> Iterator[SubprocessEmbedder]:
    emb = SubprocessEmbedder("fake-model", "cpu", idle_s=60, factory=FAKE)
    yield emb
    emb.close()


def _wait_until(pred: object, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():  # type: ignore[operator]
            return True
        time.sleep(0.05)
    return False


def test_encode_round_trips_through_the_child_and_forwards_kwargs(
    embedder: SubprocessEmbedder,
) -> None:
    out = embedder.encode(["ab", "abcd"], batch_size=7, normalize_embeddings=True)
    assert out.dtype == np.float32
    assert out.tolist() == [[2.0, 7.0], [4.0, 7.0]]
    assert embedder.alive
    assert embedder.device == "cpu"


def test_idle_child_is_stopped_about_idle_s_after_its_last_call() -> None:
    idle = 4.0
    emb = SubprocessEmbedder("fake-model", "cpu", idle_s=idle, factory=FAKE)
    try:
        emb.encode(["x"])
        time.sleep(idle / 2)
        emb.encode(["x"])  # the last call lands mid-way through the window
        last = time.monotonic()
        time.sleep(idle / 2)
        assert emb.alive, "child stopped before the idle window ended"
        # Upper bound, not just "eventually": a reaper on a fixed interval
        # missed a mid-window call and let the child live 1.5x idle_s here.
        limit = 1.3 * idle - (time.monotonic() - last)
        assert _wait_until(lambda: not emb.alive, timeout=limit), \
            "idle child outlived 1.3x idle_s after its last call"
        assert emb.encode(["xyz"]).tolist() == [[3.0, -1.0]]
        assert emb.alive
    finally:
        emb.close()


class _InterruptFirstRecv:
    """Connection wrapper whose first recv() raises KeyboardInterrupt."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self._armed = True

    def recv(self) -> Any:
        if self._armed:
            self._armed = False
            raise KeyboardInterrupt
        return self._conn.recv()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


def test_interrupt_during_startup_does_not_shift_later_replies(
    embedder: SubprocessEmbedder,
) -> None:
    real = embedder._ctx
    calls = {"n": 0}

    def pipe() -> tuple[Any, Any]:
        parent, child = real.Pipe()
        calls["n"] += 1
        return (_InterruptFirstRecv(parent) if calls["n"] == 1 else parent), child

    embedder._ctx = SimpleNamespace(Pipe=pipe, Process=real.Process)  # type: ignore[assignment]
    with pytest.raises(KeyboardInterrupt):
        embedder.ensure_started()
    # An unread "ready" left in the pipe would make each call return the
    # previous call's vectors.
    assert embedder.encode(["a"]).tolist() == [[1.0, -1.0]]
    assert embedder.encode(["bbb"]).tolist() == [[3.0, -1.0]]


def test_interrupt_mid_encode_does_not_shift_later_replies(
    embedder: SubprocessEmbedder,
) -> None:
    embedder.encode(["warm"])
    embedder._conn = _InterruptFirstRecv(embedder._conn)  # type: ignore[assignment]
    with pytest.raises(KeyboardInterrupt):
        embedder.encode(["a"])
    assert embedder.encode(["bbb"]).tolist() == [[3.0, -1.0]]


def test_close_stops_the_child_and_the_next_call_starts_one(
    embedder: SubprocessEmbedder,
) -> None:
    embedder.encode(["warm"])
    embedder.close()
    assert not embedder.alive
    assert embedder.encode(["ab"]).tolist() == [[2.0, -1.0]]
    assert embedder.alive


def test_the_server_process_never_imports_torch_on_the_idle_unload_path(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    script = tmp_path / "probe.py"
    script.write_text(textwrap.dedent("""
        import os, sys
        os.environ["BRAIN_EMBED_IDLE_UNLOAD_S"] = "60"
        os.environ["BRAIN_EMBED_DEVICE"] = "cpu"
        import ingest_lib.embed_worker as ew
        ew.DEFAULT_FACTORY = "embed_fakes:FakeModel"
        from ingest_lib import semantic

        def main():
            model, _ = semantic._load_embedder()
            semantic.encode_query(model, "hello")
            model.close()
            print("TORCH_LOADED" if "torch" in sys.modules else "TORCH_ABSENT")

        if __name__ == "__main__":
            main()
    """))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    out = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "TORCH_ABSENT"


def test_child_crash_raises_then_recovers(embedder: SubprocessEmbedder) -> None:
    embedder.encode(["warm"])
    with pytest.raises(RuntimeError, match="died"):
        embedder.encode(["__crash__"])
    assert not embedder.alive
    assert embedder.encode(["ok"]).tolist() == [[2.0, -1.0]]


def test_encode_error_propagates_and_keeps_the_child(
    embedder: SubprocessEmbedder,
) -> None:
    embedder.encode(["warm"])
    with pytest.raises(RuntimeError, match="bad input"):
        embedder.encode(["__raise__"])
    assert embedder.alive


def test_load_failure_surfaces_from_ensure_started() -> None:
    emb = SubprocessEmbedder(
        "fake-model", "cpu", idle_s=60, factory="embed_fakes:failing_factory"
    )
    with pytest.raises(RuntimeError, match="weights missing"):
        emb.ensure_started()
    assert not emb.alive


def test_idle_unload_seconds_parses_the_env() -> None:
    assert idle_unload_seconds({}) == 0.0
    assert idle_unload_seconds({"BRAIN_EMBED_IDLE_UNLOAD_S": "600"}) == 600.0
    assert idle_unload_seconds({"BRAIN_EMBED_IDLE_UNLOAD_S": "soon"}) == 0.0
    assert idle_unload_seconds({"BRAIN_EMBED_IDLE_UNLOAD_S": "-5"}) == 0.0


def test_load_embedder_uses_the_child_and_never_loads_the_model_in_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ingest_lib.embed_worker as ew

    monkeypatch.setenv("BRAIN_EMBED_IDLE_UNLOAD_S", "60")
    monkeypatch.setenv("BRAIN_EMBED_DEVICE", "cpu")
    monkeypatch.setattr(ew, "DEFAULT_FACTORY", FAKE)
    monkeypatch.setattr(semantic, "_SUBPROCESS_EMBEDDER", None)
    monkeypatch.setattr(semantic, "_EMBEDDER_CACHE", None)
    # Any in-process load would import this module; make that loud.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    model, device = semantic._load_embedder()
    try:
        assert isinstance(model, SubprocessEmbedder)
        assert device == "cpu"
        assert semantic._EMBEDDER_CACHE is None
        again, _ = semantic._load_embedder()
        assert again is model
        q = semantic.encode_query(model, "hello")
        assert q.shape == (1, 2)
    finally:
        model.close()  # type: ignore[attr-defined]


def test_load_embedder_raises_on_a_failed_child_load_so_search_can_degrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ingest_lib.embed_worker as ew

    monkeypatch.setenv("BRAIN_EMBED_IDLE_UNLOAD_S", "60")
    monkeypatch.setenv("BRAIN_EMBED_DEVICE", "cpu")
    monkeypatch.setattr(ew, "DEFAULT_FACTORY", "embed_fakes:failing_factory")
    monkeypatch.setattr(semantic, "_SUBPROCESS_EMBEDDER", None)
    # semantic.search wraps _load_embedder in try/except and falls back to
    # lexical; that only works if the load failure is raised HERE, not on
    # the first encode outside the try.
    with pytest.raises(RuntimeError, match="weights missing"):
        semantic._load_embedder()


def test_large_inputs_are_sent_in_slices_and_reassembled_in_order(
    monkeypatch: pytest.MonkeyPatch, embedder: SubprocessEmbedder
) -> None:
    import ingest_lib.embed_worker as ew

    monkeypatch.setattr(ew, "_SLICE", 2)
    sent: list[int] = []
    real = embedder._encode_slice

    def spy(sentences: list[str], kwargs: dict[str, object]) -> np.ndarray:
        sent.append(len(sentences))
        return real(sentences, kwargs)

    monkeypatch.setattr(embedder, "_encode_slice", spy)
    out = embedder.encode(["a", "bb", "ccc", "dddd", "eeeee"])
    assert sent == [2, 2, 1]
    assert out[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_missing_model_package_raises_import_error_without_spawning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ingest_lib.embed_worker as ew

    monkeypatch.setenv("BRAIN_EMBED_IDLE_UNLOAD_S", "60")
    monkeypatch.setattr(ew, "DEFAULT_FACTORY", "no_such_module_xyz:Model")
    monkeypatch.setattr(semantic, "_SUBPROCESS_EMBEDDER", None)
    # ImportError (not RuntimeError) keeps build_index's install hint.
    with pytest.raises(ImportError):
        semantic._load_embedder()
    assert semantic._SUBPROCESS_EMBEDDER is None
