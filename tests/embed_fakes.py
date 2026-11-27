"""Fake model factories for tests/test_embed_worker.py.

A separate importable module (not the test file) because the embedder child
is a *spawned* interpreter: it imports the factory by "module:attr" name and
inherits the parent's sys.path, on which pytest has put tests/.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np


class FakeModel:
    """Encodes each sentence as [len(sentence), batch_size kwarg or -1]."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        if "__crash__" in sentences:
            os._exit(3)  # the process dies mid-call, as an OOM kill would
        if "__raise__" in sentences:
            raise ValueError("bad input")
        bs = float(kwargs.get("batch_size", -1))
        return np.array([[float(len(s)), bs] for s in sentences])


def failing_factory(model_name: str, device: str) -> FakeModel:
    raise OSError("weights missing")
