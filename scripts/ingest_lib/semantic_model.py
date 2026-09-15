"""The canonical embedding-model name (AUD-121 split).

Kept in its own module, rather than defined once in ``semantic.py`` and
re-exported, because ``_embed_model_tag``/``_embed_text``/``_meta_row`` (in
``semantic_meta.py``) need the SAME name — and a sibling module may not
import ``semantic.py`` itself (see that module's docstring on why: nearly
everything else it defines is reached through a direct
``monkeypatch.setattr(semantic, "_name", ...)`` in the test suite, which
only ever patches ``semantic.__dict__`` and so only affects code literally
defined in that file). ``_MODEL_NAME`` carries no such coupling — nothing
monkeypatches ``semantic_meta``'s or ``semantic_chunking``'s copy of it — so
both ``semantic.py`` and its siblings import it from here instead.
"""
from __future__ import annotations

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
