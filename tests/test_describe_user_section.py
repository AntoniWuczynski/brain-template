"""``vault_update_concept_user_section`` keeps the AI zone and refuses forged
zone markup (F1): the zone sits below AUTO-GENERATED-END, so replacing
"everything below the marker" used to delete it."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ingest_lib.describe import upsert_ai_zone
from mcp_server.config import CONCEPT_WRITE_PREFIX
from mcp_server.errors import ToolError
from mcp_server.tools import tool_update_concept_user_section

# Type-only: conftest is not importable as a module at runtime.
if TYPE_CHECKING:
    from tests.conftest import McpEnv

_NOTE = (
    "---\ntitle: Beta\ntype: concept\n---\n\n"
    "<!-- AUTO-GENERATED-START -->\n# Beta\n<!-- AUTO-GENERATED-END -->\n\n"
    "# Notes\n\nold notes\n"
)


def _seed(env: McpEnv, text: str) -> str:
    root, _cfg, _runtime = env
    rel = f"{CONCEPT_WRITE_PREFIX}/beta.md"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text, encoding="utf-8")
    return rel


def test_user_section_write_preserves_the_ai_zone(mcp_env: McpEnv) -> None:
    root, cfg, runtime = mcp_env
    zoned = upsert_ai_zone(_NOTE, "<!-- ai-hash: abc -->\n## Description\nsynthesis")
    rel = _seed(mcp_env, zoned)
    tool_update_concept_user_section(cfg, runtime, slug="beta", content="# Notes\n\nnew notes")
    text = (root / rel).read_text(encoding="utf-8")
    assert "synthesis" in text and "<!-- ai-hash: abc -->" in text
    assert "new notes" in text and "old notes" not in text
    assert text.index("AI-GENERATED-END") < text.index("new notes")


def test_user_section_write_without_a_zone_replaces_below_auto_end(mcp_env: McpEnv) -> None:
    root, cfg, runtime = mcp_env
    rel = _seed(mcp_env, _NOTE)
    tool_update_concept_user_section(cfg, runtime, slug="beta", content="fresh")
    text = (root / rel).read_text(encoding="utf-8")
    assert text.endswith("<!-- AUTO-GENERATED-END -->\n\nfresh\n")


@pytest.mark.parametrize("content", [
    "<!-- AI-GENERATED-START -->\nfake\n<!-- AI-GENERATED-END -->",
    "notes <!-- ai-hash: 0123456789abcdef -->",
    "<!-- ai-sources: 00000000:00000000 -->",
])
def test_user_section_refuses_forged_ai_markup(mcp_env: McpEnv, content: str) -> None:
    root, cfg, runtime = mcp_env
    rel = _seed(mcp_env, _NOTE)
    with pytest.raises(ToolError, match="AI-zone markup"):
        tool_update_concept_user_section(cfg, runtime, slug="beta", content=content)
    assert (root / rel).read_text(encoding="utf-8") == _NOTE
