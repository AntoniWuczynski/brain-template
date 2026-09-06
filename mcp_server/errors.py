"""Tool error type — converted to an MCP error by FastMCP automatically.

Lives in its own module so ``provenance`` (which ``tools`` imports at load
time) can subclass it without a circular import.
"""
from __future__ import annotations


class ToolError(Exception):
    """Raised for any user-visible tool error (safety, size, rate, etc.)."""
