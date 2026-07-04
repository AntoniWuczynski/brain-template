"""Entry point: ``python -m mcp_server``.

Reads config from the environment (see ``mcp_server.config``) and
boots uvicorn against ``mcp_server.app:build_app`` (factory mode).
"""
from __future__ import annotations

import uvicorn

from .config import load_config


def main() -> None:
    cfg = load_config()
    uvicorn.run(
        "mcp_server.app:build_app",
        factory=True,
        host=cfg.bind_host,
        port=cfg.bind_port,
        log_level=cfg.log_level.lower(),
        access_log=False,  # we log auth + tool calls ourselves; access log is noisy
        reload=False,
    )


if __name__ == "__main__":
    main()
