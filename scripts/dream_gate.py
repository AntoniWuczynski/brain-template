#!/usr/bin/env python3
"""brain dream-pass gate CLI.

Examples:
    uv run python scripts/dream_gate.py                 # gate check: exit 0 dream / 1 skip
    uv run python scripts/dream_gate.py --dry-run       # check without recording the pending marker
    uv run python scripts/dream_gate.py --emit-packet   # print the dream worklist JSON
    uv run python scripts/dream_gate.py --propose f.json  # write the adjudicated contradiction proposals
    uv run python scripts/dream_gate.py --mark-done     # after a successful dream session

The logic lives in ``scripts/ingest_lib/dream.py`` — deterministic git +
metadata arithmetic, no LLM. The LLM session (.claude/skills/dream-pass/)
calls this CLI; it never touches metadata/dream.json itself. Scheduled
runs are logged by scripts/dream.sh into logs/dream-*.log (AGENTS.md
rule 5); manual runs log to stderr.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

# Make ``ingest_lib`` importable when running this file directly
# (i.e. without ``uv run`` having installed the package yet).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest_lib.config import VaultPaths, default_paths  # noqa: E402
from ingest_lib.dream import (  # noqa: E402
    GitError,
    ProposalOutcome,
    build_packet,
    evaluate_gate,
    mark_done,
    parse_adjudicated,
    record_pending,
    write_proposals,
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _parse_as_of(raw: str) -> datetime:
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected ISO 8601 timestamp, got {raw!r}") from exc
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brain-dream-gate",
        description=(
            "Deterministic gate for the LLM dream pass: decide whether enough "
            "new information landed since the last dream, emit the session "
            "worklist, and record run state. No LLM involved."
        ),
        epilog=(
            "Exit codes for the default (check) mode: 0 = dream, 1 = skip, "
            "2 = git error or bad --propose input. --emit-packet, --propose and "
            "--mark-done always exit 0 on success."
        ),
    )
    parser.add_argument("--as-of", type=_parse_as_of, default=None, metavar="ISO8601",
                        help="Reference time (default: now, UTC). For tests and replays.")
    parser.add_argument("--threshold", type=int, default=_env_int("BRAIN_DREAM_THRESHOLD", 5),
                        help="Changes needed to dream (default: 5, env BRAIN_DREAM_THRESHOLD).")
    parser.add_argument("--stale-days", type=int, default=_env_int("BRAIN_DREAM_STALE_DAYS", 7),
                        help="Dream on any change after this many days (default: 7, env BRAIN_DREAM_STALE_DAYS).")
    parser.add_argument("--pairs", type=int, default=_env_int("BRAIN_DREAM_PAIRS", 10),
                        help="Candidate connection pairs in the packet (default: 10, env BRAIN_DREAM_PAIRS).")
    parser.add_argument("--emit-packet", action="store_true",
                        help="Print the dream worklist JSON (regardless of the gate verdict).")
    parser.add_argument("--propose", metavar="JSON", default=None,
                        help=(
                            "Write the adjudicated contradiction proposals in this JSON "
                            "file (or '-' for stdin) into knowledge/assistant/inbox/. "
                            "The array's entries are the packet's own contradiction "
                            "findings plus a 'title' and 'body'."
                        ))
    parser.add_argument("--mark-done", action="store_true",
                        help="Record a completed dream: advance metadata/dream.json to HEAD.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check without recording the pending marker.")
    return parser


def _run_propose(
    source: str, paths: VaultPaths, *, now: datetime, logger: logging.Logger
) -> list[ProposalOutcome]:
    """Write the LLM-adjudicated contradiction survivors as fact notes.

    The dream session adjudicates, this writes: every proposal goes through
    ``ingest_lib.propose.propose_fact``, so the memory-fact contract and the
    ``written_via: script`` provenance have exactly one owner, and re-running
    over the same findings is a no-op (the inbox filename is a hash of the
    proposal's content).
    """
    raw_text = (
        sys.stdin.read() if source == "-"
        else Path(source).read_text(encoding="utf-8")
    )
    try:
        payload: object = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--propose input is not valid JSON: {exc}") from None
    findings = parse_adjudicated(payload)
    return write_proposals(paths, findings, now=now, logger=logger)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    logger = logging.getLogger("brain.dream")
    paths = default_paths()
    as_of = args.as_of if args.as_of is not None else datetime.now(tz=UTC)
    try:
        if args.propose is not None:
            outcomes = _run_propose(args.propose, paths, now=as_of, logger=logger)
            print(json.dumps(outcomes, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.mark_done:
            state = mark_done(paths, now=as_of, logger=logger)
            print(json.dumps(asdict(state), ensure_ascii=False, sort_keys=True))
            return 0
        verdict = evaluate_gate(
            paths, logger=logger, as_of=as_of,
            threshold=args.threshold, stale_days=args.stale_days,
        )
        if args.emit_packet:
            packet = build_packet(paths, verdict=verdict, top_k=args.pairs, logger=logger)
            print(json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if verdict.should_dream and not args.dry_run:
            record_pending(paths, now=as_of)
        print(json.dumps(asdict(verdict), ensure_ascii=False, sort_keys=True))
        return 0 if verdict.should_dream else 1
    except GitError as exc:
        logger.error("%s", exc)
        return 2
    except (ValueError, OSError) as exc:
        # --propose input problems: a malformed file, a finding missing a
        # field, an edited fact. A clean message, not a traceback.
        logger.error("%s", exc)
        return 2
    except Exception:  # noqa: BLE001 — a crash must exit 2, not read as "skip"
        logger.exception("dream gate failed unexpectedly")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
