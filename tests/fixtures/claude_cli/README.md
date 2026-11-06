# Claude Code CLI stream-json fixtures

Transcripts of `claude -p --output-format stream-json --verbose` runs, used by
`tests/test_claude_cli.py`. They were captured from Claude Code 2.1.318 under
the lockdown profiles in `scripts/ingest_lib/claude_cli.py`.

Every fixture was scrubbed the same way: `cwd`, `original_cwd` and
`messaging_socket_path` read `/tmp/x`, every `session_id` and `uuid` is one
fixed UUID, every message `id` is `msg_00000000000000000000000000` and every
tool-use `id` and `tool_use_id` is `toolu_00000000000000000000000000`, and
the account-linked keys `credential_expires_at`, `request_id`
and `rate_limit_info` were removed. No key was added or retyped, except the
single change named below for each derived fixture.

## Captured

| Fixture | Source |
|---|---|
| `text_ok.jsonl` | text profile run on Sonnet, every line kept |
| `schema_ok.jsonl` | schema profile run with the `DocSummary` schema, every line kept |
| `image_ok.jsonl` | image profile run reading `./image.png`, every line kept |

## Captured result, init line borrowed

These results were captured with `--output-format json`, which emits no init
event, so each carries the init line of the matching profile.

| Fixture | Init line from | Result |
|---|---|---|
| `is_error_model.jsonl` | `text_ok` | a run with a model id that does not exist |
| `not_logged_in.jsonl` | `text_ok` | a run with no login |
| `schema_gave_up.jsonl` | `schema_ok` | a run whose schema no value can satisfy |
| `budget.jsonl` | `text_ok` | a run stopped by `--max-budget-usd` |

## Derived (no real capture)

| Fixture | Based on | Single change |
|---|---|---|
| `usage_limit.jsonl` | `is_error_model` | `result` reads `You've hit your session limit · resets 3:45pm` |
| `throttle.jsonl` | `is_error_model` | `result` reads `API Error: Server is temporarily limiting requests (not your usage limit)` |
| `prompt_too_long.jsonl` | `is_error_model` | `result` reads `Prompt is too long`, `terminal_reason` is `prompt_too_long` |
| `max_tokens.jsonl` | `text_ok` | the result event's `stop_reason` is `max_tokens` |
| `extra_tool.jsonl` | `text_ok` | `Bash` added to the init event's `tools` |
| `unexpected_tool_use.jsonl` | `text_ok` | one extra assistant event holding a `Bash` `tool_use` |
| `image_read_other.jsonl` | `image_ok` | the `Read` `file_path` is `../canary.txt` |
