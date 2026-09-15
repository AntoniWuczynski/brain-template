# Deploying the brain MCP server

This is the recipe for running `mcp_server/` on a Linux box you control, as **user** systemd units (no `sudo`, no dedicated system account), reachable over your own Tailscale network. About 30 minutes end to end.

Three decisions this doc assumes:

- **Ingestion stays on the Mac (or wherever your working clone lives) — always, not just at first.** MinerU, faster-whisper, the vision extractors and their multi-GB weights never run on the server. `pull.py`/`ingest.py` keep running there against the Obsidian working clone; the server only ever *serves* the vault (the 18 MCP tools) and runs the maintenance/dream passes over what's already there. That is the whole reason the install below needs no ML stack.
- **One unit set, user-scoped.** All three scheduled units (`brain-mcp`, `brain-maintenance`, `brain-dream`) are `systemd --user` units under `~/services/brain` — a plain user account, no `sudo`, no `/srv`, no `/etc/brain-mcp`, no dedicated system user.
- **No extraction stack on the server, but embeddings YES.** MinerU, whisper and the vision extractors stay on the Mac with ingestion. The embedding model does run here, because `memory_search` and the dense half of `vault_search` are the server's whole job. The lockfile's Linux `torch` is the CUDA build (several GB of `nvidia-*` libraries for a box with no GPU), so the default install below excludes those and adds a CPU-only torch; the fully locked alternative and the lexical-only fallback are spelled out beside it (AUD-125 tracks locking the CPU build).

## Prerequisites

- A Linux box you control, with a normal user account and `systemd --user` support (any modern distro; the commands below use Debian/Ubuntu `apt` syntax where one is needed — adjust for yours).
- `git` and [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- [Tailscale](https://tailscale.com/) installed and logged into the same account as your other devices, so the server has a stable tailnet hostname. (A public hostname for the claude.ai connector is a separate, optional step — see the very end of this doc.)
- Write access to the private `brain` GitHub repo, to add a deploy key.

## 1. Clone the repo and install Python 3.12 (no ML stack)

```bash
git clone git@github.com:<you>/brain.git ~/services/brain
cd ~/services/brain
```

`pyproject.toml` pins `requires-python = ">=3.12,<3.13"` (MinerU's PaddlePaddle has no 3.13/3.14 wheels yet) — irrelevant to *running* MinerU here since it never runs here, but the pin is enforced at resolve time regardless, and your box may well ship a newer system Python than that (3.14 is common on current distro images). Get a matching interpreter without touching the system Python:

```bash
uv python install 3.12
```

Install the locked environment **without** the CUDA stack, then add a CPU-only torch and
the embedding package — this is the default, because the server serves search:

```bash
# Run this in bash. $EXCLUDES relies on word-splitting an unquoted
# variable into multiple arguments — zsh (macOS's default shell) doesn't
# do that by default, and passes the whole string as one argument instead.
EXCLUDES=$(grep -v '^\s*#' .github/workflows/ci.yml | grep -oE -- '--no-install-package [^ \\]+' | tr '\n' ' ')
uv sync --locked $EXCLUDES
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install sentence-transformers
```

The exclusion list is extracted from the CI workflow rather than hand-copied, so it cannot
drift from what CI tests. It removes `torch`, `sentence-transformers` and every CUDA/`triton`
package the lockfile carries (`nvidia-*`, `cuda-*`, reachable only through the CUDA torch
build). The two `uv pip install` lines put back a CPU torch and the embedder. Those two are
**not** locked or hashed — the one gap in an otherwise locked install, and the reason AUD-125
exists (pin a CPU torch for Linux in `uv.lock` so this collapses to one locked command).

Two alternatives, both honest:

- **Fully locked, heavy.** Plain `uv sync --locked` installs the lockfile as is: the CUDA torch
  and its NVIDIA libraries, several GB, unused on a GPU-less box, but every byte hashed. Works.
- **Lexical-only, minimal.** The `uv sync --locked $EXCLUDES` line alone. `vault_search` and
  `memory_search` still answer, but with no embedder the dense half of hybrid search cannot
  load, so every search silently degrades to BM25 (`semantic.py:search()` does this by design,
  and the boot log line from AUD-124 says so once at start). Choose this only for a box that
  will never serve search.

Smoke-test the import before wiring any unit:

```bash
uv run --no-sync python -c "import mcp_server.app; print('ok')"
```

(`python -m mcp_server` itself won't help here — it reads `BRAIN_MCP_VAULT_ROOT` and the rest
of `.env` before importing `app.py`, so it fails on missing config before it would ever hit a
missing package.)

First `vault_search` after that downloads the ~90 MB embedding model into `HF_HOME` — pointed at the service dir by the unit file (step 4), not the default `~/.cache`, so a `ProtectHome`-sandboxed run and a stray `$HOME` can't send it somewhere the service can't read back.

## 2. Generate a deploy key — outside the vault

The server commits and pushes on every write, so it needs its own SSH key. Keep it **outside** `~/services/brain`: `mcp_server/config.py`'s read tools already deny any file literally named `id_rsa`/`id_ed25519` or `.env`/`.env.local` by name, and separately allow only `knowledge/`, `archive/`, `inbox/`, `metadata/` and a few named root docs by path — but defence in depth costs nothing here, and a key that never lives under the vault root can't be reached by a future bug in either check.

```bash
mkdir -p ~/.ssh
ssh-keygen -t ed25519 -f ~/.ssh/brain-mcp-deploy -N "" -C "brain-mcp deploy key ($(hostname))"
ssh-keyscan github.com >> ~/.ssh/known_hosts   # pin GitHub's host key; verify the fingerprint against
                                                # https://docs.github.com/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints
cat ~/.ssh/brain-mcp-deploy.pub
```

In the private `brain` repo: **Settings → Deploy keys → Add deploy key**. Paste the public key, **check "Allow write access"**, save.

Test it as yourself (the unit will run as you too — no `sudo -u` needed):

```bash
GIT_SSH_COMMAND="ssh -i ~/.ssh/brain-mcp-deploy -o IdentitiesOnly=yes -o UserKnownHostsFile=~/.ssh/known_hosts -o StrictHostKeyChecking=yes" \
  git -C ~/services/brain push origin main
```

Should succeed (or say "everything up-to-date") without prompting.

## 3. Write the service's `.env`

`.env.example` at the repo root carries the template for this, below its `MCP server (mcp_server/)` divider — copy just that file to the service directory, not the repo's own `.env` (they're for two different processes: yours locally loads the connector-config half via python-dotenv; the server reads its own `.env` straight from the environment through `EnvironmentFile=`):

```bash
cp .env.example ~/services/brain/.env
chmod 600 ~/services/brain/.env
```

Fill in every key under the `MCP server` divider:

| Key | Value here |
|---|---|
| `BRAIN_MCP_VAULT_ROOT` | `/home/<you>/services/brain` (absolute — `~` does not expand in a systemd `EnvironmentFile`) |
| `BRAIN_MCP_TOKENS` | `claude-code=<token>,codex=<token>,claude-ai=<token>` — one fresh `openssl rand -hex 32` per client, never reused from the Mac's old tokens (see step 7) |
| `BRAIN_MCP_BIND_HOST` / `_PORT` | `127.0.0.1` / `8765` — loopback only; Tailscale/a tunnel is what makes it reachable, not a public bind |
| `BRAIN_MCP_ALLOWED_HOSTS` | your tailnet MagicDNS hostname, e.g. `myhost.tailXXXX.ts.net` (no port — see step 5 on why) — **required**: `TrustedHostMiddleware` rejects any request whose `Host` header isn't loopback or in this list, so a request that reaches the box over the tailnet is rejected at the guard, not your code, until this is set |
| `BRAIN_MCP_GIT_PUSH_ON_WRITE` | `1` |
| `BRAIN_MCP_GIT_REMOTE` / `_BRANCH` | `origin` / `main` — `_BRANCH` must equal the branch actually checked out here, or every write's commit step refuses (`committed: false`, note still written to disk) |
| `GIT_SSH_COMMAND` | points at the step-2 key (the commented-out line in `.env.example` is ready to uncomment) |
| `BRAIN_MCP_LOG_LEVEL` | `info` |
| `HF_HOME` | set by the unit file unconditionally (step 4) — leave commented out here unless running without systemd |

## 4. Install the units (AUD-123)

```bash
mkdir -p ~/.config/systemd/user
cp mcp_server/systemd/brain-mcp.service \
   mcp_server/systemd/brain-maintenance.service mcp_server/systemd/brain-maintenance.timer \
   mcp_server/systemd/brain-dream.service mcp_server/systemd/brain-dream.timer \
   ~/.config/systemd/user/
systemctl --user daemon-reload
```

So the units keep running after you log out of the SSH session (user units otherwise stop when your last session ends):

```bash
loginctl enable-linger "$(whoami)"
```

Start everything:

```bash
systemctl --user enable --now brain-mcp.service brain-maintenance.timer brain-dream.timer
systemctl --user status brain-mcp
```

`brain-dream.service` runs the `claude` CLI itself (subscription auth — see `scripts/dream.sh`), which needs your normal `~/.claude` config and credentials, so — unlike `brain-mcp.service` — neither it nor `brain-maintenance.service` is filesystem-sandboxed with `ProtectHome`/`ProtectSystem`; see the comments in those two unit files for why that line was drawn where it was. `brain-mcp.service` doesn't need that latitude (its I/O surface is just the vault, the git push, and — with the optional ML stack — a model load), so it keeps the tighter hardening.

`active (running)` means you're past the local part:

```bash
curl -s http://127.0.0.1:8765/health
# {"status":"ok"}
journalctl --user -u brain-mcp -n 20   # confirm the boot line: "brain MCP server starting (vault=..., ...)"
                                        # and the extractor-availability line right after it (step 1's degrade, spelled out)
```

## 5. Tailscale

```bash
sudo tailscale up   # once per machine; prints a URL to authenticate in a browser
```

`BRAIN_MCP_BIND_HOST=127.0.0.1` (step 3) means the server is reachable from this box alone by default — Tailscale being up does not, on its own, expose a loopback-bound port to the tailnet. Use [`tailscale serve`](https://tailscale.com/kb/1242/tailscale-serve) to proxy it, rather than rebinding to `0.0.0.0`, which would keep the security posture "reachable over the tailnet only" while widening it to "reachable from anywhere the box's network stack sees":

```bash
tailscale serve --bg https / http://127.0.0.1:8765
```

This needs HTTPS certificates enabled once for the tailnet (Tailscale admin console → **DNS** → **HTTPS Certificates**) — `tailscale serve` then issues and renews the cert itself. The server becomes reachable at `https://<this box's MagicDNS name>/` (port 443, no `:8765` — `tailscale serve` is the thing listening on 443 and forwarding to your local port). Confirm the exact name:

```bash
tailscale status   # this box's MagicDNS name, e.g. myhost.tailXXXX.ts.net
```

Confirm `BRAIN_MCP_ALLOWED_HOSTS` in `.env` already has that hostname (with no port — the client's `Host` header carries `myhost.tailXXXX.ts.net`, since `tailscale serve` terminates TLS at 443), then restart if you changed it after step 4:

```bash
systemctl --user restart brain-mcp
```

## 6. Post-cutover probe (scripts/mcp_probe.py)

Run this from the Mac (or anywhere with `uv`/Python and network access to the tailnet) after cutover, and again after any token rotation. It checks `/health`, runs one full MCP `initialize` → `tools/list` handshake, and confirms every given token authenticates — exiting non-zero if anything is wrong, so it's the one command to point at "is the server actually OK":

```bash
uv run --no-sync python scripts/mcp_probe.py \
    --url https://myhost.tailXXXX.ts.net \
    claude-code=<token> codex=<token> claude-ai=<token>
```

```
mcp-probe: [ok] health: 200
mcp-probe: [ok] mcp_initialize: 200
mcp-probe: [ok] tools_list: 18 tools
mcp-probe: [ok] token:claude-code: 200
mcp-probe: [ok] token:codex: 200
mcp-probe: [ok] token:claude-ai: 200
mcp-probe: server exposes 18 tool(s)
```

A `[FAIL]` line names exactly which check broke rather than hiding it behind an overall failure — a single bad token doesn't hide a genuinely down server, and vice versa.

## 7. Point the Mac's clients at the server, and rotate tokens

**Rotate every token first.** Generate three fresh ones (`openssl rand -hex 32` each) and put them in the server's `.env` (step 3) — never copy the Mac's existing launchd-era tokens across; they've been sitting in plaintext in a running config for a while and this cutover is the natural moment to retire them.

Claude Code (`~/.claude.json` on the Mac), the `brain` MCP server entry:

```json
{
  "mcpServers": {
    "brain": {
      "url": "https://myhost.tailXXXX.ts.net/mcp",
      "headers": { "Authorization": "Bearer <claude-code token>" }
    }
  }
}
```

Codex, the equivalent `http_headers` map in its own MCP server config:

```toml
[mcp_servers.brain]
url = "https://myhost.tailXXXX.ts.net/mcp"
http_headers = { Authorization = "Bearer <codex token>" }
```

(Exact key names depend on your Codex config version — check its current MCP-server block; the shape is the same `url` + bearer header regardless.)

Re-run the probe (step 6) with the new tokens before removing the old ones from anywhere, then re-run it once more from the Mac after editing the configs above, to confirm the clients themselves — not just curl — can reach the server.

## 8. Retire the Mac's launchd jobs

Only once the server has been serving for a few days without incident. The corrected Mac-side unit lives at `mcp_server/launchd/com.brain.mcp.plist` — the server side of this deploy does not touch it; this step is entirely on the Mac:

```bash
launchctl bootout gui/$(id -u)/com.brain.mcp
launchctl bootout gui/$(id -u)/com.brain.dream
launchctl bootout gui/$(id -u)/com.brain.maintenance
```

**`~/brain` on the Mac stays** as the Obsidian working clone — you edit notes there by hand, `git pull` picks up the server's commits, and anything you push is absorbed by the server's pull-before-write on its next tool call (AUD-122). Ingestion (`scripts/ingest.py`, `scripts/pull.py`) also keeps running from this same Mac clone — that's the whole point of IMP-028: the Mac is the only machine with the sources (Granola/justREC exports, `~/.claude/projects` transcripts, the MinerU weights), so it stays the only machine that ingests, indefinitely, not just until the server is stable.

## 9. Updating the server later

```bash
cd ~/services/brain
git pull origin main
# Run this in bash. $EXCLUDES relies on word-splitting an unquoted
# variable into multiple arguments — zsh (macOS's default shell) doesn't
# do that by default, and passes the whole string as one argument instead.
EXCLUDES=$(grep -v '^\s*#' .github/workflows/ci.yml | grep -oE -- '--no-install-package [^ \\]+' | tr '\n' ' ')
uv sync --locked $EXCLUDES
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # a bare sync prunes the two unlocked packages
uv pip install sentence-transformers
systemctl --user restart brain-mcp
```

Re-copy any unit file that changed (`mcp_server/systemd/*`) to `~/.config/systemd/user/` and `daemon-reload` first if the update touched one.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `systemctl --user start brain-mcp` fails with "BRAIN_MCP_VAULT_ROOT must be set" | `.env` wasn't read; check the `EnvironmentFile=` path resolves — `%h` expands to your home directory, so it must be `~/services/brain/.env` exactly |
| `auth: rejected request from <ip>` in `journalctl --user -u brain-mcp` | client and server tokens differ; rotate one to match the other, then re-run the probe |
| A request over the tailnet gets no response / a rebind rejection | `BRAIN_MCP_ALLOWED_HOSTS` doesn't list the tailnet hostname the client actually connected with — `tailscale status` shows the exact one |
| Writes commit but never reach GitHub | the async push worker is failing; check `journalctl --user -u brain-mcp` for `push worker:` lines and `logs/mcp-audit.jsonl` for the write outcomes |
| `git push exited 128: Permission denied` in logs after a write | the step-2 deploy key is missing "Allow write access" on GitHub |
| Boot log doesn't mention extractor availability | you're looking at an old build — the line was added under AUD-124; `git pull` and restart |
| `mcp_probe.py` reports `mcp_initialize: status 0, request failed: ...` | the URL is unreachable from where you're running the probe — check Tailscale is up on both ends, and that you used the tailnet hostname, not `127.0.0.1` |
| Health works but `/mcp` returns 404 | client URL is `/mcp`, not `/mcp/mcp` — a double-prefix bug in the client config |

Logs:

```bash
journalctl --user -u brain-mcp -f
journalctl --user -u brain-maintenance -f
journalctl --user -u brain-dream -f
```

## Optional: a public hostname, for claude.ai only

Skip this section entirely if only your own devices (over Tailscale) need to reach the server — that's the complete, done state as of step 8.

claude.ai runs in Anthropic's cloud, not on a device of yours, so it cannot reach the tailnet; reaching it needs a public hostname on a domain you control, via [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/):

```bash
cloudflared tunnel login
cloudflared tunnel create brain-mcp
```

This prints a tunnel UUID and writes credentials to `~/.cloudflared/<UUID>.json`. Write the config (`~/.cloudflared/config.yml`):

```yaml
tunnel: <UUID>
credentials-file: /home/<you>/.cloudflared/<UUID>.json
ingress:
  - hostname: brain.yourdomain.example
    service: http://127.0.0.1:8765
  - service: http_status:404
```

```bash
cloudflared tunnel route dns brain-mcp brain.yourdomain.example
cloudflared service install   # needs sudo: installs a SYSTEM unit for cloudflared itself
                              # (unlike brain-mcp, which stays a user unit)
```

Add the public hostname to `BRAIN_MCP_ALLOWED_HOSTS` in `.env` (alongside the tailnet one, not instead of it) and restart `brain-mcp`. Verify: `curl -s https://brain.yourdomain.example/health`.

**Two independent auth layers, deliberately not Access service tokens:**

1. A Cloudflare Access policy (**Zero Trust → Access → Applications**) restricting the hostname's `/mcp*` path to Anthropic's published egress range, `160.79.104.0/21` ([anthropic's IP list](https://platform.claude.com/docs/en/api/ip-addresses)) — leave `/health` outside the policy so it stays a plain liveness check.
2. This server's own `claude-ai` bearer token (step 3's `BRAIN_MCP_TOKENS`), in the `Authorization` header claude.ai sends via its `static_headers` connector config.

Access **service tokens** are the wrong tool here: claude.ai's `static_headers` support accepts the `authorization`/`x-api-key` header names without review but needs Anthropic's approval for any other header, so `CF-Access-Client-Id`/`CF-Access-Client-Secret` are out — and Cloudflare's single-header fallback (`read_service_tokens_from_header: Authorization`) would then consume the very header this server's own bearer token needs. The IP-range policy avoids the collision entirely and keeps both layers independent. (`static_headers` is in beta on claude.ai as of this writing — confirm it's available on your account before relying on this path.)

## What this deploy does NOT do

- **No automated ingestion, ever — not just until the server is stable.** PDFs dropped via `vault_drop_inbox_file` land in `inbox/`; ingestion happens on the Mac, where MinerU's weights live, when you run `scripts/ingest.py --inbox`. Per IMP-028 this is a standing property of the design, not a temporary gap — see step 8.
- **Embeddings are on by default; the CPU torch and embedder are the one unlocked part** of the install (see step 1). Lexical-only is a deliberate, documented downgrade, not the default.
- **No remote summarisation.** If you want LLM summaries generated on this box (so ingestion — were it ever to run here — wouldn't need an internet API key), wire the `local` provider against an Ollama instance running on it. See the root `README.md` → "Requirements" for the provider list.
- **No multi-user.** `BRAIN_MCP_TOKENS` gives each *agent* its own identity over the shared vault (attributed commits + audit lines), but every token has the same full read/write access — this is one person's vault, a single trust domain. If multiple humans need different identities or permissions, that's a Cloudflare Access / Tailscale ACL policy question, not something per-agent tokens solve.
