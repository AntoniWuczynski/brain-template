# Deploying the brain MCP server

This is the recipe for putting `mcp_server/` on a Linux box behind a Cloudflare Tunnel so your devices (and, when you want, hosted agents like claude.ai) can reach it. About 30 minutes end to end if you've never used CF Tunnel before.

## Prerequisites

- A Linux server you control (any distro with systemd). The instructions below use Ubuntu/Debian apt syntax; adjust for your distro.
- `git`, `python3.12`, and [`uv`](https://docs.astral.sh/uv/) installed.
- A clone of your private brain repo at `/srv/brain` (you can pick a different path; just be consistent).
- A Cloudflare account with a domain you control (any TLD; you don't need to pay anything beyond domain registration).
- `cloudflared` installed (`curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb && sudo dpkg -i /tmp/cloudflared.deb`).

## 1. Generate a bearer token

The server requires a static bearer token on every request (inner ring of auth — Cloudflare Access is the outer ring). Generate one:

```bash
openssl rand -hex 32
```

Keep the output safe. You'll paste it into the env file in the next step and into every MCP client config that connects.

## 2. Configure the service environment

Create a non-root user and a config dir:

```bash
sudo useradd --system --home /srv/brain --shell /usr/sbin/nologin brain
sudo chown -R brain:brain /srv/brain
sudo install -d -m 750 -o root -g brain /etc/brain-mcp
```

Write the env file (`/etc/brain-mcp/env`):

```ini
# Path to your private brain checkout
BRAIN_MCP_VAULT_ROOT=/srv/brain

# Bearer token from step 1. This single token maps to the agent name
# "default" in commits and audit logs.
BRAIN_MCP_BEARER_TOKEN=PASTE-YOUR-TOKEN-HERE

# Optional: named per-agent tokens instead of (or alongside) the single
# token above. Each agent commits as `mcp(<name>): ...` and is attributed
# in logs/mcp-{audit,access}.jsonl. Names: lowercase slug, max 32 chars;
# tokens: min 24 chars, no '=' or ','. A duplicate token or a second
# "default" is refused at startup.
#BRAIN_MCP_TOKENS=claude-code=PASTE-TOKEN-1,codex=PASTE-TOKEN-2

# Optional: byte budget for knowledge/assistant/PROFILE.md writes via the
# profile_update tool (default 4096).
#BRAIN_PROFILE_MAX_BYTES=4096

# Bind to localhost ONLY. The server trusts the CF-Connecting-IP header
# for logging and assumes the only way in is the Cloudflare Tunnel. Do not
# bind 0.0.0.0 without the tunnel + Access in front, or that trust (and the
# single-bearer model) is exposed to the whole network.
BRAIN_MCP_BIND_HOST=127.0.0.1
BRAIN_MCP_BIND_PORT=8765

# Push writes back to origin/main so your laptop sees them on `git pull`.
# The commit is synchronous; the push runs on a background worker and
# retries failures with capped backoff (30s -> 5min). A failed push never
# fails the write — check the server log and logs/mcp-audit.jsonl.
BRAIN_MCP_GIT_PUSH_ON_WRITE=1
BRAIN_MCP_GIT_REMOTE=origin
BRAIN_MCP_GIT_BRANCH=main

# SSH key for git push. Kept OUTSIDE the vault (see step 3) so a
# compromised agent can never read it through the read tools.
GIT_SSH_COMMAND=ssh -i /etc/brain-mcp/ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/etc/brain-mcp/ssh/known_hosts -o StrictHostKeyChecking=yes

# warning | info | debug
BRAIN_MCP_LOG_LEVEL=info

# Public hostname(s) the tunnel presents (comma-separated). Required, or
# the DNS-rebinding guard rejects tunnel traffic whose Host header is not
# localhost. Use the same hostname as the Cloudflare ingress below.
BRAIN_MCP_ALLOWED_HOSTS=mcp.yourdomain.example
```

Lock it down so only the service user can read it:

```bash
sudo chmod 640 /etc/brain-mcp/env
sudo chown root:brain /etc/brain-mcp/env
```

## 3. Set up git push from the server

The MCP server commits write tool results and pushes them to `origin`. It needs an SSH deploy key kept **outside the vault** — if it lived under `/srv/brain` a compromised agent could read it through the read tools and take over the repo. Put it under `/etc/brain-mcp`, which the service mounts read-only:

```bash
sudo install -d -m 750 -o root -g brain /etc/brain-mcp/ssh
sudo ssh-keygen -t ed25519 -f /etc/brain-mcp/ssh/id_ed25519 -N "" -C "brain-mcp deploy key"
sudo chgrp brain /etc/brain-mcp/ssh/id_ed25519 && sudo chmod 640 /etc/brain-mcp/ssh/id_ed25519
# Pin GitHub's host key (avoids trust-on-first-use). Verify the fingerprint
# against https://docs.github.com/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints
sudo sh -c 'ssh-keyscan github.com > /etc/brain-mcp/ssh/known_hosts'
sudo chgrp brain /etc/brain-mcp/ssh/known_hosts && sudo chmod 644 /etc/brain-mcp/ssh/known_hosts
sudo cat /etc/brain-mcp/ssh/id_ed25519.pub
```

In your GitHub private brain repo: **Settings → Deploy keys → Add deploy key**. Paste the public key, **check "Allow write access"**, save.

The `GIT_SSH_COMMAND` line you added to `/etc/brain-mcp/env` in step 2 points git at this key. Test as the service user with the same command:

```bash
sudo -u brain env GIT_SSH_COMMAND="ssh -i /etc/brain-mcp/ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/etc/brain-mcp/ssh/known_hosts -o StrictHostKeyChecking=yes" \
  git -C /srv/brain push origin main
```

Should succeed without prompting. The key never lives under `/srv/brain`, so it sits outside both the read allowlist and the agent-writable area.

## 4. Build the venv and install the systemd unit

Build the virtualenv as the `brain` user (the unit runs `/srv/brain/.venv/bin/python` directly):

```bash
sudo -u brain bash -lc 'cd /srv/brain && uv sync'
```

This creates `/srv/brain/.venv` with the server's dependencies (FastAPI, the MCP SDK, uvicorn, sentence-transformers). The first `vault_search` downloads the ~100 MB embedding model into `/srv/brain/.cache/huggingface` (pinned by the unit's `HF_HOME`), so make sure `/srv/brain` has room and is writable by `brain`.

Then install the unit:

```bash
sudo cp /srv/brain/mcp_server/systemd/brain-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now brain-mcp
sudo systemctl status brain-mcp
```

If the status shows `active (running)` you're past the local part. Verify the health endpoint:

```bash
curl -s http://127.0.0.1:8765/health
# {"status":"ok"}
```

## 5. Cloudflare Tunnel

Log into Cloudflare on the server:

```bash
cloudflared tunnel login
```

Create a tunnel:

```bash
cloudflared tunnel create brain-mcp
```

This prints a tunnel UUID and writes credentials to `~/.cloudflared/<UUID>.json`. Copy them where the system service can read them:

```bash
sudo mkdir -p /etc/cloudflared && sudo cp ~/.cloudflared/<UUID>.json /etc/cloudflared/
```

Then write the config (`/etc/cloudflared/config.yml`):

```yaml
tunnel: <UUID>
credentials-file: /etc/cloudflared/<UUID>.json
ingress:
  - hostname: mcp.yourdomain.example
    service: http://127.0.0.1:8765
  - service: http_status:404
```

Route DNS:

```bash
cloudflared tunnel route dns brain-mcp mcp.yourdomain.example
```

Install + start the cloudflared service:

```bash
sudo cloudflared service install
sudo systemctl status cloudflared
```

Verify externally:

```bash
curl -s https://mcp.yourdomain.example/health
# {"status":"ok"}
```

## 6. Cloudflare Access (outer auth ring)

In the Cloudflare dashboard: **Zero Trust → Access → Applications → Add an application → Self-hosted**.

- Application domain: `mcp.yourdomain.example`
- Session duration: as you prefer (24h is reasonable)
- Path: `/mcp*`  (gate only the MCP endpoint; leave `/health` open so Cloudflare and systemd liveness probes are not challenged — it returns only `{"status":"ok"}`)

Create a policy:

- Name: "Owner"
- Action: Allow
- Rules → Include → "Emails ending in `@yourdomain.example`" *(or just your single email)*
- Optionally: enable "Service Auth" if you want to let claude.ai or another hosted agent reach the tunnel via a service token.

Save. Now browser visits prompt for SSO; agents need either a service token in their headers or a different Access policy.

## 7. Connect a client

Any MCP client speaking Streamable HTTP can reach it now. Example with the official Python client:

```python
import asyncio, os
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    headers = {
        "Authorization": f"Bearer {os.environ['BRAIN_MCP_BEARER_TOKEN']}",
        # If CF Access is in front of the tunnel, also include a service token:
        # "CF-Access-Client-Id":     "...",
        # "CF-Access-Client-Secret": "...",
    }
    async with streamablehttp_client(
        "https://mcp.yourdomain.example/mcp",
        headers=headers,
    ) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            print((await s.list_tools()).tools)

asyncio.run(main())
```

For claude.ai or Claude Code: register the URL + bearer header as a remote MCP server in their config. They'll auto-discover the seventeen tools listed in `mcp/README.md`. Give each client its own token via `BRAIN_MCP_TOKENS` so its writes are attributed to it.

## 8. Updating the server later

When you pull framework updates from `brain-template`:

```bash
sudo -u brain git -C /srv/brain pull origin main
sudo -u brain bash -lc 'cd /srv/brain && uv sync'
sudo systemctl restart brain-mcp
```

Same shape if you change `mcp_server/` code on your laptop and push: pull on the server, sync, restart.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `systemctl start brain-mcp` fails with "BRAIN_MCP_VAULT_ROOT must be set" | env file wasn't read; check the `EnvironmentFile=` path in the service unit |
| `auth: rejected request from <ip>` in `journalctl -u brain-mcp` | client and server tokens differ; rotate one to match the other |
| Writes commit but never reach GitHub | the async push worker is failing; check `journalctl -u brain-mcp` for `push worker:` lines (it retries with backoff — a fresh write retries immediately) and `logs/mcp-audit.jsonl` for the write outcomes |
| `git push exited 128: Permission denied` in logs after a write | deploy key missing "Allow write access" on GitHub |
| `Session terminated` on every client call | the lifespan didn't run; check that the FastAPI app was built via `mcp_server.app.build_app()` and the systemd unit didn't override `ExecStart` |
| Health works but `/mcp` returns 404 | client URL is `/mcp` not `/mcp/mcp`; double-prefix bug |

Logs:

```bash
sudo journalctl -u brain-mcp -f          # server
sudo journalctl -u cloudflared -f        # tunnel
```

## What this deploy does NOT do

- **No automated ingestion.** PDFs dropped via `vault_drop_inbox_file` land in `inbox/`; ingestion happens on your laptop (where MinerU + its 14 GB of model weights live) when you run `scripts/ingest.py --inbox`. The MCP server's job ends at the drop.
- **No remote summarisation.** If you want LLM summaries on the server side (so a Linux box without an internet API key can still tag topics), wire `local` provider against an Ollama instance running on the box. See `_template/README.md` → "LLM provider".
- **No multi-user.** `BRAIN_MCP_TOKENS` gives each *agent* its own identity over the shared vault (attributed commits + audit lines), but every token has the same full read/write access — this is one person's vault, a single trust domain. If multiple humans need different identities or permissions, put per-person policies in Cloudflare Access; per-agent tokens are "machine credentials", not user accounts.
