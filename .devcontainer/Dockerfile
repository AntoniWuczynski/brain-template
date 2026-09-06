# Minimal devcontainer for the brain vault.
# - Python 3.12 (PaddlePaddle-compatible)
# - uv as the package manager (per repo convention)
# - non-root vscode user
#
# No machine-specific assumptions: no GPU, no CUDA. MinerU runs in
# CPU mode; for faster extraction on a workstation, use a GPU image
# defined in a separate compose override (out of scope here).
#
# MinerU itself is NOT installed here, and `uv sync` never installs it: it is
# not in uv.lock (see CLAUDE.md "Tooling defaults"). A fresh container
# therefore extracts PDFs through the pypdf fallback and marks them
# `status: partial`. To get full extraction, run the pinned install inside the
# container — 65 unlocked, unhashed packages, which is why it is opt-in:
#   uv pip install --prerelease=allow "mineru[pipeline]==2.7.6" six
# The brain-mineru-models volume (devcontainer.json) exists so the ~14 GB of
# weights that install then downloads survives container rebuilds.

# Pinned by digest, not just by the floating `1-3.12` tag, so a rebuild months
# from now resolves the same base layer. Refresh with:
#   docker buildx imagetools inspect mcr.microsoft.com/devcontainers/python:1-3.12
FROM mcr.microsoft.com/devcontainers/python:1-3.12@sha256:7876580dc67fd460fd962f004cbeb480027e9bbc0657096f1087db11f9eaff39

# System packages MinerU/PaddleOCR need at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Install uv to /usr/local/bin, pinned to an exact release so the build does
# not silently pick up whatever uv shipped this morning. Must stay >= the
# `required-version` in pyproject.toml; bump both together.
RUN curl -LsSf https://astral.sh/uv/0.12.1/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

USER vscode
WORKDIR /workspaces/brain
