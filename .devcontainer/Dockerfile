# Minimal devcontainer for the brain vault.
# - Python 3.12 (PaddlePaddle-compatible)
# - uv as the package manager (per repo convention)
# - non-root vscode user
#
# No machine-specific assumptions: no GPU, no CUDA. MinerU runs in
# CPU mode; for faster extraction on a workstation, use a GPU image
# defined in a separate compose override (out of scope here).

FROM mcr.microsoft.com/devcontainers/python:1-3.12

# System packages MinerU/PaddleOCR need at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Install uv to /usr/local/bin. This installs whatever the LATEST uv release
# is at build time (the install script is not version-pinned), so container
# builds are not perfectly reproducible across uv releases. Pin the URL to a
# specific version (e.g. https://astral.sh/uv/0.7.13/install.sh) if you need
# byte-reproducible dependency-management behavior.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

USER vscode
WORKDIR /workspaces/brain
