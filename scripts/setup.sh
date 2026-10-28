#!/bin/zsh
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo 'Imagen requires an Apple Silicon Mac.' >&2
  exit 1
fi
if ! xcrun --find swift >/dev/null 2>&1; then
  echo 'Install Xcode Command Line Tools first: xcode-select --install' >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
  exit 1
fi

# Match the source and Metal arithmetic used for the recorded parity checks.
uv tool install --python 3.12 \
  --from git+https://github.com/mflux-community/mflux.git@8a7fd0a486635f670ef90e02efe1c051ed3f133c \
  --with mlx==0.32.4 --with mlx-metal==0.32.4 \
  mflux --force
