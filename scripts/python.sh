#!/bin/zsh
set -euo pipefail
cli_path="$(command -v mflux-generate-flux3)" || {
  echo 'MFLUX is not installed. Run ./setup.sh and add uv tool bin to PATH.' >&2
  exit 1
}
IFS= read -r python_path < "$cli_path"
python_path="${python_path#\#!}"
if [[ ! -x "$python_path" ]]; then
  echo "Cannot resolve the MFLUX CLI's Python interpreter." >&2
  exit 1
fi
exec "$python_path" "$@"
