#!/bin/zsh
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: $0 OUTPUT.png 'edit instruction' SOURCE_IMAGE [SOURCE_IMAGE ...]" >&2
  exit 2
fi

# Keep the existing command name; all references share the common 512px policy.
script_dir="${0:A:h}"
exec "$script_dir/generate.sh" "$@"
