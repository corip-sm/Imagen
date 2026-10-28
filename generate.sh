#!/bin/zsh
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 OUTPUT.png 'full scene caption' [reference.png ...]" >&2
  exit 2
fi

output_path="$1"
prompt="$2"
shift 2
script_dir="${0:A:h}"

seed="${SEED:-7}"
width="${WIDTH:-${SIZE:-512}}"
height="${HEIGHT:-${SIZE:-512}}"
steps="${STEPS:-16}"
model_path="${GAON_IMAGE_MODEL:-$HOME/.cache/gaon-image/flux3-klein-q8}"

if [[ ! -d "$model_path" ]]; then
  echo "Q8 model directory not found: $model_path" >&2
  echo 'See README.md for Hugging Face login and mflux-save, or set IMAGEN_MODEL.' >&2
  exit 1
fi

# Normalize every reference before activating the model runtime.
exec "$script_dir/scripts/python.sh" "$script_dir/src/imagen/references/launch_generation.py" \
  --output "$output_path" --prompt "$prompt" --model "$model_path" \
  --seed "$seed" --steps "$steps" --width "$width" --height "$height" \
  -- "$@"
