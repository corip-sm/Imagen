#!/bin/zsh
set -euo pipefail
script_dir="${0:A:h}"
project_root="${script_dir:h}"
mlx_path="$("$script_dir/python.sh" -c 'import mlx.core; from pathlib import Path; print(Path(mlx.core.__file__).parent)')"
build_dir="$project_root/.cache/imagen-native"
mkdir -p "$build_dir"
clang++ -std=c++20 -O3 -I"$mlx_path/include" "$project_root/src/imagen/native/flux3_denoise.cpp" \
  -L"$mlx_path/lib" -lmlx -Wl,-rpath,"$mlx_path/lib" -o "$build_dir/imagen-native"
