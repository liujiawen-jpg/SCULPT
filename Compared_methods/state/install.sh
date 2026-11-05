#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?Provide the pinned official STATE checkout}
environment_dir=${2:?Provide a NEW isolated environment directory}
if [ -e "$environment_dir" ]; then
  echo "Refusing to modify an existing environment" >&2
  exit 1
fi
if [ "$(git -C "$source_dir" rev-parse HEAD)" != "9bbfe78a434a55205e4de834e1ea99f85f7a3add" ]; then
  echo "STATE revision does not match registry.json" >&2
  exit 1
fi
python3.11 -m venv "$environment_dir"
py="$environment_dir/bin/python"
"$py" -m pip install --upgrade pip
"$py" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
"$py" -m pip install -c "$(dirname "$0")/requirements-runtime.txt" -e "$source_dir"
