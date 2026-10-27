#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?Provide the pinned official scPILOT checkout}
environment_dir=${2:?Provide a NEW isolated environment directory}
if [ -e "$environment_dir" ]; then
  echo "Refusing to modify an existing environment: $environment_dir" >&2
  exit 1
fi
expected_commit="ff83e3283f091324107a55f12f6801c4e72dfb05"
if [ "$(git -C "$source_dir" rev-parse HEAD)" != "$expected_commit" ]; then
  echo "The official checkout revision does not match registry.json" >&2
  exit 1
fi
python3.12 -m venv "$environment_dir"
py="$environment_dir/bin/python"
"$py" -m pip install --upgrade pip
"$py" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
"$py" -m pip install -r "$(dirname "$0")/requirements-runtime.txt"
"$py" -m pip install --no-deps -e "$source_dir"
