#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?Provide the pinned official TxPert checkout}
environment_dir=${2:?Provide a NEW isolated environment directory}
if [ -e "$environment_dir" ]; then
  echo "Refusing to modify an existing environment: $environment_dir" >&2
  exit 1
fi
expected_commit="08d82eea86746b044cf7531f4ec8c5f60e1cb73f"
if [ "$(git -C "$source_dir" rev-parse HEAD)" != "$expected_commit" ]; then
  echo "The official checkout revision does not match registry.json" >&2
  exit 1
fi
python3.12 -m venv "$environment_dir"
py="$environment_dir/bin/python"
"$py" -m pip install --upgrade pip
"$py" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$py" -m pip install -r "$(dirname "$0")/requirements-runtime.txt"
"$py" -m pip install torch-scatter==2.1.2 torch-sparse==0.6.18 \
  --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html
"$py" -m pip install --no-deps -e "$source_dir"
