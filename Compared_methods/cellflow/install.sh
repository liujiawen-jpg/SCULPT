#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?Provide the separately obtained pinned official checkout}
environment_dir=${2:?Provide a NEW isolated environment directory}
if [ -e "$environment_dir" ]; then
  echo "Refusing to modify an existing environment: $environment_dir" >&2
  exit 1
fi
expected_commit="446ed6073c60ac2e8db13c4ea096a43cdec288b2"
if [ "$(git -C "$source_dir" rev-parse HEAD)" != "$expected_commit" ]; then
  echo "The upstream checkout revision does not match this registration" >&2
  exit 1
fi
python3.11 -m venv "$environment_dir"
py="$environment_dir/bin/python"
"$py" -m pip install --upgrade pip
"$py" -m pip install 'jax[cuda12]==0.6.2'
"$py" -m pip install -r "$(dirname "$0")/requirements-runtime.txt"
"$py" -m pip install --no-deps -e "$source_dir"
