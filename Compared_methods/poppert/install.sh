#!/usr/bin/env bash
set -euo pipefail
source_dir=${1:?Provide the separately obtained pinned official checkout}
environment_dir=${2:?Provide a NEW isolated environment directory}
if [ -e "$environment_dir" ]; then
  echo "Refusing to modify an existing environment: $environment_dir" >&2
  exit 1
fi
expected_commit="248fc3e1428de96335bef1d0271d30ca1e4d5565"
if [ "$(git -C "$source_dir" rev-parse HEAD)" != "$expected_commit" ]; then
  echo "The upstream checkout revision does not match this registration" >&2
  exit 1
fi
python3.12 -m venv "$environment_dir"
py="$environment_dir/bin/python"
"$py" -m pip install --upgrade pip
"$py" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
"$py" -m pip install -r "$(dirname "$0")/requirements-runtime.txt"
# PopPert has no installable package metadata. configure_upstream() imports the
# separately obtained, hash-verified sources; no vendor copy is created here.
