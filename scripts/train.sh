#!/usr/bin/env bash
set -euo pipefail
# See --help for required data/prior/graph/output paths. No task starts on import.
python scripts/run_benchmark.py "$@"
