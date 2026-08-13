#!/usr/bin/env bash
# Run search on one config, then immediately re-evaluate the resulting front
# on the true test-set corpus (writes logs/<run>/data/results.csv).
#
# run_name is pinned in these experiment configs (see evolmc/rundir.py), so
# the run directory is predictable as <log.root>/<log.run_name> -- but only
# the first time that name is used. RunDir appends -2, -3, ... on a
# collision (e.g. a stale prior run already sitting under that name), and
# this script does not re-detect that; it assumes a clean first run.
#
#   scripts/search_then_eval.sh <config.yaml> [run_search.py flags...]

set -euo pipefail
cd "$(dirname "$0")/.."

cfg="$1"; shift

python3 scripts/run_search.py "$cfg" "$@"

# `| tail -n 1`: some environments print import-time warnings (e.g. autograd,
# matplotlib) to stdout, not stderr, which would otherwise land inside this
# capture ahead of the actual path and corrupt it.
run_dir="$(python3 -c "
import os, sys
sys.path.insert(0, '.')
from evolmc import Config
from evolmc.rundir import default_run_name
cfg = Config.from_yaml('$cfg')
print(os.path.join(cfg.log.root, cfg.log.run_name or default_run_name(cfg)))
" | tail -n 1)"

python3 scripts/run_eval.py "$cfg" "$run_dir"
