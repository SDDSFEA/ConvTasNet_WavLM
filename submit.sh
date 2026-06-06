#!/bin/bash
set -euo pipefail

EXCLUDE_FILE="/lustre/teams/mmai/mmai-job-setting/exclude_nodes.txt"

EXCLUDE_NODES=""
if [[ -f "$EXCLUDE_FILE" ]]; then
  EXCLUDE_NODES=$(grep -v '^\s*#' "$EXCLUDE_FILE" | grep -v '^\s*$' | paste -sd,)
fi

echo "[INFO] Submitting with --exclude=${EXCLUDE_NODES}"
# sbatch --exclude="$EXCLUDE_NODES" run_film.sh
# sbatch --exclude="$EXCLUDE_NODES" run_att2.sh
sbatch --exclude="$EXCLUDE_NODES" run_up_unfreeze.sh



