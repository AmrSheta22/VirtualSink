#!/bin/bash
# Source after the batch script's #SBATCH header.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit through Slurm on the HPC." >&2
    exit 1
fi
status() { printf '[%s] job=%s %s\n' "$(date -Is)" "$SLURM_JOB_ID" "$*"; }
heartbeat_pid=''
finish() {
    result=$?
    trap - EXIT
    if [[ -n "$heartbeat_pid" ]]; then
        kill "$heartbeat_pid" 2>/dev/null || true
        wait "$heartbeat_pid" 2>/dev/null || true
    fi
    if (( result == 0 )); then status 'COMPLETED exit_code=0'; else status "FAILED exit_code=$result"; fi
    exit "$result"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
status "STARTED host=$(hostname)"
cd "$HOME/data/VirtualSink"
status 'Activating virtual environment'
source "$HOME/data/VirtualSink/venv/bin/activate"
export PYTHONUNBUFFERED=1
status "ENV_READY python=$(command -v python)"
(
    while true; do
        sleep 10
        status 'HEARTBEAT batch script active; inspect application metrics for progress'
    done
) &
heartbeat_pid=$!
