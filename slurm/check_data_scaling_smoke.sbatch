#!/bin/bash
#SBATCH --job-name=data-scaling-smoke-gate-v0
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
cd "$PROJECT_ROOT"

PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing uv environment: $PYTHON" >&2
    echo "Run 'uv sync --locked' from the repository root before submitting." >&2
    exit 1
fi

SCRIPT="scripts/check_data_scaling_smoke.py"

if [[ ! -f "$SCRIPT" ]]; then
    echo "Smoke gate does not exist: $SCRIPT" >&2
    exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

"$PYTHON" "$SCRIPT" "$@"
