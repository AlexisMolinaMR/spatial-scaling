#!/bin/bash
#SBATCH --job-name=data-scaling-spatial-90pass-summary-v0
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

"$PYTHON" scripts/summarize_data_scaling_spatial_90pass_convergence.py "$@"
