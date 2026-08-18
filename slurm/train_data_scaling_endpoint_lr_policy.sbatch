#!/bin/bash
#SBATCH --job-name=data-scaling-lr-policy-v0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
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

MANIFEST="${1:?Pass the endpoint LR-policy manifest as argument 1}"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Endpoint LR-policy manifest does not exist: $MANIFEST" >&2
    exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

"$PYTHON" scripts/run_data_scaling.py \
    --manifest "$MANIFEST" \
    --task-id "${SLURM_ARRAY_TASK_ID:?Missing SLURM array task ID}"

