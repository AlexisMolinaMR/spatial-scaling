#!/bin/bash
#SBATCH --job-name=synthetic-pilot-v0
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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

if [[ $# -ne 2 ]]; then
    echo "Usage: sbatch slurm/generate_synthetic_pilot.sbatch CONFIG OUTPUT_DIR" >&2
    exit 2
fi

CONFIG_PATH="$1"
OUTPUT_DIR="$2"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Synthetic configuration does not exist: $CONFIG_PATH" >&2
    exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

"$PYTHON" scripts/generate_synthetic.py \
    --config "$CONFIG_PATH" \
    --output "$OUTPUT_DIR"
