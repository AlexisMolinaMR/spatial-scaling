#!/bin/bash
#SBATCH --job-name=ssl-focal-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
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

CONFIG_PATH="configs/pilot/ssl_focal.yaml"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "SSL configuration does not exist: $CONFIG_PATH" >&2
    exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

"$PYTHON" scripts/train_focal_ssl.py \
    --config "$CONFIG_PATH" \
    --override "experiment_name=pilot_ssl_small_smoke" \
    --override "output_dir=results/pilot_ssl/small_smoke" \
    --override "data.corpus_path=outputs/validation_synthetic_spatial_small" \
    --override "masking.fraction=0.50" \
    --override "training.batch_size=128" \
    --override "training.steps=100" \
    --override "training.evaluation_interval=25" \
    --override "training.mask_epoch_steps=25" \
    --override "evaluation.batch_size=512" \
    --override "evaluation.max_cells=512"

