#!/bin/bash
#SBATCH --job-name=ssl-focal-calibration
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5%6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

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

case "${SLURM_ARRAY_TASK_ID:?Missing SLURM array task ID}" in
    0)
        EXPERIMENT_NAME="pilot_ssl_random_015"
        OUTPUT_DIR="results/pilot_ssl/calibration/random_015"
        POLICY="random"
        FRACTION="0.15"
        CORPUS="outputs/pilot_v0_synthetic_spatial"
        STEPS="2000"
        ;;
    1)
        EXPERIMENT_NAME="pilot_ssl_random_030"
        OUTPUT_DIR="results/pilot_ssl/calibration/random_030"
        POLICY="random"
        FRACTION="0.30"
        CORPUS="outputs/pilot_v0_synthetic_spatial"
        STEPS="2000"
        ;;
    2)
        EXPERIMENT_NAME="pilot_ssl_random_050"
        OUTPUT_DIR="results/pilot_ssl/calibration/random_050"
        POLICY="random"
        FRACTION="0.50"
        CORPUS="outputs/pilot_v0_synthetic_spatial"
        STEPS="2000"
        ;;
    3)
        EXPERIMENT_NAME="pilot_ssl_random_070"
        OUTPUT_DIR="results/pilot_ssl/calibration/random_070"
        POLICY="random"
        FRACTION="0.70"
        CORPUS="outputs/pilot_v0_synthetic_spatial"
        STEPS="2000"
        ;;
    4)
        EXPERIMENT_NAME="pilot_ssl_high_random_085"
        OUTPUT_DIR="results/pilot_ssl/calibration/high_random_085"
        POLICY="high_random"
        FRACTION="0.85"
        CORPUS="outputs/pilot_v0_synthetic_spatial"
        STEPS="2000"
        ;;
    5)
        EXPERIMENT_NAME="pilot_ssl_null_random_030"
        OUTPUT_DIR="results/pilot_ssl/null_smoke/random_030"
        POLICY="random"
        FRACTION="0.30"
        CORPUS="outputs/pilot_v0_synthetic_null"
        STEPS="500"
        ;;
    *)
        echo "Unsupported array task: $SLURM_ARRAY_TASK_ID" >&2
        exit 2
        ;;
esac

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

"$PYTHON" scripts/train_focal_ssl.py \
    --config "$CONFIG_PATH" \
    --override "experiment_name=$EXPERIMENT_NAME" \
    --override "output_dir=$OUTPUT_DIR" \
    --override "data.corpus_path=$CORPUS" \
    --override "masking.policy=$POLICY" \
    --override "masking.fraction=$FRACTION" \
    --override "training.steps=$STEPS"
