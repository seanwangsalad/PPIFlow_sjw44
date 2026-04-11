#!/usr/bin/env bash
#SBATCH --job-name=ppiflow-nanobody
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
IMAGE_PATH="${REPO_DIR}/apptainer/ppiflow_cuda121.sif"
CONFIG_PATH="${REPO_DIR}/apptainer/examples/pipeline_nanobody.example.yaml"
OUTPUT_DIR="${REPO_DIR}/outputs/nanobody_${SLURM_JOB_ID}"
CHECKPOINT_DIR="${REPO_DIR}/checkpoints"

mkdir -p "${REPO_DIR}/logs"

module purge
module load apptainer

bash "${REPO_DIR}/apptainer/run_pipeline.sh" \
    "${IMAGE_PATH}" \
    "${CONFIG_PATH}" \
    "${OUTPUT_DIR}" \
    "${CHECKPOINT_DIR}" \
    5
