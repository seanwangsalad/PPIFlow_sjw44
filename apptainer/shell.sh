#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 IMAGE [CHECKPOINT_DIR]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_PATH="$(realpath "$1")"
CHECKPOINT_DIR="${2:-${REPO_DIR}/checkpoints}"

mkdir -p "${CHECKPOINT_DIR}"

apptainer shell \
    --nv \
    --bind "${REPO_DIR}:/workspace" \
    --bind "$(realpath -m "${CHECKPOINT_DIR}"):/checkpoints" \
    "${IMAGE_PATH}"
