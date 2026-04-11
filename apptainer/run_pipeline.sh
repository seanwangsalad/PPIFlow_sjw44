#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_pipeline.sh IMAGE CONFIG OUTPUT_DIR CHECKPOINT_DIR [NUM_SAMPLES]

Example:
  ./apptainer/run_pipeline.sh \
    ./apptainer/ppiflow_cuda121.sif \
    ./apptainer/examples/pipeline_nanobody.example.yaml \
    ./outputs/nanobody_demo \
    /shared/checkpoints \
    5
EOF
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
    usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_PATH="$(realpath "$1")"
CONFIG_PATH="$(realpath "$2")"
OUTPUT_DIR="$(realpath -m "$3")"
CHECKPOINT_DIR="$(realpath "$4")"
NUM_SAMPLES="${5:-5}"

mkdir -p "${OUTPUT_DIR}"

apptainer exec \
    --nv \
    --bind "${REPO_DIR}:/workspace" \
    --bind "$(dirname "${CONFIG_PATH}"):/config_ro" \
    --bind "${OUTPUT_DIR}:/output" \
    --bind "${CHECKPOINT_DIR}:/checkpoints" \
    "${IMAGE_PATH}" \
    /opt/bin/activate_ppiflow.sh \
    bash -lc "
        set -euo pipefail
        cd /workspace
        export PYTHONPATH=/workspace:\${PYTHONPATH:-}
        python pipeline.py \
            --config /config_ro/$(basename "${CONFIG_PATH}") \
            --output /output \
            --num_samples ${NUM_SAMPLES}
    "
