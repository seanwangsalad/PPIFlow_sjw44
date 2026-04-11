#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_PATH="${1:-${REPO_DIR}/apptainer/ppiflow_cuda121.sif}"
DEF_FILE="${REPO_DIR}/apptainer/ppiflow.def"
ENV_FILE="$(realpath "${REPO_DIR}/environment.yml")"
TMP_DEF="$(mktemp "${REPO_DIR}/apptainer/ppiflow.XXXXXX.def")"
BUILD_ROOT="${APPTAINER_BUILD_ROOT:-${REPO_DIR}/apptainer/.build}"
APPTAINER_TMPDIR_DEFAULT="${BUILD_ROOT}/tmp"
APPTAINER_CACHEDIR_DEFAULT="${BUILD_ROOT}/cache"

if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is not on PATH" >&2
    exit 1
fi

cleanup() {
    rm -f "${TMP_DEF}"
}
trap cleanup EXIT

mkdir -p "$(dirname "${IMAGE_PATH}")"
mkdir -p "${APPTAINER_TMPDIR_DEFAULT}" "${APPTAINER_CACHEDIR_DEFAULT}"

export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${APPTAINER_TMPDIR_DEFAULT}}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${APPTAINER_CACHEDIR_DEFAULT}}"

sed "s|__ENVIRONMENT_YML__|${ENV_FILE}|g" "${DEF_FILE}" > "${TMP_DEF}"

cd "${REPO_DIR}"
apptainer build "${IMAGE_PATH}" "${TMP_DEF}"

echo "Built ${IMAGE_PATH}"
echo "Apptainer temp dir: ${APPTAINER_TMPDIR}"
echo "Apptainer cache dir: ${APPTAINER_CACHEDIR}"
