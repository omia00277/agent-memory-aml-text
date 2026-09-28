#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${LOCOMO_EXPORT_DIR}"

exec "${LOCOMO_PYTHON_BIN}" -c 'from export import main; main()' \
  --dataset-path "${LOCOMO_DATASET_PATH}" \
  --output-dir "${LOCOMO_EXPORT_DIR}" \
  "$@"
