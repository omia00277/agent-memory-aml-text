#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LOCOMO_BENCHMARK_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_LOCOMO_REPO_ROOT="${DEFAULT_LOCOMO_BENCHMARK_ROOT}"

export LOCOMO_BENCHMARK_ROOT="${LOCOMO_BENCHMARK_ROOT:-${DEFAULT_LOCOMO_BENCHMARK_ROOT}}"
export LOCOMO_REPO_ROOT="${LOCOMO_REPO_ROOT:-${DEFAULT_LOCOMO_REPO_ROOT}}"

if [[ -f "${LOCOMO_REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${LOCOMO_REPO_ROOT}/.env"
  set +a
fi

export LOCOMO_ASSETS_DIR="${LOCOMO_ASSETS_DIR:-${LOCOMO_BENCHMARK_ROOT}/data/public}"
export LOCOMO_DATASET_PATH="${LOCOMO_DATASET_PATH:-${LOCOMO_BENCHMARK_ROOT}/data/raw/locomo_refined.json}"
export LOCOMO_OUTPUT_DIR="${LOCOMO_OUTPUT_DIR:-${LOCOMO_BENCHMARK_ROOT}/outputs}"
export LOCOMO_EXPORT_DIR="${LOCOMO_EXPORT_DIR:-${LOCOMO_OUTPUT_DIR}/exported}"
export LOCOMO_QUESTIONS_PATH="${LOCOMO_QUESTIONS_PATH:-${LOCOMO_ASSETS_DIR}/questions.jsonl}"
export LOCOMO_SUBMISSION_TEMPLATE_PATH="${LOCOMO_SUBMISSION_TEMPLATE_PATH:-${LOCOMO_ASSETS_DIR}/submission_template.jsonl}"
export LOCOMO_MANIFEST_PATH="${LOCOMO_MANIFEST_PATH:-${LOCOMO_ASSETS_DIR}/manifest.json}"
export LOCOMO_PREDICTIONS_PATH="${LOCOMO_PREDICTIONS_PATH:-${LOCOMO_OUTPUT_DIR}/predictions.jsonl}"

if [[ -n "${LOCOMO_PYTHON_BIN:-}" ]]; then
  export LOCOMO_PYTHON_BIN
elif [[ -x "${LOCOMO_REPO_ROOT}/.venv/bin/python" ]]; then
  export LOCOMO_PYTHON_BIN="${LOCOMO_REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  export LOCOMO_PYTHON_BIN="python3"
else
  export LOCOMO_PYTHON_BIN="python"
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${LOCOMO_BENCHMARK_ROOT}/src:${PYTHONPATH}"
else
  export PYTHONPATH="${LOCOMO_BENCHMARK_ROOT}/src"
fi
