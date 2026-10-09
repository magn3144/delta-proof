#!/usr/bin/env bash

set -euo pipefail

export UV_PROJECT_ENVIRONMENT="$PWD/.venv-ucloud-b200"
test -x "$UV_PROJECT_ENVIRONMENT/bin/python"
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

uv pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
nvidia-smi -L
XLA_PYTHON_CLIENT_PREALLOCATE=false "$UV_PROJECT_ENVIRONMENT/bin/python" -m scripts.batch.UCloud.preflight_b200
