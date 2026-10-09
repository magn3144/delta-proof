#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
test "$(uname -sm)" = "Linux x86_64"

# Install only the isolated environment, never the root uv project.
uv venv --allow-existing --python 3.13 .venv-ucloud-b200
uv pip sync --python "$PWD/.venv-ucloud-b200/bin/python" \
    --torch-backend cu128 requirements-ucloud-b200.lock
source scripts/batch/UCloud/common_b200.sh
