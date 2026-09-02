#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/pipeline_env.sh"
cd "$ROOT"
exec "$QWEN_PYTHON" "$ROOT/infer_public.py" "$@"
