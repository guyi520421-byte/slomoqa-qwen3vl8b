#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/pipeline_env.sh"
configure_split

case "$SPLIT" in
  public)
    EXPECTED_SHA256=6fbaa7aeb24ad21483800bb26b6c8c5aaf9dc3506eb067ac8d732f018ca0712a
    DEFAULT_OUTPUT=$ROOT/outputs/qwen3vl8b_base512_public3.json
    ;;
  private)
    EXPECTED_SHA256=66fe3508dcf73462676ee394f3b2d0f413cb60516ebae165fc55c085ff3b95ff
    DEFAULT_OUTPUT=$ROOT/outputs/qwen3vl8b_base_private_notfullregen_evidence2.json
    ;;
esac

SELECTIONS=$ROOT/inputs/${SPLIT}_selections.jsonl
CLEAN_EVIDENCE=$ROOT/inputs/${SPLIT}_evidence_ocr_clean.jsonl
OUTPUT=${OUTPUT:-$DEFAULT_OUTPUT}
ALLOW_CUSTOM_PUBLIC_INPUTS=0
export SPLIT GPU_ID QUESTIONS VIDEO_DIR SELECTIONS CLEAN_EVIDENCE OUTPUT
export ALLOW_CUSTOM_PUBLIC_INPUTS

mkdir -p "$(dirname -- "$OUTPUT")"
bash "$ROOT/run_05_infer_base.sh"
"$QWEN_PYTHON" "$ROOT/validate_artifacts.py" \
  --split "$SPLIT" --questions "$QUESTIONS" \
  --selections "$SELECTIONS" --evidence "$CLEAN_EVIDENCE" \
  --predictions "$OUTPUT"

ACTUAL_SHA256=$(sha256sum "$OUTPUT" | awk '{print $1}')
if [[ "$ACTUAL_SHA256" == "$EXPECTED_SHA256" ]]; then
  echo "Historical prediction hash: MATCH ($ACTUAL_SHA256)"
else
  echo "Historical prediction hash: DIFFERENT" >&2
  echo "  expected: $EXPECTED_SHA256" >&2
  echo "  actual:   $ACTUAL_SHA256" >&2
  echo "Coverage is complete, but runtime output is not byte-identical." >&2
fi
echo "Predictions: $OUTPUT"
