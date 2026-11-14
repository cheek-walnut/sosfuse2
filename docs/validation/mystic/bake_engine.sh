#!/usr/bin/env bash
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python}
if (( $# < 1 )); then
  printf '%s\n' 'Usage: bake_engine.sh WORK [--arms A0,A1,A2] [--states ID,...] [--config FILE]' >&2
  exit 2
fi
mkdir -p "$1"
WORK=$(cd "$1" && pwd)
shift
ARMS=A0
STATES=
CONFIG="$HERE/config.json"
while (( $# )); do
  case "$1" in
    --arms) ARMS=$2; shift 2 ;;
    --states) STATES=$2; shift 2 ;;
    --config) CONFIG=$2; shift 2 ;;
    *) printf 'E_ARGUMENT: %s\n' "$1" >&2; exit 2 ;;
  esac
done
export SOSFUSE2_WORK_ROOT=${SOSFUSE2_WORK_ROOT:-"$WORK/sosfuse2_root"}
export SOSFUSE2_DETERMINISTIC=1 SOSFUSE2_GROUND_SH_L=24
flags=$("$PYTHON" -B -m sosfuse2.backends.flags_env C3) || exit 2
eval "$flags"
if [[ -z $STATES ]]; then
  STATES=$("$PYTHON" -B -c $'import json,sys\nprint(",".join(json.load(open(sys.argv[1]))["state_ids"]))' \
    "$CONFIG") || exit 2
fi
IFS=, read -r -a selected <<< "$ARMS"
failed=0
for selected_id in "${selected[@]}"; do
  case "$selected_id" in
    A0|A1|A2) ;;
    *) printf 'E_UNKNOWN_ARM: %s\n' "$selected_id" >&2; exit 2 ;;
  esac
  mkdir -p "$WORK/engine/$selected_id"
  "$PYTHON" -B "$HERE/prepare_arm.py" --work "$WORK" --config "$CONFIG" --arm "$selected_id" --status RUNNING
  rc=0
  "$PYTHON" -B "$HERE/prepare_arm.py" --work "$WORK" --config "$CONFIG" --arm "$selected_id" \
    > "$WORK/engine/$selected_id/prepare.log" 2>&1 || rc=$?
  if (( rc == 0 )); then
    "$PYTHON" -B -m sosfuse2.jobs.pipeline --states "$WORK/states.json" --ids "$STATES" \
      --work-root "$WORK/engine/$selected_id/work" --settings-json "$WORK/engine/$selected_id/settings.json" \
      --backend cupy --persistent --workers auto --gpu-active 1 \
      > "$WORK/engine/$selected_id/pipeline.log" 2>&1 || rc=$?
  fi
  if (( rc == 0 )); then
    "$PYTHON" -B "$HERE/prepare_arm.py" --work "$WORK" --config "$CONFIG" --arm "$selected_id" --status DONE
  else
    failed=1
    "$PYTHON" -B "$HERE/prepare_arm.py" --work "$WORK" --config "$CONFIG" --arm "$selected_id" \
      --status FAILED --reason "E_ARM_EXIT_$rc"
  fi
done
exit "$failed"
