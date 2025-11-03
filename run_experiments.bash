#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_script.py}

NAME=${NAME:-NN}
SEEDS=${SEEDS:-"10,20,30"}
PROFILE=${PROFILE:-DacExperiments}
CONFIG_DIR="configs"
DRY_RUN=${DRY_RUN:-0}

usage() {
  echo "Usage: $0 [-n NAME] [-s SEEDS] [-p PROFILE] [-d CONFIG_DIR] [--dry-run]"
  exit 1
}

# parse short flags
while getopts ":n:s:p:d:-:" opt; do
  case "$opt" in
    n) NAME="$OPTARG" ;;
    s) SEEDS="$OPTARG" ;;
    p) PROFILE="$OPTARG" ;;
    d) CONFIG_DIR="$OPTARG" ;;
    -)
      case "$OPTARG" in
        dry-run) DRY_RUN=1 ;;
        *) usage ;;
      esac
      ;;
    \?) usage ;;
  esac
done
shift $((OPTIND-1))

[[ -d "$CONFIG_DIR" ]] || { echo "No such dir: $CONFIG_DIR" >&2; exit 1; }

mapfile -d '' CONFIGS < <(find "$CONFIG_DIR" -type f -name '*.json' -print0 | sort -z)
(( ${#CONFIGS[@]} > 0 )) || { echo "No .json under $CONFIG_DIR" >&2; exit 1; }

slugify() {
  echo -n "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | tr -c 'a-z0-9._-' '-' \
    | tr -s '-' \
    | sed -E 's/^-+//; s/-+$//'
}

for cfg in "${CONFIGS[@]}"; do
  cfg_base=$(basename "$cfg" .json)
  cfg_slug=$(slugify "$cfg_base")
  name_slug=$(slugify "$NAME")

  parts=()
  [[ -n "$cfg_slug"  ]] && parts+=("$cfg_slug")
  [[ -n "$name_slug" ]] && parts+=("$name_slug")
  run_name=$(IFS=-; echo "${parts[*]}")

  echo ">>> Running with config: $cfg (name: $run_name)"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "$PYTHON_BIN $TRAIN_SCRIPT -n \"$run_name\" -s \"$SEEDS\" -p \"$PROFILE\" -j \"$cfg\""
  else
    set -x
    "$PYTHON_BIN" "$TRAIN_SCRIPT" -n "$run_name" -s "$SEEDS" -p "$PROFILE" -j "$cfg"
    { set +x; } 2>/dev/null
  fi
  echo
done
