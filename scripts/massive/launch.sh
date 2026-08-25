#! /usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]] || ! [[ $1 =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 <n_runs>" >&2
  exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

sbatch --export=SCRIPT_DIR=$SCRIPT_DIR "$SCRIPT_DIR/calibrate.sh" "$1"
