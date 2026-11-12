#! /usr/bin/env bash
# Usage: launch.sh <n_runs> <walltime> [<task id to resume>]
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
sbatch --time="$2" --export=SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/calibrate.sh" "$1" "${@:3}"
