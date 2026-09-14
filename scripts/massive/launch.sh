#! /usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
sbatch --time="$2" --export=SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/calibrate.sh" "$1"
