#! /usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
sbatch "$SCRIPT_DIR/calibrate.sh"
