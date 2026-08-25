#!/bin/bash

#SBATCH --job-name=calibrate_vnm
#SBATCH --account=sh30

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096

cd "$SCRIPT_DIR/../.."

pixi run python scripts/massive/calibrate.py "$SLURM_JOB_ID" "$1"
