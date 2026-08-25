#!/bin/bash

#SBATCH --job-name=test_tb
#SBATCH --account=sh30

#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096

# SCRIPT_DIR is exported by launch.sh from the login-node path.
# Do not recompute it from BASH_SOURCE: Slurm executes a copy under /var/spool.
cd "$SCRIPT_DIR/../.."

pixi run python scripts/massive/calibrate.py "$SLURM_JOB_ID"
