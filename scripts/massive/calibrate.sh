#!/bin/bash

#SBATCH --job-name=test_tb
#SBATCH --account=sh30

#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

cd $SCRIPT_DIR/../..

pixi run python scripts/massive/calibrate.py