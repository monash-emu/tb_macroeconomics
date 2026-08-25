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

# 4 JAX host devices (chains) x 2 threads, matching --cpus-per-task=8.
# XLA_FLAGS must be set before Python starts so numpyro can pmap chains.
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_force_host_platform_device_count=4"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

pixi run python scripts/massive/calibrate.py "$SLURM_JOB_ID"
