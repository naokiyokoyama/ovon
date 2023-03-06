#!/bin/bash
#SBATCH --job-name=ovon
#SBATCH --output=slurm_logs/dataset-%j.out
#SBATCH --error=slurm_logs/dataset-%j.err
#SBATCH --gpus 2
#SBATCH --nodes 1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node 1
#SBATCH --constraint=a40
#SBATCH --partition=short
#SBATCH --exclude=conroy
#SBATCH --signal=USR1@100
#SBATCH --requeue

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

srun python ovon/dataset/objectnav_generator.py \
  --split val \
  --num-scenes 
