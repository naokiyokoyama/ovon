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
#SBATCH --exclude=conroy,ig-88
#SBATCH --signal=USR1@100
#SBATCH --requeue

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

SPLIT=$1
NUM_TASKS=$2
NUM_SCENES=-1

srun python ovon/dataset/objectnav_generator.py \
  --split $SPLIT \
  --num-scenes $NUM_SCENES \
  --tasks-per-gpu $NUM_TASKS \
  --episodes-per-object 200 \
  --episodes-per-scene 50 \
  --multiprocessing
