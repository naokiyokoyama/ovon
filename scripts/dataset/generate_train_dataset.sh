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

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate ovon

SPLIT=$1
NUM_TASKS=$2
OUTPUT_PATH=$3
NUM_SCENES=-1

srun python ovon/dataset/objectnav_generator.py \
  --split $SPLIT \
  --num-scenes $NUM_SCENES \
  --tasks-per-gpu $NUM_TASKS \
  --output-path $OUTPUT_PATH \
  --multiprocessing
