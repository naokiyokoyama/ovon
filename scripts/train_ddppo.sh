#!/bin/bash
#SBATCH --job-name=ovon
#SBATCH --output=slurm_logs/ovon-ddppo-%j.out
#SBATCH --error=slurm_logs/ovon-ddppo-%j.err
#SBATCH --gpus 4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 4
#SBATCH --constraint=a40
#SBATCH --partition=short
#SBATCH --signal=USR1@100
#SBATCH --requeue


export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

TENSORBOARD_DIR="tb/objectnav/ddppo/resnetclip/seed_1"
CHECKPOINT_DIR="data/new_checkpoints/objectnav/ddppo/resnetclip/seed_1"

srun python -um ovon.run \
  --run-type train \
  --exp-config config/experiments/ver_objectnav.yaml \
  habitat_baselines.rl.policy.name=PointNavResNetCLIPPolicy \
  habitat_baselines.rl.ddppo.train_encoder=False \
  habitat_baselines.rl.ddppo.backbone=resnet50_clip_avgpool \
  habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
  habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
