#!/bin/bash
#SBATCH --job-name=ovon
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gpus a40:4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node 4
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@90
#SBATCH --exclude=crushinator,major,chappie,deebot
#SBATCH --requeue
#SBATCH --partition=cvmlp-lab
#SBATCH --qos=short

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

source /srv/flash1/rramrakhya3/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate ovon

cd /srv/flash1/rramrakhya3/spring_2023/ovon

export PYTHONPATH=/srv/flash1/rramrakhya3/spring_2023/habitat-sim/src_python/

TENSORBOARD_DIR="tb/objectnav/ddppo/vc1_llama/seed_1/debug/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav/ddppo/vc1_llama/seed_1/debug/"
DATA_PATH="data/datasets/objectnav/hm3d/v2"

srun python ovon/run.py --exp-config config/experiments/rl_transformer_hm3d.yaml \
    --run-type train \
    habitat_baselines.checkpoint_folder=$CHECKPOINT_DIR/ \
    habitat_baselines.tensorboard_dir=$TENSORBOARD_DIR \
    habitat_baselines.num_environments=2 \
    habitat_baselines.rl.policy.transformer_config.inter_episodes_attention=False \
    habitat_baselines.rl.policy.transformer_config.add_sequence_idx_embed=False  \
    habitat_baselines.rl.policy.transformer_config.reset_position_index=False    \
    habitat_baselines.rl.policy.transformer_config.max_position_embeddings=2000    \
    habitat.environment.max_episode_steps=500 \
    habitat.dataset.data_path=${DATA_PATH}/train/train.json.gz
