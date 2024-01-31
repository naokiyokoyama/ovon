#!/bin/bash
#SBATCH --job-name=ovon
#SBATCH --output=slurm_logs/eval/ovon-tf-%j.out
#SBATCH --error=slurm_logs/eval/ovon-tf-%j.err
#SBATCH --gpus a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --partition=cvmlp-lab,overcap
#SBATCH --qos=short
#SBATCH --signal=USR1@100

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

source /srv/flash1/rramrakhya3/summer_2023/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate ovon-v2

export PYTHONPATH= #/srv/flash1/rramrakhya3/spring_2023/habitat-sim/src_python/

DATA_PATH="data/datasets/objectnav/hm3d/v2/"
eval_ckpt_path_dir="data/new_checkpoints/objectnav/ddppo/vc1_llama/seed_3/ckpt.0.pth"
tensorboard_dir="tb/objectnav/ddppo/vc1_llama/seed_2/eval_debug/"
split="val"

echo "Evaluating ckpt: ${eval_ckpt_path_dir}"
echo "Data path: ${DATA_PATH}/${split}/${split}.json.gz"

python -um ovon.run \
  --run-type eval \
  --exp-config config/experiments/rl_transformer_hm3d.yaml \
  -cvt \
  habitat_baselines.trainer_name=transformer_ddppo \
  habitat_baselines.tensorboard_dir=$tensorboard_dir \
  habitat_baselines.eval_ckpt_path_dir=$eval_ckpt_path_dir \
  habitat_baselines.checkpoint_folder=$eval_ckpt_path_dir \
  habitat.dataset.data_path="${DATA_PATH}/${split}/${split}.json.gz" \
  habitat_baselines.num_environments=2 \
  habitat_baselines.rl.ppo.training_precision="float32" \
  habitat_baselines.rl.policy.transformer_config.inter_episodes_attention=False \
  habitat_baselines.rl.policy.transformer_config.add_sequence_idx_embed=False  \
  habitat_baselines.rl.policy.transformer_config.reset_position_index=False    \
  habitat_baselines.rl.policy.transformer_config.max_position_embeddings=2000    \
  habitat.environment.max_episode_steps=500 \
  habitat.dataset.data_path="data/datasets/objectnav/hm3d/v2/val/val.json.gz" \
  habitat.dataset.split=val \
  habitat_baselines.eval.use_ckpt_config=False \
  habitat_baselines.load_resume_state_config=False \
  habitat.simulator.habitat_sim_v0.allow_sliding=False \
  habitat_baselines.eval.split=$split
