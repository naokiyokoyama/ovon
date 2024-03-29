#!/bin/bash
#SBATCH --job-name=ovon
#SBATCH --output=slurm_logs/ovon-ver-%j.out
#SBATCH --error=slurm_logs/ovon-ver-%j.err
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --constraint=a40
#SBATCH --partition=short
#SBATCH --exclude=cheetah,samantha,xaea-12,kitt
#SBATCH --signal=USR1@100
#SBATCH --requeue

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

source /srv/flash1/rramrakhya3/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate ovon

export PYTHONPATH=/srv/flash1/rramrakhya3/spring_2023/habitat-sim/src_python/

TENSORBOARD_DIR="tb/ovon/ver/resnet_scratch_clip_goal/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/ovon/ver/resnet_scratch_clip_goal/seed_1/"
DATA_PATH="data/datasets/ovon/hm3d/v5_final"

srun python -um ovon.run \
  --run-type train \
  --exp-config config/experiments/ver_objectnav.yaml \
  habitat_baselines.trainer_name="ver" \
  habitat_baselines.num_environments=32 \
  habitat_baselines.rl.policy.name=OVRLPolicy \
  habitat_baselines.rl.ddppo.train_encoder=True \
  habitat_baselines.rl.policy.backbone=resnet50 \
  habitat_baselines.rl.policy.freeze_backbone=False \
  habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
  habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
  habitat.dataset.data_path=${DATA_PATH}/train/train.json.gz \
  +habitat/task/lab_sensors@habitat.task.lab_sensors.clip_objectgoal_sensor=clip_objectgoal_sensor \
  ~habitat.task.lab_sensors.objectgoal_sensor \
  habitat.task.lab_sensors.clip_objectgoal_sensor.cache=data/clip_embeddings/ovon_stretch_final_cache.pkl \
  habitat.task.measurements.success.success_distance=0.25 \
  habitat.dataset.type="OVON-v1" \
  habitat.task.measurements.distance_to_goal.type=OVONDistanceToGoal \
  habitat.simulator.type="OVONSim-v0"
