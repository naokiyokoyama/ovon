#!/bin/bash
#SBATCH --job-name=ovon
#SBATCH --output=slurm_logs/eval/ovon-ver-%j.out
#SBATCH --error=slurm_logs/eval/ovon-ver-%j.err
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --constraint="a40|rtx_6000"
#SBATCH --partition=short
#SBATCH --exclude=cheetah,samantha,xaea-12,kitt,calculon,vicki,neo,kipp,ripl-s1,tars
#SBATCH --signal=USR1@100

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

source /srv/flash1/rramrakhya3/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate ovon

export PYTHONPATH=/srv/flash1/rramrakhya3/spring_2023/habitat-sim/src_python/

DATA_PATH="data/datasets/ovon/hm3d/v5_final/"
eval_ckpt_path_dir="data/new_checkpoints/ovon/ver/resnetclip_rgb_text/seed_1/ckpt.80.pth"
tensorboard_dir="tb/ovon/ver/resnetclip_rgb_text/seed_1/eval_val_seen_debug/"
split="val_seen"
OVON_EPISODES_JSON="data/analysis/episode_metrics/rl_ckpt_80_additional_metrics_${split}.json"

echo "Evaluating ckpt: ${eval_ckpt_path_dir}"
echo "Data path: ${DATA_PATH}/${split}/${split}.json.gz"

srun python -um ovon.run \
  --run-type eval \
  --exp-config config/experiments/ver_objectnav.yaml \
  -cvt \
  habitat_baselines.num_environments=2 \
  habitat_baselines.test_episode_count=4 \
  habitat_baselines.trainer_name=ver_pirlnav \
  habitat_baselines.rl.policy.name=PointNavResNetCLIPPolicy \
  habitat_baselines.tensorboard_dir=$tensorboard_dir \
  habitat_baselines.eval_ckpt_path_dir=$eval_ckpt_path_dir \
  habitat_baselines.checkpoint_folder=$eval_ckpt_path_dir \
  habitat.dataset.data_path="${DATA_PATH}/${split}/${split}.json.gz" \
  +habitat/task/lab_sensors@habitat.task.lab_sensors.clip_objectgoal_sensor=clip_objectgoal_sensor \
  habitat.task.lab_sensors.clip_objectgoal_sensor.cache=data/clip_embeddings/ovon_stretch_cache.pkl \
  habitat.task.measurements.success.success_distance=0.25 \
  habitat.dataset.type="OVON-v1" \
  habitat.task.measurements.distance_to_goal.type=OVONDistanceToGoal \
  +habitat/task/measurements@habitat.task.measurements.ovon_object_goal_id=ovon_object_goal_id \
  +habitat/task/measurements@habitat.task.measurements.failure_modes=failure_modes \
  habitat.simulator.type="OVONSim-v0" \
  habitat_baselines.eval.use_ckpt_config=False \
  habitat_baselines.load_resume_state_config=False \
  habitat.simulator.habitat_sim_v0.allow_sliding=False \
  habitat_baselines.eval.split=$split

touch $checkpoint_counter
