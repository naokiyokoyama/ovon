#!/bin/bash
#SBATCH --job-name=ovon-dgen
#SBATCH --output=slurm_logs/dataset-%j.out
#SBATCH --error=slurm_logs/dataset-%j.err
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --constraint="a40|rtx_6000|2080_ti"
#SBATCH --partition=short
#SBATCH --exclude calculon,alexa,cortana,bmo,c3po,ripl-s1,t1000,hal,irona,fiona
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

# SPLIT=$1
# NUM_TASKS=$2
# OUTPUT_PATH=$3
# NUM_SCENES=-1
export PYTHONPATH=/srv/flash1/rramrakhya3/spring_2023/habitat-sim/src_python/
export HOME=/srv/flash1/rramrakhya3/summer_2023/

srun python ovon/dataset/languagenav_generator.py \
  --scene $scene \
  --split $split \
  --num-scenes $num_scenes \
  --tasks-per-gpu $num_tasks \
  --output-path $output_path \
  --start-poses-per-object $start_poses_per_object \
  --episodes-per-object $episodes_per_object \
  --with-start-poses
