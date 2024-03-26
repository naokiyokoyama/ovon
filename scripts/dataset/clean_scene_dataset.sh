#!/bin/bash
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 7
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@100
#SBATCH --requeue
#SBATCH --constraint="a40|rtx_6000|2080_ti"
#SBATCH --partition=short
#SBATCH --exclude calculon,alexa,cortana,bmo,c3po,ripl-s1,t1000,hal,irona,fiona

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

source /srv/flash1/rramrakhya3/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate ovon

echo "\n"
echo $scene_path
echo $(which python)
echo "ola"

srun python ovon/dataset/clean_episodes.py --path $scene_path --output-path $output_path