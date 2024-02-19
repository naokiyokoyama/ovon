#!/bin/bash

split=$1
content_dir="data/datasets/ovon/hm3d/v3_shuffled/${split}/content"
output_path="data/datasets/ovon/hm3d/v3_shuffled_cleaned/${split}/content"

gz_files=`ls ${content_dir}/*json.gz`
for i in ${gz_files[@]}
do
  scene_id=`basename $i`
  base=${scene_id%.*}  # remove .gz
  base=${base%.*}  # remove .json

  if [ -f "${output_path}/${base}.json.gz" ]; then
    echo "Skipping ${base}"
    continue
  fi
  echo "Submitting ${base}"

  sbatch --job-name=clean-${split}-${base} \
    --output=slurm_logs/dataset/clean-${split}-${base}.out \
    --error=slurm_logs/dataset/clean-${split}-${base}.err \
    --export=ALL,scene_path=$i,output_path=$output_path \
    scripts/dataset/clean_scene_dataset.sh
done

