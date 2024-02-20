#!/bin/bash

path=$1
split=$2
model=$3

export OPENAI_APIKEY=

count=0
prompt_meta_files=`ls ${path}/${split}/content/*_meta.json`
for i in ${prompt_meta_files[@]}
do
  scene_id=`basename $i`
  base=${scene_id%.*}  # remove .gz
  base=${base%.*}  # remove .json

  if [ -f "${path}/${split}/content/${base}_annotated.json" ]; then
    echo "Skipping ${base}"
    continue
  fi

  meta_path="${path}/${split}/content/${base}.json"
  meta_output_path="${path}/${split}/content/${base}_annotated.json"

  echo "Submitting ${base} - ${count}"
  python ovon/dataset/generate_captions.py --path $meta_path --output-path $meta_output_path --model $model
  count=$((count + 1))
done
