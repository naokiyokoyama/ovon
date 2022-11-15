python -m habitat_baselines.run \
  --run-type train \
  --exp-config habitat_baselines/config/objectnav/ddppo_objectnav_hm3d.yaml \
  TASK_CONFIG.DATASET.DATA_PATH ${OVON_DATASET}/{split}/{split}.json.gz
