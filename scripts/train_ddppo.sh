python -m habitat_baselines.run \
  --run-type train \
  --exp-config habitat-baselines/habitat_baselines/config/objectnav/ddppo_objectnav_hm3d.yaml \
  habitat.dataset.data_path ${OVON_DATASET}/{split}/{split}.json.gz
