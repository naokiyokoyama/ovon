# Record of how the environment was set up
conda_env_name=ovon
mamba create -n $conda_env_name python=3.7 cmake=3.14.0 -y
mamba install -n $conda_env_name \
  habitat-sim withbullet headless pytorch pytorch-cuda=11.6 \
  -c pytorch -c nvidia -c conda-forge -c aihabitat -y
git clone git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
mamba activate $conda_env_name
pip install -e habitat-lab
pip install -e habitat-baselines
