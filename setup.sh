# Record of how the environment was set up
conda_env_name=ovon
mamba create -n $conda_env_name python=3.7 cmake=3.14.0 -y
mamba install -n $conda_env_name \
  habitat-sim withbullet headless pytorch cudatoolkit=11.3 \
  -c pytorch -c nvidia -c conda-forge -c aihabitat -y
git clone git@github.com:vincentpierre/habitat-lab.git
cd habitat-lab
git checkout 9fd904925cf3ad222f3c88b5a3a0b90f5505f017
git cherry-pick 6adc3b78cac8b98bbfa94d61cfa9f18250593651
mamba activate $conda_env_name
pip install -e habitat-lab
pip install -e habitat-baselines
