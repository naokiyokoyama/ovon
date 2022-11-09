# Record of how the environment was set up
mamba create -n ovon python=3.7 cmake=3.14.0 -y
mamba activate ovon
mamba install habitat-sim withbullet headless -c conda-forge -c aihabitat -y
git clone git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
pip uninstall -y torch
mamba install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia -y
