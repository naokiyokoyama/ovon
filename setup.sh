# Record of how the environment was set up
# Create conda environment. Mamba is recommended for faster installation.
conda_env_name=ovon
mamba create -n $conda_env_name python=3.7 cmake=3.14.0 -y
mamba install -n $conda_env_name \
  habitat-sim=0.2.3 headless pytorch cudatoolkit=11.3 \
  -c pytorch -c nvidia -c conda-forge -c aihabitat -y

# Install this repo as a package
mamba activate $conda_env_name
pip install -e .

# Install distributed_dagger and frontier_exploration
cd frontier_exploration && pip install -e . && cd ..

# Install habitat-lab
git clone --branch v0.2.3 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines

pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
