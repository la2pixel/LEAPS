# Usage: source setup.sh   (SCONE with Hyfydy must already be installed, see README)
conda env create -f environment.yml
conda activate leaps

# pinned forks: sconegym (LEAPS reward variants/envs) and depRL (MPO/DEP-MPO, imports leaps.envs)
mkdir -p deps
git clone https://github.com/la2pixel/sconegym.git deps/sconegym
git -C deps/sconegym checkout 5fe39ac4fcaa90c12ed45c5d254c547e7cd8e392
git clone https://github.com/la2pixel/depRL.git deps/depRL
git -C deps/depRL checkout 485fc9375ad76cfc2969a081b88ae951fe0d821e

pip install -e deps/sconegym -e deps/depRL -e ".[dev,wandb]"
