set -eux

module load python/3.8;
module load SciEnv/2020;
virtualenv ${HOME}/envs/lmpl;

source ${HOME}/envs/lmpl/bin/activate;

pip install -r requirements.txt

