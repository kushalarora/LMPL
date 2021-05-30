set -eux

virtualenv ${HOME}/envs/lmpl;

source ${HOME}/envs/lmpl/bin/activate;

pip install -r requirements.txt

