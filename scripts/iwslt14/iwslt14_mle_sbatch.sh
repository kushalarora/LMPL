#/bin/sh

set -eux

module load httpproxy
export DISTRIBUTED="true";
export NUM_GPUS=4;
export DEBUG="false";
sbatch -J iwslt_mle -t 6:00:00 --gres=gpu:4 ./scripts/iwslt14/iwslt14_mle.sh