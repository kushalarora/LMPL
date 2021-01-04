#!/bin/sh

set -eux
export run_id=$(date '+%Y_%m_%d_%H_%M')
OUT_DIR=~/scratch/lmpl/results/iwslt/mle/$run_id/

sbatch -J iwslt_transformer -t 11:00:00 ./launcher_basic.sh allennlp train training_configs/iwslt/iwslt14_de_en.jsonnet -s ${OUT_DIR}/mle/ --include-package lmpl
