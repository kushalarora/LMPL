#!/bin/sh

set -ex
export WARM_START_MODEL=/home/karora/scratch/lmpl/results/iwslt/mle/2021_05_27_21_08/mle
/;

./scripts/launcher_basic.sh allennlp train training_configs/iwslt/iwslt14_de_en_reinforce.jsonnet "iwslt_reinforce"
