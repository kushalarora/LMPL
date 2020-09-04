#!/bin/sh

set -eux

export run_id=$(date '+%Y_%m_%d_%H_%M')
OUT_DIR=results/iwslt/reinforce/$run_id/

#allennlp train training_configs/iwslt/iwslt14_de_en.jsonnet -s ${OUT_DIR}/mle/ --include-package lmpl 

WARM_START_MODEL=${OUT_DIR}/mle allennlp train training_configs/iwslt/iwslt14_de_en_reinforce.jsonnet -s ${OUT_DIR}/reinforce/ --include-package lmpl 
