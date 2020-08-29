#!/bin/sh

set -eux

OUT_DIR=results/iwslt/reinforce/$(date '+%Y_%m_%d_%H_%M')/

allennlp train training_configs/iwslt/iwslt14_de_en.jsonnet -s ${OUT_DIR}/mle/ --include-package lmpl 
