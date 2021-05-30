#!/bin/sh

set -eux

./scripts/launcher_basic.sh training_configs/iwslt/hyperparam_search_configs/iwslt14_de_en_transformer_lstm.jsonnet "iwslt_mle_transformer_lstm"
