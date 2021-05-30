#!/bin/sh

set -eux
./scripts/launcher_basic.sh training_configs/iwslt/iwslt14_de_en.jsonnet "iwslt_mle_lstm"
