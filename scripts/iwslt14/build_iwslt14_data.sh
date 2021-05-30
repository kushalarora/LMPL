#!/bin/sh

set -eux

mkdir -p data/iwslt14
cd data/iwslt14
#bash ../../scripts/iwslt14/prepare-iwslt14.sh
python ../../scripts/convert_src_tgt_tsv.py . 
python ../../scripts/split_into_4.py train.tsv
python ../../scripts/split_into_4.py valid.tsv
python ../../scripts/split_into_4.py test.tsv
cd ../../
