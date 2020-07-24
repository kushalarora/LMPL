# Language Modeling as Policy Learning (LMPL) library.
# Installation:
```
python setup.py --editable
```

## Download spacy model:

```
python -m spacy download en
```

or

```
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-1.2.0/en_core_web_sm-2.2.0.tar.gz
```

## Install Apex for half precision training:
```pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --force-reinstall git+https://github.com/nvidia/apex.git```


## Running PTB experiments:
```
  allennlp train training_configs/natural_lang/ptb.jsonnet -s result/lm/ptb --include-package lmpl
```

## Running IWSLT14 de-en experiments:
```
  allennlp train training_configs/iwslt/iwslt_de_en.jsonnet -s result/iwslt/ --include-package lmpl
```
