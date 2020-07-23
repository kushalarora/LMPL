local config = import "iwslt14_de_en.jsonnet";

config + {
  "train_data_path": "data/iwslt/dev.de-en.tsv",
}