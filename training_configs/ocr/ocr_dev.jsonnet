local config = import "ocr.jsonnet";

config + {
  "train_data_path": "data/ocr/valid.txt",
}