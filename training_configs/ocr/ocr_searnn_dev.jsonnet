local config = import "ocr_searnn.jsonnet";

config + {
  "train_data_path": "data/ocr/valid.txt",
}