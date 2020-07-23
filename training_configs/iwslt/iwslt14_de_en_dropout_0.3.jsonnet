local config = import "iwslt14_de_en.json";

config + {
  "model" +: {
    "dropout": 0.3
  }
}