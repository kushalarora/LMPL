local config = import "iwslt14_de_en.jsonnet";

config + {
  "model"+: {
    "decoder"+: {
        "type": "lmpl_composed_lm",
        "generation_batch_size": 32, 
        "rollin_mode": "teacher_forcing",
        "rollout_mode": "reference",
        "temperature": 5,
        "num_neighbors_to_add": 0,
        "num_tokens_to_rollout": 25,
        "rollout_cost_function": {
          "type": "bleu",
        },
    },
  },
  "data_loader"+: {
    "batch_sampler"+: {
      "batch_size": 32,
    },
  },
}