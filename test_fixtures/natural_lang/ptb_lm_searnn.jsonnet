local emnlp_gpt2_searnn_config = import "ptb_lm.jsonnet";

local rollout_cost_function = {
          "type": "bleu",
        };

local loss_criterion = {
          "type": "searnn-kl",
          "rollout_cost_function": rollout_cost_function,
      };
      
emnlp_gpt2_searnn_config + {
  'model'+: {
    'decoder'+: {
          "type": "lmpl_searnn_decoder",
          "generation_batch_size": 128,
          "loss_criterion": loss_criterion,
          "rollin_mode":  "teacher_forcing",
          "rollout_mode": "reference",
          "num_tokens_to_rollout": 100,
          "rollout_ratio": 1.0,
          "sample_rollouts": false,
          "decode_rollouts": true,
    },
  },
  "data_loader"+: {
    "batch_sampler"+: {
      "batch_size": 10,
    },
  },
  "trainer"+: {
    "validation_metric": "-loss",
  },
}