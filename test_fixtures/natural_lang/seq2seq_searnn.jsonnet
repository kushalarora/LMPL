local searnn_config = import "seq2seq_mle.jsonnet";

local rollout_cost_function = { "type" : "hamming"};

local loss_criterion = {
          "type": "searnn-kl",
          "rollout_cost_function": rollout_cost_function,
      };

searnn_config + {
  'model'+: {
    'decoder'+: {
          "type": "lmpl_searnn_decoder",
          "generation_batch_size": 128,
          "loss_criterion": loss_criterion,
          "rollin_mode":  "teacher_forcing",
          "rollout_mode": "reference",
          "num_tokens_to_rollout": 200,
          "rollout_ratio": 1.0,
          "decode_rollouts": true,
    },
  },
  "data_loader"+: {
    "batch_sampler"+: {
      "batch_size": 96,
    },
  },
  "trainer"+: {
    "validation_metric": "-loss",
    "optimizer": {
      "type": "adam",
      "lr": 0.01
    },
  },
}