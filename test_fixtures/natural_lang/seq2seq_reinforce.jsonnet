local rl_config = import "seq2seq_mle.jsonnet";

local rollout_cost_function = { "type" : "bleu"};
local loss_criterion = {
          "type": "reinforce",
          "temperature": 1,
          "rollout_cost_function": rollout_cost_function,
      };
rl_config + {
      "model"+: {
        "decoder"+: {
          "type": "lmpl_reinforce_decoder",
          "generation_batch_size": 128,
          "loss_criterion": loss_criterion,
          "rollout_ratio": 1.0,
          "rollin_rollout_mixing_coeff": 0.0,
          "detach_rollin_logits": false,
        },
      },
      "data_loader"+: {
        "batch_sampler"+: {
          "batch_size": 96,
        },
      },
      "trainer"+: {
        "num_epochs": 10,
      },
    }