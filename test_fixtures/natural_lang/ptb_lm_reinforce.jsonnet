local emnlp_gpt2_rl_config = import "ptb_lm.jsonnet";

local rollout_cost_function = {
          "type": "bleu",
        };

local loss_criterion = {
          "type": "reinforce",
          "rollout_cost_function": rollout_cost_function,
      };

emnlp_gpt2_rl_config + {
      "model"+: {
        "decoder"+: {
          "type": "lmpl_reinforce_decoder",
          "generation_batch_size": 128,
          "loss_criterion": loss_criterion,
          "rollout_ratio": 1.0,
          "rollin_rollout_mixing_coeff": 0.0,
          "detach_rollin_logits": false,
          "sample_rollouts": false,
          "decode_rollouts": true,
        },
      },
      "data_loader"+: {
        "batch_sampler"+: {
          "batch_size": 16,
        },
      },
      "trainer"+: {
        "num_epochs": 10,
        "validation_metric": "-loss",
      },
    }