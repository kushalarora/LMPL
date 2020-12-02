local rl_config = import "iwslt14_de_en.jsonnet";
local warm_start_model = std.extVar("WARM_START_MODEL");

local rollout_cost_function = { "type" : "bleu"};
local loss_criterion = {
          "type": "reinforce",
          "temperature": 1,
          "rollout_cost_function": rollout_cost_function,
          "detach_rollin_logits": false,
          "alpha": 0.99,
      };

rl_config + {
      // "vocabulary": {
      //   "type": "from_files",
      //   "directory": warm_start_model + "/vocabulary",
      // },
      "model"+: {
        "decoder"+: {
          "type": "lmpl_reinforce_decoder",
          "generation_batch_size": 128,
          "loss_criterion": loss_criterion,
          "rollout_ratio": 0.10,
          "rollin_rollout_mixing_coeff": 0.5,
        },
        // "initializer": {
        //   "regexes": [
        //     [".*embedder*.*|_decoder._decoder_net.*|_decoder._output_projection_layer.*|_encoder.*",
        //       {
        //         "type": "pretrained",
        //         "weights_file_path": warm_start_model + "/best.th",
        //       },
        //     ],
        //   ],
        // },
      },
      "data_loader"+: {
        "batch_sampler"+: {
          "batch_size": 32,
        },
      },
      "trainer"+: {
        "num_epochs": 80,
        "grad_clipping": 5.0,
        "grad_norm": 100.0,
        "optimizer": {
          "type": "adamax",
          // "lr": 0.01,
          // "momentum": 0.95,
          // "type": "sgd",
          // "lr": 0.50,
          // "momentum": 0.95,
        },
        "epoch_callbacks": [{
          "type": 'log_metrics_to_wandb',
          "project_name": "lmpl_debug",
          "run_name": "reinforce",
          "sync_tensorboard": false,
        },],
        // "num_gradient_accumulation_steps": 4,
      },
      "distributed": {
        "cuda_devices": [0, 1, 2, 3],
      },
    }
