local emnlp_gpt2_rl_config = import "emnlp_news_gpt2.jsonnet";

local rollout_cost_function = {
                                "type": "noisy_oracle",
                                "add_brevity_penalty": true,
                                "oracle": {
                                  "type": "gpt2_oracle",
                                  "model_name": "distilgpt2",
                                  "batch_size": 16,
                                  "cuda_device": -2,
                                }
                              };

emnlp_gpt2_rl_config + {
      "vocabulary": {
        "type": "from_files",
        "directory": std.extVar("VOCAB_PATH"),
      },
      "model"+: {
        "decoder"+: {
          "type": "lmpl_reinforce_decoder",
          "generation_batch_size": 128,
          "rollout_cost_function": rollout_cost_function,
          "rollout_ratio": 0.33,
          "rollin_rollout_mixing_coeff": 0.5,
          "detach_rollin_logits": false,
        },
        "initializer": {
          "regexes": [
            ["_decoder._decoder_net.*|_decoder._output_projection*|_decoder.target_embedder*|_decoder._dropout",
              {
                "type": "pretrained",
                "weights_file_path": std.extVar("WEIGHT_FILE_PATH"),
              },
            ],
          ],
        },
      },
      "data_loader"+: {
        "batch_sampler"+: {
          "batch_size": 16,
        },
      },
      "trainer"+: {
        "validation_metric": "-loss",
      },
    }