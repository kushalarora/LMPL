{
  "dataset_reader": {
    "type": "quant_exp_seq2seq",
    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "source_tokens"
      }
    },
    "target_token_indexers": {
      "tokens": {
        "namespace": "target_tokens"
      }
    }
  },
  "vocabulary": {
    "max_vocab_size": { 
        "source_tokens": 32009, 
        "target": 22822
    }
  },
  "train_data_path": "data/iwslt/100.tsv",
  "validation_data_path": "data/iwslt/100.tsv",
  "model": {
    "type": "quant_exp_composed_lm",
    "use_in_seq2seq_mode": true,
    "decoder": {
        "type": "quant_exp_searnn_decoder",
        "max_decoding_steps": 50,
        "rollin_mode":  std.extVar("rollin_mode"),
        "rollout_mode": std.extVar("rollout_mode"),
        "decoder_net": {
            "type": "quant_exp_bias_lstm_cell",
            "decoding_dim": 512, 
            "target_embedding_dim": 256,
            "attention": {
                "type": "additive",
                "vector_dim": 512,
                "matrix_dim": 512
            },
        },
        "target_embedder": {
          "vocab_namespace": "target_tokens",
          "embedding_dim": 256, 
        },
        "use_in_seq2seq_mode": true, 
        "target_namespace": "target_tokens",
        "beam_size": 1,
        "use_bleu" : true,
        "dropout": 0.2,
        "rollout_cost_function": {
          "type": "bleu",
        },
        "temperature": 200,
        "num_neighbors_to_add": 5,
        "num_tokens_to_rollout": 10,
        "rollin_rollout_combination_mode": "kl",
    },
    "source_embedder": {
      "tokens": {
        "type": "embedding",
        "vocab_namespace": "source_tokens",
        "embedding_dim": 256,
        "trainable": true
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 256,
      "hidden_size": 256,
      "num_layers": 1,
      "dropout": 0,
      "bidirectional": true
    },
    "initializer": [
        ["embedder*.*weight", {"type": "kaiming_uniform"}],
        [".*projection_layer.*weight", {"type": "xavier_uniform"}],
        [".*projection_layer.*bias", {"type": "zero"}],
        [".*weight_ih.*", {"type": "xavier_uniform"}],
        [".*weight_hh.*", {"type": "orthogonal"}],
        [".*bias_ih.*", {"type": "zero"}],
        [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
    ]
  },
  "iterator": {
    "type": "bucket",
    "padding_noise": 0.0,
    "batch_size" : 4,
    "sorting_keys": [["source_tokens", "num_tokens"]],
    "max_instances_in_memory": 10000
  },
  "trainer": {
    "num_epochs": 80,
    "patience": 10,
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.0001
    },
    "learning_rate_scheduler": {
      "type": "exponential",
      "gamma": 0.99
    },
    "should_log_learning_rate": true,
    "should_log_parameter_statistics": false
  }
}
