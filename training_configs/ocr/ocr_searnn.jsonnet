{
  "dataset_reader": {
    "type": "lmpl_seq2seq",
    "source_token_indexers": {
      "tokens": {
        "type": "ocr_indexer",
       "binary_string_dim": 128,
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
  "train_data_path": "data/ocr/train.txt",
  "validation_data_path": "data/ocr/valid.txt",
  "test_data_path": "data/ocr/test.txt",
  "evaluate_on_test": true,
  "model": {
    "type": "lmpl_composed_lm",
    "use_in_seq2seq_mode": true,
    "decoder": {
        "type": "lmpl_searnn_decoder",
        "max_decoding_steps": 14,
        "generation_batch_size": 1024, 
        "rollin_mode":  std.extVar("rollin_mode"),
        "rollout_mode": std.extVar("rollout_mode"),
        "decoder_net": {
            "type": "lmpl_lstm_cell",
            "decoding_dim": 256, 
            "target_embedding_dim": 128,
        },
        "target_embedder": {
          "vocab_namespace": "target_tokens",
          "embedding_dim": 128, 
        },
        "use_in_seq2seq_mode": true, 
        "target_namespace": "target_tokens",
        "beam_size": 1,
        "use_hamming" : true,
        "dropout": 0.2,
        "rollout_cost_function": {
            "type": "hamming",
        },
        "temperature": 1000,
    },
    "source_embedder": {
      "tokens": {
        "type": "ocr_token_embedder",
        "hidden_dim": 128,
        "binary_str_size": 128,
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 128,
      "hidden_size": 128,
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
    "batch_size" : 48,
    "sorting_keys": [["source_tokens", "num_tokens"]],
  },
  "trainer": {
    "num_epochs": 150,
    "patience": 10,
    "cuda_device": 0,
    "validation_metric": "-hamming",
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "min",
      "patience": 2
    },
    "should_log_learning_rate": true,
    "should_log_parameter_statistics": false
  }
}
