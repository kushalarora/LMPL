{
  "dataset_reader": {
    "type": "lmpl_seq2seq",
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
  "train_data_path": "data/iwslt/train.de-en.tsv",
  "validation_data_path": "data/iwslt/dev.de-en.tsv",
  "model": {
    "type": "lmpl_composed_lm",
    "use_in_seq2seq_mode": true,
    "decoder": {
        "type": "lmpl_auto_regressive_seq_decoder",
        "max_decoding_steps": 50,
        "decoder_net": {
            "type": "lmpl_lstm_cell",
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
    },
    "source_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "source_tokens",
          "embedding_dim": 256,
          "trainable": true
        }
      },
    },
    "encoder": {
      "type": "lstm",
      "input_size": 256,
      "hidden_size": 256,
      "num_layers": 1,
      "dropout": 0,
      "bidirectional": true
    },
    "initializer": {
      "regexes": [
        ["embedder*.*weight", {"type": "kaiming_uniform"}],
        [".*projection_layer.*weight", {"type": "xavier_uniform"}],
        [".*projection_layer.*bias", {"type": "zero"}],
        [".*weight_ih.*", {"type": "xavier_uniform"}],
        [".*weight_hh.*", {"type": "orthogonal"}],
        [".*bias_ih.*", {"type": "zero"}],
        [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
      ],
    }
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "padding_noise": 0.0,
      "batch_size": 48,
    }
  },
  "trainer": {
    "num_epochs": 80,
    "patience": 10,
    "opt_level": "O2",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.0001
    },
    "learning_rate_scheduler": {
      "type": "exponential",
      "gamma": 0.99
    },
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    },
  }
}
