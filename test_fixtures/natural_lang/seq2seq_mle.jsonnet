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
  "train_data_path": "test_fixtures/natural_lang/seq2seq_copy.tsv",
  "validation_data_path": "test_fixtures/natural_lang/seq2seq_copy.tsv",
  "model": {
    "type": "lmpl_composed_lm",
    "use_in_seq2seq_mode": true,
    "decoder": {
        "type": "lmpl_auto_regressive_seq_decoder",
        "max_decoding_steps": 50,
        "decoder_net": {
            "type": "lmpl_lstm_cell",
            "decoding_dim": 12, 
            "target_embedding_dim": 12,
            "attention": {
              "type": "dot_product"
            }
        },
        "target_embedder": {
          "vocab_namespace": "target_tokens",
          "embedding_dim": 12, 
        },
        "loss_criterion": {
          "type": "mle",
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
          "embedding_dim": 12,
          "trainable": true
        }
      },
    },
    "encoder": {
      "type": "lstm",
      "input_size": 12,
      "hidden_size": 12,
      "num_layers": 1,
      "dropout": 0,
      "bidirectional": false,
    },
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "padding_noise": 0.0,
      "batch_size": 48,
    }
  },
  "trainer": {
    "num_epochs": 1,
    "cuda_device": -1,
    "optimizer": {
      "type": "adam",
      "lr": 0.0001
    },
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    },
    "batch_callbacks": ["update_epoch_iter"],
  }
}
