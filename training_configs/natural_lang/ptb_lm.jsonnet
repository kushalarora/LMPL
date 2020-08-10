{
    "dataset_reader": {
      "type": "lmpl_language_modeling",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "namespace": "target_tokens"
        },
      },
      "start_tokens": ["<S>"],
      "end_tokens": ["</S>"]
    },
    "train_data_path": "data/ptb/ptb.train.txt",
    "validation_data_path": "data/ptb/ptb.valid.txt",
    "model": {
      "type": "lmpl_composed_lm",
      "use_in_seq2seq_mode": false,
      "decoder": {
        "type": "lmpl_auto_regressive_seq_decoder",
        "max_decoding_steps": 400,
        "generation_batch_size": 32, 
        "decoder_net": {
          "type": "lmpl_lstm_cell",
          "decoding_dim": 1200, 
          "target_embedding_dim": 400,
          # This doesn't seem to be working as of
          # now.
          // "num_decoder_layers": 4,
        },
        "loss_criterion": {
          "type": "mle",
        },
        "target_embedder": {
          "vocab_namespace": "target_tokens",
          "embedding_dim": 400
        },
        "use_in_seq2seq_mode": false,
        "target_namespace": "target_tokens",
        "beam_size": 1,
        "use_bleu" : false,
        "dropout": 0.2,
        "start_token": "<S>",
        "end_token": "</S>",
      },
      // "initializer": {
      //   "regexes": [
      //     ["embedder*.*weight", {"type": "kaiming_uniform"}],
      //     [".*projection_layer.*weight", {"type": "xavier_uniform"}],
      //     [".*projection_layer.*bias", {"type": "zero"}],
      //     [".*weight_ih.*", {"type": "xavier_uniform"}],
      //     [".*weight_hh.*", {"type": "orthogonal"}],
      //     [".*bias_ih.*", {"type": "zero"}],
      //     [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
      //   ],
      // }
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "padding_noise": 0.0,
      "batch_size": 128,
    }
  },
  "trainer": {
    "num_epochs": 100,
    "opt_level": 'O2',
    "cuda_device" : 0,
    "validation_metric": "-perplexity",
    "optimizer": {
      "type": "sgd",
      "lr": 2,
    },
    "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "min",
        "patience": 0
    },
    "patience": 5,
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    },
  }
}
