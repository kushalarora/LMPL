{
    "dataset_reader": {
      "type": "quant_exp_language_modeling",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "namespace": "target_tokens"
        },
      },
      "tokenizer": {
        "type": "pretrained_transformer",
        "model_name": "gpt2",
        "start_tokens": ["<S>"],
        "end_tokens": ["</S>"],
        "do_lowercase": false,
      },
    },
    "train_data_path": "data/ptb/ptb.train.txt",
    "validation_data_path": "data/ptb/ptb.valid.txt",
    "model": {
      "type": "quant_exp_composed_lm",
      "use_in_seq2seq_mode": false,
      "decoder": {
        "type": "quant_exp_auto_regressive_seq_decoder",
        "max_decoding_steps": 100,
        "generation_batch_size": 32, 
        "decoder_net": {
          "type": "quant_exp_bias_lstm_cell",
          "decoding_dim": 128, 
          "target_embedding_dim": 128,
          # This doesn't seem to be working as of
          # now.
          "num_decoder_layers": 2,
        },
        "target_embedder": {
          "vocab_namespace": "target_tokens",
          "embedding_dim": 128,
        },
        "use_in_seq2seq_mode": false,
        "target_namespace": "target_tokens",
        "beam_size": 1,
        "use_bleu" : false,
        "sample_output": true,
        "start_token": "<S>",
        "end_token": "</S>",
        "oracle": {
          "type": "gpt2_oracle",
          "model_name": "gpt2"
        },
        "detokenizer": {
          "type": "gpt2_detokenizer",
          "model_name": "gpt2"
        },
      }
  },
  "iterator": {
      "type": "bucket",
      "sorting_keys": [["target_tokens", "num_tokens"]],
      "batch_size": 128,
      // This is needed stupidly for bucket iterator to work.
      "max_instances_in_memory": 50000
  },
  "trainer": {
    "num_epochs": 10,
    "cuda_device" : 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.001,
    },
    "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "min",
        "patience": 0
    },
    "patience": 5,
    "should_log_learning_rate": true,
    "log_batch_size_period": 500,
  }
}
