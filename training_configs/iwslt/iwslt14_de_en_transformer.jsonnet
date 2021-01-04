local encoder_input_dim = 512;
local num_encoder_layers = 6;
local encoder_feedforward_hidden_dim = 1024;
local encoder_num_attention_heads = 4;
local encoder_dropout_ratio = 0.1;
local decoder_dropout_ratio = 0.1;

local dropout_ratio = 0.1;
local decoder_embedding_dim = 512;
local decoder_feedforward_hidden_dim = 1024;
local decoder_hidden_dim = 512;
local num_decoder_layers = 1;
local decoder_num_attention_heads = 4;
local loss_criterion = {
          "type": "mle",
          "labeling_smooting_ratio": 0.1,
        };
local dataset_reader = {
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
    },
    // "cache_directory": "data/iwslt/",
    "target_max_tokens": 175,
    "source_max_tokens": 175,
    "source_to_target_len_max_ratio": 1.5,
  };
{
  "dataset_reader": {
    "type": "sharded",
    "base_reader": dataset_reader,
  },
  // "vocabulary": {
  //   "max_vocab_size": { 
  //       "source_tokens": 32009, 
  //       "target": 22822
  //   }
  // },
  "train_data_path": "data/iwslt/train_*.tsv",
  "validation_data_path": "data/iwslt/valid_*.tsv",
  // "test_data_path": "data/iwslt/test_*.tsv",
  // "evaluate_on_test": true,
  "model": {
    "type": "lmpl_composed_lm",
    "use_in_seq2seq_mode": true,
    "decoder": {
        "type": "lmpl_auto_regressive_seq_decoder",
        "max_decoding_steps": 200,
        "decoder_net": {
            "type": "transformer",
            "decoding_dim": decoder_hidden_dim, 
            "target_embedding_dim": decoder_embedding_dim,
            "feedforward_hidden_dim": decoder_feedforward_hidden_dim,
            "num_layers": num_decoder_layers,
            "num_attention_heads": decoder_num_attention_heads,
            "dropout_prob": decoder_dropout_ratio,
        },
        "target_embedder": {
          "vocab_namespace": "target_tokens",
          "embedding_dim": decoder_embedding_dim, 
        },
        "loss_criterion": loss_criterion,
        "use_in_seq2seq_mode": true, 
        "target_namespace": "target_tokens",
        "beam_size": 1,
        "use_bleu" : true,
        "dropout": dropout_ratio,
        "eval_beam_size": 5,
    },
    "source_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "source_tokens",
          "embedding_dim": encoder_input_dim,
          "trainable": true
        }
      },
    },
    "encoder": {
      "type": "pytorch_transformer",
      "input_dim": encoder_input_dim,
      "num_layers": num_encoder_layers,
      "feedforward_hidden_dim": encoder_feedforward_hidden_dim,
      "positional_encoding": "sinusoidal",
      "num_attention_heads": encoder_num_attention_heads,
      "dropout_prob": encoder_dropout_ratio,
    },
    "initializer": {
        "regexes": [
          // [".norm*.*", {"type": "kaiming_uniform"}],
          ["_encoder._transformer.*.weight", {"type": "xavier_uniform"}],
          ["._decoder_net*.*weight", {"type": "xavier_uniform"}],
          ["embedder*.*weight", {"type": "kaiming_uniform"}],
          [".bias", {"type": "zero"}],
        ],
        "prevent_regexes": [".norm*.*"],
    },
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "padding_noise": 0.0,
      "batch_size": 64,
      "sorting_keys": ["target_tokens"],
    },
    "num_workers": 1,
    "pin_memory": true,
  },
  "trainer": {
    "validation_metric": "+BLEU",
    "num_epochs": 80,
    "patience": 10,
    // "use_amp": true,
    // "opt_level": "O2",
    "cuda_device": 0,
    "optimizer": {
      // "type": "adamax",
      "type": "adam",
      "lr": 5e-4,
      "betas": [0.9, 0.98],
      "eps": 1e-9,
      // 'weight_decay': 0.0001,
    },
    // "learning_rate_scheduler": {
    //   "type": "noam",
    //   "model_size": 512,
    //   "warmup_steps": 4000,
    // },
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    },
    "epoch_callbacks": [{
      "type": 'log_metrics_to_wandb',
      "project_name": "lmpl_debug",
      "run_name": "mle_transformer",
      "sync_tensorboard": false,
    },],
    "num_gradient_accumulation_steps": 4,
  },

  // "distributed": {
  //   "cuda_devices": [0, 1, 2, 3],
  // },
}
