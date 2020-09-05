local bidirection_input = true;
local encoder_input_dim = 512;
local encoder_hidden_dim = 512;
local decoder_embedding_dim = 512;
local decoder_hidden_dim = if bidirection_input then encoder_hidden_dim * 2 else encoder_hidden_dim;
local num_decoder_layers = 2;
local num_encoder_layers = 2;
local dropout_ratio = 0.3;
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
    },
    // "cache_directory": "data/iwslt/",
    "target_max_tokens": 50,
  },
  // "vocabulary": {
  //   "max_vocab_size": { 
  //       "source_tokens": 32009, 
  //       "target": 22822
  //   }
  // },
  "train_data_path": "data/iwslt/train.tsv",
  "validation_data_path": "data/iwslt/valid.tsv",
  "test_data_path": "data/iwslt/test.tsv",
  "evaluate_on_test": true,
  "model": {
    "type": "lmpl_composed_lm",
    "use_in_seq2seq_mode": true,
    "decoder": {
        "type": "lmpl_auto_regressive_seq_decoder",
        "max_decoding_steps": 55,
        "decoder_net": {
            "type": "lmpl_lstm_cell",
            "decoding_dim": decoder_hidden_dim, 
            "target_embedding_dim": decoder_embedding_dim,
            "bidirectional_input": bidirection_input,
            "num_decoder_layers": num_decoder_layers,
            "dropout": dropout_ratio,
            "attention": {
                // "type": "additive",
                // "vector_dim": decoder_hidden_dim,
                // "matrix_dim": decoder_hidden_dim,
                "type": "dot_product",
            },
        },
        "target_embedder": {
          "vocab_namespace": "target_tokens",
          "embedding_dim": decoder_embedding_dim, 
        },
        "loss_criterion": {
          "type": "mle",
        },
        "use_in_seq2seq_mode": true, 
        "target_namespace": "target_tokens",
        "beam_size": 1,
        "use_bleu" : true,
        "dropout": dropout_ratio,
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
      "type": "lstm",
      "input_size": encoder_input_dim,
      "hidden_size": encoder_hidden_dim,
      "num_layers": num_encoder_layers,
      "dropout": dropout_ratio,
      "bidirectional": bidirection_input,
    },
    "initializer": {
      "regexes": [
        ["embedder*.*weight", {"type": "kaiming_uniform"}],
        [".*projection_layer.*weight", {"type": "xavier_uniform"}],
        [".*projection_layer.*bias", {"type": "zero"}],
        [".*weight_ih.*", {"type": "xavier_uniform"}],
        [".*weight_hh.*", {"type": "orthogonal"}],
        [".*bias_ih.*", {"type": "zero"}],
        [".*bias_hh.*", {"type": "lstm_hidden_bias"}],
        [".*attention.*", {"type": "zero"}],
      ],
    }
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "padding_noise": 0.0,
      "batch_size": 36,
      "sorting_keys": ["target_tokens", "source_tokens"],
    },
    "num_workers": 4,
    "pin_memory": true,
  },
  "trainer": {
    "validation_metric": "+BLEU",
    "num_epochs": 50,
    "patience": 10,
    // "use_amp": true,
    // "opt_level": "O2",
    "cuda_device": 0,
    "grad_norm": 0.1,
    "optimizer": {
      "type": "sgd",
      "lr": 0.5,
      "momentum": 0.95
    },
    "learning_rate_scheduler": {
      "type": "multi_step",
      "milestones": [10, 20, 30, 40],
    },
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    },
    "epoch_callbacks": [{
      "type": 'log_metrics_to_wandb',
      "project_name": "lmpl_debug",
      "run_name": "mle",
      "sync_tensorboard": false,
    },],
  }
}
