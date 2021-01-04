local bidirection_input = true;
local encoder_input_dim = 512;
local encoder_hidden_dim = 512;
local decoder_embedding_dim = 512;
local decoder_hidden_dim = if bidirection_input then encoder_hidden_dim * 2 else encoder_hidden_dim;
local num_decoder_layers = 2;
local num_encoder_layers = 2;
local dropout_ratio = 0.1;
local lr = std.parseJson(std.extVar('lr'));

local gpus(ngpu) =
  if ngpu == 1 then [0]
  else if ngpu == 2 then [0, 1]
  else if ngpu == 3 then [0, 1, 2]
  else if ngpu == 4 then [0, 1, 2, 3]
  else error "invalid option: " + std.manifestJson(ngpu);

local stringToBool(s) =
  if s == "true" then true
  else if s == "false" || s == '' || s == null then false
  else error "invalid boolean: " + std.manifestJson(s);

{
  seq2seq_dataset_reader(target_max_tokens=175, 
                                  source_max_tokens=175, 
                                  source_to_target_len_max_ratio=1.5)::
    {
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
      "target_max_tokens": target_max_tokens,
      "source_max_tokens": source_max_tokens,
      "source_to_target_len_max_ratio": source_to_target_len_max_ratio,
    },

  sharded_dataset_reader(dataset_reader):: 
    {
      "type": "sharded",
      "base_reader": dataset_reader,
    },

  lstm_cell_decoder_net(decoder_hidden_dim=256, 
                            encoder_hidden_dim=128,
                            decoder_embedding_dim=128,
                            bidirection_input=true,
                            num_decoder_layers=1,
                            dropout_ratio=0.1)::
    {
      "type": "lmpl_lstm_cell",
      "decoding_dim": decoder_hidden_dim, 
      "target_embedding_dim": decoder_embedding_dim,
      "bidirectional_input": bidirection_input,
      "num_decoder_layers": num_decoder_layers,
      "dropout": dropout_ratio,
      // TODO: Allow attention as a functional argument.
      "attention": {
          "type": "bilinear",
          "vector_dim": decoder_hidden_dim,
          "matrix_dim": encoder_hidden_dim,
          // "type": "dot_product",
      },
    },

  transformer_decoder_net(decoder_embedding_dim=512,
                          decoder_feedforward_hidden_dim=1024,
                          decoder_hidden_dim=512,
                          num_decoder_layers=6,
                          decoder_num_attention_heads=4,
                          decoder_dropout_ratio=0.1)::
    {
      "type": "transformer",
      "decoding_dim": decoder_hidden_dim, 
      "target_embedding_dim": decoder_embedding_dim,
      "feedforward_hidden_dim": decoder_feedforward_hidden_dim,
      "num_layers": num_decoder_layers,
      "num_attention_heads": decoder_num_attention_heads,
      "dropout_prob": decoder_dropout_ratio,
    },

  lstm_encoder(encoder_input_dim=128, 
               encoder_hidden_dim=128,
               num_encoder_layers=1,
               dropout_ratio=0.1,
               bidirection_input=true) ::
    {
      "type": "lstm",
      "input_size": encoder_input_dim,
      "hidden_size": encoder_hidden_dim,
      "num_layers": num_encoder_layers,
      "dropout": dropout_ratio,
      "bidirectional": bidirection_input,
    },

  transformer_encoder(encoder_input_dim=512, 
                      encoder_feedforward_hidden_dim=512, 
                      num_encoder_layers=6,
                      encoder_num_attention_heads=4,
                      encoder_dropout_ratio=0.1,
                    )::
    {
      "type": "pytorch_transformer",
      "input_dim": encoder_input_dim,
      "num_layers": num_encoder_layers,
      "feedforward_hidden_dim": encoder_feedforward_hidden_dim,
      "positional_encoding": "sinusoidal",
      "num_attention_heads": encoder_num_attention_heads,
      "dropout_prob": encoder_dropout_ratio,
    },

  lstm_initializer()::
    {
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
    },
  
  transformer_initializer()::
    {
      "regexes": [
        // [".norm*.weight", {"type": "zero"}],
        ["_encoder._transformer.*.weight", {"type": "xavier_uniform"}],
        ["._decoder_net*.*weight", {"type": "xavier_uniform"}],
        ["embedder*.*weight", {"type": "kaiming_uniform"}],
        [".bias", {"type": "zero"}],
      ],
      "prevent_regexes": [".norm*.*"],
    },
  
  transformer_lstm_initializer()::
    {
      "regexes": [
        ["_encoder._transformer.*.weight", {"type": "xavier_uniform"}],
        ["embedder*.*weight", {"type": "kaiming_uniform"}],
        [".bias", {"type": "zero"}],
        ["_decoder_net.*projection_layer.*weight", {"type": "xavier_uniform"}],
        ["_decoder_net.*projection_layer.*bias", {"type": "zero"}],
        ["_decoder_net.*weight_ih.*", {"type": "xavier_uniform"}],
        ["_decoder_net.*weight_hh.*", {"type": "orthogonal"}],
        ["_decoder_net.*bias_ih.*", {"type": "zero"}],
        ["_decoder_net.*bias_hh.*", {"type": "lstm_hidden_bias"}],
        ["_decoder_net.*attention.*", {"type": "zero"}],
      ],
      "prevent_regexes": [".norm*.*"],
    },
  
  mle_loss_criterion(label_smoothing_ratio=0.0) :: 
    {
      "type": "mle",
      "labeling_smooting_ratio": label_smoothing_ratio,
    },
  
  wandb_epoch_callback(project_name='lmpl_debug', run_name='mle', sync_tensorboard=false) ::
    [{
          "type": 'log_metrics_to_wandb',
          "project_name": project_name,
          "run_name": run_name,
          "sync_tensorboard": sync_tensorboard,
    },],


  seq2seq_config(train_path, valid_path, dataset_reader, encoder, decoder_net, 
                  loss_criterion, optimizer, batch_size,  num_epochs,
                  decoder_embedding_dim=128, encoder_input_dim=128, 
                  max_decoding_steps=175, initializer= null, learning_rate_scheduler=null,
                  patience=5, distributed="false", ngpus=1, grad_clipping=5.0, epoch_callbacks=[], use_amp=false,
                  encoder_vocab_namespace='source_tokens', decoder_vocab_namespace='target_tokens',
                  dropout_ratio = 0.1, eval_beam_size=5, 
                  evaluate_on_test=false, test_path=null,
                  ) ::
    {
      "dataset_reader": dataset_reader,
      "train_data_path": train_path,
      "validation_data_path": valid_path,
      [if test_path != null then "test_data_path"]: test_path,
      "evaluate_on_test": evaluate_on_test,
      "model": {
        "type": "lmpl_composed_lm",
        "use_in_seq2seq_mode": true,
        "decoder": {
            "type": "lmpl_auto_regressive_seq_decoder",
            "max_decoding_steps": max_decoding_steps,
            "decoder_net": decoder_net,
            "target_embedder": {
              "vocab_namespace": decoder_vocab_namespace,
              "embedding_dim": decoder_embedding_dim, 
            },
            "loss_criterion": loss_criterion,
            "use_in_seq2seq_mode": true, 
            "target_namespace": "target_tokens",
            "beam_size": 1,
            "use_bleu" : true,
            "dropout": dropout_ratio,
            "eval_beam_size": eval_beam_size,
        },
        "source_embedder": {
          "token_embedders": {
            "tokens": {
              "type": "embedding",
              "vocab_namespace": encoder_vocab_namespace,
              "embedding_dim": encoder_input_dim,
              "trainable": true
            }
          },
        },
        "encoder": encoder,
        [if initializer != null then "initializer"]: initializer,
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
        "num_epochs": num_epochs,
        "patience": patience,
        "use_amp": use_amp,
        "cuda_device": 0,
        "grad_clipping": grad_clipping,
        "optimizer": optimizer,
        [if learning_rate_scheduler != null then "learning_rate_scheduler"]: learning_rate_scheduler,
        "checkpointer": {
          "num_serialized_models_to_keep": 5,
        },
        "epoch_callbacks": epoch_callbacks,
      },
      [if stringToBool(distributed) then "distributed"]:   { "cuda_devices": gpus(ngpus),},
    },
}