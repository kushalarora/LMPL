local base = import "iwslt14_de_en_base.jsonnet";

local warm_start_model = std.extVar("WARM_START_MODEL");

local optimizer = {
      "type": "adam",
      "lr": 0.0001,
      "betas": [0.9, 0.98],
      "eps": 1e-9,
      'weight_decay': 0.0,
    };

local train_path = "data/iwslt/train_*.tsv";
local valid_path = "data/iwslt/valid_*.tsv";
local test_path = "data/iwslt/test_*.tsv";

local evaluate_on_test = true;


local decoder_dropout_ratio = 0.4;
// local decoder_dropout_ratio = std.parseJson(std.extVar('weight_decay'));;

local num_encoder_layers = 2;
local num_decoder_layers = 2;
// local num_decoder_layers = std.parseJson(std.extVar('num_decoder_layers'));

local dropout_ratio = 0.4;

local encoder_hidden_dim = 256;
local decoder_embedding_dim = 512;
local encoder_input_dim = 512;
local decoder_hidden_dim = 512;

local batch_size = 8;
local num_epochs = 20;

local NUM_GPUS = std.parseJson(std.extVar("NUM_GPUS"));
local ngpus = if NUM_GPUS != null then NUM_GPUS else 1;

local DISTRIBUTED = std.parseJson(std.extVar("DISTRIBUTED"));
local distributed = if DISTRIBUTED == "true"  || DISTRIBUTED == true then "true"  else "false";
local dataset_reader = base.sharded_dataset_reader(base.seq2seq_dataset_reader());

local encoder = base.lstm_encoder(encoder_input_dim=encoder_input_dim, 
                                    encoder_hidden_dim=encoder_hidden_dim,
                                    num_encoder_layers=num_encoder_layers,);

local decoder_net = base.lstm_cell_decoder_net(
                            decoder_embedding_dim=decoder_embedding_dim,
                            encoder_hidden_dim=encoder_hidden_dim,
                            decoder_hidden_dim=decoder_hidden_dim,
                            bidirectional_input=true,
                            num_decoder_layers=num_decoder_layers,
                            dropout_ratio=decoder_dropout_ratio);

local epoch_callbacks = base.wandb_epoch_callback(run_name='reinforce');

local initializer = {
          "regexes": [
            [".*embedder*.*|_decoder._decoder_net.*|_decoder._output_projection_layer.*|_encoder.*",
              {
                "type": "pretrained",
                "weights_file_path": warm_start_model + "/best.th",
              },
            ],
          ],
        };
local learning_rate_scheduler = {
        "type": "reduce_on_plateau",
        "factor": 0.25,
        "mode": "max",
        "patience": 2
    };


local rollout_cost_function = { "type" : "bleu"};
local loss_criterion = {
          "type": "reinforce",
          "temperature": 1,
          "rollout_cost_function": rollout_cost_function,
          "detach_rollin_logits": false,
          // "alpha": 0.99,
          "rollin_rollout_mixing_coeff": 0.0,
      };

local decoder_extra_args = {
          "generation_batch_size": 32,
          "loss_criterion": loss_criterion,
          "max_num_contexts": 1,
          "include_first": true,
          "beam_size": 4,
          "do_max_rollout_steps": false,
      };

local model_extra_args = {
  "log_output_every_iteration": 1000,
};
base.seq2seq_config(
      train_path=train_path, valid_path=valid_path, test_path=test_path,
      dataset_reader=dataset_reader, encoder=encoder, decoder_net=decoder_net, decoder_extra_args=decoder_extra_args, 
      decoder_type='lmpl_reinforce_decoder', evaluate_on_test=true, loss_criterion=loss_criterion, optimizer=optimizer, 
      encoder_input_dim=encoder_input_dim, decoder_embedding_dim=decoder_embedding_dim, initializer=initializer, 
      dropout_ratio=dropout_ratio, tie_output_embedding=true, tied_source_embedder_key="tokens", beam_size=10,
      learning_rate_scheduler=learning_rate_scheduler, batch_size=batch_size, num_epochs=num_epochs,
      num_gradient_accumulation_steps=4, epoch_callbacks=epoch_callbacks, ngpus=ngpus, 
      distributed=distributed, use_amp=true,
)