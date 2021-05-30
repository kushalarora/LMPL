local base = import "../iwslt14_de_en_base.jsonnet";

local lr = 0.001; #std.parseJson(std.extVar('lr'));
local weight_decay = 0; #.0001; #std.parseJson(std.extVar('weight_decay'));
local warmup_steps = 8000; #std.parseJson(std.extVar('warmup_steps'));

local decoder_dropout_ratio = 0.4;
// local decoder_dropout_ratio = std.parseJson(std.extVar('weight_decay'));;

local num_decoder_layers = 2;
// local num_decoder_layers = std.parseJson(std.extVar('num_decoder_layers'));

local dropout_ratio = 0.4;

local decoder_hidden_dim = 512;
// local decoder_hidden_dim = std.parseJson(std.extVar('decoder_hidden_dim'));

local decoder_embedding_dim = 512;
// local decoder_embedding_dim = std.parseJson(std.extVar('decoder_embedding_dim'));

local optimizer=  {
      "type": "adam",
      "lr": lr,
      "betas": [0.9, 0.98],
      "eps": 1e-9,
      'weight_decay': weight_decay,
};

// local learning_rate_scheduler = {
//       "type": "noam",
//       "model_size": 512,
//       "warmup_steps": warmup_steps,
//     };
local learning_rate_scheduler = {
        "type": "reduce_on_plateau",
        "factor": 0.25,
        "mode": "max",
        "patience": 3
    };

local loss_criterion = {
          "type": "mle",
          "labeling_smooting_ratio": 0.1,
};
local train_path = "data/iwslt/train_*.tsv";
local valid_path = "data/iwslt/valid_*.tsv";
local test_path = "data/iwslt/test_*.tsv";
local dataset_reader = base.sharded_dataset_reader(base.seq2seq_dataset_reader());
// local encoder = base.transformer_encoder();
local encoder = base.lstm_encoder(encoder_input_dim=512, 
                                    encoder_hidden_dim=256,
                                    num_encoder_layers=2,);
local decoder_net = base.lstm_cell_decoder_net(decoder_embedding_dim=512,
                                                encoder_hidden_dim=256,
                                                decoder_hidden_dim=512,
                                                bidirectional_input=true,
                                                num_decoder_layers=num_decoder_layers,
                                                dropout_ratio=decoder_dropout_ratio);
local initializer = base.lstm_initializer();

// local optuna_epoch_callbacks = [{ type: 'optuna_pruner',},];
local epoch_callbacks = base.wandb_epoch_callback(run_name='transformer_lstm_mle');

base.seq2seq_config(train_path, valid_path, dataset_reader, encoder, decoder_net, loss_criterion, optimizer, batch_size=60, num_epochs=80, tie_output_embedding=true,
encoder_input_dim=512, decoder_embedding_dim=512, tied_source_embedder_key="tokens",
epoch_callbacks=epoch_callbacks, learning_rate_scheduler=learning_rate_scheduler, dropout_ratio=dropout_ratio, initializer=initializer, patience=15, use_amp=true, test_path=test_path)