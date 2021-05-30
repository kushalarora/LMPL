local base = import "iwslt14_de_en_base.jsonnet";
local optimizer=  {
      "type": "adam",
      "lr": 0.001,
      "betas": [0.9, 0.98],
      "eps": 1e-9,
      'weight_decay': 0.0,
};
local train_path = "data/iwslt/train_*.tsv";
local valid_path = "data/iwslt/valid_*.tsv";
local test_path = "data/iwslt/test_*.tsv";


local decoder_dropout_ratio = 0.4;
// local decoder_dropout_ratio = std.parseJson(std.extVar('weight_decay'));

local num_encoder_layers = 2;
local num_decoder_layers = 2;
// local num_decoder_layers = std.parseJson(std.extVar('num_decoder_layers'));

local dropout_ratio = 0.4;

local encoder_hidden_dim = 256;
local decoder_embedding_dim = 512;
local encoder_input_dim = 512;
local decoder_hidden_dim = 512;

local batch_size = std.parseJson(std.extVar('BATCH_SIZE'));
// local batch_size = 60; 
local num_epochs = 80;

local ngpus = std.parseJson(std.extVar("NUM_GPUS"));

local debug = base.stringToBool(std.extVar("DEBUG"));
local distributed = base.stringToBool(std.extVar("DISTRIBUTED"));

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

local epoch_callbacks = if !debug then base.wandb_epoch_callback();
local initializer = base.lstm_initializer();

local learning_rate_scheduler = {
        "type": "reduce_on_plateau",
        "factor": 0.25,
        "mode": "max",
        "patience": 2
    };

local loss_criterion = {
          "type": "mle",
          "labeling_smooting_ratio": 0.1,
};

base.seq2seq_config(train_path, valid_path, dataset_reader, encoder, decoder_net, loss_criterion, optimizer, test_path=test_path, evaluate_on_test=true, batch_size=batch_size, num_epochs=num_epochs, encoder_input_dim=encoder_input_dim, 
decoder_embedding_dim=decoder_embedding_dim, tie_output_embedding=true, tied_source_embedder_key="tokens", epoch_callbacks=epoch_callbacks, learning_rate_scheduler=learning_rate_scheduler, dropout_ratio=dropout_ratio, initializer=initializer, patience=15, use_amp=true,)
