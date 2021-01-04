local base = import "../iwslt14_de_en_base.jsonnet";

local lr = std.parseJson(std.extVar('lr'));
local weight_decay = std.parseJson(std.extVar('weight_decay'));
local warmup_steps = std.parseJson(std.extVar('warmup_steps'));

local dropout_ratio = 0.1;

local optimizer=  {
      "type": "adam",
      "lr": lr,
      "betas": [0.9, 0.98],
      "eps": 1e-9,
      'weight_decay': weight_decay,
};

local learning_rate_scheduler = {
      "type": "noam",
      "model_size": 512,
      "warmup_steps": warmup_steps,
    };

local loss_criterion = {
          "type": "mle",
          "labeling_smooting_ratio": 0.1,
};
local train_path = "data/iwslt/train_*.tsv";
local valid_path = "data/iwslt/valid_*.tsv";

local dataset_reader = base.sharded_dataset_reader(base.seq2seq_dataset_reader());
local encoder = base.transformer_encoder();
local decoder_net = base.transformer_decoder_net();
local initializer = base.transformer_initializer();

local optuna_epoch_callbacks = [{ type: 'optuna_pruner',},];

base.seq2seq_config(train_path, valid_path, dataset_reader, encoder, decoder_net, loss_criterion, optimizer, batch_size=64, num_epochs=10, 
encoder_input_dim=512, decoder_embedding_dim=512, 
epoch_callbacks=optuna_epoch_callbacks, learning_rate_scheduler=learning_rate_scheduler, dropout_ratio=dropout_ratio, initializer=initializer)