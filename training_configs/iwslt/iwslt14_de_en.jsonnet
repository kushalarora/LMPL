local base = import "iwslt14_de_en_base.jsonnet";
local optimizer = {
      "type": "adam",
      "lr": 0.001,
    };
local train_path = "data/iwslt/train_*.tsv";
local valid_path = "data/iwslt/valid_*.tsv";

local dataset_reader = base.sharded_dataset_reader(base.seq2seq_dataset_reader());

local encoder = base.lstm_encoder();
local decoder_net = base.lstm_cell_decoder_net();
local loss_criterion = base.mle_loss_criterion();
local epoch_callbacks = base.wandb_epoch_callback();
local learning_rate_scheduler = {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "max",
        "patience": 3
    };
base.seq2seq_config(train_path, valid_path, dataset_reader, encoder, decoder_net, loss_criterion, optimizer, batch_size=64, num_epochs=80, epoch_callbacks=epoch_callbacks, learning_rate_scheduler=learning_rate_scheduler)