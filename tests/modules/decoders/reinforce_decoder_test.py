from typing import Any, Iterable, Dict
from functools import partial

import math
import os
import pytest
import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from tests import ModelTestCase
from allennlp.common.util import END_SYMBOL, prepare_environment, START_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Embedding
from allennlp.training.metrics import BLEU, Metric

from lmpl.modules.criterions import MaximumLikelihoodLossCriterion
from lmpl.modules.decoders.decoder_net import LstmCellDecoderNet
from lmpl.modules.decoders import LMPLReinforceDecoder
from lmpl.modules.decoders.searnn_decoder import expand_tensor, \
                                                 rollout_mixing_functional, \
                                                 get_neighbor_tokens, \
                                                 extend_targets_by_1

def create_vocab_decoder_net_and_criterion(decoder_input_dim, symbols=["A", "B"]):
    vocab = Vocabulary()
    vocab.add_tokens_to_namespace(symbols + [START_SYMBOL, END_SYMBOL])

    decoder_net = LstmCellDecoderNet(
        decoding_dim=decoder_input_dim,
        target_embedding_dim=decoder_input_dim,
    )

    loss_criterion = MaximumLikelihoodLossCriterion()

    return vocab, decoder_net, loss_criterion

def build_searnn_decoder(decoder_input_dim, embedder=None, symbols=["A", "B"], **kwargs):

    vocab, decoder_net, loss_criterion = \
        create_vocab_decoder_net_and_criterion(decoder_input_dim, 
                                               symbols=symbols)
    
    embedder = embedder or Embedding(num_embeddings=vocab.get_vocab_size(), 
                                embedding_dim=decoder_input_dim)

    return LMPLReinforceDecoder(vocab, 10, decoder_net, 
                                        embedder, loss_criterion, **kwargs)


class TestReinforceDecoder(ModelTestCase):
    def setup_method(self):
      super().setup_method()
      self.set_up_model(
          ModelTestCase.FIXTURES_ROOT / "natural_lang" / "ptb_lm_reinforce.jsonnet",
          ModelTestCase.FIXTURES_ROOT / "natural_lang" / "sentences.txt",
      )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-2)

    def test_searnn_init(self):
        decoder_input_dim = 4

        # Test you can build LMPLReinforceDecoder object.
        build_searnn_decoder(decoder_input_dim)