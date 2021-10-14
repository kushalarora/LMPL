from typing import Any, Iterable, Dict
from functools import partial

import math
import os
import pytest
import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.common import Params
from tests import ModelTestCase
from allennlp.common.util import END_SYMBOL, prepare_environment, START_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Embedding
from allennlp.training.metrics import BLEU, Metric

from lmpl.modules.criterions import MaximumLikelihoodLossCriterion
from lmpl.modules.decoders.decoder_net import LstmCellDecoderNet
from lmpl.modules.decoders import LMPLSEARNNDecoder
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

    return LMPLSEARNNDecoder(vocab, 10, decoder_net, 
                                        embedder, loss_criterion, **kwargs)


class TestSEARNNDecoder(ModelTestCase):
    def setup_method(self):
      super().setup_method()
      self.set_up_model(
          ModelTestCase.FIXTURES_ROOT / "natural_lang" / "ptb_lm_searnn.jsonnet",
          ModelTestCase.FIXTURES_ROOT / "natural_lang" / "sentences.txt",
      )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-2)

    def test_searnn_init(self):
        decoder_input_dim = 4

        # Test you can build LMPLSearnnDecoder object.
        build_searnn_decoder(decoder_input_dim)

    def test_get_num_tokens_to_rollout(self):
        decoder_input_dim = 4
        decoder = build_searnn_decoder(decoder_input_dim, 
                                        symbols=["A", "B", "C", 
                                                    "D", "E", "F", "G",
                                                    "H", "I", "J"],
                                        mask_padding_and_start=False)
        assert decoder.get_num_tokens_to_rollout() == decoder._vocab.get_vocab_size('tokens')

        decoder = build_searnn_decoder(decoder_input_dim, 
                                        symbols=["A", "B", "C", 
                                                    "D", "E", "F", "G",
                                                    "H", "I", "J"],
                                        mask_padding_and_start=True)
        assert decoder.get_num_tokens_to_rollout() == \
                decoder._vocab.get_vocab_size('tokens')- 2

        decoder = build_searnn_decoder(decoder_input_dim, 
                                        symbols=["A", "B", "C", 
                                                    "D", "E", "F", "G",
                                                    "H", "I", "J"],
                                        num_tokens_to_rollout=3)
        assert decoder.get_num_tokens_to_rollout() == 3

    def test_get_neighbor_tokens(self):
      num_decoding_steps = 30
      step = 10
      targets = torch.randint(0, 10, (2, num_decoding_steps))
    
      # Test num_neighbors_to_add=6
      neighbors = get_neighbor_tokens(num_neighbors_to_add=6,
                                      num_decoding_steps=num_decoding_steps, 
                                      step=step, targets=targets)
      true_neighbor_index = [7, 8, 9, 11, 12, 13]
      assert torch.all(targets[:, true_neighbor_index] == neighbors)

      # Test support num_neighbors_to_add=0
      with pytest.raises(AssertionError):
        neighbors = get_neighbor_tokens(num_neighbors_to_add=0,
                                        num_decoding_steps=num_decoding_steps, 
                                        step=step, targets=targets)

      # Test support for odd num_neighbors_to_add
      neighbors = get_neighbor_tokens(num_neighbors_to_add=5,
                                      num_decoding_steps=num_decoding_steps, 
                                      step=step, targets=targets)
      true_neighbor_index = [8, 9, 11, 12, 13]
      assert torch.all(targets[:, true_neighbor_index] == neighbors)
    
      # Test support for negative (limited to 0) left context
      neighbors = get_neighbor_tokens(num_neighbors_to_add=22,
                                      num_decoding_steps=num_decoding_steps, 
                                      step=step, targets=targets)
      true_neighbor_index = list(range(0, 10)) + list(range(11, 23))
      assert torch.all(targets[:, true_neighbor_index] == neighbors)

      # Test support for overshooting right context
      step = 25
      neighbors = get_neighbor_tokens(num_neighbors_to_add=22,
                                      num_decoding_steps=num_decoding_steps, 
                                      step=step, targets=targets)
      true_neighbor_index = list(range(14, 25)) + list(range(26, num_decoding_steps))
      assert torch.all(targets[:, true_neighbor_index] == neighbors)

    def test_extend_targets_by_1(self):
        num_decoding_steps = 30
        targets = torch.randint(0, 10, (2, num_decoding_steps))

        extended_targets = extend_targets_by_1(targets)
        assert extended_targets.size(1) == num_decoding_steps + 1
        assert torch.all(extended_targets[:, -1] == targets[:, -1])
        assert torch.all(extended_targets[:, -2] == targets[:, -1])

    @pytest.mark.skip(reason="not implemented")
    def test_get_contexts_to_rollout(self):
      pass

    @pytest.mark.skip(reason="not implemented")
    def test_get_next_tokens(self):
        assert False

    def test_get_prediction_prefixes(self):
        decoder_input_dim = 4
        num_decoding_steps = 30
        step = 10
        decoder = build_searnn_decoder(decoder_input_dim, 
                                        symbols=["A", "B", "C", 
                                                    "D", "E", "F", "G",
                                                    "H", "I", "J"],
                                        mask_padding_and_start=False)
        vocab_size = decoder._vocab.get_vocab_size('tokens')

        targets = torch.randint(0, vocab_size, (2, num_decoding_steps))
        rollin_predictions = torch.randint(0, vocab_size, (2, num_decoding_steps))
        
        # rollin_mode == 'teacher_forcing'
        decoder._rollin_mode = 'teacher_forcing'
        prediction_prefixes = decoder.get_prediction_prefixes(
                                            targets=targets,
                                            rollin_predictions=rollin_predictions,
                                            step=step, 
                                            num_tokens_to_rollout=1)

        assert torch.all(prediction_prefixes == targets[:, :step])

        # rollin_mode == 'mixed'
        decoder._rollin_mode = 'mixed'
        prediction_prefixes = decoder.get_prediction_prefixes(
                                            targets=targets,
                                            rollin_predictions=rollin_predictions,
                                            step=step, 
                                            num_tokens_to_rollout=1)

        assert torch.all(prediction_prefixes == targets[:, :step])

        # rollin_mode == 'learned'
        decoder._rollin_mode = 'learned'
        prediction_prefixes = decoder.get_prediction_prefixes(
                                            targets=targets,
                                            rollin_predictions=rollin_predictions,
                                            step=step, 
                                            num_tokens_to_rollout=1)

        assert torch.all(prediction_prefixes == rollin_predictions[:, :step])

        # When step=0, prediction_prefixes should be None.
        decoder._rollin_mode = 'learned'
        prediction_prefixes = decoder.get_prediction_prefixes(
                                            targets=targets,
                                            rollin_predictions=rollin_predictions,
                                            step=0, 
                                            num_tokens_to_rollout=1)

        assert prediction_prefixes == None

    def test_get_rollout_steps(self):
        decoder_input_dim = 4
        num_decoding_steps = 30
        step = 10
        decoder = build_searnn_decoder(decoder_input_dim, 
                                        symbols=["A", "B", "C", 
                                                    "D", "E", "F", "G",
                                                    "H", "I", "J"],
                                        mask_padding_and_start=False)
        vocab_size = decoder._vocab.get_vocab_size('tokens')

        decoder._do_max_rollout_steps = False
        rollout_steps = decoder.get_rollout_steps(
                                    num_decoding_steps=num_decoding_steps, 
                                    step=step)
        assert rollout_steps ==  num_decoding_steps + 1 - step

        decoder._do_max_rollout_steps = True
        decoder._max_decoding_steps = 40
        rollout_steps = decoder.get_rollout_steps(
                                    num_decoding_steps=num_decoding_steps, 
                                    step=step)
        assert rollout_steps ==  num_decoding_steps + 1 - step + 5

    @pytest.mark.skip(reason="not implemented")
    def test_get_rollout_iterator(self):
        decoder_input_dim = 4
        num_decoding_steps = 30
        step = 10
        decoder = build_searnn_decoder(decoder_input_dim, 
                                        symbols=["A", "B", "C", 
                                                    "D", "E", "F", "G",
                                                    "H", "I", "J"],
                                        mask_padding_and_start=False)
        num_tokens_to_rollout = vocab_size = decoder._vocab.get_vocab_size('tokens')

        targets = torch.randint(0, vocab_size, (2, num_decoding_steps))
        rollin_predictions = torch.randint(0, vocab_size, (2, num_decoding_steps))

        rollin_logits = torch.rand((2, num_tokens_to_rollout))
        rollout_iter = decoder.get_rollout_iterator(
                                rollin_logits=rollin_logits, 
                                rollin_predictions=rollin_predictions,
                                targets=targets,
                                num_decoding_steps=num_decoding_steps,
                                num_tokens_to_rollout=num_tokens_to_rollout)

