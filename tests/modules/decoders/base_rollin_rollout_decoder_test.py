from typing import Any, Iterable, Dict
from functools import partial

import math
import pytest
import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import END_SYMBOL, prepare_environment, START_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Embedding
from allennlp.training.metrics import BLEU, Metric

from lmpl.modules.criterions import MaximumLikelihoodLossCriterion
from lmpl.modules.decoders import LstmCellDecoderNet
from lmpl.modules.decoders import BaseRollinRolloutDecoder

def create_vocab_decoder_net_and_criterion(decoder_input_dim, symbols=["A", "B"]):
    vocab = Vocabulary()
    vocab.add_tokens_to_namespace(symbols + [START_SYMBOL, END_SYMBOL])

    decoder_net = LstmCellDecoderNet(
        decoding_dim=decoder_input_dim,
        target_embedding_dim=decoder_input_dim,
    )

    loss_criterion = MaximumLikelihoodLossCriterion()

    return vocab, decoder_net, loss_criterion

def build_decoder(decoder_input_dim, embedder=None, symbols=["A", "B"], **kwargs):

    vocab, decoder_net, loss_criterion = \
        create_vocab_decoder_net_and_criterion(decoder_input_dim, 
                                               symbols=symbols)
    
    embedder = embedder or Embedding(num_embeddings=vocab.get_vocab_size(), 
                                embedding_dim=decoder_input_dim)

    return BaseRollinRolloutDecoder(vocab, 10, decoder_net, 
                                        embedder, loss_criterion)

class TestBaseRollinRolloutDecoder(AllenNlpTestCase):
    def test_rollin_rollout_decoder_init(self):
        decoder_input_dim = 4

        # Test you can build BaseRollinRolloutDecoder object.
        build_decoder(decoder_input_dim)

        # Test init. raise error when decoder input annd embedding dim
        # are not same.
        with pytest.raises(ConfigurationError):
            vocab, _, _ = create_vocab_decoder_net_and_criterion(decoder_input_dim)
            embedder = Embedding(num_embeddings=vocab.get_vocab_size(), 
                            embedding_dim=decoder_input_dim + 1)
            build_decoder(decoder_input_dim, embedder)

        # Test init. raises error when embedding are tied and 
        # projection layers size is not same as 
        # embedding layer (transposed).

        # Test embeding hidden dim should be same as decoder output dim (
        # or output projection layer's input dim)
        with pytest.raises(ConfigurationError):
            vocab = Vocabulary()
            vocab.add_tokens_to_namespace(["A", "B", START_SYMBOL, END_SYMBOL])

            decoder_net = LstmCellDecoderNet(
                decoding_dim=decoder_input_dim,
                target_embedding_dim=decoder_input_dim + 1,
            )

            loss_criterion = MaximumLikelihoodLossCriterion()

            embedder = Embedding(num_embeddings=vocab.get_vocab_size(), 
                            embedding_dim=decoder_input_dim)

            BaseRollinRolloutDecoder(vocab, 10, decoder_net, 
                                        embedder, loss_criterion,
                                        tie_output_embedding=True)

        # Test output projection layers output dim is same as vocab (or embed's input dim)
        with pytest.raises(ConfigurationError):
            vocab = Vocabulary()
            vocab.add_tokens_to_namespace(["A", "B", START_SYMBOL, END_SYMBOL])

            decoder_net = LstmCellDecoderNet(
                decoding_dim=decoder_input_dim,
                target_embedding_dim=decoder_input_dim,
            )

            loss_criterion = MaximumLikelihoodLossCriterion()

            embedder = Embedding(num_embeddings=vocab.get_vocab_size() + 1, 
                            embedding_dim=decoder_input_dim)

            BaseRollinRolloutDecoder(vocab, 10, decoder_net, 
                                        embedder, loss_criterion, 
                                        tie_output_embedding=True)
        
    def test_get_start_predictions(self):
        decoder_input_dim: int = 4

        decoder = build_decoder(decoder_input_dim)

        start_idx = decoder._vocab._token_to_index['tokens'][START_SYMBOL]

        ground_truth = torch.LongTensor([4, 4])
        # Generate start predictions from target tokens.
        target_tokens = {'tokens': {'tokens': torch.LongTensor([[1,2,3,4], 
                                                                [5,6,7,8]])}}

        start_predictions = decoder._get_start_predictions(state={},
                                            target_tokens=target_tokens)
        assert torch.all(ground_truth == start_predictions)

        # Generate start predictions from generation_batch_size:
        start_predictions = decoder._get_start_predictions(state={},
                                                generation_batch_size=2)
        assert torch.all(ground_truth == start_predictions)

        # TODO: #9 (Kushal) Generate start predictions from source encoder mask.


    def test_rollin_rollout_decoder_forward_cannot_be_called(self):
        decoder_input_dim: int = 4

        base_decoder: BaseRollinRolloutDecoder = build_decoder(decoder_input_dim)

        with pytest.raises(NotImplementedError):
            target_tokens = {'tokens': 
                                {'tokens': torch.LongTensor([[1,2,3,4], 
                                                             [5,6,7,8]])}}
            base_decoder(target_tokens=target_tokens)

    def test_apply_scheduled_sampling(self):
        decoder_input_dim: int = 4

        decoder: BaseRollinRolloutDecoder = build_decoder(decoder_input_dim)

        # Test uniform doesn't change with iteration.
        decoder._scheduled_sampling_type = "uniform"
        decoder._scheduled_sampling_ratio = 0.2

        assert decoder.training_iteration == 0

        decoder.training_iteration = 100

        decoder._apply_scheduled_sampling()
        
        assert decoder._scheduled_sampling_ratio == 0.2

        # Test Linear
        decoder: BaseRollinRolloutDecoder = build_decoder(decoder_input_dim)
        
        decoder._scheduled_sampling_type = "linear"
        decoder._scheduled_sampling_ratio = 0.2
        decoder._scheduled_sampling_k = 0.01

        decoder._apply_scheduled_sampling()
        
        assert decoder._scheduled_sampling_ratio == 0

        decoder.training_iteration = 100
        decoder._apply_scheduled_sampling()

        assert decoder._scheduled_sampling_ratio == 0.01

        # Test max value
        decoder.training_iteration = 10000
        decoder._apply_scheduled_sampling()

        assert decoder._scheduled_sampling_ratio == 0.95

        # Test exponential
        decoder: BaseRollinRolloutDecoder = build_decoder(decoder_input_dim)
        decoder._scheduled_sampling_type = "exponential"
        decoder._scheduled_sampling_ratio = 0.2
        decoder._scheduled_sampling_k = 0.1

        decoder._apply_scheduled_sampling()
        
        assert decoder._scheduled_sampling_ratio == 0.1

        decoder.training_iteration = 3

        decoder._apply_scheduled_sampling()

        assert decoder._scheduled_sampling_ratio - 0.0001 < 1e-8

        # Test inverse_sigmoid:
        decoder: BaseRollinRolloutDecoder = build_decoder(decoder_input_dim)
        decoder._scheduled_sampling_type = "inverse_sigmoid"
        decoder._scheduled_sampling_ratio = 0.2
        decoder._scheduled_sampling_k = 100

        decoder._apply_scheduled_sampling()
        
        assert decoder._scheduled_sampling_ratio - 1/101 < 1e-8

        decoder.training_iteration = 3

        decoder._apply_scheduled_sampling()
        
        e = math.exp(1)
        assert decoder._scheduled_sampling_ratio - e/(100 + e) < 1e-8


    def test_rollin_rollout_decoder_post_process(self):
        decoder_inout_dim = 4

        decoder = build_decoder(decoder_inout_dim)
        
        predictions = torch.tensor([[3, 2, 5, 0, 0], [2, 2, 3, 5, 0]])

        tokens_ground_truth = [["B", "A"], ["A", "A", "B"]]

        output_dict = {"predictions": predictions}
        predicted_tokens = decoder.post_process(output_dict)["predicted_tokens"]
        assert predicted_tokens == tokens_ground_truth

    def test_rollin_policy(self):
        decoder_input_dim = 4

        decoder = build_decoder(decoder_input_dim)
        target_tokens: Dict[str, torch.Tensor] = \
            { 'tokens': {
                'tokens': torch.LongTensor([[1,2,3,4], 
                                            [5,6,7,8]])}}
        last_predictions = torch.LongTensor([10, 14])
        token_at_timstep = torch.LongTensor([3, 7])
        timestep=2

        # Test "learned" rollin mode
        decoder._rollin_mode = "learned"
        input_choices = decoder.rollin_policy(
                                    timestep=timestep,
                                    last_predictions=last_predictions,
                                    target_tokens=target_tokens
                                )

        assert torch.all(last_predictions == input_choices)

        # Test "mixed/scheduled sampling" rollin mode
        decoder._rollin_mode = "mixed"
        pass

        # Test "teacher_forcing" rollin mode
        decoder._rollin_mode = "teacher_forcing"
        input_choices = decoder.rollin_policy(
                                    timestep=timestep,
                                    last_predictions=last_predictions,
                                    target_tokens=target_tokens
                                )
        assert torch.all(token_at_timstep == input_choices)

    def test_copy_reference_policy(self):
        decoder_input_dim = 4

        target_tokens: Dict[str, torch.Tensor] = \
            {'tokens': {'tokens': torch.LongTensor([[5,6,7,8,4], 
                                                    [8,7,6,5,4]])}}
        last_predictions = torch.LongTensor([3, 4])
        state = {}
        
        # Test case when timestep < seq_len
        decoder = build_decoder(decoder_input_dim, 
                                    symbols=["A", "B", "C", 
                                                "D", "E", "F", "G",
                                                "H", "I", "J"])
        timestep=2
        target_logits, state = decoder.copy_reference_policy(
                                            timestep=timestep,
                                            last_predictions=last_predictions,
                                            state=state,
                                            target_tokens=target_tokens,
                                            )
        target_logits_exp = torch.exp(target_logits)
        assert target_logits_exp[0].sum() == 1
        assert target_logits_exp[1].sum() == 1

        assert target_logits_exp[0, 8] == 1
        assert target_logits_exp[1, 5] == 1

        # Test case when timestep > seq_len
        decoder = build_decoder(decoder_input_dim, 
                                    symbols=["A", "B", "C", 
                                                "D", "E", "F", "G",
                                                "H", "I", "J"])
        timestep=8
        target_logits, state = decoder.copy_reference_policy(
                                            timestep=timestep,
                                            last_predictions=last_predictions,
                                            state=state,
                                            target_tokens=target_tokens,
                                            )
        target_logits_exp = torch.exp(target_logits)
        assert target_logits_exp[0].sum() == 1
        assert target_logits_exp[1].sum() == 1

        assert target_logits_exp[0, 4] == 1
        assert target_logits_exp[1, 4] == 1

    def test_rollout_policy(self):
        decoder_input_dim = 4

        target_tokens: Dict[str, torch.Tensor] = \
            { 'tokens': {
                'tokens': torch.LongTensor([[5,6,7,8,4], 
                                            [8,7,6,5,4]])}}
        last_predictions = torch.LongTensor([3, 4])
        state = {}
        
        token_at_timstep = torch.LongTensor([3, 7])
        timestep=2

        # Test "learned" rollout mode
        decoder = build_decoder(decoder_input_dim, 
                            symbols=["A", "B", "C", 
                                        "D", "E", "F", "G",
                                        "H", "I", "J"])
        decoder._rollout_mode = 'learned'

        logits = torch.randn((2, decoder._num_classes))
        copy_reference_policy = partial(decoder.copy_reference_policy, 
                                        target_tokens)

        output_logits, state = decoder.rollout_policy(timestep=timestep, 
                                            last_predictions=last_predictions,
                                            state=state,
                                            logits=logits,
                                            reference_policy=copy_reference_policy,
                                          )

        assert torch.all(logits == output_logits)

        # Test "reference" rollout mode
        decoder = build_decoder(decoder_input_dim, 
                                    symbols=["A", "B", "C", 
                                                "D", "E", "F", "G",
                                                "H", "I", "J"])
        decoder._rollout_mode = 'reference'

        logits = torch.randn((2, decoder._num_classes))

        referece_logits, state = decoder.copy_reference_policy(
                                                timestep=timestep,
                                                last_predictions=last_predictions,
                                                state=state,
                                                target_tokens=target_tokens,
                                                )

        copy_reference_policy = partial(decoder.copy_reference_policy, 
                                        target_tokens=target_tokens)
        output_logits, state = decoder.rollout_policy(timestep=timestep, 
                                            last_predictions=last_predictions,
                                            state=state,
                                            logits=logits,
                                            reference_policy=copy_reference_policy,
                                          )

        assert torch.all(F.softmax(referece_logits, dim=-1) - 
                                F.softmax(output_logits, dim=-1) < 1e-8)

        # Test "mixed" rollout mode
        decoder = build_decoder(decoder_input_dim, 
                                    symbols=["A", "B", "C", 
                                                "D", "E", "F", "G",
                                                "H", "I", "J"])
        decoder._rollout_mode = 'mixed'

        logits = torch.randn((2, decoder._num_classes))

        referece_logits, state = decoder.copy_reference_policy(
                                                timestep=timestep,
                                                last_predictions=last_predictions,
                                                state=state,
                                                target_tokens=target_tokens,
                                                )

        copy_reference_policy = partial(decoder.copy_reference_policy, 
                                        target_tokens=target_tokens)

        rollout_mixing_func = lambda : torch.tensor([0, 1.])
        output_logits, state = decoder.rollout_policy(timestep=timestep, 
                                            last_predictions=last_predictions,
                                            state=state,
                                            logits=logits,
                                            reference_policy=copy_reference_policy,
                                            rollout_mixing_func=rollout_mixing_func)
        
        assert torch.all(logits[0] == output_logits[0])
        assert torch.all(F.softmax(referece_logits[1], dim=-1) - 
                                F.softmax(output_logits[1], dim=-1) < 1e-8)

    @pytest.mark.skip(reason="not implemented")
    def test_prepare_output_projections(self):
        assert False

    @pytest.mark.skip(reason="not implemented")
    def test_take_step(self):
        assert False


    @pytest.mark.skip(reason="not implemented")
    def test_rollin(self):
        assert False


    @pytest.mark.skip(reason="not implemented")
    def test_rollout(self):
        assert False

    @pytest.mark.skip(reason="not implemented")
    def test_get_mask(self):
        assert False