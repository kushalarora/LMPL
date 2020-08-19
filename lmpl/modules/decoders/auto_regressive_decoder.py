from typing import Dict, List, Tuple, Optional, Callable, Iterable
from overrides import overrides
from functools import partial

import copy
import logging
import math
import numpy
import torch
import torch.nn.functional as F
import time
from torch.nn import Linear

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.modules import Embedding

from allennlp.nn import util
from allennlp.training.metrics import BLEU, Perplexity, Average
from allennlp.training.metrics import Metric

from allennlp_models.generation.modules.decoder_nets import DecoderNet
from allennlp_models.generation.modules.seq_decoders import SeqDecoder

from lmpl.modules.cost_functions.cost_function import CostFunction
from lmpl.modules.criterions.base_loss_criterion import LossCriterion
from lmpl.modules.detokenizers.detokenizer import DeTokenizer, default_tokenizer

from lmpl.modules.decoders import BaseRollinRolloutDecoder


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@SeqDecoder.register("lmpl_auto_regressive_seq_decoder")
class LMPLAutoRegressiveSeqDecoder(BaseRollinRolloutDecoder):
    """
    An autoregressive decoder.
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    decoder_net : ``DecoderNet``, required
        Module that contains implementation of neural network for decoding output elements
    max_decoding_steps : ``int``
        Maximum length of decoded sequences.
    target_embedder : ``Embedding``
        Embedder for target tokens.
    target_namespace : ``str``, optional (default = 'target_tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    beam_size : ``int``, optional (default = 4)
        Width of the beam for beam search.
    tensor_based_metric : ``Metric``, optional (default = None)
        A metric to track on validation data that takes raw tensors when its called.
        This metric must accept two arguments when called: a batched tensor
        of predicted token indices, and a batched tensor of gold token indices.
    token_based_metric : ``Metric``, optional (default = None)
        A metric to track on validation data that takes lists of lists of tokens
        as input. This metric must accept two arguments when called, both
        of type `List[List[str]]`. The first is a predicted sequence for each item
        in the batch and the second is a gold sequence for each item in the batch.
    scheduled_sampling_ratio : ``float`` optional (default = 0)
        Defines ratio between teacher forced training and real output usage. If its zero
        (teacher forcing only) and `decoder_net`supports parallel decoding, we get the output
        predictions in a single forward pass of the `decoder_net`.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 max_decoding_steps: int,
                 decoder_net: DecoderNet,
                 target_embedder: Embedding,
                 loss_criterion: LossCriterion,
                
                 generation_batch_size: int = 200,
                 use_in_seq2seq_mode: bool = False,
                 target_namespace: str = "tokens",
                 beam_size: int = None,
                 scheduled_sampling_ratio: float = 0.0,
                 scheduled_sampling_k: int = 100,
                 scheduled_sampling_type: str = 'uniform',

                 dropout: float = None,
                 start_token: str =START_SYMBOL,
                 end_token: str = END_SYMBOL,
                 num_decoder_layers: int = 1,
                 mask_pad_and_oov: bool = False,
                 tie_output_embedding: bool = False,

                 use_bleu: bool = False,
                 use_hamming: bool = False,

                 sample_rollouts: bool = False,
                 beam_search_sampling_temperature: float = 1.,
                 top_k=0, 
                 top_p=0,
                 detokenizer: DeTokenizer = default_tokenizer,
                 tensor_based_metric: Metric = None,
                 tensor_based_metric_mask: Metric = None,
                 token_based_metric: Metric = None,
                ) -> None:
        super().__init__(
            vocab=vocab,
            max_decoding_steps=max_decoding_steps,
            decoder_net=decoder_net,
            target_embedder=target_embedder,
            loss_criterion=loss_criterion,

            generation_batch_size=generation_batch_size,
            use_in_seq2seq_mode=use_in_seq2seq_mode,
            target_namespace=target_namespace,
            beam_size=beam_size,
            scheduled_sampling_ratio=scheduled_sampling_ratio,
            scheduled_sampling_k=scheduled_sampling_k,
            scheduled_sampling_type=scheduled_sampling_type,
            rollin_mode="teacher_forcing",

            dropout=dropout,
            start_token=start_token,
            end_token=end_token,
            num_decoder_layers=num_decoder_layers,
            mask_pad_and_oov=mask_pad_and_oov,
            tie_output_embedding=tie_output_embedding,
                        
            use_bleu=use_bleu,
            use_hamming=use_hamming,

            sample_rollouts=sample_rollouts,
            beam_search_sampling_temperature=beam_search_sampling_temperature,
            top_k=top_k,
            top_p=top_p,
            detokenizer=detokenizer,
            tensor_based_metric=tensor_based_metric,
            tensor_based_metric_mask=tensor_based_metric_mask,
            token_based_metric=token_based_metric,
        )

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      start_predictions: torch.LongTensor,
                      num_decoding_steps: int,
                      target_tokens: Dict[str, torch.LongTensor] = None,
                     ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        rollin_output_dict: Dict[str, torch.Tensor] = {}
        rollout_output_dict_iter: Iterable[Dict[str, torch.Tensor]] = None

        rollin_output_dict.update(self.rollin(state,
                                                start_predictions,
                                                rollin_steps=num_decoding_steps,
                                                target_tokens=target_tokens,))
        decoder_output: Dict[str, torch.Tensor] = rollin_output_dict
        return (decoder_output, rollin_output_dict, rollout_output_dict_iter)
