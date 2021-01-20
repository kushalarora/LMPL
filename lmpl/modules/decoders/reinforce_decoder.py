from typing import Dict, List, Tuple, Optional, Callable
from overrides import overrides

import torch
import torch.nn.functional as F
import sys
import numpy as np

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.modules import Embedding
from allennlp.nn import util
from allennlp.training.metrics import Metric

from allennlp_models.generation.modules.decoder_nets import DecoderNet
from allennlp_models.generation.modules.seq_decoders import SeqDecoder

from lmpl.modules.decoders.searnn_decoder import LMPLSEARNNDecoder
from lmpl.modules.criterions import LossCriterion
from lmpl.modules.utils import expand_tensor

@SeqDecoder.register("lmpl_reinforce_decoder")
class LMPLReinforceDecoder(LMPLSEARNNDecoder):

    def __init__(self,
                 vocab: Vocabulary,
                 max_decoding_steps: int,
                 decoder_net: DecoderNet,
                 target_embedder: Embedding,
                 loss_criterion: LossCriterion,

                 generation_batch_size: int = 32,
                 use_in_seq2seq_mode: bool = False,
                 target_namespace: str = "tokens",
                 beam_size: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 scheduled_sampling_k: int = 100,
                 scheduled_sampling_type: str = 'uniform',

                 dropout: float = None,
                 sample_rollouts: bool = False,
                 start_token: str =START_SYMBOL,
                 end_token: str = END_SYMBOL,
                 num_decoder_layers: int = 1,
                 mask_pad_and_oov: bool = True,
                 tie_output_embedding: bool = False,

                 use_bleu: bool = False,
                 use_hamming: bool = False,


                 beam_search_sampling_temperature: float = 1.,
                 top_k=0, 
                 top_p=0,
                 eval_beam_size: int = 1, 
                 tensor_based_metric: Metric = None,
                 tensor_based_metric_mask: Metric = None,
                 token_based_metric: Metric = None,

                 rollin_steps: int = 50,
                 rollout_mixing_prob: float = 0.5,
                 do_max_rollout_steps: bool = True,
                 rollin_rollout_mixing_coeff: float = 0.5,
                 rollout_ratio: float = 1.0,
                 detach_rollin_logits: bool = True,
                 max_num_contexts: int = sys.maxsize,
                 min_num_contexts: int = 1,
                include_first: bool = True,
                include_last: bool = False,
                rollout_iter_start_pct: int = 0,
                rollout_iter_end_pct: int = 100,
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

            dropout=dropout,
            sample_rollouts=sample_rollouts,
            start_token=start_token,
            end_token=end_token,
            num_decoder_layers=num_decoder_layers,
            mask_pad_and_oov=mask_pad_and_oov,
            tie_output_embedding=tie_output_embedding,

            use_bleu=use_bleu,
            use_hamming=use_hamming,


            beam_search_sampling_temperature=beam_search_sampling_temperature,
            top_k=top_k,
            top_p=top_p,
            eval_beam_size=eval_beam_size,
            tensor_based_metric=tensor_based_metric,
            tensor_based_metric_mask=tensor_based_metric_mask,
            token_based_metric=token_based_metric,

            rollin_mode='teacher_forcing',
            rollout_mode='learned',
            rollin_steps=rollin_steps,
            rollout_mixing_prob=rollout_mixing_prob,
            num_tokens_to_rollout=1,
            num_neighbors_to_add=0,
            do_max_rollout_steps=do_max_rollout_steps,
            rollout_iter_function=lambda x: range(1, x),
            rollout_iter_start_pct=rollout_iter_start_pct,
            rollout_iter_end_pct=rollout_iter_end_pct,
            mask_padding_and_start=False,
            must_include_target_token=True,
            rollout_ratio=rollout_ratio,
            detach_rollin_logits=detach_rollin_logits,
            rollin_rollout_mixing_coeff=rollin_rollout_mixing_coeff,
            include_first=include_first,
            include_last=include_last,
            max_num_contexts=max_num_contexts,
            min_num_contexts=min_num_contexts,
    )

