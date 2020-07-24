from typing import Dict, List, Tuple, Optional, Callable
from overrides import overrides

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import copy

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.modules import Embedding
from allennlp.nn import util

from allennlp_models.generation.modules.decoder_nets import DecoderNet
from allennlp_models.generation.modules.seq_decoders import SeqDecoder

from lmpl.oracles.oracle_base import Oracle
from lmpl.modules.decoders.searnn_decoder import LMPLSEARNNDecoder
from lmpl.modules.cost_functions.cost_function import CostFunction
from lmpl.modules.cost_functions.noise_oracle_likelihood_cost_function import NoiseOracleCostFunction
from lmpl.modules.detokenizers.detokenizer import DeTokenizer, default_tokenizer

torch.autograd.set_detect_anomaly(True)


@SeqDecoder.register("lmpl_reinforce_decoder")
class LMPLReinforceDecoder(LMPLSEARNNDecoder):

    def __init__(self,
                 vocab: Vocabulary,
                 max_decoding_steps: int,
                 decoder_net: DecoderNet,
                 target_embedder: Embedding,
                 use_in_seq2seq_mode: bool = False,
                 generation_batch_size: int = 32,
                 target_namespace: str = "tokens",
                 beam_size: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 scheduled_sampling_k: int = 100,
                 scheduled_sampling_type: str = 'uniform',
                 use_bleu: bool = False,
                 use_hamming: bool = False,
                 dropout: float = None,
                 sample_output: bool = False,
                 start_token: str =START_SYMBOL,
                 end_token: str = END_SYMBOL,
                 num_decoder_layers: int = 1,
                 mask_pad_and_oov: bool = True,
                 tie_output_embedding: bool = False,
                 label_smoothing_ratio: Optional[float] = None,

                 oracle: Oracle = None,
                 rollout_cost_function: CostFunction = None,
                 rollin_steps: int = 50,
                 rollin_rollout_combination_mode='rl',
                 rollout_mixing_prob: float = 0.5,
                 detokenizer: DeTokenizer = default_tokenizer,
                 temperature: int = 1,
                 do_max_rollout_steps: bool = True,
                 num_mle_iters: int = 0,
                 rollin_rollout_mixing_coeff: float = 0.5,
                 detach_rollin_logits: bool = True,
                 rollout_ratio: float = 1.0,
                 ) -> None:

        super().__init__(vocab=vocab,
                         max_decoding_steps=max_decoding_steps,
                         generation_batch_size=generation_batch_size,
                         decoder_net=decoder_net,
                         target_embedder=target_embedder,
                         use_in_seq2seq_mode=use_in_seq2seq_mode,
                         target_namespace=target_namespace,
                         beam_size=beam_size,
                         scheduled_sampling_ratio=scheduled_sampling_ratio,
                         scheduled_sampling_k=scheduled_sampling_k,
                         scheduled_sampling_type=scheduled_sampling_type,
                         rollin_mode='teacher_forcing',
                         rollout_mode='learned',
                         use_bleu=use_bleu,
                         use_hamming=use_hamming,
                         dropout=dropout,
                         sample_output=sample_output,
                         start_token=start_token,
                         end_token=end_token,
                         num_decoder_layers=num_decoder_layers,
                         mask_pad_and_oov=mask_pad_and_oov,
                         tie_output_embedding=tie_output_embedding,
                         label_smoothing_ratio=label_smoothing_ratio,

                         oracle=oracle,
                         rollout_cost_function=rollout_cost_function,
                         rollin_rollout_combination_mode=rollin_rollout_combination_mode,
                         rollout_mixing_prob=rollout_mixing_prob,
                         num_tokens_to_rollout=1,
                         num_neighbors_to_add=0,
                         detokenizer=detokenizer,
                         must_include_target_token=False,
                         do_max_rollout_steps=do_max_rollout_steps,
                         rollout_ratio=rollout_ratio,
                         detach_rollin_logits=detach_rollin_logits,
                         )
        self._rollin_steps = rollin_steps
        self._num_mle_iters = num_mle_iters
        self._rollin_rollout_mixing_coeff = rollin_rollout_mixing_coeff

    def _get_mask(self, predictions):
        # SEARNN with KL might not produce the sequences that
        # match target sequence on length. This is especially true
        # with LM done with learned rollins. The pattern observed
        # here is that sequence lengths keep shrinking.

        # This code computes mask from predicted tokens by observing
        # first time eos token is produced. Everything after that is
        # masked out.
        mask = predictions.new_ones(predictions.shape)
        for i, indices in enumerate(predictions.detach().cpu().tolist()):
            if self._end_index in indices:
                end_idx = indices.index(self._end_index)
                mask[i, :end_idx + 1] = 1
                mask[i, end_idx + 1:] = 0
        return mask

    @overrides
    def _combine_rollin_rollout_losses(self,
                                       rollin_output_dict: Dict[str, torch.Tensor],
                                       rollout_output_dict: Dict[str, torch.Tensor],
                                       state: Dict[str, torch.Tensor],
                                       target_tokens: Dict[str, torch.LongTensor]):

        if self._combiner_mode == 'rl':
            # rollin_logits: (batch_size, num_rollin_steps, num_classes)
            rollin_logits = rollin_output_dict['logits'].squeeze(1)
            batch_size, num_rollin_steps, num_classes = rollin_logits.shape
            num_tokens_to_rollout = len(rollout_output_dict['rollout_steps'])
            rollin_rollout_logits = []
            for i, step in enumerate(rollout_output_dict['rollout_steps']):
                rollin_logits_prefix = rollin_logits[:, :step + 1, :] \
                                            .unsqueeze(1) 

                if self._detach_rollin_logits:
                    rollin_logits_prefix = rollin_logits_prefix.detach()

                rollout_output_logits = rollout_output_dict['logits'][i]

                rollin_rollout_logits.append(torch.cat([rollin_logits_prefix, 
                                                        rollout_output_logits],
                                                       dim=2))
                
            rollin_rollout_logits = torch.stack(rollin_rollout_logits, dim=1)
                        
            # rollout_output_dict['baseline_rewards'] = self._baseline_regressor(state['decoder_hiddens'].detach()).squeeze(-1)
            output_dict = {'predictions': rollin_output_dict['predictions']}
            if target_tokens:
                # rollout_loss_batch : (batch_size,rollout_steps)
                rollout_reward_batch = -1 * rollout_output_dict['loss_batch']

                if self.training_iteration < self._num_mle_iters:
                    loss_batch = rollin_output_dict['loss_batch']
                else:
                    # rewards = rollout_reward_batch.detach()
                    # rewards = F.softmax(rewards * self._temperature, dim=1) 
                    rewards = torch.exp(rollout_reward_batch.detach()) 
                    # rewards = (rewards - rewards.min())/(rewards.max() - rewards.min())
                
                    # predictions: (batch_size, rollout_steps, rollout_steps - 1)
                    predictions = rollout_output_dict["predictions"].squeeze(dim=2)
                    predictions = predictions[:, :, 1:]

                    # rollout_logits: (batch_size, rollout_steps, rollout_steps - 1)
                    rollout_logits = F.log_softmax(rollin_rollout_logits.squeeze(dim=2),
                                                   dim=-1)

                    log_probs = torch.gather(rollout_logits, -1,
                                             predictions.unsqueeze(dim=3))\
                                     .squeeze(dim=3)

                    batch_size, rollout_size, seq_size = log_probs.shape

                    # Get mask expects first detects </S> and considers all the tokens before this
                    # and masks out everything after this.
                    log_prob_mask_flattened = self._get_mask(predictions.reshape(batch_size*rollout_size, seq_size))
                    log_prob_mask = log_prob_mask_flattened.reshape(batch_size, 
                                                                    rollout_size, 
                                                                    seq_size)

                    log_probs *= log_prob_mask

                    # We are trying to maximize the reward, hence minimizing the log prob * reward.
                    summed_reward_log_probs = (-1 * log_probs * rewards).sum(dim=-1)
                    num_tokens_per_seq = log_prob_mask.sum(dim=-1)

                    rollout_rl_loss_batch = (summed_reward_log_probs/num_tokens_per_seq).sum(dim=-1)
                    if self.training_iteration % 100 == 1:
                        # import pdb; pdb.set_trace()
                        for i, x in enumerate(self._decode_tokens( predictions[2], vocab_namespace=self._target_namespace, truncate=True)):
                             print(f"{' '.join(x)}:: {rewards[2][i].item():.5f}")

                    loss_batch = self._rollin_rollout_mixing_coeff * rollin_output_dict['loss_batch'] + \
                                  (1 - self._rollin_rollout_mixing_coeff) * rollout_rl_loss_batch

                # shape : (batch_size,)
                target_mask = util.get_text_field_mask(target_tokens)
                target_mask = target_mask[:, 1:].float()
                non_batch_dims = tuple(range(1, len(target_mask.shape)))

                target_mask_sum = target_mask.sum(dim=non_batch_dims)
                num_non_empty_sequences = ((target_mask_sum > 0).float().sum() + 1e-13)
                loss = loss_batch.sum()/num_non_empty_sequences

                # output_dict['loss'] = rollin_output_dict['loss'] if self.training_iteration < 10 else rollin_output_dict['loss'] + loss
                # output_dict['loss'] = loss + 0 * baseline_reward_regressor_loss
                output_dict['loss'] = loss
            return output_dict
        return None
