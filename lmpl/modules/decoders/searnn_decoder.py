from typing import Dict, List, Tuple, Optional, Callable, Iterable
from overrides import overrides

import torch
import torch.nn.functional as F

import numpy as np
import random
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.modules import Embedding
from allennlp.nn import util

from quant_exp_bias.modules.decoders.decoder_net import DecoderNet
from quant_exp_bias.oracles.oracle_base import Oracle
from quant_exp_bias.modules.decoders.auto_regressive_decoder import QuantExpAutoRegressiveSeqDecoder
from quant_exp_bias.modules.decoders.seq_decoder import SeqDecoder
from quant_exp_bias.modules.cost_functions.cost_function import CostFunction
from quant_exp_bias.modules.cost_functions.noise_oracle_likelihood_cost_function import NoiseOracleCostFunction
from quant_exp_bias.modules.detokenizers.detokenizer import DeTokenizer, default_tokenizer

@SeqDecoder.register("quant_exp_searnn_decoder")
class QuantExpSEARNNDecoder(QuantExpAutoRegressiveSeqDecoder):

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
                 rollin_mode: str = 'teacher_forcing',
                 rollout_mode: str = 'reference',
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
                 rollin_rollout_combination_mode='kl',
                 rollout_mixing_prob: float = 0.5,
                 num_tokens_to_rollout:int = -1,
                 detokenizer: DeTokenizer = default_tokenizer,
                 temperature: int = 1,
                 num_neighbors_to_add: int = 0,
                 do_max_rollout_steps: bool = False,
                 mask_padding_and_start: bool = True,
                 must_include_target_token: bool = True,
                 rollout_iter_function: Callable[[int], Iterable[int]]=lambda x: range(1, x),
                 rollout_ratio:float = 1.0,
                 detach_rollin_logits: bool = False,
                 rollin_rollout_mixing_coeff: float = 0.25,
                 rollout_reference_policy:str = 'copy',
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
                         rollin_mode=rollin_mode,
                         rollout_mode=rollout_mode,
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
                         detokenizer=detokenizer,
                        )

        self._rollin_steps = rollin_steps
        self._rollin_mode = rollin_mode
        self._rollout_mode = rollout_mode
        self._combiner_mode = rollin_rollout_combination_mode

        self._combiner_loss = None
        if self._combiner_mode == 'kl':
            self._combiner_loss = torch.nn.KLDivLoss(reduction='none')

        self._num_tokens_to_rollout = num_tokens_to_rollout

        self._temperature = temperature

        self._rollout_mask = torch.tensor([self._padding_index, self._start_index],
                                           device=self.current_device)

        self._num_neighbors_to_add = num_neighbors_to_add

        self._do_max_rollout_steps = do_max_rollout_steps

        self._mask_padding_and_start = mask_padding_and_start
        self._must_include_target_token = must_include_target_token

        self._rollout_iter_function = rollout_iter_function
        self._rollout_ratio = rollout_ratio

        self._detach_rollin_logits = detach_rollin_logits
        
        self._rollin_rollout_mixing_coeff = rollin_rollout_mixing_coeff

        self._rollout_reference_policy = rollout_reference_policy
    @overrides
    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      start_predictions: torch.LongTensor,
                      num_decoding_steps,
                      target_tokens: Dict[str, torch.LongTensor] = None,
                     ) -> Dict[str, torch.Tensor]:
        output_dict:  Dict[str, torch.Tensor] = {}
        rollin_steps = num_decoding_steps

        self._decoder_net._accumulate_hidden_states = True
        rollin_output_dict = self.rollin(state,
                                         start_predictions,
                                         rollin_mode=self._rollin_mode,
                                         rollin_steps=rollin_steps,
                                         target_tokens=target_tokens,)

        batch_size, beam_size, num_rollin_steps, num_classes = rollin_output_dict['logits'].size()

        # rollin_logits: (batch_size, num_rollin_steps, num_classes)
        rollin_logits = rollin_output_dict['logits'].squeeze(1)

        # rollin_predictions: (batch_size, num_rollin_steps)
        rollin_predictions = rollin_output_dict['predictions'].squeeze(1)

        # decoder_hidden: (batch_size, num_rollin_steps, hidden_state_size)
        rollin_decoder_hiddens = state['decoder_hiddens']
        batch_size, num_rollin_steps, hidden_size = rollin_decoder_hiddens.size()

        # decoder_context: (batch_size, num_rollin_steps,  hidden_state_size)
        rollin_decoder_context = state['decoder_contexts']

        # If num_tokens_to_rollout is not specified (default value: -1), consider all tokens for next step.
        # We do not want to rollout pad, oov, eos and sos tokens.

        num_tokens_to_rollout = (self._num_tokens_to_rollout \
                                    if self._num_tokens_to_rollout > 0  else \
                                        (num_classes - self._rollout_mask.size(0) if self._mask_padding_and_start \
                                            else num_classes))

        def rollout_mixing_func():
            """ This function generates batch specific random mask
                which is used to decide either pick the next target token
                or pick next prediction.
            """
            return torch.bernoulli(torch.ones(batch_size) * self._rollout_mixing_prob) \
                        .unsqueeze(1) \
                        .expand(batch_size, num_tokens_to_rollout) \
                        .reshape(-1)

        # Reshape/expand source_mask and encoder output
        # to effective batch size of batch_size * num_tokens_to_rollout.
        source_mask = state.get('source_mask', None)
        if source_mask is not None:
            batch_size, source_length = source_mask.shape
            source_mask = source_mask \
                            .unsqueeze(1) \
                            .expand(batch_size, num_tokens_to_rollout, source_length) \
                            .reshape(batch_size * num_tokens_to_rollout, source_length)
            state['source_mask'] = source_mask

        encoder_outputs = state.get('encoder_outputs', None)
        if encoder_outputs is not None:
            batch_size, source_length, hidden_size = encoder_outputs.shape
            encoder_outputs = encoder_outputs \
                                .unsqueeze(1) \
                                .expand(batch_size, num_tokens_to_rollout, source_length, hidden_size) \
                                .reshape(batch_size * num_tokens_to_rollout, source_length, hidden_size)
            state['encoder_outputs'] = encoder_outputs

        rollout_loss_batch = []
        rollout_logits = []
        rollout_predictions = []
        next_tokens_list = []
        rollout_output_dict = {}
        rollout_steps_list = []
        # For SEARNN, we will have one extra step as we look at
        # rollout happening given certain actions were taken.
        # So, we extend targets by 1 to get cost for last action
        # Which should ideally be a padding or end token.
        targets_plus_1 = None
        if target_tokens is not None:
            # targets Shape: (batch_size, num_decoding_steps + 1)
            targets = target_tokens['tokens']

            # targets_plus_1 Shape: (batch_size, num_decoding_steps + 2)
            targets_plus_1 = torch.cat([targets, targets[:, -1].unsqueeze(1)], dim=-1)

        rollout_contexts = []
        all_rollouts = []

        for i, step in enumerate(self._rollout_iter_function(num_decoding_steps + 1)):
            all_rollouts.append(step)
            # Always do rollout for first step and the last step. 
            # Do not rollout for (1 - self._rollout_ratio) steps.
            if i > 0 and step < num_decoding_steps and \
                 random.random() < (1 - self._rollout_ratio):
                continue
            rollout_contexts.append(step)

        while len(rollout_contexts) < 2:
            rollout_idx = random.randint(0, len(all_rollouts)-1)
            rollout_contexts.append(all_rollouts[rollout_idx])

        for step in rollout_contexts:
            # There might be a case where max_decoding_steps < num_decoding_steps, in this 
            # case we want to rollout beyond max_decoding_steps
            rollout_steps = (max(self._max_decoding_steps, num_decoding_steps)  + 1 - step) \
                                if self._do_max_rollout_steps else \
                                    (num_decoding_steps + 1 - step)

            # Do not select masked tokens and always select target token.
            # This will set masked tokens values to be really low and
            # selected tokens value to be really high.
            # So that topk or sampling doesn't return masked values and always returns selected values.
            masked_step_logits = rollin_logits[:, step - 1, :].clone().detach()
            rollout_steps_list.append(step - 1)

            if self._num_neighbors_to_add > 0:
                # Select these self._num_neighbors_to_add tokens.
                # We add _num_neighbors_to_add/2 both on left and the right side.
                # Neighbors are previous and next words in the context.
                num_neighbors_to_add = (self._num_neighbors_to_add // 2) * 2
                if target_tokens is not None:
                    left_context = min(step, num_neighbors_to_add//2)
                    right_context = min(num_decoding_steps - step, num_neighbors_to_add - left_context)
                    neighbor_tokens = target_tokens['tokens'][:, step-left_context: step+right_context]

                    masked_step_logits = masked_step_logits.scatter_(dim=1,
                                                                    index=neighbor_tokens,
                                                                    value=1e2)
            
            # These masks should be done after sampling and including 
            # random words and neighbors as they might include start
            # and EOS and padding tokens, we should do this after selecting
            # those.
            if self._mask_padding_and_start:
                masked_step_logits.scatter_(dim=1,
                                            index=self._rollout_mask.expand(
                                                rollin_logits.size(0), -1),
                                            value=-1e2)

            if self._must_include_target_token:
                masked_step_logits.scatter_(dim=1,
                                            index=targets[:, step].unsqueeze(
                                                1),
                                            value=1e3)

            # softmax of masked step logits + some noise to break ties while topk.
            masked_step_probabilities = F.softmax(masked_step_logits, dim=-1)  + \
                                            1e-10 * masked_step_logits \
                                                        .new_zeros(masked_step_logits.shape) \
                                                        .uniform_(0,1)

            # _, searnn_next_step_tokens = torch.topk(masked_step_logits,
            #                                         num_tokens_to_rollout,
            #                                         dim=-1)

            # searnn_next_step_tokens: (batch_size, num_tokens_to_rollout)
            searnn_next_step_tokens = torch.multinomial(masked_step_probabilities, 
                                                        num_tokens_to_rollout)

            next_tokens_list.append(searnn_next_step_tokens)


            # shape (rollin_start_predictions) : (batch_size * num_tokens_to_rollout)
            rollin_start_predictions = searnn_next_step_tokens.reshape(-1)

            target_tokens_truncated = None
            target_prefixes = None
            if targets_plus_1 is not None:
                # targets Shape: (batch_size, num_decoding_steps + 1)
                targets = target_tokens['tokens']
                target_len = targets_plus_1.size(1)

                targets_expanded = targets_plus_1 \
                                    .unsqueeze(1) \
                                    .expand(batch_size, num_tokens_to_rollout, target_len) \
                                    .reshape(batch_size * num_tokens_to_rollout, target_len)

                targets_step_onwards_expanded = targets_expanded[:, step:]
                target_tokens_truncated = {'tokens': targets_step_onwards_expanded}

                # This is needed to compute the cost which are based on target
                # such as BLEU score or hamming loss.
                target_prefixes = targets_expanded[:, :step]

            # decoder_hidden_step: (batch_size, hidden_state_size)
            decoder_hidden_step = rollin_decoder_hiddens[:, step - 1, :]

            # decoder_hidden_step_expanded: (batch_size *  num_tokens_to_rollout, 1, hidden_state_size)
            decoder_hidden_step_expanded = decoder_hidden_step \
                                            .unsqueeze(1) \
                                            .expand(batch_size, num_tokens_to_rollout, hidden_size) \
                                            .reshape(-1, 1,  hidden_size)

            # decoder_context_step: (batch_size, hidden_state_size)
            decoder_context_step = rollin_decoder_context[:, step - 1, :]

            # decoder_hidden_step_expanded: (batch_size *  num_tokens_to_rollout, 1, hidden_state_size)
            decoder_context_step_expanded = decoder_context_step \
                                                .unsqueeze(1) \
                                                .expand(batch_size, num_tokens_to_rollout, hidden_size) \
                                                .reshape(-1, 1, hidden_size)

            # decoder_hidden: (batch_size * num_tokens_to_rollout, 1, hidden_state_size)
            state['decoder_hiddens'] = decoder_hidden_step_expanded

            # decoder_context: (batch_size *  num_tokens_to_rollout, 1, hidden_state_size)
            state['decoder_contexts'] = decoder_context_step_expanded

            # TODO (Kushal): Maybe do mixed properly by using scatter function.
            if self._rollin_mode == 'teacher_forcing' or \
                self._rollin_mode == 'mixed':
                prediction_prefixes = targets[:, :step] \
                                        .unsqueeze(1) \
                                        .expand(batch_size, num_tokens_to_rollout, step) \
                                        .reshape(batch_size * num_tokens_to_rollout, step) \
                                            if step > 0 else None

            else:
                prediction_prefixes = rollin_predictions[:, :step] \
                                        .unsqueeze(1) \
                                        .expand(batch_size, num_tokens_to_rollout, step) \
                                        .reshape(batch_size * num_tokens_to_rollout, step) \
                                            if step > 0 else None

            self._decoder_net._accumulate_hidden_states = False

            output = self.rollout(state,
                                    rollin_start_predictions,
                                    rollout_steps=rollout_steps,
                                    rollout_mode=self._rollout_mode,
                                    target_tokens=target_tokens_truncated,
                                    prediction_prefixes=prediction_prefixes,
                                    target_prefixes=target_prefixes,
                                    truncate_at_end_all=False,
                                    rollout_mixing_func=rollout_mixing_func,
                                    reference_policy_type=self._rollout_reference_policy,
                                 )

            predictions = output['predictions']\
                            .reshape(batch_size, num_tokens_to_rollout, -1)
            rollout_predictions.append(predictions.unsqueeze(1))

            # shape is (batch_size, 1, num_tokens_to_rollout) so that we can concat them
            # together at the end of the loop.
            loss_batch =  output['loss_batch'] \
                                .reshape(batch_size, 1, num_tokens_to_rollout)
            rollout_loss_batch.append(loss_batch)
            
            # output['logits'] shape: (batch_size * num_tokens_to_rollout, 1, rollout_steps, num_classes)
            rollout_logits.append(output['logits'])

        rollout_output_dict['loss_batch'] = torch.cat(rollout_loss_batch, dim=1)
        rollout_output_dict['predictions'] = torch.cat(rollout_predictions, dim=1)
        rollout_output_dict['next_tokens'] = torch.stack(next_tokens_list, dim=1)
        rollout_output_dict['rollout_steps'] = torch.tensor(rollout_steps_list,
                                                            device=self.current_device)

        rollout_output_dict['logits'] = rollout_logits
        return rollin_output_dict, rollout_output_dict

    @overrides
    def _combine_rollin_rollout_losses(self,
                                        rollin_output_dict: Dict[str, torch.Tensor],
                                        rollout_output_dict: Dict[str, torch.Tensor],
                                        state: Dict[str, torch.Tensor],
                                        target_tokens: Dict[str, torch.LongTensor]):
        # rollin_logits: (batch_size, num_rollin_steps, num_classes)
        logits = rollin_output_dict['logits'].squeeze(1)

        # For whole batch we rollout only these contexts.  
        # By default this will be for all, but in certain cases of
        # filtering, we might only consider a select few.
        # rollout_contexts: (num_rollout_contexts,)
        rollout_contexts = rollout_output_dict['rollout_steps']

        # next_tokens: (batch_size, num_rollout_contexts)
        next_tokens = rollout_output_dict['next_tokens']


        logits = F.log_softmax(logits, dim=-1)

        # Only consider those logits which you did rollout for.
        # rolled_out_logits: (batch_size, num_rollout_contexts, num_classes)
        rolled_out_logits = logits[:, rollout_contexts, :]

        target_mask = util.get_text_field_mask(target_tokens)

        # Similarly, only update logits (or not mask logits) for steps
        # we rollout out for,
        target_mask = target_mask[:, 1:][:, rollout_contexts].float()

        non_batch_dims = tuple(range(1, len(target_mask.shape)))

        # rolled_out_logits: (batch_size, num_rollout_contexts, num_next_tokens)
        scattered_logits = torch.gather(input=rolled_out_logits, dim=-1, index=next_tokens)

        predictions = rollin_output_dict['predictions'].squeeze(1)
        output_dict = {'predictions': predictions.data.cpu()}

        # loss_batch: (batch_size, num_rollout_contexts, num_next_tokens)
        loss_batch = rollout_output_dict['loss_batch']
        
        if self._combiner_mode == 'kl':
            # x: (batch_size, num_rollout_contexts, num_next_tokens)
            x = scattered_logits # F.log_softmax(scattered_logits, dim=-1)

            # y: (batch_size, num_rollout_contexts, num_next_tokens)
            y = F.softmax(-1 * self._temperature * loss_batch, dim=-1)
            
            # kl_losses: (batch_size, num_rollout_contexts)
            kl_losses = self._combiner_loss(x, y).sum(dim=-1)

            # kl_loss_batch: (batch_size,)
            kl_loss_batch_unnormalized = (kl_losses * target_mask).sum(dim=non_batch_dims)

            # Generate denominator for normalizing loss across batch.
            # Ideally this will be equal to batch_size, but this is a
            # safer way to do this. Here, we ignore sequences with all
            # pad tokens.

            # shape : (batch_size,)
            target_mask_sum = target_mask.sum(dim=non_batch_dims)

            kl_loss_batch = kl_loss_batch_unnormalized/target_mask_sum

            loss_batch = self._rollin_rollout_mixing_coeff * rollin_output_dict['loss_batch'] + \
                                (1 - self._rollin_rollout_mixing_coeff) * kl_loss_batch
            
            num_non_empty_sequences = ((target_mask_sum > 0).float().sum() + 1e-13)
            loss = loss_batch.sum()/num_non_empty_sequences
            output_dict['loss'] = loss

            return output_dict

        elif self._combiner_mode == 'mle':
            output_dict['loss'] = rollin_output_dict['loss']
            return output_dict
        return output_dict
