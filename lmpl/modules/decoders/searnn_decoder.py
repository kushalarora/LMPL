from typing import Dict, List, Tuple, Optional, Callable, Iterable, Union
from overrides import overrides

import logging

import torch
import torch.nn.functional as F
import sys
import numpy as np
import random
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.modules import Embedding
from allennlp.nn import util
from allennlp.training.metrics import Metric

from allennlp_models.generation.modules.decoder_nets import DecoderNet
from allennlp_models.generation.modules.seq_decoders import SeqDecoder

from lmpl.modules.decoders.auto_regressive_decoder import BaseRollinRolloutDecoder
from lmpl.modules.criterions import LossCriterion
from lmpl.modules.utils import expand_tensor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def rollout_mixing_functional(batch_size, rollout_mixing_prob, num_tokens_to_rollout):
    def rollout_mixing_func():
        """ This function generates batch specific random mask
            which is used to decide either pick the next target token
            or pick next prediction.
        """
        return torch.bernoulli(torch.ones(batch_size) * rollout_mixing_prob) \
                    .unsqueeze(1) \
                    .expand(batch_size, num_tokens_to_rollout) \
                    .reshape(-1)
    return rollout_mixing_func

def reshape_encoder_output(rollin_state: Dict[str, torch.Tensor], 
                            num_tokens_to_rollout: int):
    """Reshape/expand source_mask and encoder output
        to effective batch size of batch_size * num_tokens_to_rollout. """
    rollout_state: Dict[str, torch.Tensor] = {}
    source_mask = rollin_state.get('source_mask', None)
    if source_mask is not None:
        rollout_state['source_mask'] = expand_tensor(source_mask,
                                            num_tokens_to_rollout)

    encoder_outputs = rollin_state.get('encoder_outputs', None)
    if encoder_outputs is not None:
        rollout_state['encoder_outputs'] = expand_tensor(encoder_outputs,
                                                num_tokens_to_rollout)
    return rollout_state

def reshape_targets(targets:torch.LongTensor, 
                        step:int,  num_tokens_to_rollout:int):
    targets_expanded = expand_tensor(targets, 
                                    num_tokens_to_rollout)
    targets_step_onwards_expanded = targets_expanded[:, step:]
    target_tokens_truncated = {'tokens': 
                {'tokens': targets_step_onwards_expanded}}

    # This is needed to compute the cost which are based on target
    # such as BLEU score or hamming loss.
    target_prefixes = targets_expanded[:, :step]
    return target_prefixes, target_tokens_truncated

def reshape_decoder_hidden_and_context(state: Dict[str, torch.Tensor],
                                        rollin_decoder_context: torch.FloatTensor, 
                                        rollin_decoder_hiddens: torch.FloatTensor,
                                        step:int, num_tokens_to_rollout:int):
    # decoder_hidden_step: (batch_size, hidden_state_size)
    # decoder_context_step: (batch_size, hidden_state_size)
    decoder_hidden_step = rollin_decoder_hiddens
    decoder_context_step = rollin_decoder_context

    # decoder_hidden_step_expanded: (batch_size *  num_tokens_to_rollout, 1, hidden_state_size)
    decoder_hidden_step_expanded = expand_tensor(decoder_hidden_step,
                                                 num_tokens_to_rollout)

    # decoder_hidden_step_expanded: (batch_size *  num_tokens_to_rollout, 1, hidden_state_size)
    decoder_context_step_expanded = expand_tensor(decoder_context_step,
                                                        num_tokens_to_rollout)

    # decoder_hidden: (batch_size * num_tokens_to_rollout, 1, hidden_state_size)
    # decoder_context: (batch_size *  num_tokens_to_rollout, 1, hidden_state_size)
    state['decoder_hidden'] = decoder_hidden_step_expanded
    state['decoder_context'] = decoder_context_step_expanded
    return state


def get_neighbor_tokens(num_neighbors_to_add:int, 
                        num_decoding_steps:int, 
                        step: int, targets: torch.LongTensor):
    assert num_neighbors_to_add > 1, "Num Neighbors should always be greater than 2, i.e., at least one from each side."
    # TODO: #20 If right_context < num_neighbors_to_add/2, we end up getting fewer neighbors.
    left_context = min(step, num_neighbors_to_add//2)
    right_context = min(num_decoding_steps - step, num_neighbors_to_add - left_context)

    neighbor_tokens = torch.cat([targets[:, step-left_context:step], # neighbors/2 left tokens
                                 targets[:, step+1:step+1+right_context]], # neighbors/2 right tokens excluding step token.
                                dim=-1)
    return neighbor_tokens

def extend_targets_by_1(targets: torch.LongTensor):
    # For SEARNN, we extend targets by 1 to get cost for last action
    # which should ideally be a padding or end token.
    # targets_plus_1 Shape: (batch_size, num_decoding_steps + 2)
    targets_plus_1 = torch.cat([targets, targets[:, -1:]], dim=-1)
    return targets_plus_1


@SeqDecoder.register("lmpl_searnn_decoder")
class LMPLSEARNNDecoder(BaseRollinRolloutDecoder):

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

                 rollin_mode: str = 'teacher_forcing',
                 rollout_mode: str = 'reference',
                 rollin_steps: int = 50,
                 rollout_mixing_prob: float = 0.5,
                 num_tokens_to_rollout:int = -1,
                 num_neighbors_to_add: int = 0,
                 do_max_rollout_steps: bool = False,
                 mask_padding_and_start: bool = True,
                 must_include_target_token: bool = True,
                 rollout_iter_function: Callable[[int], Iterable[int]]=lambda x: range(1, x),
                 rollout_iter_start_pct: int = 0,
                 rollout_iter_end_pct: int = 100,
                 rollout_ratio:float = 1.0,
                 detach_rollin_logits: bool = False,
                 rollin_rollout_mixing_coeff: float = 0.25,
                 rollout_reference_policy:str = 'copy',
                 sort_next_tokens:bool = False,
                 include_first: bool = False,
                 include_last: bool = False,
                 max_num_contexts: int = sys.maxsize,
                 min_num_contexts: int = 2,
                 num_random_tokens_to_add = 0,
                 add_noise_to_sampling = True,
                 max_sampling_noise = 1e-5,
                 sampling_temperature=10,
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

            rollin_mode=rollin_mode,
            rollout_mode=rollout_mode,
            rollout_mixing_prob=rollout_mixing_prob,
        )

        # TODO: #18 See how num_decoding_steps relate to rollin_steps
        self._rollin_steps = rollin_steps

        self._num_tokens_to_rollout = num_tokens_to_rollout
        
        # TODO #15 (Kushal): Verify if we need _rollout_mask, if we do, add justification for it.
        self._rollout_mask = torch.tensor([self._padding_index, self._start_index],
                                           device=self.current_device)

        self._num_neighbors_to_add = num_neighbors_to_add
        
        self._num_random_tokens_to_add = num_random_tokens_to_add

        self._do_max_rollout_steps = do_max_rollout_steps

        self._mask_padding_and_start = mask_padding_and_start
        self._must_include_target_token = must_include_target_token
        self._rollout_iter_function = rollout_iter_function
        self._rollout_iter_start_pct = rollout_iter_start_pct
        self._rollout_iter_end_pct = rollout_iter_end_pct

        self._rollout_ratio = rollout_ratio

        self._detach_rollin_logits = detach_rollin_logits
        
        self._rollin_rollout_mixing_coeff = rollin_rollout_mixing_coeff

        self._rollout_reference_policy = rollout_reference_policy

        self._sort_next_tokens = sort_next_tokens

        self._include_first = include_first
        self._include_last = include_last
        self._max_num_contexts = max_num_contexts
        self._min_num_contexts = min_num_contexts
        self._add_noise_to_sampling = add_noise_to_sampling
        self._max_sampling_noise = max_sampling_noise
        self._sampling_temperature = sampling_temperature

    def get_contexts_to_rollout(self,
                                context_iterator:Iterable[int], 
                                num_decoding_steps:int,
                                rollout_ratio: float):
        # TODO: #16 (@kushalarora) Simplify the context rollout computation logic.
        rollout_contexts = []
        context_iterator_len =  len(context_iterator)
        for i, step in enumerate(context_iterator):
            # Always do rollout for first step and the last step. 
           
            if (i == 0) and self._include_first or \
               (i == context_iterator_len - 1) and self._include_last or \
               random.random() < rollout_ratio:
                    rollout_contexts.append(step)
   
        num_rollout_contexts = len(rollout_contexts)
        if num_rollout_contexts < self._min_num_contexts:
            num_additional_contexts = max(int(rollout_ratio * context_iterator_len),
                                            self._min_num_contexts) \
                                         - len(rollout_contexts)
            
            remaining_contexts = [x for x in context_iterator if x not in rollout_contexts]

            rollout_contexts += np.random.choice(remaining_contexts,
                                                    num_additional_contexts,
                                                    replace=False).tolist()

        num_rollout_contexts = len(rollout_contexts)
        if num_rollout_contexts > self._max_num_contexts:
            sorted_rollout_contexts = sorted(rollout_contexts)
            rollout_contexts = []
            num_contexts = self._max_num_contexts

            if self._include_first:
                rollout_contexts.append(sorted_rollout_contexts[0])
                sorted_rollout_contexts = sorted_rollout_contexts[1:]
                num_contexts -= 1

            if self._include_last:
                rollout_contexts.append(sorted_rollout_contexts[-1])
                sorted_rollout_contexts = sorted_rollout_contexts[:-1]
                num_contexts -= 1

            rollout_contexts += np.random.choice(sorted_rollout_contexts,
                                                    num_contexts,
                                                    replace=False).tolist()

        # Sorting it here as while accumulating rollin decoder hidden state, we expect
        # the rollins to be sorted in increasing order of timestep.
        return sorted(rollout_contexts)

    def get_num_tokens_to_rollout(self):
        num_tokens_to_rollout = min(self._num_tokens_to_rollout, self._num_classes)
        # If num_tokens_to_rollout is not specified (default value: -1), 
        # consider all tokens for next step.
        if num_tokens_to_rollout < 0:
            num_tokens_to_rollout = self._num_classes
            # If masking padding and start, we will be rolling out 
            # all tokens except for padding and start.
            if self._mask_padding_and_start:
                num_tokens_to_rollout -= self._rollout_mask.size(0)
        return num_tokens_to_rollout

    def get_next_tokens(self, 
            rollin_logits: torch.FloatTensor, 
            step: int, num_decoding_steps: int, 
            targets: torch.LongTensor,
            num_tokens_to_rollout: int,
        ):
        """Do not select masked tokens and always select target token.
            This will set masked tokens values to be really low and
            selected tokens value to be really high.
            So that topk or sampling doesn't return masked values 
            and always returns selected values.
        """
        searnn_next_step_tokens = []
        # If num_tokens_to_rollout >= num_classes, we return all the tokens in logits.
        # This saves computation. Additionally, torch.multinomial for large 
        # num_tokens_to_rollout, sometime ends up returning duplicates 
        # despite replacement=False
        # See, https://github.com/pytorch/pytorch/issues/25030.
        if num_tokens_to_rollout >= self._num_classes:
            batch_size = rollin_logits.size(0)
            return torch.LongTensor([range(0, self._num_classes)] * batch_size) \
                            .to(rollin_logits.device)

        # step_logits: (batch_size, vocab_size)
        step_logits = rollin_logits[:, step - 1, :].clone().detach()
        step_logits[torch.isnan(step_logits)] = -1e30
        
        # step_unnorm_probabilities: (batch_size, vocab_size)
        step_unnorm_probabilities = F.softmax(step_logits/self._sampling_temperature, dim=-1)

        if self._num_neighbors_to_add > 0:
            # Select these self._num_neighbors_to_add tokens.
            # We add _num_neighbors_to_add/2 both on left and the right side.
            # Neighbors are previous and next words in the context.
            neighbor_tokens = get_neighbor_tokens(self._num_neighbors_to_add, 
                                                    num_decoding_steps, 
                                                    step, targets)
            searnn_next_step_tokens += [neighbor_tokens]
            num_tokens_to_rollout -= neighbor_tokens.size(1)

        if self._num_random_tokens_to_add > 0:
            top_k_probs, top_k_indices = torch.topk(step_unnorm_probabilities, k=250)
            random_token_indices = torch.multinomial(torch.ones_like(top_k_probs), self._num_random_tokens_to_add)
            random_tokens = torch.gather(top_k_indices, -1, random_token_indices)
            searnn_next_step_tokens += [random_tokens]
            num_tokens_to_rollout -= random_tokens.size(1)

        # These masks should be done after sampling and including 
        # random words and neighbors as they might include start,
        # EOS and padding tokens, we should do this after selecting those.
        # TODO (@kushalarora) Do we need to mask start, end and padding tokens?
        if self._mask_padding_and_start:
            # rollout_masks: (batch_size, num_tokens_to_mask)
            rollout_masks = self._rollout_mask.expand(step_unnorm_probabilities.size(0), -1)
            step_unnorm_probabilities.scatter_(dim=1, index=rollout_masks, value=0)
            add_noise = True

        if self._must_include_target_token:
            # target_token: (batch_size, 1)
            target_token = targets[:, step:step+1]
            searnn_next_step_tokens += [target_token]
            num_tokens_to_rollout -= 1

        # softmax of masked step logits + some noise to break ties while topk.
        # noise: (batch_size, vocab_size)
        if self._add_noise_to_sampling:
            noise = self._max_sampling_noise * torch.empty_like(step_unnorm_probabilities).uniform_(0,1)

            step_unnorm_probabilities += noise

        if num_tokens_to_rollout > 0:
            # searnn_next_step_tokens: (batch_size, num_tokens_to_rollout)
            logit_sampled_tokens = torch.multinomial(step_unnorm_probabilities, 
                                                        num_tokens_to_rollout)
            searnn_next_step_tokens += [logit_sampled_tokens]


        if self._sort_next_tokens:
            searnn_next_step_tokens, _ = torch.sort(searnn_next_step_tokens)
        
        if False and self.training_iteration % 300 == 0:
            logger.warn(f"Next Tokens: {searnn_next_step_tokens}")
            logger.warn(f"Top 10: {torch.topk(step_unnorm_probabilities, k=num_tokens_to_rollout)}")

        searnn_next_step_tokens = torch.cat(searnn_next_step_tokens, dim=-1)
        return searnn_next_step_tokens

    def get_prediction_prefixes(self, 
                                targets: torch.LongTensor,
                                rollin_predictions: torch.LongTensor,
                                step: int, num_tokens_to_rollout: int):
        # TODO (Kushal): Replace this with topk 1 (along dim=-1) on logits.
        # That will handle the case weather it was teacher forcing, mixed or learned
        # rollin policy.
        prediction_prefixes = None
        if step > 0: 
            if self._rollin_mode == 'teacher_forcing' or \
                                self._rollin_mode == 'mixed':
                prediction_prefixes = expand_tensor(targets[:, :step], 
                                                    num_tokens_to_rollout)
            else:
                prediction_prefixes = expand_tensor(rollin_predictions[:, :step],
                                                    num_tokens_to_rollout)
        return prediction_prefixes

    def get_rollout_steps(self, 
                         num_decoding_steps: int, 
                         step: int, ):
        # TODO #19 (@kushalarora) Verify if we need do_max_rollout_steps.
        rollout_steps = num_decoding_steps + 1 - step
        if self._do_max_rollout_steps:
            # There might be a case where max_decoding_steps < num_decoding_steps, in this 
            # case we want to rollout beyond max_decoding_steps
            # TODO: #21 Add assertion to ensure rollout_steps is not negative.
            rollout_steps = min(self._max_decoding_steps, rollout_steps + 5)
        return rollout_steps

    def get_rollout_iterator(self, 
                             rollout_contexts: List[int],
                             rollin_state: Dict[str, torch.Tensor],
                             rollin_logits: torch.FloatTensor,
                             rollin_predictions: torch.LongTensor,
                             rollin_decoder_context: torch.FloatTensor,
                             rollin_decoder_hiddens: torch.FloatTensor,
                             targets: torch.LongTensor,
                             num_decoding_steps: int, 
                             num_tokens_to_rollout: int,
                             ) -> Iterable[Dict[str, Union[torch.Tensor, int]]]:
        """ Get rollout iterator.
        """
        self._decoder_net._accumulate_hidden_states = False
        
        for i, step in enumerate(rollout_contexts):

            # Reshape/expand source_mask and encoder output
            # to effective batch size of batch_size * num_tokens_to_rollout.
            rollout_state = reshape_encoder_output(rollin_state, num_tokens_to_rollout)

            searnn_next_step_tokens = self.get_next_tokens(
                                                rollin_logits=rollin_logits, 
                                                step=step, 
                                                num_decoding_steps=num_decoding_steps, 
                                                num_tokens_to_rollout=num_tokens_to_rollout,
                                                targets=targets)
            
            batch_size: int = rollin_logits.size(0)
            
            # shape (rollout_start_predictions) : (batch_size * num_tokens_to_rollout)
            rollout_start_predictions = searnn_next_step_tokens.reshape(-1)

            accumulated_decoder_context_length = rollin_decoder_context.size(1)
            accumulated_decoder_hiddens_length = rollin_decoder_hiddens.size(1)

            assert accumulated_decoder_context_length == accumulated_decoder_context_length, \
                    "decoder's accumulated hidden and context lengths during rollins should be same"
            
            if i < accumulated_decoder_context_length - 1:
                rollin_decoder_context_ith = rollin_decoder_context[:, i]
                rollin_decoder_hiddens_ith = rollin_decoder_hiddens[:, i]
            else:
                rollin_decoder_context_ith = rollin_decoder_context[:, -1]
                rollin_decoder_hiddens_ith = rollin_decoder_hiddens[:, -1]

            target_prefixes, target_tokens_truncated   = \
                    reshape_targets(targets, step, num_tokens_to_rollout)
            rollout_state = reshape_decoder_hidden_and_context(
                                    state=rollout_state, 
                                    rollin_decoder_context=rollin_decoder_context_ith,
                                    rollin_decoder_hiddens=rollin_decoder_hiddens_ith,
                                    step=step, 
                                    num_tokens_to_rollout=num_tokens_to_rollout)

            prediction_prefixes = self.get_prediction_prefixes(
                                            targets=targets, 
                                            step=step,
                                            rollin_predictions=rollin_predictions,  
                                            num_tokens_to_rollout=num_tokens_to_rollout)
            
            rollout_steps = self.get_rollout_steps(num_decoding_steps, step)

            rollout_mixing_function = rollout_mixing_functional(batch_size, 
                                                                self._rollin_rollout_mixing_coeff,
                                                                num_tokens_to_rollout)

            rollout_output_dict = self.rollout(rollout_state,
                                        rollout_start_predictions,
                                        rollout_steps=rollout_steps,
                                        target_tokens=target_tokens_truncated,
                                        prediction_prefixes=prediction_prefixes,
                                        target_prefixes=target_prefixes,
                                        truncate_at_end_all=False,
                                        rollout_mixing_func=rollout_mixing_function,
                                        # TODO: #23 Maybe make it default, why make it an argument.
                                        reference_policy_type=self._rollout_reference_policy,
                                        sampled=self._sample_rollouts,
                                    )
                
            rollout_output_dict['num_tokens_to_rollout'] = num_tokens_to_rollout
            rollout_output_dict['step'] = step
            rollout_output_dict['next_tokens'] = searnn_next_step_tokens

            yield rollout_output_dict

    @overrides
    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      start_predictions: torch.LongTensor,
                      num_decoding_steps,
                      target_tokens: Dict[str, torch.LongTensor] = None,
        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        output_dict: Dict[str, torch.Tensor] = {}
        rollout_output_dict_list: List[Dict[str, torch.Tensor]] = []
        # Do rollin with state accumulation to be able to do rollouts from them.
        # TODO #14 (Kushal): During roll-in for SEARNN, only  accumulate/preserve states 
        # for which we need to do rollouts later.
        self._decoder_net._accumulate_hidden_states = True

        # +1 because we start at 1 and we need to rollout num_decoding_steps which is usually length - 1. 
        rollout_iter_start = max(1, int(self._rollout_iter_start_pct/100. * num_decoding_steps))
        rollout_iter_end = int(self._rollout_iter_end_pct/100. * num_decoding_steps)
        context_iter=range(rollout_iter_start, rollout_iter_end + 1)
        rollout_contexts = self.get_contexts_to_rollout(rollout_ratio=self._rollout_ratio,
                                                    num_decoding_steps=num_decoding_steps,
                                                    context_iterator=context_iter)
        state['timesteps_to_accumulate'] = set(rollout_contexts)

        rollin_output_dict = self.rollin(state=state, 
                                         start_predictions=start_predictions,
                                         rollin_steps=num_decoding_steps,
                                         target_tokens=target_tokens)

        # decoder_context: (batch_size, num_rollin_steps,  hidden_state_size)
        # decoder_hidden: (batch_size, num_rollin_steps, hidden_state_size)
        rollin_decoder_context = state['decoder_accumulated_hiddens']
        rollin_decoder_hiddens = state['decoder_accumulated_contexts']

        # rollin_logits: (batch_size, beam_size, num_rollin_steps, num_classes)
        # rollin_predictions: (batch_size, beam_size, num_rollin_steps)
        rollin_logits = rollin_output_dict['logits']
        rollin_predictions = rollin_output_dict['predictions']
        
        output_dict['predictions'] = rollin_predictions.data.cpu()

        # rollin_logits: (batch_size, num_rollin_steps, num_classes)
        # rollin_predictions: (batch_size, num_rollin_steps)
        rollin_logits = rollin_logits.squeeze(1)
        rollin_predictions = rollin_predictions.squeeze(1)

        num_tokens_to_rollout = self.get_num_tokens_to_rollout()
        
        rollin_state = state

        # targets Shape: (batch_size, num_decoding_steps + 1)
        targets = util.get_token_ids_from_text_field_tensors(target_tokens)

        # TODO #17 Verify if target_plus_1 logic is essential for SEARNN, if not, get rid of it. (@kushalarora)
        targets = extend_targets_by_1(targets)

        rollout_output_dict_iter = self.get_rollout_iterator(
                                                rollout_contexts=rollout_contexts,
                                                rollin_state=rollin_state,
                                                rollin_logits=rollin_logits,
                                                rollin_predictions=rollin_predictions,
                                                rollin_decoder_context=rollin_decoder_context,
                                                rollin_decoder_hiddens=rollin_decoder_hiddens, targets=targets,
                                                num_decoding_steps=num_decoding_steps, 
                                                num_tokens_to_rollout=num_tokens_to_rollout)

        return output_dict, rollin_output_dict, rollout_output_dict_iter

