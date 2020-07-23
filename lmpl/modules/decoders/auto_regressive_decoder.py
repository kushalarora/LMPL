from typing import Dict, List, Tuple, Optional, Callable
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

from allennlp_models.generation.modules.seq_decoders import SeqDecoder
from allennlp_models.generation.modules.decoder_nets import DecoderNet

from lmpl.models.sampled_beam_search import SampledBeamSearch
from lmpl.modules.cost_functions.cost_function import CostFunction
from lmpl.modules.detokenizers.detokenizer import DeTokenizer, default_tokenizer

from lmpl.metrics.hamming_loss import HammingLoss



logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

RolloutMixingProbFuncType = Callable[[], torch.Tensor]
DeTokenizerType = Callable[[List[List[str]]], List[str]]

RollinPolicyType = Callable[[int, torch.LongTensor, Dict[str, torch.Tensor]], torch.LongTensor]
# By default, if no rollin_policy is specified, just return the last prediction.
default_rollin_policy = lambda x,y,z: y

RolloutPolicyType = Callable[[int, torch.LongTensor, Dict[str, torch.Tensor], torch.FloatTensor], torch.FloatTensor]
# By default, if no rollout_policy is specified, just return the same class logits.
default_rollout_policy = lambda u,v,w,x: (x,w)

ReferencePolicyType = Callable[[int, torch.LongTensor,Dict[str, torch.Tensor]], torch.FloatTensor]

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-1e30):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    #assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        sorted_logits += sorted_indices_to_remove.int() * filter_value
        logits.scatter_(-1, sorted_indices, sorted_logits)
    return logits

@SeqDecoder.register("lmpl_auto_regressive_seq_decoder")
class QuantExpAutoRegressiveSeqDecoder(SeqDecoder):
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
                 generation_batch_size: int = 200,
                 use_in_seq2seq_mode: bool = False,
                 target_namespace: str = "tokens",
                 beam_size: int = None,
                 scheduled_sampling_ratio: float = 0.0,
                 scheduled_sampling_k: int = 100,
                 scheduled_sampling_type: str = 'uniform',
                 rollin_mode: str = 'mixed',
                 rollout_mode: str = 'learned',
                 use_bleu: bool = False,
                 use_hamming: bool = False,
                 dropout: float = None,
                 sample_rollouts: bool = False,
                 start_token: str =START_SYMBOL,
                 end_token: str = END_SYMBOL,
                 num_decoder_layers: int = 1,
                 mask_pad_and_oov: bool = False,
                 tie_output_embedding: bool = False,
                 label_smoothing_ratio: Optional[float] = None,

                 rollout_cost_function: CostFunction = None,
                 rollin_rollout_combination_mode='mle',
                 rollout_mixing_prob:float = 0.5,
                 detokenizer: DeTokenizer = default_tokenizer,
                 temperature: float = 1,
                 top_k=0, 
                 top_p=0,
                ) -> None:
        super().__init__(target_embedder)

        self.current_device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self._vocab = vocab
        self._seq2seq_mode = use_in_seq2seq_mode

        # Decodes the sequence of encoded hidden states into e new sequence of hidden states.
        self._max_decoding_steps = max_decoding_steps
        self._generation_batch_size = generation_batch_size
        self._decoder_net = decoder_net

        self._target_namespace = target_namespace
        self._label_smoothing_ratio = label_smoothing_ratio

        # TODO (Kushal): Maybe make them modules so that we can add more of these later.
        self._rollin_mode = rollin_mode
        self._rollout_mode = rollout_mode

        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._scheduled_sampling_k = scheduled_sampling_k
        self._scheduled_sampling_type = scheduled_sampling_type
        self._sample_rollouts = sample_rollouts
        self._mask_pad_and_oov = mask_pad_and_oov

        self._rollout_mixing_prob = rollout_mixing_prob

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self._vocab.get_token_index(start_token, self._target_namespace)
        self._end_index = self._vocab.get_token_index(end_token, self._target_namespace)

        self._padding_index = self._vocab.get_token_index(DEFAULT_PADDING_TOKEN, self._target_namespace)
        self._oov_index = self._vocab.get_token_index(DEFAULT_OOV_TOKEN, self._target_namespace)

        if self._mask_pad_and_oov:
            self._vocab_mask = torch.ones(self._vocab.get_vocab_size(self._target_namespace),
                                            device=self.current_device) \
                                    .scatter(0, torch.tensor([self._padding_index, self._oov_index, self._start_index],
                                                                device=self.current_device),
                                                0)
        if use_bleu:
            pad_index = self._vocab.get_token_index(self._vocab._padding_token, self._target_namespace)  # pylint: disable=protected-access
            self._bleu = BLEU(exclude_indices={pad_index, self._end_index, self._start_index})
        else:
            self._bleu = None

        if use_hamming:
            self._hamming = HammingLoss()
        else:
            self._hamming = None

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        beam_size = beam_size or 1

        # TODO(Kushal): Pass in the arguments for sampled. Also, make sure you do not sample in case of Seq2Seq models.
        self._beam_search = SampledBeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size, temperature=temperature)

        self._num_classes = self._vocab.get_vocab_size(self._target_namespace)

        if self.target_embedder.get_output_dim() != self._decoder_net.target_embedding_dim:
            raise ConfigurationError(
                "Target Embedder output_dim doesn't match decoder module's input."
            )

        self._perplexity = Perplexity()

        self._ss_ratio = Average()

        self._rollout_cost_function = rollout_cost_function
        if self._rollout_cost_function:
            self._rollout_cf_avg = Average()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        self.training_iteration = 0
        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(self._decoder_net.get_output_dim(), self._num_classes)

        if tie_output_embedding:
            if self._output_projection_layer.weight.shape != self.target_embedder.weight.shape:
                raise ConfigurationError(
                    "Can't tie embeddings with output linear layer, due to shape mismatch"
                )
            self._output_projection_layer.weight = self.target_embedder.weight

        self._combiner_mode = rollin_rollout_combination_mode

        self._detokenizer = detokenizer

        self._top_k = top_k
        self._top_p = top_p

    def get_output_dim(self):
        return self._decoder_net.get_output_dim()

    def rollin_policy(self,
                      timestep: int,
                      last_predictions: torch.LongTensor,
                      state: Dict[str, torch.Tensor],
                      rollin_mode = 'mixed') -> torch.LongTensor:
        """ Roll-in policy to use.
            This takes in targets, timestep and last_predictions, and decide
            which to use for taking next step i.e., generating next token.
            What to do is decided by rolling mode. Options are
                - teacher_forcing,
                - learned,
                - mixed,

            By default the mode is mixed with scheduled_sampling_ratio=0.0. This 
            defaults to teacher_forcing. You can also explicitly run with teacher_forcing
            mode.

        Arguments:
            timestep {int} -- Current timestep decides which target token to use.
                              In case of teacher_forcing this is usually {t-1}^{th} timestep
                              for predicting t^{th} token.
            last_predictions {torch.LongTensor} -- {t-1}^th token predicted by the model.

        Keyword Arguments:
            targets {torch.LongTensor} -- Targets value if it is available. This will be
                                           available in training mode but not in inference mode. (default: {None})
            rollin_mode {str} -- Rollin mode. Options are
                                  teacher_forcing, learned, scheduled-sampling (default: {'teacher_forcing'})
        Returns:
            torch.LongTensor -- The method returns input token for predicting next token.
        """
        target_tokens = state['rollin_params'].get('target_tokens', None)

        # For first timestep, you are passing start token, so don't do anything smart.
        if (timestep == 0 or
           # If no targets, no way to do teacher_forcing, so use your own predictions.
           target_tokens is None  or
           rollin_mode == 'learned'):
            # shape: (batch_size,)
            return last_predictions

        targets = util.get_token_ids_from_text_field_tensors(target_tokens)
        if rollin_mode == 'teacher_forcing':
            # shape: (batch_size,)
            input_choices = targets[:, timestep]
        elif rollin_mode == 'mixed':
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - self._scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]
        else:
            raise ConfigurationError(f"invalid configuration for rollin policy: {rollin_mode}")
        return input_choices

    def copy_reference_policy(self,
                                timestep,
                                last_predictions: torch.LongTensor,
                                state: Dict[str, torch.Tensor],
                              ) -> torch.FloatTensor:
        targets = util.get_token_ids_from_text_field_tensors(state['rollout_params']['target_tokens'])
        seq_len = targets.size(1)
        
        batch_size = last_predictions.shape[0]
        # So this must be prediction of last step,
        # where there are no more
        if seq_len > timestep + 1:  # + 1 because timestep is an index, indexed at 0.
            # As we might be overriding  the next/predicted token/
            # We have to use the value corresponding to {t+1}^{th}
            # timestep.
            target_at_timesteps = targets[:, timestep + 1]
        else:
            # We have overshot the seq_len, so just repeat the
            # last token which is either _end_token or _pad_token.
            target_at_timesteps = targets[:, -1]

        # TODO: Add support to allow other types of reference policies.
        # target_logits: (batch_size, num_classes).
        # This tensor has 0 at targets and (near) -inf at other places.
        target_logits = (target_at_timesteps.new_zeros((batch_size, self._num_classes)) + 1e-45) \
                            .scatter_(dim=1,
                                      index=target_at_timesteps.unsqueeze(1),
                                      value=1.0).log()
        return target_logits, state
    

    def oracle_reference_policy(self, 
                                timestep: int,
                                last_predictions: torch.LongTensor,
                                state: Dict[str, torch.Tensor],
                                token_to_idx: Dict[str, int],
                                idx_to_token: Dict[int, str],
                               ) -> torch.FloatTensor:
        # TODO(Kushal): This is a temporary fix. Ideally, we should have
        # an individual oracle for this which is different from cost function.
        assert hasattr(self._rollout_cost_function, "_oracle"), \
                "For oracle reference policy, we will need noisy oracle loss function"

        start_time = time.time()
        target_logits, state = self._rollout_cost_function \
                                    ._oracle \
                                    .reference_step_rollout(
                                        step=timestep,
                                        last_predictions=last_predictions,
                                        state=state,
                                        token_to_idx=token_to_idx,
                                        idx_to_token=idx_to_token)
        end_time = time.time()
        logger.info(f"Oracle Reference time: {end_time - start_time} s")
        return target_logits, state
    
    def rollout_policy(self,
                       timestep: int,
                       last_predictions: torch.LongTensor, 
                       state: Dict[str, torch.Tensor],
                       logits: torch.FloatTensor,
                       reference_policy:ReferencePolicyType,
                       rollout_mode: str = 'learned',
                       rollout_mixing_func: RolloutMixingProbFuncType = None,
                      ) -> torch.FloatTensor:
        """Rollout policy to use.
           This takes in predicted logits at timestep {t}^{th} and
           depending upon the rollout_mode replaces some of the predictions
           with targets.

           The options for rollout mode are:
               - learned,
               - reference,
               - mixed.

        Arguments:
            timestep {int} -- Current timestep decides which target token to use.
                              In case of reference this is usually {t-1}^{th} timestep
                              for predicting t^{th} token.
            logits {torch.LongTensor} -- Logits generated by the model for {t}^{th} timestep.
                                         (batch_size, num_classes).

        Keyword Arguments:
            targets {torch.LongTensor} -- Targets value if it is available. This will be
                                available in training mode but not in inference mode. (default: {None})
            rollout_mode {str} -- Rollout mode: Options are:
                                    learned, reference, mixed. (default: {'learned'})
            rollout_mixing_func {RolloutMixingProbFuncType} -- Function to get mask to choose predicted logits vs targets in case of mixed
                                    rollouts.  (default: {0.5})

        Returns:
            torch.LongTensor -- The method returns logits with rollout policy applied.
        """
        output_logits = logits
        targets = state['rollout_params']['target_tokens'].get('tokens', None)
        assert rollout_mode == 'learned' or targets is not None, \
            f"Rollout mode {rollout_mode} needs targets to be specified."

        if rollout_mode == 'learned' or targets is None:
            # For learned rollout policy, just return the same logits.
            return output_logits, state

        target_logits, state = reference_policy(timestep, 
                                         last_predictions, 
                                         state,)

        batch_size = logits.size(0)
        if rollout_mode == 'reference':
             output_logits += target_logits
        elif rollout_mode == 'mixed':
            # Based on the mask (Value=1), copy target values.

            if rollout_mixing_func is not None:
                rollout_mixing_prob_tensor = rollout_mixing_func()
            else:
                # This returns a (batch_size, num_classes) boolean map where the rows are either all zeros or all ones.
                rollout_mixing_prob_tensor = torch.bernoulli(torch.ones(batch_size) * self._rollout_mixing_prob)

            rollout_mixing_mask = rollout_mixing_prob_tensor \
                                    .unsqueeze(1) \
                                    .expand(logits.shape) \
                                    .to(self.current_device)

            # The target_logits ranges from (-inf , 0), so, by adding those to logits,
            # we turn the values that are not target tokens to -inf, hence making the distribution
            # skew towards the target.
            output_logits += rollout_mixing_mask * target_logits
        else:
            raise ConfigurationError(f"Incompatible rollout mode: {rollout_mode}")
        return output_logits, state

    def take_step(self,
                  timestep: int,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor],
                  rollin_policy:RollinPolicyType=default_rollin_policy,
                  rollout_policy:RolloutPolicyType=default_rollout_policy,
                 ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the predictions
            during the last time step.
        state : ``Dict[str, torch.Tensor]``
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape ``(group_size, *)``, where ``*`` can be any other number
            of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
            is a tensor of shape ``(group_size, num_classes)`` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though ``group_size`` is not necessarily
            equal to ``batch_size``, since the group may contain multiple states
            for each source sentence in the batch.
        """
        input_choices = rollin_policy(timestep, last_predictions, state)
        # shape: (group_size, num_classes)
        class_logits, state = self._prepare_output_projections(input_choices, state)

        if not self.training and self._mask_pad_and_oov:
            # This implementation is copied from masked_log_softmax from allennlp.nn.util.
            mask = (self._vocab_mask.expand(class_logits.shape) + 1e-45).log()
            # shape: (group_size, num_classes)
            class_logits = class_logits + mask

        # shape: (group_size, num_classes)
        class_logits, state = rollout_policy(timestep, last_predictions, state, class_logits)
        class_logits = top_k_top_p_filtering(class_logits,  self._top_k, self._top_p, 1e-30)
        return class_logits, state

    @overrides
    def forward(self,  # type: ignore
                encoder_out: Dict[str, torch.LongTensor] = {},
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        source_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        output_dict:  Dict[str, torch.Tensor] = {}
        state:  Dict[str, torch.Tensor] = {}
        decoder_init_state: Dict[str, torch.Tensor] = {}

        # In Seq2Seq setting, we will encode the source sequence,
        # and init the state object with encoder output and decoder
        # cell will use these encoder outputs for attention/initing
        # the decoder states.
        if self._seq2seq_mode:
            state = encoder_out
            decoder_init_state = self._decoder_net.init_decoder_state(state)
            state.update(decoder_init_state)

       # Initialize target predictions with the start index.
        # shape: (batch_size,)
        start_predictions = self._get_start_predictions(state,
                                                        target_tokens,
                                                        self._generation_batch_size)
        
        # In case we have target_tokens, roll-in and roll-out
        # only till those many steps, otherwise we roll-out for
        # `self._max_decoding_steps`.
        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = util.get_token_ids_from_text_field_tensors(target_tokens)

            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps


        # TODO (Kushal): Clean this if else loop jungle.
        # This is training loop.
        if target_tokens:
            rollin_output_dict, rollout_output_dict = \
                    self._forward_loop(state,
                                        start_predictions,
                                        num_decoding_steps=num_decoding_steps,
                                        target_tokens=target_tokens)

            output_dict.update(self._combine_rollin_rollout_losses(rollin_output_dict,
                                                                    rollout_output_dict,
                                                                    state,
                                                                    target_tokens,))

            # The rollin loss (w or w/o teacher_forcing is perplexity.)
            self._perplexity(rollin_output_dict['loss'])

        if not self.training:
            # While validating or testing we need to roll out the learned policy and the output
            # of this rollout is used to compute the secondary metrics
            # like BLEU.
            state = encoder_out
            state.update(decoder_init_state)

            rollout_output_dict = self.rollout(state,
                                                start_predictions,
                                                rollout_steps=num_decoding_steps,
                                                rollout_mode='learned',
                                                sampled=self._sample_rollouts,
                                                # TODO (Kushal): Add a reason why truncate_at_end_all is False here.
                                                truncate_at_end_all=False)

            output_dict['logits'] = rollout_output_dict['logits']

            # shape: (batch_size, beam_size, max_sequence_length)
            top_k_predictions = rollout_output_dict["predictions"]
            output_dict['predictions'] = top_k_predictions

            # shape: (batch_size, max_predicted_sequence_length)
            best_predictions = top_k_predictions[:, 0, :]

            predicted_tokens = self._decode_tokens(best_predictions,
                                                    vocab_namespace=self._target_namespace,
                                                    truncate=True)
            output_dict['predicted_tokens'] = predicted_tokens

            top_k_log_probabilities = rollout_output_dict["class_log_probabilities"]
            prediction_loss = top_k_log_probabilities[:,0].float().data.cpu()
            normalized_prediction_losses = [torch.exp(pred_loss/(len(pred_tokens) + 1))
                                                for pred_loss, pred_tokens in zip(prediction_loss, predicted_tokens)]
            output_dict['model_sampled_model_probs'] =  normalized_prediction_losses

            # TODO (Kushal): Verify if this works. I doubt it does. 
            # Also, do we need target_tokens here? We should be able to compute
            # rollout_cost_function in some cases without the target tokens.
            if self._rollout_cost_function and target_tokens:
                if self._rollout_cost_function.takes_decoded_input():
                    targets = util.get_token_ids_from_text_field_tensors(target_tokens)

                    decoded_targets = self._decode_tokens(targets,
                                                          vocab_namespace=self._target_namespace,
                                                          truncate=True)

                    loss_batch = self._rollout_cost_function(predicted_tokens, decoded_targets)
                else:
                    # This is for rollout cost function like hamming loss for OCR.
                    target_mask = util.get_text_field_mask(target_tokens)
                    targets = util.get_token_ids_from_text_field_tensors(target_tokens)
                    loss_batch = self._rollout_cost_function(best_predictions, 
                                                             targets, 
                                                             target_mask)

                mask = util.get_text_field_mask({'predictions': best_predictions})
                non_batch_dims = tuple(range(1, len(mask.shape)))
                # shape : (batch_size,)
                mask_sum = mask.sum(dim=non_batch_dims)
                num_non_empty_sequences = ((mask_sum > 0).float().sum() + 1e-13)
                loss = loss_batch.sum()/num_non_empty_sequences

                self._rollout_cf_avg(float(loss.cpu()))

            if self._bleu and target_tokens:
                targets = util.get_token_ids_from_text_field_tensors(target_tokens)
                self._bleu(best_predictions, targets)

            if  self._hamming and target_tokens:
                target_mask = util.get_text_field_mask(target_tokens)
                targets = util.get_token_ids_from_text_field_tensors(target_tokens)
                self._hamming(best_predictions, targets, target_mask)

            output_dict['prediction_loss'] = prediction_loss.data.cpu()
            output_dict['model_sampled_predicted_tokens'] = self._detokenizer(predicted_tokens)

        return output_dict

    def _decode_tokens(self,
                       predicted_indices: torch.Tensor,
                       vocab_namespace:str ='tokens',
                       truncate=False) -> List[str]:
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]

            # We add start token to the predictions.
            # In case it is present at position 0, remove it.
            if self._start_index == indices[0]:
                indices = indices[1:]

            indices = list(indices)
            # Collect indices till the first end_symbol
            if truncate and self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self._vocab.get_token_from_index(x, namespace=vocab_namespace)
                                for x in indices]

            all_predicted_tokens.append(predicted_tokens)
        return all_predicted_tokens

    @overrides
    def post_process(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        all_predicted_tokens = self._decode_tokens(predicted_indices, 
                                                    vocab_namespace=self._target_namespace,
                                                    truncate=True)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def _apply_scheduled_sampling(self):

        if not self.training:
            raise RuntimeError("Scheduled Sampling can only be applied during training.")

        k = self._scheduled_sampling_k
        if self._scheduled_sampling_type == 'uniform':
            # This is same scheduled sampling ratio set by config.
            pass
        elif self._scheduled_sampling_type == 'quantized':
            self._scheduled_sampling_ratio =  1 -  k/(k + math.exp(self.training_iteration//k))
        elif self._scheduled_sampling_type == 'linear':
            self._scheduled_sampling_ratio =  1 -  k/(k + math.exp(self.training_iteration/k))
        else:
            raise ConfigurationError(f"{self._scheduled_sampling_type} is not a valid scheduled sampling type.")

        self._ss_ratio.reset()
        self._ss_ratio(self._scheduled_sampling_ratio)
        self.training_iteration += 1

    def _combine_rollin_rollout_losses(self,
                                       rollin_output_dict: Dict[str, torch.Tensor],
                                       rollout_output_dict: Dict[str, torch.Tensor],
                                       state: Dict[str, torch.Tensor],
                                       target_tokens) -> Dict[str, torch.LongTensor]:
        """ Given rollin and rollout, how to combine loss from rollin and
            rollout to compute final loss. This will be used to learning local
            loss such that it reflects the global loss as well.

        Arguments:
            rollin_output_dict {Dict[str, torch.Tensor]} -- Dictionary with rollin computations.
            rollout_output_dict {Dict[str, torch.Tensor]} -- Dictionary with rollout computations.
            state {Dict[str, torch.Tensor]} -- State dictionary.
            target_tokens {Dict[str, torch.Tensor]} -- Target tokens dict.
        Returns:
             output_dict {Dict[str, torch.LongTensor]} -- Updated outptut dict with global and local
                                                          loss combined.
        """

        # Here, we just do rollin for training and rollout for validation, so nothing to compute.
        return {'loss': rollin_output_dict['loss']}

    def rollin(self,
               state: Dict[str, torch.Tensor],
               start_predictions: torch.LongTensor,
               rollin_steps: int,
               rollin_mode: str = 'mixed',
               target_tokens: Dict[str, torch.LongTensor] = None,
               beam_size:int = 1,
               per_node_beam_size: int = None,
               sampled: bool = False,
               truncate_at_end_all: bool = False,
              ):

        # We cannot make a class variable as default, so making default value
        # as None and in case it is None, setting it to num_classes.
        per_node_beam_size: int = per_node_beam_size or self._num_classes

        if self.training:
            self._apply_scheduled_sampling()

        state['rollin_params'] = {}
        state['rollin_params']['target_tokens'] = target_tokens
        rollin_policy = partial(self.rollin_policy,
                                rollin_mode=rollin_mode)

        rolling_policy=partial(self.take_step,
                               rollin_policy=rollin_policy)

        # shape (step_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        # shape (logits): (batch_size, beam_size, num_decoding_steps, num_classes)
        step_predictions, log_probabilities, logits = \
                    self._beam_search.search(start_predictions,
                                                state,
                                                rolling_policy,
                                                max_steps=rollin_steps,
                                                beam_size=beam_size,
                                                per_node_beam_size=per_node_beam_size,
                                                sampled=sampled,
                                                truncate_at_end_all=truncate_at_end_all)

        logits = torch.cat(logits, dim=2)

        batch_size, beam_size, _ = step_predictions.shape
        start_prediction_length = start_predictions.size(0)
        step_predictions = torch.cat([start_predictions.unsqueeze(1) \
                                        .expand(batch_size, beam_size) \
                                        .reshape(batch_size, beam_size, 1),
                                        step_predictions],
                                        dim=-1)

        output_dict = {"predictions": step_predictions,
                        "logits": logits,
                        "class_log_probabilities": log_probabilities,}

        if target_tokens:
            targets = util.get_token_ids_from_text_field_tensors(target_tokens)
            # shape: (batch_size, num_decoding_steps)
            best_logits = logits[:, 0, :, :].squeeze(1)
            target_mask = util.get_text_field_mask(target_tokens)

            rollin_targets = targets[:, :rollin_steps+1]
            rollin_target_masks = target_mask[:, :rollin_steps+1]

            # Compute loss.
            loss_batch = self._get_loss(best_logits, rollin_targets, rollin_target_masks)
            output_dict["loss_batch"] = loss_batch

            # Generate denominator for normalizing loss across batch.
            # Ideally this will be equal to batch_size, but this is a
            # safer way to do this. Here, we ignore sequences with all
            # pad tokens.
            non_batch_dims = tuple(range(1, len(rollin_target_masks.shape)))
            # shape : (batch_size,)
            target_mask_sum = target_mask.sum(dim=non_batch_dims)
            num_non_empty_sequences = ((target_mask_sum > 0).float().sum() + 1e-13)
            loss = loss_batch.sum()/num_non_empty_sequences

            output_dict['loss'] = loss
        return output_dict

    def rollout(self,
                state: Dict[str, torch.Tensor],
                start_predictions: torch.LongTensor,
                rollout_steps: int,
                rollout_mode: str = 'learned',
                beam_size: int = None,
                per_node_beam_size: int = None,
                target_tokens: Dict[str, torch.LongTensor] = None,
                sampled: bool = True,
                truncate_at_end_all: bool = True,
                # shape (prediction_prefixes): (batch_size, prefix_length)
                prediction_prefixes: torch.LongTensor = None,
                target_prefixes: torch.LongTensor = None,
                rollout_mixing_func: RolloutMixingProbFuncType = None,
                reference_policy_type:str = "copy",
               ):

        state['rollout_params'] = {}
        if reference_policy_type == 'oracle':
            reference_policy = partial(self.oracle_reference_policy,
                                        token_to_idx=self._vocab._token_to_index['target_tokens'],
                                        idx_to_token=self._vocab._index_to_token['target_tokens'],
                                       )
            num_steps_to_take = rollout_steps
            state['rollout_params']['rollout_prefixes'] = prediction_prefixes
        else:
            reference_policy = self.copy_reference_policy            
            num_steps_to_take = rollout_steps
            state['rollout_params']['target_tokens'] = target_tokens or {}

        rollout_policy = partial(self.rollout_policy,
                                    rollout_mode=rollout_mode,
                                    rollout_mixing_func=rollout_mixing_func,
                                    reference_policy=reference_policy,
                                )
        rolling_policy=partial(self.take_step,
                               rollout_policy=rollout_policy)

        # This is a hack to speed up performance
        if rollout_mode == 'reference':
            rolling_policy = lambda ts,lp,st,pp=None: reference_policy(ts, lp, st)


        # shape (step_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        # shape (logits): (batch_size, beam_size, num_decoding_steps, num_classes)
        step_predictions, log_probabilities, logits = \
                    self._beam_search.search(start_predictions,
                                                state,
                                                rolling_policy,
                                                max_steps=num_steps_to_take,
                                                beam_size=beam_size,
                                                per_node_beam_size=per_node_beam_size,
                                                sampled=sampled,
                                                truncate_at_end_all=truncate_at_end_all)

        logits = torch.cat(logits, dim=2)

        # Concatenate the start tokens to the predictions.They are not
        # added to the predictions by default.
        batch_size, beam_size, _ = step_predictions.shape

        start_prediction_length = start_predictions.size(0)
        step_predictions = torch.cat([start_predictions.unsqueeze(1) \
                                        .expand(batch_size, beam_size) \
                                        .reshape(batch_size, beam_size, 1),
                                        step_predictions],
                                        dim=-1)

        # There might be some predictions which might have been made by
        # rollin policy. If passed, concatenate them here.
        if prediction_prefixes is not None:
            prefixes_length = prediction_prefixes.size(1)
            step_predictions = torch.cat([prediction_prefixes.unsqueeze(1)\
                                            .expand(batch_size, beam_size, prefixes_length), 
                                         step_predictions],
                                         dim=-1)

        step_targets = None
        if target_tokens is not None:
            step_targets = util.get_token_ids_from_text_field_tensors(target_tokens)

            if target_prefixes is not None:
                prefixes_length = target_prefixes.size(1)
                step_targets = torch.cat([target_prefixes, step_targets], dim=-1)

        output_dict = {"predictions": step_predictions,
                       "logits": logits,
                       "class_log_probabilities": log_probabilities,}

        if step_targets is not None and \
           self._rollout_cost_function is not None:
            # TODO (Kushal): Maybe return loss batches for
            # all beams.
            top_k_predictions = output_dict["predictions"]
            # shape: (batch_size, max_predicted_sequence_length)
            best_predictions = top_k_predictions[:, 0, :]

            predicted_tokens = self._decode_tokens(best_predictions,
                                                    vocab_namespace=self._target_namespace,
                                                    truncate=True)

            output_dict['predicted_tokens'] = predicted_tokens

            if self._rollout_cost_function.takes_decoded_input():
                # This is for rollout cost function like BLEU or Noisy Oracle for OCR.
                decoded_targets = None
                if step_targets is not None:
                    decoded_targets = self._decode_tokens(step_targets,
                                            vocab_namespace=self._target_namespace,
                                            truncate=True)

                loss_batch = self._rollout_cost_function(predicted_tokens, decoded_targets)
            else:
                # This is for rollout cost function like hamming loss for OCR.
                target_mask = util.get_text_field_mask({'tokens': step_targets})
                loss_batch = self._rollout_cost_function(best_predictions, step_targets, target_mask)

            output_dict["loss_batch"] = loss_batch
        return output_dict

    def compute_sentence_probs(self,
                               sequences_dict: Dict[str, torch.LongTensor],
                              ) -> torch.FloatTensor:
        """ Given a batch of tokens, compute the per-token log probability of sequences
            given the trained model.

        Arguments:
            sequences_dict {Dict[str, torch.LongTensor]} -- The sequences that needs to be scored.

        Returns:
            seq_probs {torch.FloatTensor} -- Probabilities of the sequence.
            seq_lens {torch.LongTensor} -- Length of the non padded sequence.
            per_step_seq_probs {torch.LongTensor} -- Probability of per prediction in a sequence
        """
        state = {}
        sequences = util.get_token_ids_from_text_field_tensors(sequences_dict)

        batch_size = sequences.size(0)
        seq_len = sequences.size(1)
        start_predictions = self._get_start_predictions(state,
                                                        sequences_dict,
                                                        batch_size)
        
        # We are now computing probability considering given the sequence,
        # So, we will use rollin_mode=teacher_forcing as we want to select
        # token from the sequences for which we need to compute the probability.
        rollin_output_dict = self.rollin(state={},
                                            start_predictions=start_predictions,
                                            rollin_steps=seq_len - 1,
                                            target_tokens=sequences_dict,
                                            rollin_mode='teacher_forcing',
                                        )

        step_log_probs = F.log_softmax(rollin_output_dict['logits'].squeeze(1), dim=-1)
        per_step_seq_probs = torch.gather(step_log_probs, 2,
                                          sequences[:,1:].unsqueeze(2)) \
                                            .squeeze(2)

        sequence_mask = util.get_text_field_mask(sequences_dict)
        per_step_seq_probs_summed = torch.sum(per_step_seq_probs * sequence_mask[:, 1:], dim=-1)
        non_batch_dims = tuple(range(1, len(sequence_mask.shape)))

        # shape : (batch_size,)
        sequence_mask_sum = sequence_mask[:, 1:].sum(dim=non_batch_dims)

        # (seq_probs, seq_lens, per_step_seq_probs)
        return torch.exp(per_step_seq_probs_summed/sequence_mask_sum), \
                sequence_mask_sum, \
                torch.exp(per_step_seq_probs)

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      start_predictions: torch.LongTensor,
                      num_decoding_steps,
                      target_tokens: Dict[str, torch.LongTensor] = None,
                     ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        rollin_output_dict = {}
        rollout_output_dict = {}

        rollin_output_dict.update(self.rollin(state,
                                                start_predictions,
                                                rollin_steps=num_decoding_steps,
                                                target_tokens=target_tokens,))

        return (rollin_output_dict, rollout_output_dict)

    def _get_start_predictions(self,
              state: Dict[str, torch.Tensor],
              target_tokens: Dict[str, torch.LongTensor] = None,
              generation_batch_size:int = None) ->  torch.LongTensor:

        if self._seq2seq_mode:
           source_mask = state["source_mask"]
           batch_size = source_mask.size()[0]
        elif target_tokens:
            targets = util.get_token_ids_from_text_field_tensors(target_tokens)
            batch_size = targets.size(0)
        else:
            batch_size = generation_batch_size

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        return torch.zeros((batch_size,),
                            dtype=torch.long,
                            device=self.current_device) \
                    .fill_(self._start_index)

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state.get("encoder_outputs", None)

        # shape: (group_size, max_input_sequence_length)
        source_mask = state.get("source_mask", None)

        # shape: (group_size, steps_count, decoder_output_dim)
        previous_steps_predictions = state.get("previous_steps_predictions", None)

        # shape: (batch_size, 1, target_embedding_dim)
        last_predictions_embeddings = self.target_embedder(last_predictions).unsqueeze(1)

        if previous_steps_predictions is None or previous_steps_predictions.shape[-1] == 0:
            # There is no previous steps, except for start vectors in `last_predictions`
            # shape: (group_size, 1, target_embedding_dim)
            previous_steps_predictions = last_predictions_embeddings
        else:
            # shape: (group_size, steps_count, target_embedding_dim)
            previous_steps_predictions = torch.cat(
                [previous_steps_predictions, last_predictions_embeddings], 1
            )

        decoder_state, decoder_output = self._decoder_net(
            previous_state=state,
            encoder_outputs=encoder_outputs,
            source_mask=source_mask,
            previous_steps_predictions=previous_steps_predictions,
        )
        
        state["previous_steps_predictions"] = previous_steps_predictions

        # Update state with new decoder state, override previous state
        state.update(decoder_state)
        
        if self._decoder_net.decodes_parallel:
            decoder_output = decoder_output[:, -1, :]
        
        # add dropout
        decoder_hidden_with_dropout = self._dropout(decoder_output)

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_hidden_with_dropout)

        return output_projections, state

    def _get_loss(self,
                  logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask, label_smoothing=self._label_smoothing_ratio, average=None)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}

        all_metrics.update({'perplexity': self._perplexity.get_metric(reset=reset),
                    'ss_ratio': self._ss_ratio.get_metric(reset=reset),
                    'training_iter': self.training_iteration})

        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))

        if self._hamming and not self.training:
            all_metrics.update({'hamming': self._hamming.get_metric(reset=reset)})

        if self._rollout_cost_function:
            all_metrics.update({self._rollout_cost_function.name: self._rollout_cf_avg.get_metric(reset=reset)})
        return all_metrics
