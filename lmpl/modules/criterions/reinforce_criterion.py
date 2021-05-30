from typing import Dict, List, Iterable, Union, Tuple
from overrides import overrides


import torch 
import torch.nn.functional as F

from allennlp.nn import util

from lmpl.modules.criterions.base_loss_criterion import LossCriterion
from lmpl.modules.cost_functions import CostFunction

@LossCriterion.register("reinforce")
class ReinforceCriterion(LossCriterion):
  def __init__(self, 
          rollout_cost_function:CostFunction,
          rollin_rollout_mixing_coeff: float = 0.,
          labeling_smooting_ratio: float = None,
          temperature: float = 1, 
          detach_rollin_logits: bool = False,
          warm_start_for_epochs: int = -1, 
          warm_start_for_batch_numbers: int = -1,
          normalize_costs: bool = False,
          entropy_regularization_coeff: float = 0,
          alpha: float = 1.0,
          include_target_in_candidates: bool = True,
        ):
    super().__init__(
                rollout_cost_function=rollout_cost_function,
                labeling_smooting_ratio=labeling_smooting_ratio,
                rollin_rollout_mixing_coeff=rollin_rollout_mixing_coeff,
                shall_compute_rollin_loss=(rollin_rollout_mixing_coeff > 0),
                shall_compute_rollout_loss=True,
                warm_start_for_epochs=warm_start_for_epochs,
                warm_start_for_batch_numbers=warm_start_for_batch_numbers,
              )
    self._temperature = temperature
    self._detach_rollin_logits = detach_rollin_logits
    self._normalize_costs = normalize_costs
    self._entropy_regularization_coeff = entropy_regularization_coeff
    assert alpha > 0 and alpha <= 1., "alpha should be in the range (0, 1]."
    self._alpha = alpha
    self._include_target_in_candidates = include_target_in_candidates
  @overrides
  def _compute_rollout_loss_single_iter(self,
                  rollin_output_dict: Dict[str, torch.Tensor],
                  rollout_output_dict: Dict[str, Union[torch.Tensor, int]],
                  state: Dict[str, torch.Tensor],
                  target_tokens: Dict[str, torch.Tensor] = None,
              ) -> Tuple[torch.FloatTensor, torch.FloatTensor, int]:

      # rollin_logits: (batch_size, beam_size, num_rollin_steps, num_classes)
      rollin_logits: torch.FloatTensor = rollin_output_dict['logits']
      rollin_batch_size, rollin_beam_size, rollin_steps, rollin_num_classes = rollin_logits.shape

      # rollin_predictions: (batch_size, beam_size, num_rollin_steps)
      rollin_predictions: torch.LongTensor = rollin_output_dict['predictions']
      if self._detach_rollin_logits:
          rollin_output_dict['logits'].detach_()

      step: int = rollout_output_dict['step']

      # cost_batch: (batch_size, beam_size)
      cost_batch = self._compute_rollout_cost(
                            rollout_output_dict=rollout_output_dict)

      # Only consider those logits which you did rollout for.
      # rollin_logits_prefix: (batch_size, step - 1, num_classes)

      # rollout_logits: (batch_size, beam_size, num_decoding_steps - step - 1, num_classes)
      rollout_logits: torch.FloatTensor = rollout_output_dict['logits']

      rollout_batch_size, rollout_beam_size, rollout_steps, rollout_num_classes = rollout_logits.shape

      assert rollout_batch_size == rollin_batch_size, f"Rollin and Rollout batch sizes should be equal." + \
                                                           f"Rollin: {rollin_batch_size}, Rollout: {rollout_batch_size}."
      
      assert rollin_num_classes == rollout_num_classes, f"Rollin and Rollout num_classes should be equal." + \
                                                           f"Rollin: {rollin_num_classes}, Rollout: {rollout_num_classes}."

      batch_size = rollin_batch_size
      # We have step - 1 here because logits predict next token given the
      # current context, so they are shifted by 1 i.e., at logit index 0, 
      # we predict token at index 1 given token at 0. So, prediction 
      # corresponding to target step=t, will be at index t-1 in logits.
      rollin_logits_prefix: torch.FloatTensor = rollin_logits[:, 0, :step, :]\
                                                    .unsqueeze(1) \
                                                    .expand(batch_size, 
                                                            rollout_beam_size,
                                                            step,
                                                            rollin_num_classes)

 
      # rollout_logits: (batch_size, beam_size, num_decoding_steps - 1, num_classes)
      rollin_rollout_logits: torch.FloatTensor = torch.cat([rollin_logits_prefix, 
                                                            rollout_logits],
                                                           dim=2)
      
      if self._normalize_costs:
        # cost_batch: (batch_size, beam_size)
        cost_batch = (cost_batch - cost_batch.min())/(cost_batch.max() - cost_batch.min())
  
      # predictions: (batch_size, beam_size, num_decoding_steps)
      predictions = rollout_output_dict["predictions"]
      
      # rollout_logits: (batch_size, beam_size, num_decoding_steps - 1, num_classes)
      rollin_rollout_logits = F.log_softmax(rollin_rollout_logits, dim=-1)

      # log_probs: (batch_size, beam_size, num_decoding_steps - 1)
      log_probs = torch.gather(rollin_rollout_logits, -1,
                                  predictions[:, :, 1:] \
                                    .unsqueeze(dim=-1)) \
                        .squeeze(dim=-1)

      # Get mask expects first detects </S> and considers all the tokens before this
      # and masks out everything after this.
      log_prob_mask = rollout_output_dict['prediction_masks'][:, :, 1:]
  
      # rewards: (batch_size, beam_size)
      rewards = (1 - cost_batch)

      if self._include_target_in_candidates:
        rewards = torch.cat([rewards, rewards.new_ones((batch_size, 1))],
                             dim=-1)
        batch_size, beam_size, num_decoding_steps = log_probs.shape
        
        # shape: (batch_size, max_target_sequence_length)
        targets = rollout_output_dict['targets']

        # shape: (batch_size, 1, max_target_sequence_length, 1)
        logits = F.log_softmax(rollin_logits, dim=-1)
        target_log_probs = torch.gather(logits, 3,
                                        targets[:, 1:].unsqueeze(1)
                                               .unsqueeze(-1))
        # shape: (batch_size, 1, max_target_sequence_length)
        target_log_probs.squeeze_(-1)

        assert log_probs.size(-1) == target_log_probs.size(-1), \
          "Sequence length for rollout and target should be same. Ensure do_max_rollout=False."
        log_probs = torch.cat([log_probs, target_log_probs], dim=1)

        # shape: (batch_size, max_target_sequence_length)
        target_log_prob_mask =  rollout_output_dict['target_masks']
        log_prob_mask = torch.cat([log_prob_mask, target_log_prob_mask[:, 1:].unsqueeze(1)], dim=1)

      log_probs *= log_prob_mask
      num_tokens_per_seq = log_prob_mask.sum(dim=-1)
      normalized_log_probs = log_probs.sum(dim=-1)/num_tokens_per_seq

      # We are trying to maximize the reward, hence minimizing the log prob * reward.
      rl_loss_batch =  -1 * (self._alpha**step) * normalized_log_probs * rewards

      # Add entropy regularization coefficient
      entropy_regularization_term = torch.zeros_like(rl_loss_batch)
      if self._entropy_regularization_coeff > 0:
        entropy_regularization_term = self._compute_entropy_regularization_term(
                                                        logits=rollin_rollout_logits, 
                                                        mask=log_prob_mask,
                                                      )
      rl_loss_batch += -1 * self._entropy_regularization_coeff * entropy_regularization_term

      # Average over beam
      rl_loss_batch = rl_loss_batch.mean(dim=-1)

      return {'loss_batch': rl_loss_batch, 
              'cost_batch': cost_batch.sum(dim=-1), # average over beam. 
              'step': step,
              'normalized_log_probs': normalized_log_probs,
              'entropy_regularization_term': entropy_regularization_term,
              }


  def _compute_entropy_regularization_term(self,
                                          logits: torch.FloatTensor, 
                                          mask: torch.LongTensor):
      # entropy_seq: (batch_size, beam_size, num_decoding_steps - 1)
      entropy_seq =  -1 * (torch.exp(logits) * logits).sum(dim=-1)
      entropy_seq *= mask
      num_tokens_per_seq = mask.sum(dim=-1)
      # normalize_entropy: (batch_size, )
      normalized_entropy = entropy_seq.sum(dim=-1)/num_tokens_per_seq
      return normalized_entropy