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
          rollin_rollout_mixing_coeff:float = 0.,
          labeling_smooting_ratio: float = None,
          temperature: float = 1, 
          detach_rollin_logits: bool = False,
          warm_start_for_epochs: int = -1, 
          warm_start_for_batch_numbers: int = -1,
          normalize_costs: bool = False,
          entropy_regularization_coeff: bool = 0,
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

  @overrides
  def _compute_rollout_loss_single_iter(self,
                  rollin_output_dict: Dict[str, torch.Tensor],
                  rollout_output_dict: Dict[str, Union[torch.Tensor, int]],
                  state: Dict[str, torch.Tensor],
                  target_tokens: Dict[str, torch.Tensor] = None,
              ) -> Tuple[torch.FloatTensor, torch.FloatTensor, int]:

      # rollin_logits: (batch_size, num_rollin_steps, num_classes)
      # rollin_predictions: (batch_size, num_rollin_steps)
      rollin_logits: torch.FloatTensor = rollin_output_dict['logits'].squeeze(1)
      rollin_predictions: torch.LongTensor = rollin_output_dict['predictions'].squeeze(1)

      if self._detach_rollin_logits:
          rollin_output_dict['logits'].detach_()

      step: int = rollout_output_dict['step']

      # cost_batch: (batch_size )
      cost_batch = self._compute_rollout_cost(
                            rollout_output_dict=rollout_output_dict)

      # Only consider those logits which you did rollout for.
      # rollin_logits_prefix: (batch_size, step - 1, num_classes)

      # We have step - 1 here because logits predict next token given the
      # current context, so they are shifted by 1 i.e., at logit index 0, 
      # we predict token at index 1 given token at 0. So, prediction 
      # corresponding to target step=t, will be at index t-1 in logits.
      rollin_logits_prefix: torch.FloatTensor = rollin_logits[:, :step, :]

      # rollout_logits: (batch_size, num_decoding_steps - step - 1, num_classes)
      rollout_logits: torch.FloatTensor = rollout_output_dict['logits'].squeeze(1)

      # rollout_logits: (batch_size, num_decoding_steps - 1, num_classes)
      rollin_rollout_logits: torch.FloatTensor = torch.cat([rollin_logits_prefix, 
                                                            rollout_logits],
                                                          dim=1)

      
      if self._normalize_costs:
        cost_batch = (cost_batch - cost_batch.min())/(cost_batch.max() - cost_batch.min())
  
      # predictions: (batch_size, num_decoding_steps)
      top_predictions = rollout_output_dict["predictions"][:, 0, :]
      
      # rollout_logits: (batch_size, num_decoding_steps - 1, num_classes)
      rollin_rollout_logits = F.log_softmax(rollin_rollout_logits, dim=-1)

      # rollout_logits: (batch_size, num_decoding_steps - 1)
      log_probs = torch.gather(rollin_rollout_logits, -1,
                                  top_predictions[:, 1:] \
                                    .unsqueeze(2)) \
                        .squeeze(2)

      # Get mask expects first detects </S> and considers all the tokens before this
      # and masks out everything after this.
      log_prob_mask = rollout_output_dict['prediction_masks'][:, 0, 1:]
      log_probs *= log_prob_mask
      num_tokens_per_seq = log_prob_mask.sum(dim=-1)
      normalize_log_probs = log_probs.sum(dim=-1)/num_tokens_per_seq
      # We are trying to maximize the reward, hence minimizing the log prob * reward.
      rl_loss_batch =  -1 * normalize_log_probs * (1 - cost_batch)

      # Add entropy regularization coefficient
      if self._entropy_regularization_coeff > 0:
        rl_loss_batch += -1 * self._entropy_regularization_coeff * \
                              self._compute_entropy_regularization_term(
                                                        logits=rollin_rollout_logits, 
                                                        mask=log_prob_mask,
                                                      )
      return rl_loss_batch, cost_batch, step



  def _compute_entropy_regularization_term(self,
                                          logits: torch.FloatTensor, 
                                          mask: torch.LongTensor):
      # entropy_seq: (batch_size, num_decoding_steps - 1)
      entropy_seq =  -1 * (F.softmax(logits, dim=-1) * logits).sum(dim=-1)
      entropy_seq *= mask
      num_tokens_per_seq = mask.sum(dim=-1)
      # normalize_entropy: (batch_size, )
      normalized_entropy = entropy_seq.sum(dim=-1)/num_tokens_per_seq
      return normalized_entropy