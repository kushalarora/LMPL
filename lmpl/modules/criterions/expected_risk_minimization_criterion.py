from typing import Dict, List
from overrides import overrides
from typing import Dict, List, Iterable, Union, Tuple
from overrides import overrides

import logging

import torch 
import torch.nn.functional as F

from allennlp.nn import util

from lmpl.modules.criterions.reinforce_criterion import ReinforceCriterion

from lmpl.modules.criterions.base_loss_criterion import LossCriterion
from lmpl.modules.cost_functions import CostFunction

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

i = 0

@LossCriterion.register("risk")
class ExpectedRiskMinimization(ReinforceCriterion):
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
          normalize_to_0_1: bool = False,
          normalize_by_mean_std: bool = False,
          normalize_by_mean: bool = False,
          entropy_augumented_reward: bool = True,
        ):
    super().__init__(
                rollout_cost_function=rollout_cost_function,
                rollin_rollout_mixing_coeff=rollin_rollout_mixing_coeff,
                labeling_smooting_ratio=labeling_smooting_ratio,
                temperature=temperature,
                detach_rollin_logits=detach_rollin_logits,
                warm_start_for_epochs=warm_start_for_epochs,
                warm_start_for_batch_numbers=warm_start_for_batch_numbers,
                normalize_costs=normalize_costs,
                entropy_regularization_coeff=entropy_regularization_coeff,
                alpha=alpha,
              )
    self.i = 0
    self._normalize_to_0_1 = normalize_to_0_1
    self._normalize_by_mean_std = normalize_by_mean_std
    self._normalize_by_mean = normalize_by_mean
    self._entropy_augumented_reward = entropy_augumented_reward

  @overrides
  def _compute_rollout_loss_batch(self,
              rollin_output_dict: Dict[str, torch.Tensor],
              rollout_output_dict_iter: Iterable[Dict[str, Union[torch.Tensor, int]]],
              state: Dict[str, torch.Tensor],
              target_tokens: Dict[str, torch.Tensor] = None) -> torch.FloatTensor:
    rollout_steps = []
    losses = []
    cost_batches = []
    normalized_log_probs = []
    entropy_regularization_terms = []
    epoch: int = state['epoch']
    batch_number: int = state['batch_number']

    for rollout_output_dict in rollout_output_dict_iter:
        rollout_loss_single_iter_output = self._compute_rollout_loss_single_iter(
                                                      rollin_output_dict=rollin_output_dict,
                                                      rollout_output_dict=rollout_output_dict,
                                                      state=state,
                                                      target_tokens=target_tokens,
                                                    )

        loss_batch = rollout_loss_single_iter_output['loss_batch']
        cost_batch = rollout_loss_single_iter_output['cost_batch']
        step = rollout_loss_single_iter_output['step']
        normalized_log_prob = rollout_loss_single_iter_output['normalized_log_probs']
        entropy_regularization_term = rollout_loss_single_iter_output['entropy_regularization_term']

        losses.append(loss_batch.squeeze(-1))
        cost_batches.append(cost_batch.squeeze(-1))
        rollout_steps.append(step)
        normalized_log_probs.append(normalized_log_prob.squeeze(-1))
        entropy_regularization_terms.append(entropy_regularization_term.squeeze(-1))

    # rollout_steps: (num_rollout_steps,)
    rollout_steps = torch.tensor(rollout_steps)

    # cost_functions: (batch_size, num_rollout_steps)
    cost_batches = torch.stack(cost_batches, dim=1)

    # rl_losses: (batch_size, num_rollout_steps)
    losses = torch.stack(losses, dim=1)

    # normalized_log_probs: (batch_size, num_rollout_steps)
    normalized_log_probs = torch.stack(normalized_log_probs, dim=1)

    entropy_regularization_terms = torch.stack(entropy_regularization_terms, dim=1)

    normalized_probs = F.softmax(normalized_log_probs, dim=-1)
    # cost_batches = F.softmax(cost_batches, dim=-1)
    if self.i % 100 == 0:
      logging.info(f"cost: {cost_batches.mean()}")
      logging.info(f"entropy term: {entropy_regularization_terms.detach().mean()}")
      logging.info(f"normalized_log_prob: {-1 * normalized_log_probs.detach().mean()}")
    self.i += 1
    
    if self._entropy_augumented_reward:
      cost_batches -= self._entropy_regularization_coeff * (-1 * normalized_log_probs.detach())
    # import pdb; pdb.set_trace()
    if self._normalize_to_0_1:
      normalized_cost_batches = (cost_batches - cost_batches.min(dim=-1, keepdim=True)[0])/(cost_batches.max(dim=-1, keepdim=True)[0] - cost_batches.min(dim=-1, keepdim=True)[0])
    elif self._normalize_by_mean_std:
      normalized_cost_batches = (cost_batches - cost_batches.mean(dim=-1, keepdim=True)[0])/cost_batches.std(dim=-1, keepdim=True)
    elif self._normalize_by_mean:
      normalized_cost_batches = (cost_batches - cost_batches.mean(dim=-1, keepdim=True)[0])
    else:
      normalized_cost_batches = cost_batches

    losses = normalized_probs * normalized_cost_batches
    # losses = -1 * entropy_regularization_terms

    # TODO: #54 Figure out if we need this target_mask based normalization? 
    # # Similarly, only update logits (or not mask logits) for steps
    # # we rollout out for,
    target_masks = util.get_text_field_mask(target_tokens)
    
    # target_masks: (batch_size, num_rollout_steps)
    target_masks = target_masks[:, rollout_steps]

    non_batch_dims = tuple(range(1, len(target_masks.shape)))
    # loss_batch: (batch_size,)
    loss_batch_unnormalized = (losses * target_masks).sum(dim=non_batch_dims)

    # Generate denominator for normalizing loss across batch.
    # Ideally this will be equal to batch_size, but this is a
    # safer way to do this. Here, we ignore sequences with all
    # pad tokens.

    # shape : (batch_size,)
    target_mask_sum = target_masks.sum(dim=non_batch_dims) + 1e-45

    loss_batch = loss_batch_unnormalized/target_mask_sum
    # loss_batch = losses.mean(dim=-1)

    cost_batch_unnormalized = (cost_batches * target_masks).sum(dim=non_batch_dims)
    average_cost =  (cost_batch_unnormalized/target_mask_sum).mean()
    self._average_cost(average_cost.cpu().item())

    return loss_batch, cost_batches, rollout_steps, losses