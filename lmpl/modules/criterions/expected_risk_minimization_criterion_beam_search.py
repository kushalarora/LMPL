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

@LossCriterion.register("risk_beam")
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
  def _compute_rollout_loss_single_iter(self,
                  rollin_output_dict: Dict[str, torch.Tensor],
                  rollout_output_dict: Dict[str, Union[torch.Tensor, int]],
                  state: Dict[str, torch.Tensor],
                  target_tokens: Dict[str, torch.Tensor] = None,
              ) -> Tuple[torch.FloatTensor, torch.FloatTensor, int]:
        rollout_loss_single_iter_output = super()._compute_rollout_loss_single_iter(
                                                      rollin_output_dict=rollin_output_dict,
                                                      rollout_output_dict=rollout_output_dict,
                                                      state=state,
                                                      target_tokens=target_tokens,
                                                    )
        cost_batches = rollout_loss_single_iter_output['cost_batch']
        step = rollout_loss_single_iter_output['step']
        normalized_log_probs = rollout_loss_single_iter_output['normalized_log_probs']
        entropy_regularization_terms = rollout_loss_single_iter_output['entropy_regularization_term']

        normalized_probs = F.softmax(normalized_log_probs, dim=-1)

        if self.i % 100 == 0:
          logging.info(f"cost: {cost_batches.mean()}")
          logging.info(f"entropy term: {entropy_regularization_terms.detach().mean()}")
          logging.info(f"normalized_log_prob: {-1 * normalized_log_probs.detach().mean()}")
        self.i += 1
        
        if self._entropy_augumented_reward:
            cost_batches -= entropy_regularization_terms.detach()
            
        if self._normalize_to_0_1:
          normalized_cost_batches = (cost_batches - cost_batches.min(dim=-1, keepdim=True)[0])/(cost_batches.max(dim=-1, keepdim=True)[0] - cost_batches.min(dim=-1, keepdim=True)[0])
        elif self._normalize_by_mean_std:
          normalized_cost_batches = (cost_batches - cost_batches.mean(dim=-1, keepdim=True)[0])/cost_batches.std(dim=-1, keepdim=True)
        elif self._normalize_by_mean:
          normalized_cost_batches = (cost_batches - cost_batches.mean(dim=-1, keepdim=True)[0])
        else:
          normalized_cost_batches = cost_batches

        mrt_loss = normalized_probs * normalized_cost_batches - self._entropy_regularization_coeff * (-1 * normalized_log_probs)

        return {'loss_batch': mrt_loss.mean(dim=-1), 
                'cost_batch': cost_batches.mean(dim=-1),
                'step': step,
                'normalized_log_probs': normalized_log_probs.mean(dim=-1),
                'entropy_regularization_term': entropy_regularization_terms.mean(dim=-1),
              }
