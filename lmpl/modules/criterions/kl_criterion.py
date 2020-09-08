from typing import Dict, List, Iterable, Union, Tuple
from overrides import overrides


import torch 
import torch.nn.functional as F

from allennlp.nn import util

from lmpl.modules.criterions.base_loss_criterion import LossCriterion
from lmpl.modules.cost_functions import CostFunction
@LossCriterion.register("searnn-kl")
class KLLossCriterion(LossCriterion):
  
  def __init__(self, 
          rollout_cost_function:CostFunction,
          rollin_rollout_mixing_coeff:float = 0.,
          labeling_smooting_ratio: float = None,
          temperature: float = 1, 
          warm_start_for_epochs: int = -1, 
          warm_start_for_batch_numbers: int = -1,
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
    self._loss = torch.nn.KLDivLoss(reduction='none')

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

      batch_size, num_rollin_steps, num_classes = rollin_logits.shape

      # cost_batch_flattened: (batch_size * num_tokens_to_rollout,)
      cost_batch_flattened: torch.FloatTensor = self._compute_rollout_cost(
                                        rollout_output_dict=rollout_output_dict, 
                                    )
      
      # rollin_logits = F.log_softmax(rollin_logits, dim=-1)
      step: int = rollout_output_dict['step']
      num_tokens_to_rollout: int = rollout_output_dict['num_tokens_to_rollout']

      # next_tokens_flattened: (batch_size * num_tokens_to_rollout,)
      next_tokens_flattened: torch.LongTensor = rollout_output_dict['next_tokens']

      # next_tokens: (batch_size, num_tokens_to_rollout)
      next_tokens: torch.LongTensor = next_tokens.reshape(batch_size, num_tokens_to_rollout)

      # cost_batch: (batch_size, num_tokens_to_rollout,)
      cost_batch: torch.FloatTensor = cost_batch_flattened \
                                          .reshape(batch_size, num_tokens_to_rollout)

      # Only consider those logits which you did rollout for.
      # step_logits: (batch_size, num_classes)

      # We have step - 1 here because logits predict next token given the
      # current context, so they are shifted by 1 i.e., at logit index 0, 
      # we predict token at index 1 given token at 0. So, prediction 
      # corresponding to target step=t, will be at index t-1 in logits.
      step_logits: torch.FloatTensor = rollin_logits[:, step - 1, :]
      step_logits = F.log_softmax(step_logits, dim=-1)

      # scattered_logits: (batch_size,  num_tokens_to_rollout)
      scattered_logits: torch.FloatTensor = torch.gather(
                                                  input=step_logits, 
                                                  dim=-1, 
                                                  index=next_tokens)

      # x: (batch_size, num_tokens_to_rollout)
      x = scattered_logits # F.log_softmax(scattered_logits, dim=-1)

      # y: (batch_size, num_tokens_to_rollout)
      y = F.softmax(-1 * self._temperature * cost_batch, dim=-1)
  
      # kl_losses: (batch_size,)
      kl_loss = self._loss(x, y).sum(dim=-1)
      
      return kl_loss, cost_batch, step