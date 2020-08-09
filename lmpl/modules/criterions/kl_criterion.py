from typing import Dict, List, Iterable, Union
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
        ):
    super().__init__(
                rollout_cost_function=rollout_cost_function,
                labeling_smooting_ratio=labeling_smooting_ratio,
                rollin_rollout_mixing_coeff=rollin_rollout_mixing_coeff,
                shall_compute_rollin_loss=(rollin_rollout_mixing_coeff > 0),
                shall_compute_rollout_loss=True)
    self._temperature = temperature
    self._loss = torch.nn.KLDivLoss(reduction='none')

  @overrides
  def _compute_rollout_loss_batch(self,
              rollin_output_dict: Dict[str, torch.Tensor],
              rollout_output_dict_iter: Iterable[Dict[str, Union[torch.Tensor, int]]],
              state: Dict[str, torch.Tensor],
              target_tokens: Dict[str, torch.Tensor] = None) -> torch.FloatTensor:

    # rollin_logits: (batch_size, num_rollin_steps, num_classes)
    # rollin_predictions: (batch_size, num_rollin_steps)
    rollin_logits: torch.FloatTensor = rollin_output_dict['logits'].squeeze(1)
    rollin_predictions: torch.LongTensor = rollin_output_dict['predictions'].squeeze(1)

    rollout_steps = []
    kl_losses = []
    kl_cost_function = []
    batch_size, num_rollin_steps, num_classes = rollin_logits.shape

    for rollout_output_dict in rollout_output_dict_iter:
      # For whole batch we rollout only these contexts.  
      # By default this will be for all, but in certain cases of
      # filtering, we might only consider a select few.
      # rollout_contexts: (num_rollout_contexts,)
      
      # cost_batch: (batch_size * num_tokens_to_rollout,)
      cost_batch = self._compute_rollout_cost(
                                        rollout_output_dict=rollout_output_dict, 
                                    )
      
      # rollin_logits = F.log_softmax(rollin_logits, dim=-1)
      step: int = rollout_output_dict['step']
      num_tokens_to_rollout: int = rollout_output_dict['num_tokens_to_rollout']

      # next_tokens: (batch_size, num_tokens_to_rollout)
      next_tokens: torch.LongTensor = rollout_output_dict['next_tokens']
      next_tokens = next_tokens.reshape(batch_size, num_tokens_to_rollout)

      # cost_batch: (batch_size, num_tokens_to_rollout,)
      cost_batch = cost_batch.reshape(batch_size, num_tokens_to_rollout)

      # Only consider those logits which you did rollout for.
      # step_logits: (batch_size, num_classes)

      # We have step - 1 here because logits predict next token given the
      # current context, so they are shifted by 1 i.e., at logit index 0, 
      # we predict token at index 1 given token at 0. So, prediction 
      # corresponding to target step=t, will be at index t-1 in logits.
      step_logits: torch.FloatTensor = rollin_logits[:, step - 1, :]
      step_logits = F.log_softmax(step_logits, dim=-1)

      # scattered_logits: (batch_size,  num_tokens_to_rollout)
      scattered_logits = torch.gather(input=step_logits, 
                                        dim=-1, 
                                        index=next_tokens)

      # x: (batch_size, num_tokens_to_rollout)
      x = scattered_logits # F.log_softmax(scattered_logits, dim=-1)

      # y: (batch_size, num_tokens_to_rollout)
      y = F.softmax(-1 * self._temperature * cost_batch, dim=-1)
  
      # kl_losses: (batch_size,)
      kl_loss = self._loss(x, y).sum(dim=-1)
      
      kl_losses.append(kl_loss)
      rollout_steps.append(step)
      kl_cost_function.append(cost_batch)

    # rollout_steps: (num_rollout_steps,)
    rollout_steps = torch.tensor(rollout_steps)

    # kl_losses: (batch_size, num_rollout_steps)
    kl_losses = torch.stack(kl_losses, dim=1)

    # Similarly, only update logits (or not mask logits) for steps
    # we rollout out for,
    target_masks = util.get_text_field_mask(target_tokens)
    
    # target_masks: (batch_size, num_rollout_steps)
    target_masks = target_masks[:, rollout_steps]

    non_batch_dims = tuple(range(1, len(target_masks.shape)))

    # kl_loss_batch: (batch_size,)
    kl_loss_batch_unnormalized = (kl_losses * target_masks).sum(dim=non_batch_dims)

    # Generate denominator for normalizing loss across batch.
    # Ideally this will be equal to batch_size, but this is a
    # safer way to do this. Here, we ignore sequences with all
    # pad tokens.

    # shape : (batch_size,)
    target_mask_sum = target_masks.sum(dim=non_batch_dims)

    kl_loss_batch = kl_loss_batch_unnormalized/target_mask_sum

    return kl_loss_batch