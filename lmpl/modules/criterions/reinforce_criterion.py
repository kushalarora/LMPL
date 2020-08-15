from typing import Dict, List, Iterable, Union
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
          detach_rollin_logits: bool = True
        ):
    super().__init__(
                rollout_cost_function=rollout_cost_function,
                labeling_smooting_ratio=labeling_smooting_ratio,
                rollin_rollout_mixing_coeff=rollin_rollout_mixing_coeff,
                shall_compute_rollin_loss=(rollin_rollout_mixing_coeff > 0),
                shall_compute_rollout_loss=True)
    
    self._temperature = temperature
    self._loss = torch.nn.KLDivLoss(reduction='none')
    self._detach_rollin_logits = detach_rollin_logits

  @overrides
  def _compute_rollout_loss_batch(self,
              rollin_output_dict: Dict[str, torch.Tensor],
              rollout_output_dict_iter: Iterable[Dict[str, Union[torch.Tensor, int]]],
              state: Dict[str, torch.Tensor],
              target_tokens: Dict[str, torch.Tensor] = None) -> torch.FloatTensor:

    rollout_steps = []
    rl_losses = []
    cost_functions = []

    # rollin_logits: (batch_size, num_rollin_steps, num_classes)
    # rollin_predictions: (batch_size, num_rollin_steps)
    rollin_logits: torch.FloatTensor = rollin_output_dict['logits'].squeeze(1)
    batch_size, num_rollin_steps, num_classes = rollin_logits.shape

    if self._detach_rollin_logits:
      rollin_logits = rollin_logits.detach()

    rollin_predictions: torch.LongTensor = rollin_output_dict['predictions'].squeeze(1)

    for rollout_output_dict in rollout_output_dict_iter:
      # For whole batch we rollout only these contexts.  
      # By default this will be for all, but in certain cases of
      # filtering, we might only consider a select few.
      # rollout_contexts: (num_rollout_contexts,)
      
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
      rollin_rollout_logits = torch.cat([rollin_logits_prefix, 
                                         rollout_logits],
                                        dim=1)

      # rollout_reward_batch : (batch_size,)
      rollout_reward_batch = -1 * cost_batch

      # rewards = rollout_reward_batch.detach()
      # rewards = F.softmax(rewards * self._temperature, dim=1) 
      rewards = torch.exp(rollout_reward_batch.detach()) 
      # rewards = (rewards - rewards.min())/(rewards.max() - rewards.min())
  
      # predictions: (batch_size, num_decoding_steps)
      top_predictions = rollout_output_dict["predictions"][:, 0, :]
      
      # rollout_logits: (batch_size, num_decoding_steps - 1, num_classes)
      rollin_rollout_logits = F.log_softmax(rollin_rollout_logits,
                                      dim=-1)

      # rollout_logits: (batch_size, num_decoding_steps - 1)
      log_probs = torch.gather(rollin_rollout_logits, -1,
                                  top_predictions[:, 1:] \
                                    .unsqueeze(2)) \
                        .squeeze(2)

      # Get mask expects first detects </S> and considers all the tokens before this
      # and masks out everything after this.
      log_prob_mask = rollout_output_dict['prediction_masks'][:, 0, 1:]
      log_probs *= log_prob_mask

      # We are trying to maximize the reward, hence minimizing the log prob * reward.
      summed_reward_log_probs = (-1 * log_probs * rewards.unsqueeze(1)).sum(dim=-1)
      num_tokens_per_seq = log_prob_mask.sum(dim=-1)

      rl_loss_batch = (summed_reward_log_probs/num_tokens_per_seq)

      rl_losses.append(rl_loss_batch)
      rollout_steps.append(step)
      cost_functions.append(cost_batch)

    # rollout_steps: (num_rollout_steps,)
    rollout_steps = torch.tensor(rollout_steps)

    # rl_losses: (batch_size, num_rollout_steps)
    rl_losses = torch.stack(rl_losses, dim=1)

    # Similarly, only update logits (or not mask logits) for steps
    # we rollout out for,
    target_masks = util.get_text_field_mask(target_tokens)
    
    # target_masks: (batch_size, num_rollout_steps)
    target_masks = target_masks[:, rollout_steps]

    non_batch_dims = tuple(range(1, len(target_masks.shape)))

    # rl_loss_batch: (batch_size,)
    rl_loss_batch_unnormalized = (rl_losses * target_masks).sum(dim=non_batch_dims)

    # Generate denominator for normalizing loss across batch.
    # Ideally this will be equal to batch_size, but this is a
    # safer way to do this. Here, we ignore sequences with all
    # pad tokens.

    # shape : (batch_size,)
    target_mask_sum = target_masks.sum(dim=non_batch_dims)

    rl_loss_batch = rl_loss_batch_unnormalized/target_mask_sum

    return rl_loss_batch