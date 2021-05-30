from typing import Dict, List, Union, Iterable, Tuple

import torch 

from allennlp.common.registrable import Registrable
from allennlp.nn import util

from lmpl.modules.cost_functions.cost_function import CostFunction
from allennlp.training.metrics import Metric, Average


class LossCriterion(Registrable):
  def __init__(self, 
          rollout_cost_function:CostFunction = None,
          labeling_smooting_ratio: float = 0.,
          shall_compute_rollin_loss: bool = False, 
          shall_compute_rollout_loss: bool = False, 
          rollin_rollout_mixing_coeff: float = 0., 
          warm_start_for_epochs: int = -1, 
          warm_start_for_batch_numbers: int = -1,
        ):
    self._rollout_cost_function = rollout_cost_function
    self._shall_compute_rollin_loss = shall_compute_rollin_loss
    self._shall_compute_rollout_loss = shall_compute_rollout_loss
    self._rollin_rollout_mixing_coeff = rollin_rollout_mixing_coeff
    self._labeling_smooting_ratio = labeling_smooting_ratio
    self._warm_start_for_epochs = warm_start_for_epochs
    self._warm_start_for_batch_numbers = warm_start_for_batch_numbers
    self._average_cost = Average()

  def __call__(self,
              rollin_output_dict: Dict[str, torch.Tensor],
              rollout_output_dict_iter: Iterable[Dict[str, torch.Tensor]],
              state: Dict[str, torch.Tensor],
              target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.LongTensor]:
    """ Given rollin and rollout, how to combine loss from rollin and
        rollout to compute final loss. This will be used to learning local
        loss such that it reflects the global loss as well.

    Arguments:
        rollin_output_dict {Dict[str, torch.Tensor]} -- Dictionary with rollin computations.
        rollout_output_dict {Dict[str, torch.Tensor]} -- Dictionary with rollout computations.
        state {Dict[str, torch.Tensor]} -- State dictionary.
        target_tokens {Dict[str, torch.Tensor]} (Optional) -- Target tokens dict. This is optional
        as there might be cases of unsupervised training where target sequence is not available.
    Returns:
          output_dict {Dict[str, torch.LongTensor]} -- Updated outptut dict with global and local
                                                      loss combined.
    """
    output_dict = {}
    loss_output_dict = self._compute_loss_batch(
                                  rollin_output_dict=rollin_output_dict,
                                  rollout_output_dict_iter=rollout_output_dict_iter,
                                  state=state,
                                  target_tokens=target_tokens,
                                )
    output_dict.update(loss_output_dict)
    loss_batch = loss_output_dict['loss_batch']
    
    # This assumes target_mask was applied to loss
    # before loss_batch computation.
    if target_tokens:
      target_mask = util.get_text_field_mask(target_tokens)
      # shape : (batch_size,)
      target_mask = target_mask[:, 1:]
      non_batch_dims = tuple(range(1, len(target_mask.shape)))

      target_mask_sum = target_mask.sum(dim=non_batch_dims)
      num_non_empty_sequences = ((target_mask_sum > 0).sum() + 1e-13)
      loss = loss_batch.sum()/num_non_empty_sequences
      output_dict['loss'] = loss
    else:
      output_dict['loss'] = loss_batch.mean()
    return output_dict

  def _compute_loss_batch(self,
              rollin_output_dict: Dict[str, torch.Tensor],
              rollout_output_dict_iter: Iterable[Dict[str, Union[torch.Tensor, int]]],
              state: Dict[str, torch.Tensor],
              target_tokens: Dict[str, torch.Tensor] = None) -> torch.FloatTensor:

    epoch: int = state['epoch']
    batch_number: int = state['batch_number']

    output_dict = {}

    assert self._shall_compute_rollin_loss or \
            self._shall_compute_rollout_loss, \
            "We need to either compute rollin or rollout losses. Both" + \
              "shall_compute_rollin_loss and shall_compute_rollout_loss cannot be false."

    rollin_loss_batch = 0
    if epoch < self._warm_start_for_epochs or \
        batch_number < self._warm_start_for_batch_numbers or \
          self._shall_compute_rollin_loss:
        rollin_loss_batch = self._compute_rollin_loss_batch(
                                      rollin_output_dict=rollin_output_dict, 
                                      state=state, 
                                      target_tokens=target_tokens)

        output_dict['rollin_loss_batch'] = rollin_loss_batch

    if epoch < self._warm_start_for_epochs or \
        batch_number < self._warm_start_for_batch_numbers:
        output_dict['loss_batch'] = rollin_loss_batch
        return output_dict

    rollout_loss_batch = 0
    if self._shall_compute_rollout_loss:
        rollout_loss_batch,  rollout_costs, rollout_steps, rollout_losses =  \
                  self._compute_rollout_loss_batch(
                              rollin_output_dict=rollin_output_dict, 
                              rollout_output_dict_iter=rollout_output_dict_iter,
                              state=state,
                              target_tokens=target_tokens)

        output_dict['rollout_loss_batch'] = rollout_loss_batch
        output_dict['rollout_costs'] = rollout_costs
        output_dict['rollout_steps'] = rollout_steps
        output_dict['rollout_losses'] = rollout_losses

    assert self._rollin_rollout_mixing_coeff >= 0. and \
            self._rollin_rollout_mixing_coeff <= 1., \
              "rollin_rollout_mixing_coeff must be between [0, 1]." + \
                f"Current value: {self._rollin_rollout_mixing_coeff}"

    loss_batch = self._rollin_rollout_mixing_coeff * rollin_loss_batch + \
                  (1 - self._rollin_rollout_mixing_coeff) * rollout_loss_batch
    
    assert loss_batch is not None, \
      f"rollin_loss_batch: {rollin_loss_batch}," + \
        f"rollout_loss_batch: {rollout_loss_batch}"

    output_dict['loss_batch'] = loss_batch
    return output_dict

  def _compute_rollout_loss_batch(self,
              rollin_output_dict: Dict[str, torch.Tensor],
              rollout_output_dict_iter: Iterable[Dict[str, Union[torch.Tensor, int]]],
              state: Dict[str, torch.Tensor],
              target_tokens: Dict[str, torch.Tensor] = None) -> torch.FloatTensor:
    rollout_steps = []
    losses = []
    cost_batches = []
    
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
        
        losses.append(loss_batch)
        cost_batches.append(cost_batch)
        rollout_steps.append(step)

    # rollout_steps: (num_rollout_steps,)
    rollout_steps = torch.tensor(rollout_steps)

    # cost_functions: (batch_size, num_rollout_steps)
    cost_batches = torch.stack(cost_batches, dim=1)

    # rl_losses: (batch_size, num_rollout_steps)
    losses = torch.stack(losses, dim=1)

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

    cost_batch_unnormalized = (cost_batches.squeeze(dim=-1) * target_masks)\
                                      .sum(dim=non_batch_dims)
    average_cost =  (cost_batch_unnormalized/target_mask_sum).mean()
    self._average_cost(average_cost.cpu().item())

    return loss_batch, cost_batches, rollout_steps, losses

  def _compute_rollout_loss_single_iter(self, 
                        rollin_output_dict: Dict[str, torch.Tensor],
                        rollout_output_dict: Dict[str, Union[torch.Tensor, int]], 
                        state: Dict[str, torch.Tensor],
                        target_tokens: Dict[str, torch.Tensor] = None,
                ) -> Tuple[torch.FloatTensor, torch.FloatTensor, int]:
    raise NotImplementedError()

  def _compute_rollin_loss_batch(self, 
          rollin_output_dict: Dict[str, torch.Tensor],
          state: Dict[str, torch.Tensor],
          target_tokens: Dict[str, torch.Tensor]) -> torch.FloatTensor:
    
    logits = rollin_output_dict['logits']
    targets = util.get_token_ids_from_text_field_tensors(target_tokens)
    # shape: (batch_size, num_decoding_steps)
    best_logits = logits[:, 0, :, :].squeeze(1)
    target_masks = util.get_text_field_mask(target_tokens)

    # Compute loss.
    loss_batch = self._get_cross_entropy_loss(best_logits, targets, target_masks)
    return loss_batch

  def _compute_rollout_cost(self, 
              rollout_output_dict: Dict[str, Union[torch.Tensor, int]], 
            ) -> torch.FloatTensor:
    """ Compute the roll out cost for rolled out predictions.
    """
    batch_size, beam_size, _, _ = rollout_output_dict['logits'].shape
    
    if self._rollout_cost_function.takes_decoded_input():
        # This is for rollout cost function like BLEU or Noisy Oracle for OCR.
        decoded_predictions = rollout_output_dict["decoded_predictions"]
        decoded_targets = rollout_output_dict.get("decoded_targets")
        def flatten(beam_size, predicted_tokens, targets=None):
          """ Flatten predictions and targets on the beam_size dimension.
              Input dim: (batch_size, beam_size, seq_lens)
              Output dim: (batch_size * beam_size, seq_lens)
          """
          for beams in predicted_tokens:
            assert beam_size == len(beams)

          flattened_predictions =  [tokens for beams in predicted_tokens for tokens in beams]
          flattened_targets = None
          if targets:
            flattened_targets = [target[0] for target in targets for _ in range(beam_size) ]
          return flattened_predictions, flattened_targets

        def unflatten(cost_batch, beam_size):
          """ Return we get is of shape (batch_size * beam). 
              We reshape the cost_batch tensor to shape (batch_size, beam_size)
          """
          return cost_batch.unsqueeze(1).reshape(batch_size, beam_size)

        flattened_predictions, flattened_targets = flatten(beam_size,
                                                            decoded_predictions, 
                                                            decoded_targets,
                                                          )
        flattened_cost_batch = self._rollout_cost_function(
                                      predictions=flattened_predictions,
                                      gold_labels=flattened_targets)
        cost_batch = unflatten(flattened_cost_batch, beam_size)
    else:
        def expand(tensor, beam_size):
            return tensor.unsqueeze(1) \
                          .expand(batch_size, beam_size, -1)

        predicted_tokens = rollout_output_dict['predictions']
        targets = rollout_output_dict['targets']
        target_masks = rollout_output_dict['target_masks']

        cost_batch = self._rollout_cost_function(
                                          predictions=predicted_tokens,
                                          gold_labels=expand(targets, beam_size),
                                          mask=expand(target_masks, beam_size))
    if 'logits' in rollout_output_dict:
      cost_batch = cost_batch.to(rollout_output_dict['logits'].dtype) \
                              .to(rollout_output_dict['logits'].device)

    return cost_batch

  def takes_decoded_input(self) -> bool:
    return self._rollout_cost_function.takes_decoded_input()

  def _get_cross_entropy_loss(self,
                logits: torch.LongTensor,
                targets: torch.LongTensor,
                target_mask: torch.LongTensor) -> torch.FloatTensor:
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

      return util.sequence_cross_entropy_with_logits(logits, 
                        relevant_targets, relevant_mask, 
                        label_smoothing=self._labeling_smooting_ratio, 
                        average=None)

  def get_metric(self, reset: bool = False) -> Dict[str, float]:
    return {'cost': self._average_cost.get_metric(reset=reset)}