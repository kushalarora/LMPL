from typing import Dict, List, Union, Iterable

import torch 

from allennlp.common.registrable import Registrable
from allennlp.nn import util

from lmpl.modules.cost_functions.cost_function import CostFunction


class LossCriterion(Registrable):
  def __init__(self, 
          rollout_cost_function:CostFunction = None,
          labeling_smooting_ratio: float = 0.,
          shall_compute_rollin_loss: bool = False, 
          shall_compute_rollout_loss: bool = False, 
          rollin_rollout_mixing_coeff: float = 0., 
        ):
    self._rollout_cost_function = rollout_cost_function
    self._shall_compute_rollin_loss = shall_compute_rollin_loss
    self._shall_compute_rollout_loss = shall_compute_rollout_loss
    self._rollin_rollout_mixing_coeff = rollin_rollout_mixing_coeff
    self._labeling_smooting_ratio = labeling_smooting_ratio

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

    assert self._shall_compute_rollin_loss or \
            self._shall_compute_rollout_loss, \
            "We need to either compute rollin or rollout losses. Both" + \
              "shall_compute_rollin_loss and shall_compute_rollout_loss cannot be false."

    output_dict = {}
    rollin_loss_batch = 0
    if self._shall_compute_rollin_loss:
      rollin_loss_batch = self._compute_rollin_loss_batch(
                                    rollin_output_dict=rollin_output_dict, 
                                    state=state, 
                                    target_tokens=target_tokens)

    rollout_loss_batch = 0
    if self._shall_compute_rollout_loss:
        rollout_loss_batch = self._compute_rollout_loss_batch(
                                      rollin_output_dict=rollin_output_dict, 
                                      rollout_output_dict_iter=rollout_output_dict_iter,
                                      state=state,
                                      target_tokens=target_tokens)

    assert self._rollin_rollout_mixing_coeff >= 0. and \
            self._rollin_rollout_mixing_coeff <= 1., \
              "rollin_rollout_mixing_coeff must be between [0, 1]." + \
                f"Current value: {self._rollin_rollout_mixing_coeff}"

    loss_batch = self._rollin_rollout_mixing_coeff * rollin_loss_batch + \
                  (1 - self._rollin_rollout_mixing_coeff) * rollout_loss_batch
    
    assert loss_batch is not None, \
      f"rollin_loss_batch: {rollin_loss_batch}," + \
        f"rollout_loss_batch: {rollout_loss_batch}"

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

  def _compute_rollout_loss_batch(self,
              rollin_output_dict: Dict[str, torch.Tensor],
              rollout_output_dict_iter: Iterable[Dict[str, Union[torch.Tensor, int]]],
              state: Dict[str, torch.Tensor],
              target_tokens: Dict[str, torch.Tensor] = None) -> torch.FloatTensor:
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
    if self._rollout_cost_function.takes_decoded_input():
        # This is for rollout cost function like BLEU or Noisy Oracle for OCR.
        decoded_predictions = rollout_output_dict["decoded_predictions"]
        decoded_targets = rollout_output_dict.get("decoded_targets")
        cost_batch = self._rollout_cost_function(
                                      predictions=decoded_predictions,
                                      gold_labels=decoded_targets)
    else:
        targets = rollout_output_dict['targets']
        target_masks = rollout_output_dict['target_masks']
        predicted_tokens = rollout_output_dict['predictions']
        top_predicted_tokens = predicted_tokens[:, 0, :]
        cost_batch = self._rollout_cost_function(
                                predictions=top_predicted_tokens,
                                gold_labels=targets,
                                mask=target_masks)
    
    if 'logits' in rollout_output_dict:
      cost_batch = cost_batch.to(rollout_output_dict['logits'].dtype)

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
