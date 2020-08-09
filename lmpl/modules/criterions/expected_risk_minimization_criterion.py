from typing import Dict, List
from overrides import overrides


import torch 

from allennlp.nn import util

from lmpl.modules.criterions.base_loss_criterion import LossCriterion

@LossCriterion.register("risk")
class ExpectedRiskMinimization(LossCriterion):
  
  @overrides
  def _compute_rollout_loss_batch(self,
                    rollout_output_dict: Dict[str, torch.Tensor],
                    state: Dict[str, torch.Tensor],
                    target_tokens: Dict[str, torch.Tensor] = None) -> Dict[str, torch.LongTensor]:
    # TODO #13 (Kushal): Implement empirical risk minimization loss from Edunov 2017 paper.

    raise NotImplementedError()