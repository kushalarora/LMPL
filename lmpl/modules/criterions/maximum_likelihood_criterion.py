from typing import Dict, List
from overrides import overrides


import torch 

from allennlp.nn import util

from lmpl.modules.criterions.base_loss_criterion import LossCriterion

@LossCriterion.register("mle")
class MaximumLikelihoodLossCriterion(LossCriterion):
  
  def __init__(self, labeling_smooting_ratio: float = 0.):
      super().__init__(
          shall_compute_rollin_loss=True, 
          rollin_rollout_mixing_coeff=1.,
          labeling_smooting_ratio=labeling_smooting_ratio)