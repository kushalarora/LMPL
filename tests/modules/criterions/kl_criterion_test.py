import torch 

from allennlp.common.testing import AllenNlpTestCase

from lmpl.modules.criterions import KLLossCriterion
from lmpl.modules.cost_functions import HammingCostFunction
import pytest 

class TestKLLossCriterion(AllenNlpTestCase):
  def test_loss_criterion(self):
    rollin_dict = {}
    rollout_dict = {}
    state = {'epoch': 1, 
             'batch_number': 1,}
    target_tokens = {'tokens':
          {'tokens': torch.LongTensor([[1,2,3,4,9,10,11,12],
                                       [5,6,7,8,13,14,15,16]])}}

    hamming_cost_functions = HammingCostFunction()
    # Test w/ target_tokens without padding.
    criterion = KLLossCriterion(
                    rollout_cost_function=hamming_cost_functions,
                  )

    criterion._compute_rollout_cost = \
      lambda rollout_output_dict: torch.rand((6,))
    

    rollin_output_dict = {
      'logits': torch.randn((2, 8, 10)),
      'predictions': torch.randint(0, 10, (2, 1, 4)),
    }

    num_tokens_to_rollout: int = 3
    rollout_output_dict_list = [
      {
        'step': 3, 
        'num_tokens_to_rollout': num_tokens_to_rollout, 
        'next_tokens': torch.LongTensor([1, 3, 4, 5, 6, 7])
      }, {
        'step': 4, 
        'num_tokens_to_rollout': num_tokens_to_rollout, 
        'next_tokens': torch.LongTensor([2, 5, 6, 8, 1, 3])
      }]

    output_dict = criterion(
                        rollin_output_dict=rollin_output_dict, 
                        rollout_output_dict_iter=rollout_output_dict_list, 
                        state=state, 
                        target_tokens=target_tokens)

    assert "loss" in output_dict
    assert output_dict.get("loss") is not None