import torch 

from allennlp.common.testing import AllenNlpTestCase

from lmpl.modules.criterions import KLLossCriterion
from lmpl.modules.cost_functions import BLEUCostFunction, HammingCostFunction
import pytest 
import torch.nn.functional as F
class TestKLLossCriterion(AllenNlpTestCase):
  def test_loss_criterion(self):
    rollin_dict = {}
    rollout_dict = {}
    state = {'epoch': 1, 
             'batch_number': 1,
             'training': False,
            }
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


  def test_kl_loss_is_equiv_to_mle_w_high_temp(self):
    state = {'epoch': 1, 
               'batch_number': 1,
                'training': False,
              }
    targets = torch.LongTensor([[1,2,3,4],
                                [5,6,7,8]])
    target_masks = torch.LongTensor([[1,1,1,1],
                                [1,1,1,1]])
    target_tokens = {'tokens':
          {'tokens': targets}}

    cost_function = HammingCostFunction()
    criterion = KLLossCriterion(
                    rollout_cost_function=cost_function,
                    temperature=1e10,
                  )
    logits = torch.randn((2, 1, 8, 10))
    predictions = torch.randint(0, 10, (2, 1, 4))
    rollin_output_dict = {
      'logits': logits, 
      'predictions': predictions,
    }

    num_tokens_to_rollout: int = 3
    rollout_output_dict_list = [
      {
        'step': 1, 
        'num_tokens_to_rollout': num_tokens_to_rollout, 
        'next_tokens': torch.LongTensor([2, 3, 4, 6, 7, 8]),
        'targets': targets,
        'predictions': targets,
        'target_masks': target_masks,
      },  {
        'step': 2, 
        'num_tokens_to_rollout': num_tokens_to_rollout, 
        'next_tokens': torch.LongTensor([2, 3, 4, 6, 7, 8]),
        'targets': targets,
        'predictions': targets,
        'target_masks': target_masks,

      },  {
        'step': 3, 
        'num_tokens_to_rollout': num_tokens_to_rollout, 
        'next_tokens': torch.LongTensor([2, 3, 4, 6, 7, 8]),
        'targets': targets,
        'predictions': targets,
        'target_masks': target_masks,
      }, ]
    
    def local_compute_rollout_cost(rollout_output_dict):
      step = rollout_output_dict['step']
      target_step = targets[:, step]
      next_tokens = rollout_output_dict['next_tokens']
      return (next_tokens != target_step[0]).float() + (next_tokens != target_step[1]).float()

    criterion._compute_rollout_cost = local_compute_rollout_cost
    
    output_dict = criterion(
                      rollin_output_dict=rollin_output_dict, 
                      rollout_output_dict_iter=rollout_output_dict_list, 
                      state=state, 
                      target_tokens=target_tokens)
    rollout_losses = output_dict['rollout_losses']
    rollin_logits = rollin_output_dict['logits'].squeeze(1)
    rollout_mle_losses = torch.gather(-1 * F.log_softmax(rollin_logits, dim=-1), 
                                      dim=-1, 
                                      index=targets[:, 1:].unsqueeze(2)).squeeze(2)
    assert torch.any(torch.abs(rollout_losses - rollout_mle_losses) < 1e-4)
