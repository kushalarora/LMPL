import torch 

from allennlp.common.testing import AllenNlpTestCase

from lmpl.modules.criterions import LossCriterion

from lmpl.modules.cost_functions import HammingCostFunction
from lmpl.modules.cost_functions import BLEUCostFunction

import pytest 

class TestBaseLossCriterion(AllenNlpTestCase):
  def test_loss_criterion(self):
    rollin_dict = {}
    rollout_dict = {}
    state = {}
    target_tokens = {'tokens':
          {'tokens': torch.LongTensor([[1,2,3,4],
                                       [5,6,7,8]])}}

    # Test w/ target_tokens without padding.
    criterion = LossCriterion(
                    shall_compute_rollin_loss=True, 
                    rollin_rollout_mixing_coeff=1.)

    criterion._compute_rollin_loss_batch = \
      lambda rollin_output_dict,state,target_tokens: torch.FloatTensor([1.,3.])
    target_tokens = {'tokens':
      {'tokens': torch.LongTensor([[1,2,3,4],
                                    [5,6,7,8]])}}
    output_dict = criterion(rollin_dict, rollout_dict, 
                                  state, target_tokens)
    assert output_dict['loss'] == 2.

    # Test w/ target_tokens with padding.
    # This assumes 
    criterion = LossCriterion(
                    shall_compute_rollin_loss=True, 
                    rollin_rollout_mixing_coeff=1.)

    criterion._compute_rollin_loss_batch = \
      lambda rollin_output_dict,state,target_tokens: torch.FloatTensor([1.,0.])
    target_tokens = {'tokens':
      {'tokens': torch.LongTensor([[1,2,3,4],
                                    [5,0,0,0]])}}
    output_dict = criterion(rollin_dict, rollout_dict, 
                                  state, target_tokens)
    assert output_dict['loss'] == 1

    # Test w/o target tokens. 
    criterion = LossCriterion(
                    shall_compute_rollin_loss=True, 
                    rollin_rollout_mixing_coeff=1.)

    criterion._compute_rollin_loss_batch = \
       lambda rollin_output_dict,state,target_tokens: \
                                    torch.FloatTensor([1.,3.])
    output_dict = criterion(rollin_dict, rollout_dict, state, 
                              target_tokens=None)
    assert output_dict['loss'] == 2.

    # Test w/ rollout loss.
    criterion = LossCriterion(
                    shall_compute_rollout_loss=True, 
                    rollin_rollout_mixing_coeff=0.)

    criterion._compute_rollout_loss_batch = \
      lambda rollin_output_dict,rollout_output_dict_iter,state,target_tokens: torch.FloatTensor([1.,3.])

    target_tokens = {'tokens':
      {'tokens': torch.LongTensor([[1,2,3,4],
                                    [5,6,7,8]])}}
    output_dict = criterion(rollin_dict, rollout_dict, 
                                  state, target_tokens)
    assert output_dict['loss'] == 2.

  def test_atleast_one_rollin_rollout_should_be_computed(self):
    rollin_dict = {}
    rollout_dict = {}
    state = {}
    target_tokens = {'tokens':
          {'tokens': torch.LongTensor([[1,2,3,4],
                                       [5,6,7,8]])}}

    # Test w/ target_tokens without padding.
    criterion = LossCriterion()
    with pytest.raises(AssertionError):
      output_dict = criterion(rollin_dict, rollout_dict, 
                              state, target_tokens)

  def test_rollin_rollout_mixing_coeff_should_be_bw_0_1(self):
    rollin_dict = {}
    rollout_dict = {}
    state = {}
    target_tokens = {'tokens':
          {'tokens': torch.LongTensor([[1,2,3,4],
                                       [5,6,7,8]])}}
    criterion = LossCriterion(
                    shall_compute_rollin_loss=True, 
                    rollin_rollout_mixing_coeff=1.1)

    criterion._compute_rollin_loss_batch = \
      lambda rollin_output_dict,state,target_tokens: torch.FloatTensor([1.,3.])
    criterion._compute_rollout_loss_batch = \
      lambda rollin_output_dict,rollout_output_dict_iter,state,target_tokens: torch.FloatTensor([1.,3.])

    with pytest.raises(AssertionError):
      output_dict = criterion(rollin_dict, rollout_dict, 
                                    state, target_tokens)

    criterion = LossCriterion(
                    shall_compute_rollin_loss=True, 
                    rollin_rollout_mixing_coeff=-0.1)

    criterion._compute_rollin_loss_batch = \
      lambda rollin_output_dict,state,target_tokens: torch.FloatTensor([1.,3.])
    criterion._compute_rollout_loss_batch = \
      lambda rollin_output_dict,rollout_output_dict_iter,state,target_tokens: torch.FloatTensor([1.,3.])

    with pytest.raises(AssertionError):
      output_dict = criterion(rollin_dict, rollout_dict, 
                                    state, target_tokens)

    criterion = LossCriterion(
                    shall_compute_rollin_loss=True, 
                    shall_compute_rollout_loss=True, 
                    rollin_rollout_mixing_coeff=0.5)

    criterion._compute_rollin_loss_batch = \
      lambda rollin_output_dict,state,target_tokens: torch.FloatTensor([1.,1.])
    criterion._compute_rollout_loss_batch = \
      lambda rollin_output_dict,rollout_output_dict_iter,state,target_tokens: torch.FloatTensor([2.,2.])
      
    output_dict = criterion(rollin_dict, rollout_dict, 
                                    state, target_tokens)
    assert output_dict['loss'] - 1.5  < 1e-10

  def test_compute_rollout_cost_batch(self):    
    # takes_decoded_input=False
    rollout_output_dict = {
      'predictions': torch.LongTensor([[[1, 2, 3, 4, 5]], 
                                       [[6, 7, 8, 9, 10]]]),
      'targets': torch.LongTensor([[1, 2, 3, 0, 0],
                                   [6, 7, 8, 9, 0]]),
      'target_masks': torch.LongTensor([[1, 1, 1, 0, 0], 
                                        [1, 1, 1, 1, 0]])
    }
    criterion = LossCriterion(
                    rollout_cost_function=HammingCostFunction(),
                    shall_compute_rollout_loss=True)
    
    hamming_loss = criterion._compute_rollout_cost(
                                    rollout_output_dict)

    assert torch.all(hamming_loss - torch.tensor([0.0, 0.]) < 1e-10)
    
    rollout_output_dict = {
      'predictions': torch.LongTensor([[[1, 2, 3, 4, 5]], 
                                       [[6, 7, 8, 9, 10]]]),
      'targets': torch.LongTensor([[1, 2, 3, 0, 0],
                                   [10, 9, 8, 7, 0]]),
      'target_masks': torch.LongTensor([[1, 1, 1, 0, 0], 
                                        [1, 1, 1, 1, 0]])
    }
    criterion = LossCriterion(
                    rollout_cost_function=HammingCostFunction(),
                    shall_compute_rollout_loss=True)
    
    hamming_loss = criterion._compute_rollout_cost(
                                    rollout_output_dict)

    # First entry of the batch is same, second entry has all different
    # except for 8 and we will mask last entry, so, error rate is 3/4.
    assert torch.all(hamming_loss - torch.tensor([0.0, 0.75]) < 1e-10)

    # takes_decoded_input=True
    rollout_output_dict = {
      'decoded_predictions': [['1', '2', '3', '4', '5'], 
                             ['6', '7', '8', '9', '10']],
      'decoded_targets': [['1', '2', '3', '4', '5'],
                          ['6', '7', '8', '9', '10']]

    }
    criterion = LossCriterion(
                    rollout_cost_function=BLEUCostFunction(),
                    shall_compute_rollout_loss=True)

    bleu_loss = criterion._compute_rollout_cost(
                                    rollout_output_dict)

    assert torch.all(bleu_loss - torch.tensor([0.0, 0.]) < 1e-10)



  @pytest.mark.skip(reason="not implemented")
  def test_get_cross_entropy_loss(self):
    assert False

  @pytest.mark.skip(reason="not implemented")
  def test_compute_rollin_loss_batch(self):
    assert False
