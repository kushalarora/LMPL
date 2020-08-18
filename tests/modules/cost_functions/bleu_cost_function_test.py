import torch 

from allennlp.common.testing import AllenNlpTestCase

from lmpl.modules.criterions import LossCriterion

from lmpl.modules.cost_functions.bleu_cost_function import BLEUCostFunction, \
                                                            compute_bleu_score, \
                                                            compute_bleu_score_decoded, \
                                                            decoded_init_pool, \
                                                            init_pool

import pytest 

class TestBaseLossCriterion(AllenNlpTestCase):
  def test_computed_bleu_score_decoded(self):
    gold_label = ['a', 'b', 'c', 'd']
    prediction = ['a', 'b', 'c', 'd']

    decoded_init_pool()
    score = compute_bleu_score_decoded(
                              gold_label=gold_label,
                              prediction=prediction,
                           )
    assert score - 0.0 < 1e-10

    prediction = ['e', 'f', 'g']
    score = compute_bleu_score_decoded(
                              gold_label=gold_label,
                              prediction=prediction,
                           )
    assert score - 1.0 < 1e-10

  def test_compute_bleu_score(self):
    gold_label = torch.tensor([10, 2, 3, 4])
    prediction = torch.tensor([10, 2, 3, 4])

    init_pool(
        pad_token = 30,
        eos_token = 40,
        unk_token = 50,
            )

    score = compute_bleu_score(
                      gold_label=gold_label,
                      prediction=prediction,
                    )
    assert score - 0.0 < 1e-10

    prediction = torch.tensor([5, 6, 7, 9, 11])
    score = compute_bleu_score(
                      gold_label=gold_label,
                      prediction=prediction,
                    )
    assert score - 1.0 < 1e-10