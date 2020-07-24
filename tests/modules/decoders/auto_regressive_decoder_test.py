from typing import Tuple
from lmpl.modules.decoders import LMPLAutoRegressiveSeqDecoder
from lmpl.modules.decoders.auto_regressive_decoder import top_k_top_p_filtering

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase

import torch
def setup_logits(batch_size: int, num_vocab: int, 
                  indices_to_set: Tuple[Tuple[int]],
                  fixed_value=1e1):
  logits = torch.rand((batch_size, num_vocab))
  for i, row in enumerate(indices_to_set):
    logits[i, row] = torch.FloatTensor(row)
  return logits


class TestAutoRegressiveSeqDecoder(AllenNlpTestCase):
  def test_top_k_filtering(self):
    logits = setup_logits(batch_size=3, 
                          num_vocab=10, 
                          indices_to_set=[[1, 2, 3,],
                                          [1,2],
                                          [1,2,3,4]])

    logits=top_k_top_p_filtering(logits, top_k=3, filter_value=0)

    # Correct top-k should be kept
    # Only value at indices 1, 2, 3 should be kept.
    assert logits[0].sum() == 6

    # One additional value in addition to 1 and 2 should be kept.
    assert logits[1].sum() > 3 and logits[1].sum() < 4

    # Only top most values should be kept.
    # i.e. indices 2, 3, 4 should be kept.
    assert logits[2].sum() == 9

  def test_top_p_filetering(self):
    logits = setup_logits(batch_size=3, 
                          num_vocab=5, 
                          indices_to_set=[[0,1,2,3],
                                          [0,1,2],
                                          [0,1,2,3,4]])

    logits=top_k_top_p_filtering(logits, top_p=0.6, filter_value=0)
    assert logits[2].sum() == 4

    # 6 < logits.sum() < 7 as only 4 can be set between (0,1).
    # 3/6 < 0.6 and 5/6 > 0.6
    # 3/7 < 0.6 and 5/7 > 0.6 
    assert logits[0].sum() == 3
