from typing import Tuple
from lmpl.modules.utils import expand_tensor, top_k_top_p_filtering

from allennlp.common.testing import AllenNlpTestCase

import torch
def setup_logits(batch_size: int, num_vocab: int, 
                  indices_to_set: Tuple[Tuple[int]],
                  fixed_value=1e1):
  logits = torch.rand((batch_size, num_vocab))
  for i, row in enumerate(indices_to_set):
    logits[i, row] = torch.FloatTensor(row)
  return logits


class TestDecoderUtils(AllenNlpTestCase):
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

    def test_expand_tensor(self):
      decoder_input_dim = 4

      # Test num_token_to_rollout = 10
      tensor = torch.randn((2,5))
      expanded_tensor = expand_tensor(tensor, 
                                        num_tokens_to_rollout=10)
      new_batch_size = expanded_tensor.size(0)
      assert new_batch_size == 20
      assert torch.all(expanded_tensor.reshape(2, -1, 5)[:, 4, :] == tensor)

      # Test num_token_to_rollout = 1
      tensor = torch.randn((2,5))
      expanded_tensor = expand_tensor(tensor, 
                                        num_tokens_to_rollout=1)
      new_batch_size = expanded_tensor.size(0)
      assert new_batch_size == 2
      assert torch.all(expanded_tensor == tensor)

      # Test no non batch dim
      tensor = torch.randn((1,))
      expanded_tensor = expand_tensor(tensor, 
                                        num_tokens_to_rollout=10)
      new_batch_size = expanded_tensor.size(0)
      assert new_batch_size == 10
      assert torch.all(expanded_tensor.reshape(2, -1)[:, 4] == tensor)

      # Test two non batch dim
      tensor = torch.randn((2,3,4))
      expanded_tensor = expand_tensor(tensor, 
                                        num_tokens_to_rollout=10)
      new_batch_size = expanded_tensor.size(0)
      assert new_batch_size == 20
      assert expanded_tensor.size(1) == 3
      assert expanded_tensor.size(2) == 4
