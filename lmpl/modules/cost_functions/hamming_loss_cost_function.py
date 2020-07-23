from typing import Dict, Optional, Tuple, Union, List

from overrides import overrides

from lmpl.modules.cost_functions import CostFunction

import torch

@CostFunction.register("hamming")
class HammingCostFunction(CostFunction):
    """ This call computes hamming loss function between prediction and 
        gold labels. This is used to train OCR model.
    """
  
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor = None,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predictions``.
        mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predictions``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)
        try:
            absolute_errors = (predictions != gold_labels).int()
        except:
            import pdb;pdb.set_trace()

        if mask is not None:
            absolute_errors *= mask
            total_batch_count = mask.sum(dim=-1)
        else:
            total_batch_count = gold_labels.size(1)
        return absolute_errors.sum(dim=-1).float()/total_batch_count

    @staticmethod
    def unwrap_to_tensors(*tensors: torch.Tensor):
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures that you're using tensors directly and that they are on
        the CPU.
        """
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)
    
    @overrides
    def takes_decoded_input(self):
        return False
    