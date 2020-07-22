from typing import Dict, Optional, Tuple, Union, List

import torch
from allennlp.common.registrable import Registrable

class CostFunction(Registrable):
    """Abstract class for cost computation during rollout.  
    """
    name: str = "CostFunction"

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor = None,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions.
        gold_labels : ``torch.Tensor``, required.
            A tensor corresponding to some gold label to evaluate against.
        mask: ``torch.Tensor``, optional (default = None).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.

        Returns
        ---------
        A vector of shape (batch_size, ) of sequence losses.
        """
        raise NotImplementedError
    
    @staticmethod
    def unwrap_to_tensors(*tensors: torch.Tensor):
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures that you're using tensors directly and that they are on
        the CPU.
        """
        return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)

    def takes_decoded_input(self):
        return self._use_decoded_inputs
