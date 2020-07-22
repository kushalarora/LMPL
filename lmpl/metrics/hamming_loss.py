from typing import Optional

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("hamming_loss")
class HammingLoss(Metric):
    """
    This ``Metric`` calculates the mean absolute hamming loss between two tensors.
    """
    def __init__(self) -> None:
        self._error = 0.0
        self._total_count = 0.0

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
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

        absolute_errors = (predictions != gold_labels).int()
        if mask is not None:
            absolute_errors *= mask
            self._total_count += torch.sum(mask)
        else:
            self._total_count += gold_labels.numel()
        self._error += torch.sum(absolute_errors)

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated mean absolute error.
        """
        hamming_loss = float(self._error) / float(self._total_count)
        if reset:
            self.reset()
        return hamming_loss

    @overrides
    def reset(self):
        self._error = 0.0
        self._total_count = 0.0
