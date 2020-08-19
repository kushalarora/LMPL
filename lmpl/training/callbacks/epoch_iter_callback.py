from typing import List, Dict, Any

from allennlp.data.dataloader import TensorDict
from allennlp.training import BatchCallback
from lmpl.models.lms.composed_lm import ComposedLMBase

@BatchCallback.register("update_epoch_iter")
class UpdateEpochAndIter(BatchCallback):
    """
    Logs the CPU and GPU memory usage to tensorboard on every batch.

    This is mainly used for debugging as it can cause a significant slowdown in training.
    """

    def __call__(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_master: bool,
    ) -> None:
      model = trainer.model 
    
      assert hasattr(model, "epoch")
      assert hasattr(model, "batch_number")

      assert isinstance(model, ComposedLMBase)
      setattr(model, "epoch", epoch)
      setattr(model, "batch_number", batch_number)

