from typing import List, Tuple, Union, Dict, Any, Optional, Callable
import logging
import json
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from allennlp.training.trainer import (  # noqa
    GradientDescentTrainer, BatchCallback, EpochCallback,
)
from allennlp.models.archival import CONFIG_NAME

logger = logging.getLogger(__name__)
Number = Union[int, float]
Value = Union[int, float, bool, str]

def flatten_dict(params: Dict[str, Any],
                 delimiter: str = ".") -> Dict[str, Value]:
    """
    Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a.b': 'c'}``.
    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'.'``.
    Returns:
        Flattened dict.
    """
    output: Dict[str, Union[str, Number]] = {}

    def populate(inp: Union[Dict[str, Any], List, str, Number, bool],
                 prefix: List[str]) -> None:

        if isinstance(inp, dict):
            for k, v in inp.items():
                populate(v, deepcopy(prefix) + [k])

        elif isinstance(inp, list):
            for i, val in enumerate(inp):
                populate(val, deepcopy(prefix) + [str(i)])
        elif isinstance(inp, (str, float, int, bool)):
            output[delimiter.join(prefix)] = inp
        else:  # unsupported type
            raise ValueError(
                f"Unsuported type {type(inp)} at {delimiter.join(prefix)} for flattening."
            )

    populate(params, [])

    return output


def get_config_from_serialization_dir(dir_: str, ) -> Dict[str, Value]:
    with open(Path(dir_) / CONFIG_NAME) as f:
        config_dict = json.load(f)
    config_dict = flatten_dict(config_dict)

    return config_dict

# Adapted from https://github.com/dhruvdcoder/wandb-allennlp/blob/master/wandb_allennlp/training/callbacks/log_to_wandb.py
@EpochCallback.register("log_metrics_to_wandb")
class LogMetricsToWandb(EpochCallback):
    def __init__(
            self,
            project_name: str,
            run_name: str,
            epoch_end_log_freq: int = 1,
            sync_tensorboard: bool = True,
    ) -> None:
        # import wandb here to be sure that it was initialized
        # before this line was executed
        super().__init__()
        import wandb  # type: ignore
        run_id = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        wandb.init(
            project=project_name,
            name=f'{run_name}/{run_id}',
            sync_tensorboard=sync_tensorboard,
            id=run_id,
        )
        self.config: Optional[Dict[str, Value]] = None

        self.wandb = wandb
        self.epoch_end_log_freq = 1
        self.current_batch_num = -1
        self.current_epoch_num = -1
        self.previous_logged_epoch = -1

    def update_config(self, trainer: GradientDescentTrainer) -> None:
        if self.config is None:
            # we assume that allennlp train pipeline would have written
            # the entire config to the file by this time
            logger.info(f"Sending config to wandb...")
            self.config = get_config_from_serialization_dir(
                trainer._serialization_dir)
            self.wandb.config.update(self.config)

    def __call__(
            self,
            trainer: GradientDescentTrainer,
            metrics: Dict[str, Any],
            epoch: int,
            is_master: bool,
    ) -> None:
        """ This should run after all the epoch end metrics have
        been computed by the metric_tracker callback.
        """

        if self.config is None:
            self.update_config(trainer)

        self.current_epoch_num += 1

        if (is_master
                and (self.current_epoch_num - self.previous_logged_epoch)
                >= self.epoch_end_log_freq):
            logger.info("Writing metrics for the epoch to wandb")
            self.wandb.log(
                {
                    **metrics,
                },
            )
            self.previous_logged_epoch = self.current_epoch_num