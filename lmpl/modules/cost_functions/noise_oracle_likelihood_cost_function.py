from typing import Dict, Optional, Tuple, Union, List

from overrides import overrides

from lmpl.modules.cost_functions.cost_function import CostFunction
from lmpl.oracles.oracle_base import Oracle

import logging
import numpy as np
import torch


@CostFunction.register("noisy_oracle")
class NoiseOracleCostFunction(CostFunction):
    """This cost function computes the cost(negative likelihood) of predicted sequence under oracle and
        returns a cost corrputed by noise.
    """
    name: str = "noisy_oracle_cf"

    def __init__(self,
                 oracle: Oracle,
                 noise_type: str = None, 
                 add_brevity_penalty: bool = False) -> None:
        self._oracle = oracle
        self._noise_type = noise_type
        self._add_brevity_penalty = add_brevity_penalty

        if self._noise_type is not None:
            # Figure out how to add noise.
            pass

    def __call__(self,
                 predictions: List[str],
                 gold_labels: List[str] = None) -> torch.Tensor:
        """ Computes cost under oracle and returns the batch cost.

        Arguments:
            predictions {List[str]} -- predictions generated by a rollout.
            gold_labels {List[str]} -- Orignal Sequence.

        """

        # This hack given 0 oracle prob to sequences of length 1.
        # This is done as GPT2 craches for length 1 sequences.

        oracle_probs_and_seq_probs = self._oracle.compute_sent_probs(
            predictions)

        oracle_probs = []
        j = 0
        for i, prediction in enumerate(predictions):
            gold_len, pred_len = (len(gold_labels[i]), len(predictions[i]))
            # This encourages model to generate sequences which are of equal
            # length as gold sequence.
            brevity_penality = 1 - float(gold_len)/max(pred_len,1)  \
                                    if gold_len  > pred_len  and  \
                                        self._add_brevity_penalty \
                               else 0
            oracle_probs.append(np.log(oracle_probs_and_seq_probs[i][0] + 1e-45) + brevity_penality)

        # We return neg log prob.
        # The objective should be minimize this cost to 0.
        return -1 * torch.FloatTensor(oracle_probs) \
            .to(torch.cuda.current_device()
                if torch.cuda.is_available() else 'cpu')

    @overrides
    def takes_decoded_input(self):
        return True
