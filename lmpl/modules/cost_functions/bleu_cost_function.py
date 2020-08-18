from typing import Dict, Optional, Tuple, Union, List

from fairseq.bleu import Scorer, SacrebleuScorer
from overrides import overrides

from functools import partial

from allennlp.common.util import END_SYMBOL
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from multiprocessing import Pool

import torch

from lmpl.modules.cost_functions import CostFunction

def compute_bleu_score_decoded(
                    gold_label: List[str],
                    prediction: List[str]):
    global scorer
    scorer.add_string(' '.join(gold_label),
                      ' '.join(prediction))
    score = 1 - scorer.score()/100.0
    scorer.reset()
    return score

def compute_bleu_score(
            gold_label: torch.LongTensor,
            prediction: torch.LongTensor
):
    global scorer
    scorer.add(gold_label.type(torch.IntTensor),
                prediction.type(torch.IntTensor))
    score = 1.0 - scorer.score()/100.
    scorer.reset()
    return score

def decoded_init_pool():
    global scorer
    scorer = SacrebleuScorer()
    return scorer

def init_pool(pad_token, eos_token, unk_token):
    global scorer
    scorer = Scorer(pad_token,
                    eos_token,
                    unk_token)
    return scorer
            

@CostFunction.register("bleu")
class BLEUCostFunction(CostFunction):
    """ This call computes BLEU loss function between prediction and 
        gold targets. This is used to train NMT model.
    """

    def __init__(self,
                 pad_token:int = 0,
                 eos_token: int = 4,
                 unk_token: int = 1,
                 use_decoded_inputs:bool = True,
                 use_parallel: bool = False,
                 num_threads:int = 64,
                 ):

        self._num_threads = num_threads

        self._use_parallel = use_parallel


        if use_parallel:
            self._pool = Pool(self._num_threads, 
                                decoded_init_pool \
                                    if use_decoded_inputs else \
                                        partial(init_pool, 
                                                pad_token=pad_token,
                                                eos_token=eos_token,
                                                unk_token=unk_token))
        
        if use_decoded_inputs:
            decoded_init_pool()
        else:
            init_pool(pad_token,
                        eos_token,
                        unk_token)

        self._use_decoded_inputs = use_decoded_inputs

    def __call__(self,
                 predictions: Union[torch.IntTensor, List[str]],
                 gold_labels: Union[torch.IntTensor, List[str]] = None,
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
        bleu_costs = []
        if self._use_decoded_inputs:
            if self._use_parallel:
                bleu_costs = self._pool.starmap(compute_bleu_score_decoded, zip(gold_labels, predictions))
            else:
                for ref, pred  in zip(gold_labels, predictions):
                  bleu_costs.append(compute_bleu_score_decoded(ref, pred))

        else:
            predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)
            if self._use_parallel:
                bleu_costs = self._pool.starmap(compute_bleu_score, zip(gold_labels, predictions))
            else:
                for prediction,gold_label in zip(predictions, gold_labels):
                    bleu_costs.append(compute_bleu_score(gold_label, prediction))
             
        return torch.tensor(bleu_costs)

    @overrides
    def takes_decoded_input(self):
        return self._use_decoded_inputs
