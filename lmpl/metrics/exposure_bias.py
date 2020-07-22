from overrides import overrides
from typing import List

from allennlp.training.metrics.metric import Metric

from quant_exp_bias.oracles.oracle_base import Oracle

import logging
import torch
import numpy as np
import random
import math
from functools import reduce, partial

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def rfn_prefix(p, q, prev_p_q, n, clipping_ratio_max=math.inf, clipping_ratio_min=0): 
    # return np.exp(np.log(prev_p_q) + np.log(max(min(clipping_ratio_max, p/q), clipping_ratio_min)))
    return max(min(clipping_ratio_max, np.exp(np.log(prev_p_q) + np.log(p) - np.log(q))), clipping_ratio_min)

def rfn_sequence(p, q, prev_p_q, n, clipping_ratio_max=math.inf, clipping_ratio_min=0): 
    return max(min(clipping_ratio_max, np.exp(n * (np.log(p) - np.log(q)))), clipping_ratio_min)


@Metric.register("exp_bias")
class ExposureBias(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """

    def __init__(self,
                 oracle: Oracle,
                 type: str = 'kl',
                 at_prefix_level: bool = True,
                 clipping_ratio_max=math.inf,
                 clipping_ratio_min=0.0,
                 ctxt_size=math.inf,
                ) -> None:
        self._total_value = 0.0
        self._df_p_q = 0.0
        self._df_q_p = 0.0
        self._count = 0
        self._oracle = oracle
        self._type = type
        self._at_prefix_level = at_prefix_level
        self._ctxt_size = ctxt_size

        # D_f(P||Q) = \sum_{x in X} f(p(X)/q(x))q(x)
        self._Df = ExposureBias.DfBuilder(type,
                        partial(rfn_prefix, 
                                clipping_ratio_max=clipping_ratio_max,
                                clipping_ratio_min=clipping_ratio_min) \
                            if at_prefix_level else \
                                partial(rfn_sequence,
                                        clipping_ratio_max=clipping_ratio_max,
                                        clipping_ratio_min=clipping_ratio_min))

    @overrides
    def __call__(self,
                 model_sampled_model_probs: torch.FloatTensor,
                 model_sampled_predictions: List[str],
                 model_sampled_model_seq_probs: torch.FloatTensor,
                 use_js: bool = False,
                 oracle_sampled_model_probs: torch.FloatTensor = None,
                 oracle_sampled_predictions: List[str] = [],
                 oracle_sampled_model_seq_probs: torch.FloatTensor = None,
                 ):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        # TODO (Kushal) Add comments to explain what is going on.
        # Compute DL(P||M)
        model_sampled_batch_size = len(model_sampled_predictions)
        df_p_q = 0
        df_p_q_count = 0
        df_p_qs = []

        model_sampled_oracle_probs = []
        model_sampled_oracle_probs_and_seq_probs = self._oracle.compute_sent_probs(model_sampled_predictions)
        for i in range(model_sampled_batch_size):

            if len(model_sampled_predictions[i]) == 0:
                continue

            values = []
            prev_p_qs = []
            seq_len = min(len(model_sampled_predictions[i]) + 1, 
                            len(model_sampled_oracle_probs_and_seq_probs[i][1]),
                                len(model_sampled_model_seq_probs[i]))

            if self._at_prefix_level:
                df_p_q_seq = 0
                prev_p_q = 1.0
                for j in range(1, seq_len):
                    if self._ctxt_size < j:
                        c = self._ctxt_size
                        P_c = model_sampled_oracle_probs_and_seq_probs[i][1][j-c]
                        Q_c = model_sampled_model_seq_probs[i][j-c].item()
                        prev_p_q = rfn_prefix(Q_c, P_c, prev_p_q, 1)

                    # Here model_sampled_model_prob is Q because the samples
                    # come from the model.
                    P = model_sampled_oracle_probs_and_seq_probs[i][1][j]
                    Q = model_sampled_model_seq_probs[i][j].item()
                    
                    value, prev_p_q = self._Df(P, Q, prev_p_q, j+1)
                    
                    values.append(value)
                    prev_p_qs.append(prev_p_q)
                    df_p_q_seq += 0.5 * value
                    df_p_q_count += 1

                df_p_q += df_p_q_seq
                df_p_qs.append(df_p_q_seq/seq_len)
                
            else:
                P = model_sampled_oracle_probs_and_seq_probs[i][0]
                Q = model_sampled_model_probs[i].item()

                value, _ = self._Df(P, Q, 1.0, seq_len)
                df_p_q += value

                df_p_qs.append(value)
                df_p_q_count += seq_len

            model_sampled_oracle_probs.append(model_sampled_oracle_probs_and_seq_probs[i][0])
        
        # oracle_sampled_batch_size = len(oracle_sampled_predictions)
        # df_q_ps = []
        # df_q_p = 0
        # df_q_p_count = 0
        
        # # Compute DL(Q||M)
        # oracle_sampled_oracle_probs = []
        # oracle_sampled_oracle_probs_and_seq_probs = self._oracle.compute_sent_probs(oracle_sampled_predictions)
        # for i in range(oracle_sampled_batch_size):
        #     if len(oracle_sampled_predictions[i]) == 0:
        #         continue
            
        #     values = []
        #     prev_q_ps = []
        #     seq_len = min(len(oracle_sampled_predictions[i]) + 1, 
        #                     len(oracle_sampled_oracle_probs_and_seq_probs[i][1]),
        #                         len(oracle_sampled_model_seq_probs[i]))

        #     if self._at_prefix_level:
        #         df_q_p_seq = 0
        #         prev_q_p = 1.0
        #         for j in range(1, seq_len):
        #             if self._ctxt_size < j:
        #                 c = self._ctxt_size
        #                 P_c = oracle_sampled_oracle_probs_and_seq_probs[i][1][j-c]
        #                 Q_c = oracle_sampled_model_seq_probs[i][j-c].item()
        #                 prev_q_p = rfn_prefix(P_c, Q_c, prev_q_p, 1.0)

        #             # Here oracle_sampled_oracle_probs is Q because the samples
        #             # come from the oracle.
        #             P = oracle_sampled_oracle_probs_and_seq_probs[i][1][j]
        #             Q = oracle_sampled_model_seq_probs[i][j].item()
                    
        #             value, prev_q_p = self._Df(Q, P, prev_q_p, j+1)
                    

        #             df_q_p_seq += 0.5 * value
        #             df_q_p_count += 1
        #             values.append(value)
        #             prev_q_ps.append(prev_q_p)
                
        #         df_q_p += df_q_p_seq
        #         df_q_ps.append(df_q_p_seq/seq_len)
        #     else:
        #         P = oracle_sampled_oracle_probs_and_seq_probs[i][0]
        #         Q = oracle_sampled_model_probs[i].item()
                
        #         value, _ = self._Df(Q, P, 1.0, seq_len)
        #         df_q_p += value

        #         df_q_ps.append(value)
        #         df_q_p_count += seq_len

        #     oracle_sampled_oracle_probs.append(oracle_sampled_oracle_probs_and_seq_probs[i][0])

        self._total_value += df_p_q/df_p_q_count # + df_q_p/df_q_p_count
        self._df_p_q += df_p_q/df_p_q_count
        # self._df_q_p += df_q_p/df_q_p_count
        logging.info(f"KL(P || M) = {df_p_q/df_p_q_count:.4f}")

        # logging.info(f"KL(P || M) = {df_p_q/df_p_q_count:.4f} \t KL(Q || M) = {df_q_p/df_q_p_count:.4f}")

        return { "model_sampled_predictions": model_sampled_predictions, 
                 "model_sampled_model_probs": model_sampled_model_probs,
                  "model_sampled_oracle_probs": model_sampled_oracle_probs, 
                  "model_sampled_scores": df_p_qs,
                } # \
            # oracle_sampled_predictions, oracle_sampled_model_probs, oracle_sampled_oracle_probs, df_q_ps

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        avg_exp_bias = self._total_value
        avg_df_p_q = self._df_p_q
        # avg_df_q_p = self._df_q_p

        if reset:
            self.reset()

        return {
            "exposure_bias": avg_exp_bias, 
            "df_p_q": avg_df_p_q #, avg_df_q_p
        }

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._df_p_q = 0.0
        # self._df_q_p = 0.0

    @staticmethod
    def DfBuilder(type='kl', rfn=rfn_sequence):
        if type == 'abs_kl':
            return lambda p, q, prev_p_q, n: (np.abs(np.log10(rfn(q, p, prev_p_q, n))), rfn(q, p, prev_p_q, n))
        if type == 'kl':
            return lambda p, q, prev_p_q, n: (np.log10(rfn(q, p, prev_p_q, n)), rfn(q, p, prev_p_q, n))
        elif type == 'hellinger_squared':
            return lambda p, q, prev_p_q, n: ((np.sqrt(rfn(p, q, prev_p_q, n)) - 1)**2, rfn(p, q, prev_p_q, n))
        elif type == 'tv':
            return lambda p, q, prev_p_q, n: (np.abs(rfn(p, q, prev_p_q, n) - 1), rfn(p, q, prev_p_q, n))
        # elif type == 'js':
        #     return lambda p, q, prev_p_q, n: (np.log10(2/(rfn(p, q, prev_p_q, n) + 1)), rfn(p, q, prev_p_q, n))
