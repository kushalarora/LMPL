from typing import Dict, List
import itertools

from overrides import overrides
import torch
import numpy

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList


@TokenIndexer.register("ocr_indexer")
class OCRTokenIndexer(TokenIndexer):
    """
    This :class:`TokenIndexer` represents tokens as single integers.

    Parameters
    ----------
    binary_string_dim : ``int``, optional (default=``128``)
        The length of the binary string. This is used to create padding token.

    token_min_padding_length : ``int``, optional (default=``0``)
        See :class:`TokenIndexer`.
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 binary_string_dim: int = 128,
                 token_min_padding_length: int = 0) -> None:
        super().__init__(token_min_padding_length)
        self._binary_string_dim = binary_string_dim

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> IndexedTokenList:

        indices: List[numpy.ndarray] = []
        for binary_token in tokens:
            indices.append(numpy.array([int(x) for x in binary_token.text]))

        return {index_name: indices}

    def get_padding_token(self) -> numpy.ndarray:
        return numpy.zeros(self._binary_string_dim)


    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def as_padded_tensor_dict(self,
                         tokens: Dict[str, List[int]],
                         desired_num_tokens: Dict[str, int],
                         padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:  # pylint: disable=unused-argument

        return {key: torch.LongTensor(pad_sequence_to_length(val, 
                                                             desired_num_tokens[key], 
                                                             default_value=self.get_padding_token))
                for key, val in tokens.items()}
