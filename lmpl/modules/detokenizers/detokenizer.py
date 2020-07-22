from typing import List

from allennlp.common import Registrable
from allennlp.data.tokenizers.token import Token


class DeTokenizer(Registrable):
    """
    A ``DeTokenizer`` joins tokens to strings of text.  
    """

    #default_implementation = "default_detokenzier"


    def __call__(self, tokens_list: List[List[str]]) -> List[str]:
        """
        Actually implements detokenization by coverting list of tokens (in str form) to a string.

        Returns
        -------
        detokenized_str : ``str``
        """
        raise NotImplementedError



def default_tokenizer(token_list_list: List[List[str]]):
    str_list = []
    for token_list in token_list_list:
        str_list.append(' '.join(token_list))
    return str_list