from typing import List

from lmpl.modules.detokenizers.detokenizer import DeTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import overrides

@DeTokenizer.register("gpt2_detokenizer")
class GPT2DeTokenizer(DeTokenizer):
    def __init__(self, model_name='gpt2'):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def __call__(self, tokens_list: List[List[str]]) -> List[str]:
        return [self.tokenizer.convert_tokens_to_string(tokens) for tokens in tokens_list]
