import torch
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data import Vocabulary
from allennlp.common import Params, Tqdm

@TokenEmbedder.register("ocr_token_embedder")
class OCRBinaryStringTokenEmbedder(TokenEmbedder):
    """
    The OCR experiments map a sequence of binary string to a 
    sequence of characters. This method convert a binary string
    into a binary number (size: binary_seq_size{deafult: 128}) 
    and project it into a hidden space (size: hidden_dim).

    Parameters
    ----------
    hidden_dim : `int`, required.
    binary_str_size: `int`, {default: 128}.

    """

    def __init__(self, hidden_dim: int, binary_str_size: int = 128) -> None:
        super().__init__()

        self._hidden_dim = hidden_dim
        self._binary_str_size = binary_str_size

        self._embedding = torch.nn.Linear(hidden_dim, binary_str_size)

    def get_output_dim(self):
        return self.hidden_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:    
        return self._embedding(inputs.float())


    @classmethod
    def from_params(  # type: ignore
        cls, vocab: Vocabulary, params: Params
    ) -> "OCRBinaryStringTokenEmbedder":

        """ Create this class.
        """

        hidden_dim = params.pop_int("hidden_dim", None)
        binary_str_size = params.pop_int("binary_str_size", 128)
        params.assert_empty(cls.__name__)
        return cls(
            hidden_dim=hidden_dim,
            binary_str_size=binary_str_size,
        )
