import math
from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from overrides import overrides
from torch import nn
from torch.autograd import Variable

from allennlp.nn import util as nn_util

from allennlp_models.generation.modules.decoder_nets import DecoderNet
from allennlp_models.lm.modules.seq2seq_encoders.bidirectional_lm_transformer import PositionalEncoding, subsequent_mask
@DecoderNet.register("transformer")
class TransformerDecoderNet(DecoderNet):
    """
    A Stacked self-attention decoder implementation.

    # Parameters

    decoding_dim : `int`, required
        Defines dimensionality of output vectors.
    target_embedding_dim : `int`, required
        Defines dimensionality of input target embeddings.  Since this model takes it's output on a previous step
        as input of following step, this is also an input dimensionality.
    feedforward_hidden_dim : `int`, required.
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention layers.
    num_layers : `int`, required.
        The number of stacked self attention -> feedfoward -> layer normalisation blocks.
    num_attention_heads : `int`, required.
        The number of attention heads to use per layer.
    use_positional_encoding : `bool`, optional, (default = True)
        Whether to add sinusoidal frequencies to the input tensor. This is strongly recommended,
        as without this feature, the self attention layers have no idea of absolute or relative
        position (as they are just computing pairwise similarity between vectors of elements),
        which can be important features for many tasks.
    dropout_prob : `float`, optional, (default = 0.1)
        The dropout probability for the feedforward network.
    residual_dropout_prob : `float`, optional, (default = 0.2)
        The dropout probability for the residual connections.
    attention_dropout_prob : `float`, optional, (default = 0.1)
        The dropout probability for the attention distributions in each attention layer.
    """

    def __init__(
        self,
        decoding_dim: int,
        target_embedding_dim: int,
        feedforward_hidden_dim: int,
        num_layers: int,
        num_attention_heads: int,
        use_positional_encoding: bool = True,
        positional_encoding_max_steps: int = 5000,
        dropout_prob: float = 0.1,
        residual_dropout_prob: float = 0.2,
        attention_dropout_prob: float = 0.1,
        activation_function: str = 'relu',
    ) -> None:

        super().__init__(
            decoding_dim=decoding_dim,
            target_embedding_dim=target_embedding_dim,
            decodes_parallel=True,
        )

        decoder_layer = torch.nn.TransformerDecoderLayer(
                          d_model=decoding_dim,
                          nhead=num_attention_heads,
                          dim_feedforward=feedforward_hidden_dim,
                          dropout=dropout_prob,
                          activation=activation_function
                        )
        layer_norm = torch.nn.LayerNorm(decoding_dim)

        self.decoder = torch.nn.TransformerDecoder(
                            decoder_layer=decoder_layer, 
                            num_layers=num_layers,
                            norm=layer_norm)

        self._positional_embedder = (
            PositionalEncoding(decoding_dim, positional_encoding_max_steps)
            if use_positional_encoding
            else None
        )
        self._embed_scale = math.sqrt(decoding_dim)
        self._dropout = nn.Dropout(dropout_prob)

    @overrides
    def init_decoder_state(
        self, encoder_out: Dict[str, torch.LongTensor]
    ) -> Dict[str, torch.Tensor]:
        return {}

    @overrides
    def forward(
        self,
        previous_state: Dict[str, torch.Tensor],
        encoder_outputs: torch.Tensor,
        source_mask: torch.BoolTensor,
        previous_steps_predictions: torch.Tensor,
        previous_steps_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        future_mask = ~subsequent_mask(previous_steps_predictions.size(-2), 
                                   device=source_mask.device).squeeze(0)
        if previous_steps_mask is not None:
          previous_steps_mask = ~previous_steps_mask
        
        if source_mask is not None:
          source_mask = ~source_mask

        previous_steps_predictions = previous_steps_predictions * self._embed_scale
        if self._positional_embedder:
            previous_steps_predictions = self._positional_embedder(previous_steps_predictions)
        previous_steps_predictions = self._dropout(previous_steps_predictions)
        
        decoded = self.decoder(
                      tgt=previous_steps_predictions.transpose(0,1),
                      tgt_mask=future_mask,
                      memory=encoder_outputs.transpose(0,1),
                      tgt_key_padding_mask=previous_steps_mask,
                      memory_key_padding_mask=source_mask,
                  )
        return {}, decoded.transpose(0,1)

