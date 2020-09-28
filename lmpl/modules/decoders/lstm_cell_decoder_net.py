from typing import Tuple, Dict, Optional
from overrides import overrides

import torch
from torch.nn import LSTMCell, LSTM

from allennlp.common.checks import ConfigurationError
from allennlp.modules import Attention
from allennlp.nn import util

from allennlp_models.generation.modules.decoder_nets import DecoderNet


@DecoderNet.register("lmpl_lstm_cell")
class LstmCellDecoderNet(DecoderNet):
    """
    This decoder net implements simple decoding network with LSTMCell and Attention.

    Parameters
    ----------
    decoding_dim : ``int``, required
        Defines dimensionality of output vectors.
    target_embedding_dim : ``int``, required
        Defines dimensionality of input target embeddings.  Since this model takes it's output on a previous step
        as input of following step, this is also an input dimensionality.
    attention : ``Attention``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    """

    def __init__(
        self,
        decoding_dim: int,
        target_embedding_dim: int,
        attention: Optional[Attention] = None,
        bidirectional_input: bool = False,
        num_decoder_layers: int = 1,
        accumulate_hidden_states: bool = False,
        dropout: float = 0.2,
    ) -> None:

        super().__init__(
            decoding_dim=decoding_dim,
            target_embedding_dim=target_embedding_dim,
            decodes_parallel=False,
        )

        # In this particular type of decoder output of previous step passes directly to the input of current step
        # We also assume that decoder output dimensionality is equal to the encoder output dimensionality
        decoder_input_dim = self.target_embedding_dim

        # Attention mechanism applied to the encoder output for each step.
        self._attention = attention

        if self._attention:
            # If using attention, a weighted average over encoder outputs will be concatenated
            # to the previous target embedding to form the input to the decoder at each
            # time step. encoder output dim will be same as decoding_dim
            decoder_input_dim += decoding_dim

        # Ensure that attention is only set during seq2seq setting.
        # if not self._seq2seq_mode and self._attention is not None:
        #     raise ConfigurationError("Attention is only specified in Seq2Seq setting.")

        self._num_decoder_layers = num_decoder_layers
        if self._num_decoder_layers > 1:
            self._decoder_cell = LSTM(input_size=decoder_input_dim, 
                                        hidden_size=self.decoding_dim, 
                                        num_layers=self._num_decoder_layers,
                                        dropout=dropout,)
        else:
            # We'll use an LSTM cell as the recurrent cell that produces a hidden state
            # for the decoder at each time step.
            # TODO (pradeep): Do not hardcode decoder cell type.
            self._decoder_cell = LSTMCell(decoder_input_dim, self.decoding_dim)

        self._bidirectional_input = bidirectional_input

        self._accumulate_hidden_states = accumulate_hidden_states

    def _prepare_attended_input(self,
                                decoder_hidden_state: torch.Tensor = None,
                                encoder_outputs: torch.Tensor = None,
                                encoder_outputs_mask: torch.Tensor = None,) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state."""
        # Ensure mask is also a FloatTensor. Or else the multiplication within
        # attention will complain.
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        # encoder_outputs_mask = encoder_outputs_mask

        # shape: (batch_size, max_input_sequence_length)
        input_weights = self._attention(decoder_hidden_state[:, -1],    # Use last layer's output.
                                        encoder_outputs, 
                                        encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input

    def init_decoder_state(self, 
                            encoder_out: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        batch_size = encoder_out["source_mask"].size(0)

        # Initialize the decoder hidden state with the final output of the encoder,
        # and the decoder context with zeros.
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
            encoder_out["encoder_outputs"],
            encoder_out["source_mask"],
            bidirectional=self._bidirectional_input,)

        # shape: (batch_size, self._num_decoder_layers, decoder_output_dim)
        decoder_hidden = final_encoder_output.view(batch_size, 1,  -1) \
                                .expand(batch_size, self._num_decoder_layers, -1) \
                                .contiguous()
        # shape: (batch_size, 1, decoder_output_dim)
        decoder_context = final_encoder_output.new_zeros(
                                batch_size, self._num_decoder_layers, self.decoding_dim)

        return {
            "decoder_hidden": decoder_hidden,  
            "decoder_context": decoder_context
        }

    @overrides
    def forward(self,
                previous_state: Dict[str, torch.Tensor],
                previous_steps_predictions: torch.Tensor,
                previous_steps_mask: Optional[torch.BoolTensor] = None,
                encoder_outputs: torch.Tensor = None,
                source_mask: torch.BoolTensor = None,
               ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        # shape (decoder_hidden): (batch_size, 1, decoder_output_dim)
        decoder_hidden = previous_state.get("decoder_hidden", None)

        # shape (decoder_context):  (batch_size, 1, decoder_output_dim)
        decoder_context = previous_state.get("decoder_context", None)

        assert decoder_hidden is None and decoder_context is None or \
            decoder_hidden is not None and decoder_context is not None, \
        "Either decoder_hidden and context should be None or both should exist."

        decoder_hidden_and_context = None
        if decoder_hidden is not None and decoder_context is not None:
            if self._num_decoder_layers > 1:                             
                # This is needed because LSTM expects input to be
                # num_layers * num_directions, batchsize, hidden_dim
                # whereas everywhere else we expect batch size to be the first dimension of the tensor.
                decoder_hidden_and_context = (decoder_hidden.transpose(0,1).contiguous(),
                                                decoder_context.transpose(0,1).contiguous())  
            else:
                decoder_hidden_and_context = (decoder_hidden[:, -1], decoder_context[:, -1])

        # shape: (group_size, output_dim)
        last_predictions_embedding = previous_steps_predictions[:, -1]

        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = previous_state.get("encoder_outputs", None)

        if encoder_outputs is not None and self._attention:
            # shape: (group_size, encoder_output_dim)
            attended_input = self._prepare_attended_input(decoder_hidden, encoder_outputs, source_mask)


            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, last_predictions_embedding), -1)
        else:
            # shape: (group_size, target_embedding_dim)
            decoder_input = last_predictions_embedding

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        if self._num_decoder_layers > 1:
            _, (decoder_hidden, decoder_context) = self._decoder_cell(decoder_input.unsqueeze(0),
                                                                    decoder_hidden_and_context)
            decoder_output = decoder_hidden[-1]

            # This is needed because LSTM expects input to be
            # num_layers * num_directions, batchsize, hidden_dim
            # whereas everywhere else we expect batch size to be the 
            # first dimension of the tensor. Here, reverting batch_size as 
            # the first dimension.
            decoder_hidden = decoder_hidden.transpose(0,1).contiguous()
            decoder_context = decoder_context.transpose(0,1).contiguous()
        else:
            decoder_hidden, decoder_context = self._decoder_cell(decoder_input, 
                                                                decoder_hidden_and_context)
            decoder_output = decoder_hidden
            # This is needed as LSTMCell returns (batch_size, hidden_dim) tensor. We unsqueeze 
            # at 1 to indicate that there is only one layer.
            decoder_hidden = decoder_hidden.unsqueeze(1)
            decoder_context = decoder_context.unsqueeze(1)

        decoder_hiddens = previous_state.get('decoder_accumulated_hiddens')
        decoder_contexts = previous_state.get('decoder_accumulated_contexts')
        if self._accumulate_hidden_states:
            timestep = previous_state['timestep']
            timesteps_to_accumulate = previous_state.get('timesteps_to_accumulate', set([]))
            if timestep in timesteps_to_accumulate:
                decoder_hidden_acc = decoder_hidden.unsqueeze(1)
                decoder_context_acc = decoder_context.unsqueeze(1)
                if decoder_hiddens is None:
                    assert decoder_contexts is None
                    decoder_hiddens = decoder_hidden_acc
                    decoder_contexts = decoder_context_acc
                else:
                    assert decoder_contexts is not None
                    decoder_hiddens = torch.cat([decoder_hiddens, decoder_hidden_acc], dim=1)
                    decoder_contexts = torch.cat([decoder_contexts, decoder_context_acc], dim=1)

        return (
            {"decoder_accumulated_hiddens": decoder_hiddens, 
             "decoder_accumulated_contexts": decoder_contexts,
             "decoder_hidden": decoder_hidden,
             "decoder_context": decoder_context
             },
            decoder_output,
        )
