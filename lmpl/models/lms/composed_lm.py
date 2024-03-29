from typing import Dict, Optional

import torch
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

from allennlp_models.generation.modules.seq_decoders import SeqDecoder
from lmpl.modules.detokenizers.detokenizer import DeTokenizer, default_tokenizer

from lmpl.modules.utils import decode_tokens
import random

@Model.register("lmpl_composed_lm")
class ComposedLMBase(Model):
    """
    This ``ComposedSeq2Seq`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    The ``ComposedSeq2Seq`` class composes separate ``Seq2SeqEncoder`` and ``SeqDecoder`` classes.
    These parts are customizable and are independent from each other.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_embedders : ``TextFieldEmbedder``, required
        Embedders for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    decoder : ``SeqDecoder``, required
        The decoder of the "encoder/decoder" model
    tied_source_embedder_key : ``str``, optional (default=``None``)
        If specified, this key is used to obtain token_embedder in `source_embedder` and
        the weights are shared/tied with the decoder's target embedding weights.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        use_in_seq2seq_mode: bool,
        decoder: SeqDecoder,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,

        source_embedder: TextFieldEmbedder = None,
        source_namespace: str = "tokens",
        source_add_start_token: bool = False,
        source_start_token: str = START_SYMBOL,
        source_end_token: str = END_SYMBOL,
        encoder: Seq2SeqEncoder = None,
        tied_source_embedder_key: Optional[str] = None,
        detokenizer: DeTokenizer = default_tokenizer,
        log_output_every_iteration: int = 100,
    ) -> None:

        super().__init__(vocab, regularizer)

        self.epoch = 0
        self.batch_number = 0

        self._seq2seq_mode = use_in_seq2seq_mode
        self._decoder = decoder
        
        self._source_embedder = source_embedder
        self._source_namespace = source_namespace
        self._encoder = encoder

        self._vocab = vocab
        self._detokenizer = detokenizer

        self._source_add_start_token = source_add_start_token
        self._source_end_token = source_end_token

        self._source_start_index = None
        if source_add_start_token:
            self._source_start_index = self._vocab.get_token_index(source_start_token, self._source_namespace)
        self._source_end_index = self._vocab.get_token_index(source_end_token, self._source_namespace)

        self._log_output_every_iteration = log_output_every_iteration
        if self._seq2seq_mode:
            if self._encoder.get_output_dim() != self._decoder.get_output_dim():
                raise ConfigurationError(
                    f"Encoder output dimension {self._encoder.get_output_dim()} should be"
                    f" equal to decoder dimension {self._decoder.get_output_dim()}."
                )
            if tied_source_embedder_key:
                # A bit of a ugly hack to tie embeddings.
                # Works only for `BasicTextFieldEmbedder`, and since
                # it can have multiple embedders, and `SeqDecoder` contains only a single embedder, we need
                # the key to select the source embedder to replace it with the target embedder from the decoder.
                if not isinstance(self._source_embedder, BasicTextFieldEmbedder):
                    raise ConfigurationError(
                        "Unable to tie embeddings,"
                        "Source text embedder is not an instance of `BasicTextFieldEmbedder`."
                    )

                source_embedder = self._source_embedder._token_embedders[tied_source_embedder_key]
                if not isinstance(source_embedder, Embedding):
                    raise ConfigurationError(
                        "Unable to tie embeddings,"
                        "Selected source embedder is not an instance of `Embedding`."
                    )
                if source_embedder.get_output_dim() != self._decoder.target_embedder.get_output_dim():
                    raise ConfigurationError(
                        f"Output Dimensions mismatch between" f"source embedder and target embedder."
                    )
                self._source_embedder._token_embedders[
                    tied_source_embedder_key
                ] = self._decoder.target_embedder
        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        source_tokens: Dict[str, torch.LongTensor] = None,
        target_tokens: Dict[str, torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Make foward pass on the encoder and decoder for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        Returns
        -------
        Dict[str, torch.Tensor]
            The output tensors from the decoder.
        """
        self.batch_number += 1
        state:  Dict[str, torch.Tensor] = {
                            "epoch": self.epoch,
                            "batch_number": self.batch_number,
                           }
        if self._seq2seq_mode:
            state.update(self._encode(source_tokens))
        output_dict = self._decoder(state, target_tokens)

        if self._seq2seq_mode:
            source_indexes=util.get_token_ids_from_text_field_tensors(source_tokens)
            decoded_sources = decode_tokens(vocab=self._vocab, 
                                end_index=self._source_end_index,
                                start_index=self._source_start_index,
                                batch_predicted_indices=source_indexes,
                                vocab_namespace=self._source_namespace,
                                truncate=True,)
            output_dict['decoded_sources'] = decoded_sources
            output_dict['detokenized_sources'] = self._detokenizer(
                                                    [source[0] for source in decoded_sources])
        
        output_dict["detokenized_predictions"] = \
                [self._detokenizer(predictions)
                    for predictions in output_dict["decoded_predictions"]]
        
        if target_tokens is not None:
            output_dict["detokenized_targets"] = self._detokenizer(
                        [target[0] for target in output_dict['decoded_targets']])

        if self.batch_number % self._log_output_every_iteration == 0:
            print()
            if self._seq2seq_mode:
                print(f"Source::  {output_dict['detokenized_sources'][0]}")
            if target_tokens is not None:
                print(f"Target::  {output_dict['detokenized_targets'][0]}")
            print(f"Top Prediction::  {output_dict['detokenized_predictions'][0][0]}")

        return output_dict

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.
        """
        return self._decoder.post_process(output_dict)

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Make foward pass on the encoder.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.

        Returns
        -------
        Dict[str, torch.Tensor]
            Map consisting of the key `source_mask` with the mask over the
            `source_tokens` text field,
            and the key `encoder_outputs` with the output tensor from
            forward pass on the encoder.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)

        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)

        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input_w_dropout = self._decoder._dropout(embedded_input)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input_w_dropout, source_mask)
        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._decoder.get_metrics(reset)