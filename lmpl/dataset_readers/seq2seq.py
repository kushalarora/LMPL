import csv
from typing import Dict, Optional
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)


@DatasetReader.register("lmpl_seq2seq")
class Seq2SeqDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``ComposedSeq2Seq`` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>

    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``SpacyTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    delimiter : str, (optional, default="\t")
        Set delimiter for tsv/csv file.
    """

    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        source_add_start_token: bool = False,
        delimiter: str = "\t",
        source_max_tokens: Optional[int] = None,
        target_max_tokens: Optional[int] = None,
        source_to_target_len_max_ratio: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._source_tokenizer = source_tokenizer or WhitespaceTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        
        self._delimiter = delimiter
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_to_target_len_max_ratio = source_to_target_len_max_ratio
        self._source_ignored = 0
        self._target_ignored = 0
        self._source_target_ratio_ignored = 0
    @overrides
    def _read(self, file_path):
        # Reset exceeded counts
        self._source_ignored = 0
        self._target_ignored = 0
        self._source_target_ratio_ignored = 0
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, row in enumerate(csv.reader(data_file, delimiter=self._delimiter)):
                if len(row) != 2:
                    raise ConfigurationError(
                        "Invalid line format: %s (line number %d)" % (row, line_num + 1)
                    )
                source_sequence, target_sequence = row

                tokenized_source = self._source_tokenizer.tokenize(source_sequence)
                if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
                        self._source_ignored += 1
                        continue
                
                if target_sequence is not None:
                    tokenized_target = self._target_tokenizer.tokenize(target_sequence)
                    if self._target_max_tokens and len(tokenized_target) > self._target_max_tokens:
                        self._target_ignored += 1
                        continue
                if self._source_to_target_len_max_ratio is not None and \
                    len(tokenized_target)/len(tokenized_source) > self._source_to_target_len_max_ratio:
                    self._source_target_ratio_ignored += 1
                    continue

                yield self.text_to_instance(source_sequence, target_sequence)

        if self._source_max_tokens and self._source_ignored:
            logger.info(
                "In %d instances, the source token length exceeded the max limit (%d) and were removed.",
                self._source_ignored,
                self._source_max_tokens,
            )
        if self._target_max_tokens and self._target_ignored:
            logger.info(
                "In %d instances, the target token length exceeded the max limit (%d) and were removed.",
                self._target_ignored,
                self._target_max_tokens,
            )
        if self._source_to_target_len_max_ratio and self._source_target_ratio_ignored:
            logger.info(
                "In %d instances, the target/source  length ratio exceeded the max limit (%.2f) and were removed.",
                self._source_target_ratio_ignored,
                self._source_to_target_len_max_ratio,
            )
    @overrides
    def text_to_instance(
        self, source_string: str, target_string: str = None
    ) -> Instance:  # type: ignore

        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)
            return Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:
            return Instance({"source_tokens": source_field})