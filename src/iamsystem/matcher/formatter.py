from typing import Tuple

from iamsystem.brat.util import get_brat_format_seq
from iamsystem.matcher.api import IAnnotation
from iamsystem.matcher.api import IBratFormatter
from iamsystem.tokenization.util import group_continuous_seq
from iamsystem.tokenization.util import multiple_seq_to_offsets
from iamsystem.tokenization.util import remove_trailing_stopwords


class TokenFormatter(IBratFormatter):
    """Default Brat Formatter: annotate a document by selecting continuous
    sequences of tokens but ignore stopwords."""

    def get_text_and_offsets(self, annot: IAnnotation) -> Tuple[str, str]:
        """Return tokens' labels and token's offsets (merge if continuous)"""
        sequences = group_continuous_seq(tokens=annot.tokens)
        offsets = multiple_seq_to_offsets(sequences=sequences)
        seq_offsets = get_brat_format_seq(offsets)
        seq_label = " ".join([token.label for token in annot.tokens])
        return seq_label, seq_offsets


class TokenStopFormatter(IBratFormatter):
    """A Brat formatter that takes into account stopwords: annotate a document
    by selecting continuous sequences of tokens/stopwords."""

    def __init__(self, remove_trailing_stop=True):
        """Create a brat formatter.

        :param remove_trailing_stop: if True, trailing stopwords in a
            discontinuous sequence will be removed.
            Ex: [['North', 'and'], ['America']] -> [['North', ['America']]
        """
        self.remove_trailing_stop = remove_trailing_stop

    def get_text_and_offsets(self, annot: IAnnotation) -> Tuple[str, str]:
        tokens = [*annot.tokens, *annot.stop_tokens]
        tokens.sort(key=lambda x: x.i)
        sequences = group_continuous_seq(tokens=tokens)
        if self.remove_trailing_stop:
            stop_i = [stop.i for stop in annot.stop_tokens]
            sequences = remove_trailing_stopwords(
                sequences=sequences, stop_i=stop_i
            )
        seq_tokens = [token for seq in sequences for token in seq]
        seq_label = " ".join([token.label for token in seq_tokens])
        offsets = multiple_seq_to_offsets(sequences=sequences)
        seq_offsets = get_brat_format_seq(offsets)
        return seq_label, seq_offsets


class SpanFormatter(IBratFormatter):
    """A simple Brat formatter that only uses start,end offsets
    of an annotation"""

    def __init__(self, text: str):
        """Create a brat formatter.

        :param text: the document of the annotation.
        """
        self.text = text

    def get_text_and_offsets(self, annot: IAnnotation) -> Tuple[str, str]:
        """Return text, offsets by start and end offsets of the annotation."""
        seq_label = self.text[annot.start : annot.end]  # noqa
        seq_offsets = f"{annot.start} {annot.end}"
        return seq_label, seq_offsets
