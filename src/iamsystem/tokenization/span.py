""" Classes that store a sequence of tokens. """
from typing import Sequence

from iamsystem.brat.util import merge_offsets_and_get_brat_format
from iamsystem.tokenization.api import ISpan
from iamsystem.tokenization.api import IToken
from iamsystem.tokenization.api import TokenT
from iamsystem.tokenization.util import concat_tokens_label
from iamsystem.tokenization.util import concat_tokens_norm_label
from iamsystem.tokenization.util import get_max_end_offset
from iamsystem.tokenization.util import get_min_start_offset
from iamsystem.tokenization.util import get_span_seq_id
from iamsystem.tokenization.util import offsets_overlap


class Span(ISpan[TokenT], IToken):
    """A class that represents a sequence of tokens in a document.
    A sequence of tokens is also seen as a single Token,
    thus it implements IToken."""

    def __init__(self, tokens: Sequence[TokenT]):
        """Create a Span.

        :param tokens: a sequence of TokenT, a generic type that implements
            :class:`~iamsystem.IToken` protocol.
        """
        self._tokens = tokens
        self.start = get_min_start_offset(self._tokens)
        self.end = get_max_end_offset(self._tokens)
        self.label = concat_tokens_label(self._tokens)
        self.norm_label = concat_tokens_norm_label(self._tokens)

    @property
    def tokens(self) -> Sequence[TokenT]:
        """The tokens of the document that matched the keywords attribute of
        this instance.

        :return: an ordered sequence of TokenT, a generic type
            that implements :class:`~iamsystem.IToken`.
        """
        return self._tokens

    def to_brat_format(self) -> str:
        """Get Brat offsets format. See https://brat.nlplab.org/standoff.html
        'The start-offset is the index of the first character of the annotated
        span in the text (".txt" file),
        i.e. the number of characters in the document preceding it.
        The end-offset is the index of the first character
        after the annotated span.'

        :return: a string format of tokens' offsets
        """
        return merge_offsets_and_get_brat_format(self._tokens)

    def __str__(self):
        """A dataclass string representation."""
        return (
            f"Span(label='{self.label}', norm_label='{self.norm_label}', "
            f"start={self.start}, end={self.end})"
        )


def is_shorter_span_of(a: Span, b: Span) -> bool:
    """True if a is the shorter span of b."""
    if a is b:
        return False
    if not offsets_overlap(a=a, b=b):
        return False
    # if both conditions are true then we can't decide which to remove so it
    # returns False. Ex: 'IRC' abbreviation is matched to two long forms
    # that have the same offsets.
    if a.start == b.start and a.end == b.end:
        return False
    # b_seq_id must contain all offsets of a_seq_id, for example:
    # 1) left: 'lung cancer' and 'lung'
    # 2) right: 'prostate cancer' and 'cancer'
    # 3) middle: 'prostate cancer undetermined' and 'cancer'
    a_seq_id = get_span_seq_id(a.tokens)
    b_seq_id = get_span_seq_id(b.tokens)
    return a_seq_id in b_seq_id
