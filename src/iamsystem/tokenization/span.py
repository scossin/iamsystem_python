""" Classes that store a sequence of tokens. """
from typing import List

from iamsystem.tokenization.api import IOffsets
from iamsystem.tokenization.api import ISpan
from iamsystem.tokenization.api import TokenT
from iamsystem.tokenization.util import concat_tokens_label
from iamsystem.tokenization.util import concat_tokens_norm_label
from iamsystem.tokenization.util import get_span_seq_id
from iamsystem.tokenization.util import offsets_overlap


class Span(ISpan[TokenT], IOffsets):
    """A class that represents a sequence of tokens in a document."""

    def __init__(self, tokens: List[TokenT]):
        """Create a Span.

        :param tokens: an ordered continuous or discontinuous sequence
            of TokenT in a document.
        """
        self._tokens = tokens
        """The start offset of the first token."""
        self.start = self.tokens[0].start
        """The start offset of the first token."""
        self.end = self.tokens[-1].end

    @property
    def tokens(self) -> List[TokenT]:
        """The tokens of the document that matched the keywords attribute of
        this instance.

        :return: an ordered sequence of TokenT, a generic type
            that implements :class:`~iamsystem.IToken`.
        """
        return self._tokens

    @property
    def start_i(self):
        """The index of the first token within the parent document."""
        return self.tokens[0].i

    @property
    def end_i(self):
        """The index of the last token within the parent document."""
        return self.tokens[-1].i

    @property
    def tokens_label(self):
        """The concatenation of each token's label."""
        return concat_tokens_label(self._tokens)

    @property
    def tokens_norm_label(self):
        """The concatenation of each token's norm_label."""
        return concat_tokens_norm_label(self._tokens)

    def get_text_substring(self, text: str) -> str:
        """Return text substring."""
        return text[self.start : self.end]  # noqa

    def __str__(self):
        """A dataclass string representation."""
        return (
            f"Span(tokens_label='{self.tokens_label}', "
            f"tokens_norm_label='{self.tokens_norm_label}',"
            f"start_i={self.start_i}, end_i={self.end_i}, "
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
