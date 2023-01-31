""" Tokenization outputs."""

from iamsystem.tokenization.api import IOffsets
from iamsystem.tokenization.api import IToken


class Offsets(IOffsets):
    """Store the start and end offsets of a token."""

    def __init__(self, start: int, end: int):
        """
        :param start: start-offset is the index of the first character
            of the annotated span.
        :param end: end-offset is the index of the first character
            **after** the annotated span.
        """
        self.start = start
        self.end = end

    def __str__(self):
        """A dataclass string representation."""
        return f"Offsets(start={self.start}, end={self.end})"


class Token(Offsets, IToken):
    """Store the label, normalized label, start and end offsets of a token."""

    def __init__(self, start: int, end: int, label: str, norm_label: str):
        """Create a token.

        :param start: start-offset is the index of the first character
          of the annotated span.
        :param end: end-offset is the index of the first character
          after the annotated span.
        :param label: the label as it is in the document.
        :param norm_label: the normalized label (used by iamsystem's algorithm
            to perform entity linking).
        """
        super().__init__(start, end)
        self.label = label
        self.norm_label = norm_label

    def __str__(self):
        """A dataclass string representation."""
        return (
            f"Token(label='{self.label}', norm_label='{self.norm_label}',"
            f" start={self.start}, end={self.end})"
        )
