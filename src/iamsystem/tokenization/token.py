""" Tokenization outputs."""

from iamsystem.tokenization.api import IOffsets
from iamsystem.tokenization.api import IToken


class Offsets(IOffsets):
    """Store the start and end offsets of a token."""

    def __init__(self, start: int, end: int):
        """
        :param start: start-offset is the index of the first character.
        :param end: end-offset is the index of the last character **+ 1**, that
            is to say the first character to exclude from the returned
            substring when slicing with [start:end]
        """
        self.start = start
        self.end = end

    def __str__(self):
        """A dataclass string representation."""
        return f"Offsets(start={self.start}, end={self.end})"


class Token(Offsets, IToken):
    """Store the label, normalized label, start and end offsets of a token."""

    def __init__(
        self, start: int, end: int, label: str, norm_label: str, i: int
    ):
        """Create a token.

        :param start: start-offset is the index of the first character.
        :param end: end-offset is the index of the last character **+ 1**, that
            is to say the first character to exclude from the returned
            substring when slicing with [start:end]
        :param label: the label as it is in the document/keyword.
        :param norm_label: the normalized label (used by iamsystem's algorithm
            to perform entity linking).
        :param i: the index of the token within the parent document.
        """
        super().__init__(start, end)
        self.label = label
        self.norm_label = norm_label
        self.i = i

    def __str__(self):
        """A dataclass string representation."""
        return (
            f"Token(label='{self.label}', norm_label='{self.norm_label}',"
            f" start={self.start}, end={self.end}, i={self.i})"
        )
