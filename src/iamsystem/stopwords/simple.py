""" Basic stopwords implementation. """
from typing import Iterable
from typing import Optional
from typing import Set

from iamsystem.stopwords.api import SimpleStopwords
from iamsystem.tokenization.api import TokenT


class Stopwords(SimpleStopwords[TokenT]):
    """A simple implementation of :class:`~iamsystem.IStopwords` protocol."""

    def __init__(self, stopwords: Optional[Iterable[str]] = None):
        """Create a Stopword instance to store stopwords.

        :param stopwords: a set of stopwords. Default to None.
        """
        self._stopwords: Set[str] = set()
        if stopwords is not None:
            self._stopwords.update(stopwords)

    @property
    def stopwords(self):
        """Get the set of stopwords."""
        return self._stopwords

    def is_stopword(self, word: str) -> bool:
        """True if, after lowercasing, the word belongs to the stopwords set"""
        word = word.lower()
        return word.isspace() or word in self._stopwords

    def add(self, words: Iterable[str]) -> None:
        """Add stopwords.

        :param words: a list of string.
        :return: None
        """
        self._stopwords.update(words)


class NoStopwords(SimpleStopwords[TokenT]):
    """Utility class. Class to use when no stopwords are used."""

    def is_token_a_stopword(self, token: TokenT) -> bool:
        """Return False."""
        return False

    def is_stopword(self, word: str) -> bool:
        """Return False."""
        return False
