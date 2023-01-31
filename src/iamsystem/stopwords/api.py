""" Stopwords interface."""
from abc import ABC
from typing import Iterable

from typing_extensions import Protocol
from typing_extensions import runtime_checkable

from iamsystem.tokenization.api import TokenT


@runtime_checkable
class IStopwords(Protocol[TokenT]):
    """Stopwords Interface."""

    def is_token_a_stopword(self, token: TokenT) -> bool:
        """Check if a token is a stopword.

        :param token: a generic Token that implements
          :class:`~iamsystem.IToken` protocol.
        :return: true if this token is a stopword.
        """
        raise NotImplementedError


@runtime_checkable
class IStoreStopwords(IStopwords[TokenT], Protocol):
    """A IStopwords that stores stopwords."""

    def add(self, words: Iterable[str]) -> None:
        """Add stopwords."""
        raise NotImplementedError


@runtime_checkable
class ISimpleStopwords(IStopwords[TokenT], Protocol):
    """A IStopwords that checks a stopword by its label."""

    def is_stopword(self, word: str) -> bool:
        """True if the word is a stopword."""
        raise NotImplementedError


class SimpleStopwords(ISimpleStopwords[TokenT], ABC):
    """A Stopword that checks adding stopwords."""

    def is_token_a_stopword(self, token: TokenT) -> bool:
        """Check if a token is a stopword by its label.

        :param token: a generic token.
        :return: True if the label of the token is a stopword.
        """
        word = token.label
        return self.is_stopword(word=word)

    def is_stopword(self, word: str) -> bool:
        """True if the word is a stopword."""
        raise NotImplementedError
