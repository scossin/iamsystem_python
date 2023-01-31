""" Module interface. """
from abc import abstractmethod
from typing import Sequence
from typing import TypeVar

from typing_extensions import Protocol
from typing_extensions import runtime_checkable


@runtime_checkable
class IOffsets(Protocol):
    """Offsets interface.
    Default implementation :class:`~iamsystem.Offsets`.
    """

    start: int
    end: int


@runtime_checkable
class IToken(IOffsets, Protocol):
    """Token interface. Default implementation :class:`~iamsystem.Token`"""

    label: str
    norm_label: str


# a generic token type
TokenT = TypeVar("TokenT", bound=IToken)


@runtime_checkable
class ISpan(Protocol[TokenT]):
    """Span interface: a class that stores a sequence of tokens.
    Default implementation :class:`~iamsystem.Span`.
    """

    @property
    @abstractmethod
    def tokens(self) -> Sequence[TokenT]:
        """Get the sequence of tokens."""
        raise NotImplementedError


@runtime_checkable
class ITokenizer(Protocol[TokenT]):
    """Tokenizer Interface.
    Default implementation :class:`~iamsystem.TokenizerImp`.
    """

    def tokenize(self, text: str) -> Sequence[TokenT]:
        """Tokenize a string.

        :param text: an unormalized string.
        :return: A sequence of generic type (TokenT) that
            implements :class:`~iamsystem.IToken` protocol.
        """
        pass
