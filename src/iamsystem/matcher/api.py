""" Matcher's API."""
from typing import List
from typing import Sequence
from typing import Tuple

from typing_extensions import Protocol
from typing_extensions import runtime_checkable

from iamsystem.fuzzy.api import ISynsProvider
from iamsystem.stopwords.api import IStopwords
from iamsystem.tokenization.api import IOffsets
from iamsystem.tokenization.api import ISpan
from iamsystem.tokenization.api import ITokenizer
from iamsystem.tokenization.api import TokenT
from iamsystem.tree.api import IInitialState


class IBratFormatter(Protocol):
    """A BratFormatter takes an annotation and decides how to annotate a
    document with the (discontinuous) sequence of tokens,
    thus how to generate a Brat format."""

    def get_text_and_offsets(self, annot: "IAnnotation") -> Tuple[str, str]:
        """Return text (document substring) and annotation's offsets in the
        Brat format"""
        raise NotImplementedError


@runtime_checkable
class IAnnotation(ISpan, IOffsets, Protocol[TokenT]):
    """Declare the attributes and methods expected by an Annotation."""

    @property
    def stop_tokens(self) -> List[TokenT]:
        """Access brat formatter."""
        raise NotImplementedError

    @property
    def brat_formatter(self) -> IBratFormatter:
        """Access brat formatter."""
        raise NotImplementedError

    @property
    def keywords(self):  #
        """Keywords linked to this annotation."""
        raise NotImplementedError


@runtime_checkable
class IBaseMatcher(Protocol):
    """Declare the API methods expected by a IAMsystem matcher."""

    def annot_text(self, text: str) -> List[IAnnotation[TokenT]]:
        """Annotate a document with the matcher's tokenizer."""
        raise NotImplementedError

    def annot_tokens(
        self, tokens: Sequence[TokenT]
    ) -> List[IAnnotation[TokenT]]:
        """Annotate a document passing its tokens produced by an external
        tokenizer."""
        raise NotImplementedError


@runtime_checkable
class IMatcher(
    ISynsProvider[TokenT],
    IStopwords[TokenT],
    ITokenizer[TokenT],
    IInitialState,
    IBaseMatcher,
    Protocol,
):
    """Declare the API of the concrete matcher implementation."""

    pass
