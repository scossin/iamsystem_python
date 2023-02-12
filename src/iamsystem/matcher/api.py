""" Matcher's API."""
from typing import List
from typing import Sequence

from typing_extensions import Protocol
from typing_extensions import runtime_checkable

from iamsystem.fuzzy.api import ISynsProvider
from iamsystem.matcher.annotation import Annotation
from iamsystem.stopwords.api import IStopwords
from iamsystem.tokenization.api import ITokenizer
from iamsystem.tokenization.api import TokenT
from iamsystem.tree.api import IInitialState


@runtime_checkable
class IBaseMatcher(Protocol):
    """Declare the API methods expected by a IAMsystem matcher."""

    def annot_text(self, text: str) -> List[Annotation[TokenT]]:
        """Annotate a document with the matcher's tokenizer."""
        raise NotImplementedError

    def annot_tokens(
        self, tokens: Sequence[TokenT]
    ) -> List[Annotation[TokenT]]:
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
