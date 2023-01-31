""" Matcher's API."""
from typing import List
from typing import Sequence

from typing_extensions import Protocol
from typing_extensions import runtime_checkable

from iamsystem.fuzzy.api import ISynsProvider
from iamsystem.keywords.api import IStoreKeywords
from iamsystem.matcher.annotation import Annotation
from iamsystem.stopwords.api import IStopwords
from iamsystem.tokenization.api import ITokenizer
from iamsystem.tokenization.api import TokenT


@runtime_checkable
class IMatcher(
    ISynsProvider[TokenT],
    IStopwords[TokenT],
    ITokenizer[TokenT],
    IStoreKeywords,
    Protocol,
):
    """Declare the API methods expected by a matcher."""

    def annot_text(self, text: str, w: int = 1) -> List[Annotation[TokenT]]:
        """Annotate a document."""
        raise NotImplementedError

    def annot_tokens(
        self, tokens: Sequence[TokenT], w: int
    ) -> List[Annotation[TokenT]]:
        """Annotate a sequence of tokens."""
        raise NotImplementedError
