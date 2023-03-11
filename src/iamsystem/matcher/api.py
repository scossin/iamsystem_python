""" Matcher's API."""
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from typing_extensions import Protocol
from typing_extensions import runtime_checkable

from iamsystem.fuzzy.api import ISynsProvider
from iamsystem.keywords.api import IKeyword
from iamsystem.stopwords.api import IStopwords
from iamsystem.tokenization.api import IOffsets
from iamsystem.tokenization.api import ISpan
from iamsystem.tokenization.api import ITokenizer
from iamsystem.tokenization.api import TokenT
from iamsystem.tree.api import IInitialState
from iamsystem.tree.nodes import INode


@runtime_checkable
class IAnnotation(ISpan, IOffsets, Protocol[TokenT]):
    """Declare the attributes and methods expected by an Annotation."""

    @property
    def algos(self) -> List[List[str]]:
        """For each token, the list of algorithms that matched.
        One to several algorithms per token."""
        raise NotImplementedError

    @property
    def stop_tokens(self) -> List[TokenT]:
        """The list of stopwords tokens inside the annotation detected by
        the Matcher stopwords instance"""
        raise NotImplementedError

    @property
    def text(self) -> Optional[str]:
        """Return the annotated text."""
        raise NotImplementedError

    @text.setter
    def text(self, value) -> Optional[str]:
        """Set the annotated text."""
        raise NotImplementedError

    @property
    def keywords(self) -> Sequence[IKeyword]:
        """Keywords linked to this annotation."""
        raise NotImplementedError

    @property
    def to_string(self) -> str:
        """A string representation of an annotation."""
        raise NotImplementedError


@runtime_checkable
class IBaseMatcher(Protocol[TokenT]):
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


class IBratFormatter(Protocol):
    """A BratFormatter takes an annotation and decides how to annotate a
    document with the (discontinuous) sequence of tokens,
    thus how to generate a Brat format."""

    def get_text_and_offsets(self, annot: IAnnotation) -> Tuple[str, str]:
        """Return text (document substring) and annotation's offsets in the
            Brat format.

        :param annot: an annotation.
        :return: A text span and its offsets:
          'The start-offset is the index of the first character of the
          annotated span in the text (".txt" file), i.e. the number of
          characters in the document preceding it. The end-offset is the index
          of the first character after the annotated span.'
        """
        raise NotImplementedError


@runtime_checkable
class IMatchingStrategy(Protocol[TokenT]):
    """Declare what a matching strategy must implement."""

    def detect(
        self,
        tokens: Sequence[TokenT],
        w: int,
        initial_state: INode,
        syns_provider: ISynsProvider,
        stopwords: IStopwords,
    ) -> List[IAnnotation[TokenT]]:
        """Main internal function that implements iamsystem's algorithm.

        :param tokens: a sequence of :class:`~iamsystem.IToken`.
        :param w: window, how many previous tokens can the algorithm look at.
        :param initial_state: a node/state in the trie, i.e. the root node.
        :param syns_provider: a class that provides synonyms for each token.
        :param stopwords: an instance of :class:`~iamsystem.IStopwords`
        that checks if a token is a stopword.
        :return: A list of :class:`~iamsystem.Annotation`.
        """
        raise NotImplementedError
