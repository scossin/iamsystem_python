""" Fuzzy algoriths abstract base classes."""
import warnings

from abc import ABC
from abc import abstractmethod
from typing import Generic
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from typing_extensions import Protocol
from typing_extensions import runtime_checkable

from iamsystem.fuzzy.util import IWords2ignore
from iamsystem.fuzzy.util import SimpleWords2ignore
from iamsystem.matcher.util import IState
from iamsystem.tokenization.api import TokenT
from iamsystem.tree.nodes import INode


# Synonym type. Ex: ('insuffisance','cardiaque')
SynType = Tuple[str, ...]

# Synonym type with an algorithm. Ex: (('insuffisance','cardiaque'), 'exact')
SynAlgo = Tuple[SynType, str]

# Synonym type with algorithms. Ex: (('insuffisance','cardiaque'), ['exact',...
SynAlgos = Tuple[SynType, List[str]]


@runtime_checkable
class ISynsProvider(Protocol[TokenT]):
    """Provides all the synonyms coming from fuzzy algorithms."""

    @abstractmethod
    def get_synonyms(
        self, tokens: Sequence[TokenT], i: int, w_states: List[List[IState]]
    ) -> Iterable[SynAlgos]:
        """Retrieve the synonyms of a token.

        :param tokens: the sequence of tokens of the document.
            Useful when the fuzzy algorithm needs context, namely the tokens
            around the token of interest given by 'i' parameter.
        :param i: the ith token of this sequence for
            which synonyms are expected.
        :param w_states: the states in which the algorithm currently is.
            Useful is the fuzzy algorithm needs to know the current states
            and the possible state transitions.
        :return: 0 to many synonyms.
        """
        raise NotImplementedError


class FuzzyAlgo(Generic[TokenT], ABC):
    """Fuzzy Algorithm base class."""

    NO_SYN: Iterable[SynType] = []  #
    "Default value to return by a fuzzy algorithm if no synonym found."

    def __init__(self, name: str):
        """Create a fuzzy algorithm to allow a partial match between
        a text token and a keyword token.

        :param name: algorithm's name.
        """
        self.name = name

    @staticmethod
    def word_to_syn(word: str) -> SynType:
        """Utility function to transform a string to expected SynType.

        :param word: a word synonym produced by the algorithm.
            Ex: word='insuffisance' for token 'ins'.
        :return: SynType, the expected output format.
        """
        return tuple([word])

    @staticmethod
    def words_seq_to_syn(words: Sequence[str]) -> SynType:
        """Utility function to transform a sequence of string
        to the expected output type.

        :param words: a sequence of words produced by the algorithm.
         Ex: words=['insuffisance', 'cardiaque'] for the token 'ic'.
        :return: SynType, the expected output format.
        """
        return tuple(words)

    @abstractmethod
    def get_synonyms(
        self, tokens: Sequence[TokenT], i: int, w_states: List[List[IState]]
    ) -> Iterable[SynAlgo]:
        """Main API function to retrieve all synonyms provided by
        a fuzzy algorithm.

        :param tokens: the sequence of tokens of the document.
            Useful when the fuzzy algorithm needs context, namely the tokens
            around the token of interest given by 'i' parameter.
        :param i: the ith token of this sequence for
            which synonyms are expected.
        :param w_states: the states in which the algorithm currently is.
            Useful is the fuzzy algorithm needs to know the current states
            and the possible state transitions.
        :return: 0 to many synonyms (SynAlgo type).
        """
        raise NotImplementedError


def get_possible_transitions(w_states: List[List[IState]]) -> Iterable[INode]:
    """Return all the states (nodes) where the algorithm can go."""
    for states in w_states:
        for state in states:
            for child_nodes in state.node.get_child_nodes():
                yield child_nodes


class ContextFreeAlgo(FuzzyAlgo[TokenT], ABC):
    """A :class:`~iamsystem.FuzzyAlgo` that doesn't take into account context,
    only the current token."""

    def __init__(self, name: str):
        super().__init__(name)

    def get_synonyms(
        self, tokens: Sequence[TokenT], i: int, w_states: List[List[IState]]
    ) -> Iterable[SynAlgo]:
        """Delegate to get_syns_of_token."""
        token = tokens[i]
        for syn in self.get_syns_of_token(token=token):
            yield syn, self.name

    @abstractmethod
    def get_syns_of_token(self, token: TokenT) -> Iterable[SynType]:
        """Returns synonyms of this token."""
        pass


class INormLabelAlgo(Protocol):
    """A fuzzy algorithm that relies only on a string."""

    name: str

    @abstractmethod
    def get_syns_of_word(self, word: str) -> Iterable[SynType]:
        """Returns the synonym of this word"""
        raise NotImplementedError


class NormLabelAlgo(ContextFreeAlgo[TokenT], INormLabelAlgo, ABC):
    """A :class:`~iamsystem.FuzzyAlgo`
    that uses only the normalized label of a token.
    These fuzzy algorithms can be put in cache to avoid calling
    them multiple times.
    See :class:`~iamsystem.CacheFuzzyAlgos`.
    """

    def get_syns_of_token(self, token: TokenT) -> Iterable[SynType]:
        """Delegate to get_syns_of_word."""
        return self.get_syns_of_word(word=token.norm_label)

    @abstractmethod
    def get_syns_of_word(self, word: str) -> Iterable[SynType]:
        """Returns synonyms of this word
        (e.g. the normalized label of a token)."""
        raise NotImplementedError


class StringDistance(NormLabelAlgo, ABC):
    """Class that computes a string distance between a token of a document
    and the keywords' tokens."""

    def __init__(
        self,
        name: str,
        min_nb_char: int,
        words2ignore: Optional[IWords2ignore] = None,
    ):

        """Create a string distance fuzzy algorithm.

        :param name: string distance algorithm name.
        :param min_nb_char: the minimum number of characters a word
          must have in order not to be ignored.
        """
        super().__init__(name)
        self._min_nb_char = min_nb_char
        self._tokens2ignore: IWords2ignore
        if words2ignore is None:
            self._tokens2ignore = SimpleWords2ignore()
        else:
            self._tokens2ignore = words2ignore

    @property
    def min_nb_char(self):
        """The minimum number of characters a word must have
        not to be ignored."""
        return self._min_nb_char

    @min_nb_char.setter
    def min_nb_char(self, value: int):
        """Set the minimum number of characters a word must have."""
        self._min_nb_char = value

    def _is_a_word_to_ignore(self, word: str) -> bool:
        """Check if this word must be ignored."""
        return len(
            word
        ) < self._min_nb_char or self._tokens2ignore.is_word_2_ignore(word)

    def add_words_to_ignore(self, words: Iterable[str]):
        """Add words that the algorithm must ignore: no string distance
        will be computed."""
        warnings.warn(
            "Deprecated 'add_words_to_ignore': "
            "pass these words in the constructor."
        )
        for word in words:
            self._tokens2ignore.add_word(word)
