""" Spellwise library wrapper."""
import warnings

from enum import Enum
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

from spellwise import CaverphoneOne
from spellwise import CaverphoneTwo
from spellwise import Editex
from spellwise import Levenshtein
from spellwise import Soundex
from spellwise import Typox
from typing_extensions import Protocol
from typing_extensions import TypedDict

from iamsystem.fuzzy.api import FuzzyAlgo
from iamsystem.fuzzy.api import StringDistance
from iamsystem.fuzzy.api import SynType
from iamsystem.fuzzy.util import IWords2ignore


class Suggestions(TypedDict):
    """Spellwise algorithm's output."""

    word: str
    distance: int


class ESpellWiseAlgo(Enum):
    """Enumerated list of spellwise library algorithms.
    See spellwise documentation for more information.
    """

    LEVENSHTEIN = Levenshtein
    SOUNDEX = Soundex
    EDITEX = Editex
    TYPOX = Typox
    CAVERPHONE_1 = CaverphoneOne
    CAVERPHONE_2 = CaverphoneTwo


class ISpellWiseAlgo(Protocol):
    """Spellwise algorithm interface."""

    def add_words(self, words: List[str]) -> None:
        """Add words to a spellwise algorithm."""
        pass

    def get_suggestions(
        self, query_word: str, max_distance: int = 2
    ) -> List[Suggestions]:
        """Returns synonyms."""
        pass


class SpellWiseWrapper(StringDistance):
    """A :class:`~iamsystem.FuzzyAlgo` that wraps an algorithm from
    the spellwise library."""

    def __init__(
        self,
        measure: Union[str, ESpellWiseAlgo],
        max_distance: int,
        min_nb_char=5,
        words2ignore: Optional[IWords2ignore] = None,
        name: str = None,
    ):
        """Create an instance to take advantage of a spellwise algorithm.

        :param measure: The measure string or a value selected
            from :class:`~iamsystem.SpellWiseAlgo` enumerated list.
        :param max_distance: maximum edit distance
          (see spellwise documentation).
        :param min_nb_char: the minimum number of characters a word
          must have in order not to be ignored.
        :param words2ignore: words that must be ignored by the algorithm to
            avoid false positives, for example English vocabulary words.
        :param name: a name given to this algorithm.
          Default: spellwise algorithm's name.
        """
        if isinstance(measure, str):
            measure = ESpellWiseAlgo[measure.upper()]
        if name is None:
            name = measure.name
        super().__init__(
            name=name, min_nb_char=min_nb_char, words2ignore=words2ignore
        )
        self._suggester: ISpellWiseAlgo = measure.value()
        self._max_distance = max_distance

    @property
    def max_distance(self):
        """Maximum edit distance (see spellwise documentation)."""
        return self._max_distance

    @max_distance.setter
    def max_distance(self, value: int):
        """Set the maximum edit distance."""
        self._max_distance = value

    def add_words(self, words: Iterable[str], warn=False) -> None:
        """A list of possible word synonyms, in general all the tokens
        of your keywords. An easy way to provide these tokens is to call
        :py:meth:`~iamsystem.Matcher.get_keywords_unigrams` method after
        you added your keywords to the matcher instance.

        :param words: A list of possible synonyms.
        :param warn: raise a warning if a word added is ignored.
          Default False.
        :return: None.
        """
        words = list(words)
        words_filtered = [
            word for word in words if not len(word) < self.min_nb_char
        ]
        n_removed = len(words) - len(words_filtered)
        if n_removed != 0 and warn:
            warnings.warn(
                f"{n_removed} words weren't added to fuzzy algo '{self.name}'"
                f" after filtering"
            )
        self._suggester.add_words(words=words_filtered)

    def get_syns_of_word(self, word: str) -> Iterable[SynType]:
        """Compute string distance if it is not a word to be ignored
        and return keywords' unigrams in the maximum distance
        from that word."""
        if self._is_a_word_to_ignore(word):
            return FuzzyAlgo.NO_SYN
        suggs: List[Suggestions] = self._suggester.get_suggestions(
            query_word=word, max_distance=self._max_distance
        )
        if len(suggs) == 0:
            return FuzzyAlgo.NO_SYN
        return [self.word_to_syn(sugg.get("word", "")) for sugg in suggs]
