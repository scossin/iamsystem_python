""" A cache to avoid calling some fuzzy algorithms multiple times."""
from __future__ import annotations

from typing import Dict
from typing import Generic
from typing import Iterable
from typing import List
from typing import Sequence

from iamsystem.fuzzy.api import FuzzyAlgo
from iamsystem.fuzzy.api import INormLabelAlgo
from iamsystem.fuzzy.api import SynAlgo
from iamsystem.matcher.util import IState
from iamsystem.tokenization.api import IToken
from iamsystem.tokenization.api import TokenT


class CacheFuzzyAlgos(FuzzyAlgo, Generic[TokenT]):
    """A :class:`~iamsystem.FuzzyAlgo` that provides a cache for
    :class:`~iamsystem.NormLabelAlgo` algorithms.
    Since these algorithms don't depend on context, their output can be cached
    to avoid calling them multiple times.
    """

    def __init__(self, name: str = "Cache"):
        super().__init__(name)
        self.algos: List[INormLabelAlgo] = []
        self.cache: Dict[str, List[SynAlgo]] = {}
        self._max_nb_of_words = 100000

    @property
    def max_nb_of_words(self):
        """The maximum number of words to put in cache.
        Default 100.000 words"""
        return self._max_nb_of_words

    def add_algo(self, algo: INormLabelAlgo) -> None:
        """Add :class:`~iamsystem.NormLabelAlgo`."""
        self.empty_cache()
        self.algos.append(algo)

    def empty_cache(self) -> None:
        """Empty the cache. Done automatically when an algorithm is added."""
        self.cache = {}

    def get_synonyms(
        self, tokens: Sequence[IToken], i: int, w_states: List[List[IState]]
    ) -> List[SynAlgo]:
        """Implements superclass abstract method."""
        token = tokens[i]
        word = token.norm_label
        return self.get_syns_of_word(word=word)

    def get_syns_of_word(self, word: str) -> List[SynAlgo]:
        """Retrieve all synonyms of fuzzy algorithms from cache or by
        calling them once."""
        if word not in self.cache:
            self.cache[word] = get_norm_label_algos_syns(
                fuzzy_algos=self.algos, norm_label=word
            )
        return self.cache[word]


def get_norm_label_algos_syns(
    fuzzy_algos: Iterable[INormLabelAlgo], norm_label: str
) -> List[SynAlgo]:
    """Retrieve all the synonyms."""
    return [
        (syn, algo.name)
        for algo in fuzzy_algos
        for syn in algo.get_syns_of_word(word=norm_label)
    ]
