""" Default fuzzy algorithm."""
from typing import Iterable

from iamsystem.fuzzy.api import NormLabelAlgo
from iamsystem.fuzzy.api import SynType
from iamsystem.tokenization.api import TokenT


class ExactMatch(NormLabelAlgo[TokenT]):
    """Default algorithm that returns the normalized token.
    Thus, to find a match, the sequence of tokens of a documuent must match
    perfectly the sequence of tokens of a keyword.
    """

    def __init__(self, name: str = "exact"):
        super().__init__(name)

    def get_syns_of_word(self, word: str) -> Iterable[SynType]:
        """Return this word."""
        return [self.word_to_syn(word=word)]
