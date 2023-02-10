""" Utility classes and methods for fuzzy algorithms."""
from typing import Iterable
from typing import Optional
from typing import Set

from typing_extensions import runtime_checkable, Protocol


@runtime_checkable
class IWords2ignore(Protocol):
    """Utility class to check if a word must be ignored by a fuzzy algorithm.
    This interface is very similar to the IStopword but its meaning is
    different.
    """

    def is_word_2_ignore(self, word: str) -> bool:
        """True if the word must be ignored."""
        raise NotImplementedError


class SimpleWords2ignore(IWords2ignore):
    """Utility class to store in memory the words that must be ignored."""

    def __init__(self, words: Optional[Iterable[str]] = None):
        """Create an instance to be shared by context independant fuzzy
        algorithms.

        :param: words: words to be ignored.
        """
        self._words2ignore: Set[str] = set()
        if words is not None:
            self._words2ignore.update(words)

    def is_word_2_ignore(self, word: str) -> bool:
        """Return true if word is known."""
        return word in self._words2ignore

    def add_word(self, word: str):
        """Add a word to be ignored by a string distance algorithm."""
        self._words2ignore.add(word)
