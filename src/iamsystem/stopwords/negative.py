""" Stores the words to keep. Unknown words are stopwords by default."""
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set

from iamsystem.stopwords.api import IStopwords
from iamsystem.tokenization.api import TokenT


is_a_word_to_keep = Callable[[TokenT], bool]


class NegativeStopwords(IStopwords[TokenT]):
    """Like a negative image (a total inversion, in which light areas appear
    dark and vice versa), every token is a stopword until proven otherwise."""

    def __init__(self, words_to_keep: Optional[Iterable[str]] = None):
        """Create a NegativeStopwords instance to store words to keep
        and/or define functions that check if a word should be kept.

        :param words_to_keep: a set of words not to ignore.
        """
        self._words_to_keep_funs: List[is_a_word_to_keep] = []
        if words_to_keep is not None:
            self.words_to_keep: Set[str] = set(words_to_keep)
        else:
            self.words_to_keep = set()

    def add_words(self, words_to_keep: Iterable[str]) -> None:
        """Add words not to be ignored.

        :param words_to_keep: a list of string.
        :return: None
        """
        self.words_to_keep.update(words_to_keep)

    def add_fun_is_a_word_to_keep(self, fun: is_a_word_to_keep) -> None:
        """Add a function that checks if a word should be kept.

        :param fun: a Callable that takes a token as a parameter and returns
            a boolean.
        :return: None.
        """
        self._words_to_keep_funs.append(fun)

    def is_token_a_stopword(self, token: TokenT) -> bool:
        """Check if it's not token to keep.

        :param token: a token.
        :return: False if the token's lowercase belongs to the set of word to
            keep or if a function :py:meth:`add_fun_is_a_word_to_keep` returns
            True.
        """
        word = token.label.lower()
        fun_want_to_keep_it = any(
            [
                is_a_word_to_keep(token)
                for is_a_word_to_keep in self._words_to_keep_funs
            ]
        )
        is_a_word_to_keep = fun_want_to_keep_it or word in self.words_to_keep
        return not is_a_word_to_keep
