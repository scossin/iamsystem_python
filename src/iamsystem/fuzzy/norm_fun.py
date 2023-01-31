""" Fuzzy algorithm that expects a normalization function."""
from collections import defaultdict
from typing import Dict
from typing import Iterable
from typing import List

from iamsystem.fuzzy.api import FuzzyAlgo
from iamsystem.fuzzy.api import NormLabelAlgo
from iamsystem.fuzzy.api import SynType
from iamsystem.tokenization.normalize import normalizeFun


class WordNormalizer(NormLabelAlgo):
    """A :class:`~iamsystem.FuzzyAlgo` to handle normalization techniques
    such as stemming and lemmatization."""

    def __init__(self, name: str, norm_fun: normalizeFun):
        """Create an instance that will store the normalized tokens of a
        set of :class:`~iamsystem.IKeyword`.

        :param name: a name given to this algorithm (ex: 'english stemmer').
        :param norm_fun: a normalizing function, for example a stemming
         function or lemmatization function.
        """
        super().__init__(name)
        self.norm_fun = norm_fun
        self.words: Dict[str, List[SynType]] = defaultdict(list)

    def add_words(self, words: Iterable[str]) -> None:
        """A list of possible word synonyms, in general all the tokens
        of your keywords. An easy way to provide these tokens is to call
        :py:meth:`~iamsystem.Matcher.get_keywords_unigrams` of the matcher.

        :param words: A list of words to normalize and store.
        :return: None.
        """
        for word in words:
            word_norm = self.norm_fun(word)
            syn = self.word_to_syn(word)
            self.words[word_norm].append(syn)

    def get_syns_of_word(self, word: str) -> Iterable[SynType]:
        """Return all the words that have the same normalized form of this word

        For example, if the normalize function is an english stemmer, and
        you provided add_words=["eating"], this instance stored the stem "eat"
        associated to the word "eating". Then, if a document contains the
        token "eats", since the stem is the same, this function returns
        the synonym "eating".

        :param word: a string, i.e. a word from a document.
        :return: word synonyms and algorithm name.
        """
        word_norm = self.norm_fun(word)
        return self.words.get(word_norm, FuzzyAlgo.NO_SYN)
