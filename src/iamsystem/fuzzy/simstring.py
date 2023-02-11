""" pysimstring library wrapper."""
import os
import tempfile

from enum import Enum
from typing import Iterable
from typing import Optional
from typing import Union

from pysimstring import simstring

from iamsystem.fuzzy.api import FuzzyAlgo
from iamsystem.fuzzy.api import StringDistance
from iamsystem.fuzzy.api import SynType
from iamsystem.fuzzy.util import IWords2ignore


class ESimStringMeasure(Enum):
    """Enumerated list of simstring measures."""

    EXACT = "exact"
    DICE = "dice"
    COSINE = "cosine"
    JACCARD = "jaccard"
    OVERLAP = "overlap"


class SimStringWrapper(StringDistance):
    """SimString algorithm interface."""

    def __init__(
        self,
        words: Iterable[str],
        measure: Union[str, ESimStringMeasure] = ESimStringMeasure.JACCARD,
        name: str = None,
        threshold=0.5,
        min_nb_char=5,
        words2ignore: Optional[IWords2ignore] = None,
    ):
        """Create a fuzzy algorithm that calls simstring.

        :param words: the words to index in the simstring database.
            An easy way to provide these words is to call
            :py:meth:`~iamsystem.Matcher.get_keywords_unigrams`.
        :param name: a name given to this algorithm. Default measure name.
        :param measure: a similarity measure string or selected from
            :class:`~iamsystem.fuzzy.simstring.ESimStringMeasure`.
            Default JACCARD.
        :param threshold: similarity measure threshold.
        :param min_nb_char: the minimum number of characters a word
          must have in order not to be ignored.
        :param words2ignore: words that must be ignored by the algorithm to
            avoid false positives, for example English vocabulary words.
        """
        # If an error occured during initialization, file is not opened
        # the __del__ is called and return an error
        # I add this attribute to check if the file was opened before trying
        # to close it.
        self.__file_is_open = False
        if isinstance(measure, str):
            measure = ESimStringMeasure[measure.upper()]
        if name is None:
            name = measure.name
        super().__init__(
            name=name, min_nb_char=min_nb_char, words2ignore=words2ignore
        )
        self.path = tempfile.mkdtemp()
        os.makedirs(self.path, exist_ok=True)
        abs_path = os.path.join(self.path, "terms.simstring")
        with SimstringWriter(abs_path=abs_path) as ss_db:
            for word in words:
                ss_db.insert(word)
        self.ss_reader = simstring.reader(abs_path)
        self.__file_is_open = True
        self.ss_reader.measure = getattr(simstring, measure.value)
        self.ss_reader.threshold = threshold

    def get_syns_of_word(self, word: str) -> Iterable[SynType]:
        """Retrieve simstring similar words."""
        if self._is_a_word_to_ignore(word):
            return FuzzyAlgo.NO_SYN
        ss_words = self.ss_reader.retrieve(word)
        return [self.word_to_syn(word=word) for word in ss_words]

    def __del__(self):
        """close the file connection to simstring db."""
        # The safer approach is to open the file for every 'get_syns_of_word'
        # call. However, it takes more time. It seems to be ok to close
        # the file here.
        # https://stackoverflow.com/questions/44142836/open-file-inside-class
        if self.__file_is_open:
            self.ss_reader.close()


class SimstringWriter:
    """Utility class to create a simstring database.
    a plagiarism of https://github.com/percevalw/pysimstring/blob/master/tests/test_simstring.py # noqa
    """

    def __init__(self, abs_path: str):
        """A context class to write a simstring database

        :param abs_path: absolute path to the file.
        """
        self.abs_path = abs_path

    def __enter__(self):
        """Open the file"""
        self.db = simstring.writer(self.abs_path, 3, False, True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the file"""
        self.db.close()

    def insert(self, term):
        """insert a term in simstring file."""
        self.db.insert(term)
