""" pysimstring library wrapper."""
import os
import tempfile

from enum import Enum
from typing import Iterable

from pysimstring import simstring

from iamsystem.fuzzy.api import NormLabelAlgo
from iamsystem.fuzzy.api import SynType


class ESimStringMeasure(Enum):
    """Enumerated list of simstring measures."""

    EXACT = "exact"
    DICE = "dice"
    COSINE = "cosine"
    JACCARD = "jaccard"
    OVERLAP = "overlap"


class SimStringWrapper(NormLabelAlgo):
    """SimString algorithm interface."""

    def __init__(
        self,
        words=Iterable[str],
        name: str = "simstring",
        measure=ESimStringMeasure.JACCARD,
        threshold=0.5,
    ):
        """Create a fuzzy algorithm that calls simstring.

        :param words: the words to index in the simstring database.
            An easy way to provide these words is to call
            :py:meth:`~iamsystem.Matcher.get_keywords_unigrams` method after
            you added your keywords to the matcher instance.
        :param name: a name given to this algorithm. Default "simstring".
        :param measure: a similarity measure selected from
            :class:`~iamsystem.fuzzy.simstring.ESimStringMeasure`.
            Default JACCARD.
        :param threshold: similarity measure threshold.
        """
        super().__init__(name=name)
        self.path = tempfile.mkdtemp()
        os.makedirs(self.path, exist_ok=True)
        abs_path = os.path.join(self.path, "terms.simstring")
        with SimstringWriter(abs_path=abs_path) as ss_db:
            for word in words:
                ss_db.insert(word)
        self.ss_reader = simstring.reader(abs_path)
        self.ss_reader.measure = getattr(simstring, measure.value)
        self.ss_reader.threshold = threshold

    def get_syns_of_word(self, word: str) -> Iterable[SynType]:
        """Retrieve simstring similar words."""
        ss_words = self.ss_reader.retrieve(word)
        return [self.word_to_syn(word=word) for word in ss_words]

    def __del__(self):
        """close the file connection to simstring db."""
        # The safer approach is to open the file for every 'get_syns_of_word'
        # call. However, it takes more time. It seems to be ok to close
        # the file here.
        # https://stackoverflow.com/questions/44142836/open-file-inside-class
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
