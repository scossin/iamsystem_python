import unittest

from iamsystem.fuzzy.api import FuzzyAlgo
from iamsystem.fuzzy.spellwise import ESpellWiseAlgo
from iamsystem.fuzzy.spellwise import SpellWiseWrapper
from iamsystem.stopwords.simple import Stopwords
from iamsystem.tokenization.tokenize import french_tokenizer
from tests.utils import get_termino_ivg


class SpellWiseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.termino = get_termino_ivg()
        self.tokenizer = french_tokenizer()
        self.unigrams = list(
            self.termino.get_unigrams(
                tokenizer=self.tokenizer, stopwords=Stopwords()
            )
        )
        self.tuple_ins = tuple(["insuffisance"])
        self.leven = self.get_algo(ESpellWiseAlgo.LEVENSHTEIN)

    def get_algo(self, algo_name: ESpellWiseAlgo):
        """Utility function to build an algorithm with unigrams."""
        self.algo: SpellWiseWrapper = SpellWiseWrapper(
            algo_name, max_distance=1
        )
        self.algo.add_words(words=self.unigrams)
        return self.algo

    def test_algo_name(self):
        """Returns the Enum value by default."""
        self.assertEqual("LEVENSHTEIN", self.leven.name)

    def test_levenshtein_get_synonyms(self):
        """a 'f' is missing in 'insuffisance'."""
        syns = self.leven.get_syns_of_word("insufisance")
        self.assertTrue(self.tuple_ins in syns)

    def test_levenshtein_get_synonyms_max_2(self):
        """two letters are missing in 'insuffisance'."""
        self.leven._max_distance = 2
        syns = self.leven.get_syns_of_word("insuisance")
        self.assertTrue(self.tuple_ins in syns)

    def test_no_suggestion(self):
        """Check it returns the empty synonym."""
        syns = self.leven.get_syns_of_word("nothing_close_to_this")
        self.assertIs(syns, FuzzyAlgo.NO_SYN)

    def test_min_nb_char_wrong_way(self):
        """a warning is generated if the user tries to a word containing
        less than 5 characters (defalt nchar value)."""
        with (self.assertWarns(Warning)):
            self.leven.add_words(words=["word"], warn=True)
        syns = self.leven.get_syns_of_word("word")
        self.assertIs(syns, FuzzyAlgo.NO_SYN)

    def test_min_nb_char_correct_way(self):
        """Set the 'min_nb_char' value before adding a word less than 5.
        No warning raise check."""
        self.leven.min_nb_char = -1
        self.leven.add_words(words=["word"])
        syns = self.leven.get_syns_of_word("word")
        self.assertTrue(tuple(["word"]) in syns)

    def test_words_to_ignore(self):
        """If a word is ignored then the algorithm returns nothing."""
        self.leven.add_words_to_ignore(words=["word"])
        self.leven.add_words(words=["word"])
        syns = self.leven.get_syns_of_word("word")
        self.assertIs(syns, FuzzyAlgo.NO_SYN)

    def test_soundex(self):
        """Sounds like 'insuffisance'."""
        soundex = self.get_algo(ESpellWiseAlgo.SOUNDEX)
        syns = soundex.get_syns_of_word("inssssuffffizzzensssse")
        self.assertTrue(self.tuple_ins in syns)

    def test_editex(self):
        """Close to 'insuffisance'."""
        editex = self.get_algo(ESpellWiseAlgo.EDITEX)
        syns = editex.get_syns_of_word("inssssuffffizzzensssse")
        self.assertTrue(self.tuple_ins not in syns)
        syns = editex.get_syns_of_word("insufizzance")
        self.assertTrue(self.tuple_ins in syns)


if __name__ == "__main__":
    unittest.main()
