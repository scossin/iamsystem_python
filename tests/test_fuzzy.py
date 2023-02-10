import unittest

from abc import ABC
from typing import Iterable
from typing import List

from iamsystem.fuzzy.abbreviations import Abbreviations
from iamsystem.fuzzy.abbreviations import token_is_upper_case
from iamsystem.fuzzy.api import NormLabelAlgo
from iamsystem.fuzzy.api import SynAlgo
from iamsystem.fuzzy.api import SynType
from iamsystem.fuzzy.cache import CacheFuzzyAlgos
from iamsystem.fuzzy.cache import get_norm_label_algos_syns
from iamsystem.fuzzy.exact import ExactMatch
from iamsystem.fuzzy.norm_fun import WordNormalizer
from iamsystem.fuzzy.regex import FuzzyRegex
from iamsystem.fuzzy.spellwise import ESpellWiseAlgo
from iamsystem.fuzzy.spellwise import SpellWiseWrapper
from iamsystem.keywords.keywords import Keyword
from iamsystem.matcher.matcher import Matcher
from iamsystem.tokenization.api import TokenT
from iamsystem.tokenization.tokenize import english_tokenizer
from iamsystem.tokenization.tokenize import french_tokenizer
from iamsystem.tokenization.tokenize import split_find_iter_closure
from tests.utils import get_termino_ulceres


class TestFuzzyAlgo(NormLabelAlgo, ABC):
    """A fuzzy algos that tests how many times it is called."""

    def __init__(self, name: str = "test_fuzzy"):
        super().__init__(name)
        self.calls = 0
        self.exact = ExactMatch()

    def get_syns_of_word(self, word: str) -> Iterable[SynType]:
        """exact match and count calls."""
        self.calls += 1
        return self.exact.get_syns_of_word(word)


class InvalidSynonymsReturnedTest(unittest.TestCase):
    def test_fuzzy_algo_returns_string(self):
        """fuzzy algorithms must return a tuple of string,
        ex: ('insuffisance', 'cardiaque'). if the algorithm returns another
        iterable like a list, it must fail fast (exception raise).
        """

        def getSynonymsWrong(token: str):
            return [token]

        fuzzy = TestFuzzyAlgo()
        fuzzy.get_syns_of_word = getSynonymsWrong
        cache_syns = CacheFuzzyAlgos()
        cache_syns.add_algo(fuzzy)
        with (self.assertRaises(TypeError)):
            cache_syns.get_synonyms("anyToken")


class AbbreviationTest(unittest.TestCase):
    def get_first_token_long_forms(self, text: str) -> List[SynType]:
        """Utility function to return the long forms of the first token."""
        tokens = self.tokenizer.tokenize(text=text)
        first_token = tokens[0]
        long_forms: List[SynType] = list(
            self.abbs.get_syns_of_token(first_token)
        )
        return long_forms

    def setUp(self) -> None:
        self.abbs = Abbreviations(
            name="my abbreviations",
            token_is_an_abbreviation=token_is_upper_case,
        )
        self.tokenizer = french_tokenizer()
        self.abbs.add(
            short_form="avc",
            long_form="accident vasculaire cerebral",
            tokenizer=self.tokenizer,
        )

    def test_uppercase(self):
        """An abbreviations instance returns the long form only
        if the token is uppercase."""
        long_forms = self.get_first_token_long_forms(text="AVC sylvien")
        self.assertEqual(1, len(long_forms))

    def test_lowercase(self):
        """Lowercase so no annotation returned (upper only)."""
        long_forms = self.get_first_token_long_forms(text="avc sylvien")
        self.assertEqual(0, len(long_forms))

    def test_custom_token_is_an_abbreviation(self):
        """Modify the function that checks if a token is an abbreviation."""

        def first_letter_is_upper(token: TokenT) -> bool:
            """Return True if the first token's characters is uppercase."""
            return token.label[0].isupper()

        self.abbs.is_token_an_abbreviation = first_letter_is_upper
        long_forms = self.get_first_token_long_forms(text="aVC sylvien")
        self.assertEqual(0, len(long_forms))

        long_forms = self.get_first_token_long_forms(text="Avc sylvien")
        self.assertEqual(1, len(long_forms))


class AbbreviationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.abbs = Abbreviations(
            name="abbs", token_is_an_abbreviation=lambda token: True
        )
        self.tokenizer = french_tokenizer()

    def test_no_syn_returned(self):
        """A global empty tuple is returned by fuzzy algorithms
        when the algorithm has no suggestions"""
        self.assertEqual(len(list(self.abbs.get_syns_of_word("avc"))), 0)

    def test_add(self):
        """Add 2 long forms, check the 2 are returned."""
        self.abbs.add(
            short_form="avc",
            long_form="accident vasculaire cerebral",
            tokenizer=self.tokenizer,
        )
        long_forms: List[SynType] = list(self.abbs.get_syns_of_word("avc"))
        self.assertEqual(1, len(long_forms))
        self.assertTrue(("accident", "vasculaire", "cerebral") in long_forms)
        # another long_form :
        self.abbs.add(
            short_form="avc",
            long_form="another abbreviation",
            tokenizer=self.tokenizer,
        )
        long_forms: Iterable[SynType] = self.abbs.get_syns_of_word("avc")
        self.assertEqual(2, len(list(long_forms)))

    def test_add_tokenized_long_form(self):
        """Test adding a sequence of words."""
        self.abbs.add_tokenized_long_form(
            short_form="avc",
            long_form=["accident", "vasculaire", "cerebral"],
        )
        long_forms: List[SynType] = list(self.abbs.get_syns_of_word("avc"))
        self.assertEqual(1, len(long_forms))
        self.assertTrue(("accident", "vasculaire", "cerebral") in long_forms)

    def test_add_lower_short_form(self):
        """Even though the given short form is uppercase,
        a lowercase word returns the long form."""
        self.abbs.add(
            short_form="AVC",
            long_form="accident vasculaire cerebral",
            tokenizer=self.tokenizer,
        )
        long_forms: List[SynType] = list(self.abbs.get_syns_of_word("avc"))
        self.assertEqual(len(long_forms), 1)


class TupleTest(unittest.TestCase):
    def test_tuple_stored_in_set(self):
        """check a tuple added twice in a set is ignored (default Python)"""
        syns_set = set()
        syn1 = tuple(["insuffisance", "respiratoire", "chronique"])
        syn2 = tuple(["insuffisance", "renale", "chronique"])
        syns_set.add(syn1)
        syns_set.add(syn2)
        self.assertEqual(2, len(syns_set))
        syns_set.add(syn2)
        self.assertEqual(2, len(syns_set))

    def test_tuple_list_vs_tuple_string(self):
        """One frequent mistake I make and that users can make is to call
        'tuple' function on string and not on a list.
        Calling it on a string returns a tuple of characters.
        """
        syns_set = set()
        syn1 = tuple(["insuffisance"])  # ('insuffisance')
        syn2 = tuple("insuffisance")  # ('i','n',...,'e')
        syns_set.add(syn1)
        syns_set.add(syn2)
        self.assertEqual(2, len(syns_set))


class ExactMatchTest(unittest.TestCase):
    def test_exact_match(self):
        """Test exact match returns the correct format."""
        exact = ExactMatch()
        syns = exact.get_syns_of_word("insuffisance")
        self.assertTrue(tuple(["insuffisance"]) in syns)


class MultipleFuzzyAlgosTest(unittest.TestCase):
    def setUp(self) -> None:
        abbs = Abbreviations(name="abbs")
        tokenizer = english_tokenizer()
        abbs.add(
            short_form="fr",
            long_form="frequence respiratoire",
            tokenizer=tokenizer,
        )
        exact_match = ExactMatch()
        self.fuzzy_algos = [exact_match, abbs]
        self.tokens = tokenizer.tokenize("augmentation de la fr")
        self.i = 3  # 'fr' token

    def test_get_norm_label_algos_syns(self):
        """Return all the synonyms of NormLabel algorithms."""
        syns: List[SynAlgo] = get_norm_label_algos_syns(
            fuzzy_algos=self.fuzzy_algos, norm_label="fr"
        )
        self.assertEqual(len(syns), 2)
        label, algo = syns[0]
        self.assertEqual(tuple(["fr"]), label)
        label, algo = syns[1]
        self.assertEqual(tuple(["frequence", "respiratoire"]), label)


class CacheFuzzyAlgosTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fuzzy = TestFuzzyAlgo()
        self.cache_syns = CacheFuzzyAlgos()
        self.tuple_ins = tuple(["insuffisance"])

    def test_number_of_calls(self):
        """The purpose of the cache is to call fuzzy algorithms only once and
        then stores the results in cache. This test is just to check that
        'calls' attribute record correctly the number of calls.
        """
        self.fuzzy.get_syns_of_word("insuffisance")
        self.fuzzy.get_syns_of_word("insuffisance")
        self.assertEqual(2, self.fuzzy.calls)

    def test_caching(self):
        """Fuzzy algorithms cached are only called once and cache
        must work (synonyms are returned)."""
        self.cache_syns.add_algo(self.fuzzy)
        self.assertEqual(0, self.fuzzy.calls)
        self.cache_syns.get_syns_of_word("insuffisance")
        self.assertEqual(1, self.fuzzy.calls)  # test_fuzzy was called
        self.cache_syns.get_syns_of_word("insuffisance")
        self.assertEqual(
            1, self.fuzzy.calls
        )  # test_fuzzy was not called twice
        self.assertTrue("insuffisance" in self.cache_syns.cache)

    def test_get_syns_of_word(self):
        """Add exact match algorithm and check it returns a synonym."""
        self.cache_syns.add_algo(algo=ExactMatch())
        syns: List[SynAlgo] = self.cache_syns.get_syns_of_word("insuffisance")
        syn: SynAlgo = syns[0]
        synonym, algo = syn
        self.assertEqual(self.tuple_ins, synonym)
        self.assertEqual(algo, "exact")

    def test_get_syns_of_word_no_algo(self):
        """By default the cache contains no algorithm so it returns no
        synonym."""
        syns = self.cache_syns.get_syns_of_word("insuffisance")
        self.assertEqual(0, len(syns))

    def test_add_spellwise(self):
        """Check it call the spellwise algorithm and stores its suggestions"""
        algo: SpellWiseWrapper = SpellWiseWrapper(
            ESpellWiseAlgo.LEVENSHTEIN, max_distance=1
        )
        algo.add_words(words=["insuffisance"])
        self.cache_syns.add_algo(algo)
        syns = self.cache_syns.get_syns_of_word("insufisance")
        syn = syns[0]
        self.assertTrue(self.tuple_ins in syn)


class FuzzyRegexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fuzzy = FuzzyRegex(
            name="regex_num",
            pattern=r"^\d*[.,]?\d*$",
            pattern_name="numval",
        )
        split = split_find_iter_closure(pattern=r"(\w|\.|,)+")
        self.tokenizer = french_tokenizer()
        self.tokenizer.split = split
        self.matcher = Matcher(tokenizer=self.tokenizer)

    def test_split_function(self):
        """[calcium, 2.1, mmol, L] expected."""
        tokens = self.tokenizer.tokenize("calcium 2.1 mmol/L")
        self.assertEqual(4, len(tokens))

    def test_replace_pattern_in_keyword(self):
        """The function replaces regex matches by the pattern_name
        In this example, span containing '2.1' is replaced by 'numval'
        """
        keyword = Keyword(label="calcium 2.1 mmol/L")
        new_keyword = self.fuzzy.replace_pattern_in_keyword(
            keyword=keyword, tokenizer=self.tokenizer
        )
        self.assertEqual("calcium numval mmol/L", new_keyword.label)

    def test_get_syns_of_word(self):
        """the function should return the patter name (numval here)."""
        syns = self.fuzzy.get_syns_of_word("2.1")
        self.assertTrue(tuple(["numval"]) in syns)

    def test_detection(self):
        """It works with a matcher."""
        keyword = Keyword(label="CALCIUM NUMVAL mmol/L")
        self.matcher.add_keywords(keywords=[keyword])
        annots = self.matcher.annot_text(text="calcium 2.1 mmol/L")
        self.assertEqual(0, len(annots))
        self.matcher.add_fuzzy_algo(self.fuzzy)
        annots = self.matcher.annot_text(text="calcium 2.1 mmol/L")
        self.assertEqual(1, len(annots))


class WordNormalizerTest(unittest.TestCase):
    def simple_stemmer(self, string: str):
        if string.startswith("gastr"):
            return "gastr"
        else:
            return string

    def setUp(self) -> None:
        self.fuzzy_stemmer = WordNormalizer(
            name="stemmer", norm_fun=self.simple_stemmer
        )
        self.tokenizer = french_tokenizer()
        self.detector = Matcher(tokenizer=self.tokenizer)
        self.detector.add_fuzzy_algo(self.fuzzy_stemmer)

    def test_get_syns_of_word(self):
        """'gastrique' and 'gastrologique' start with 'gastr'
        so 'gastrique' is a synonym of 'gastrologique'.
        """
        self.fuzzy_stemmer.add_words(words=["gastrique", "gastriques"])
        synonyms = self.fuzzy_stemmer.get_syns_of_word("gastrologique")
        self.assertTrue(tuple(["gastrique"]) in synonyms)

    def test_detection_stem(self):
        """'gastrique' and 'gastrologique' start with 'gastr'
        Matcher matches it.
        """
        self.fuzzy_stemmer.add_words(words=["gastrique"])
        self.detector.add_keywords(keywords=get_termino_ulceres())
        text = "Ulcère gastrologique"
        annots = self.detector.annot_text(text)
        self.assertEqual(1, len(annots))


if __name__ == "__main__":
    unittest.main()
