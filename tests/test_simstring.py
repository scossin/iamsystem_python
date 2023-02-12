import unittest

from iamsystem.fuzzy.cache import CacheFuzzyAlgos
from iamsystem.fuzzy.simstring import ESimStringMeasure
from iamsystem.fuzzy.simstring import SimStringWrapper
from iamsystem.matcher.matcher import Matcher


class MatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.matcher = Matcher()
        self.matcher.add_keywords(keywords=["paracetamol", "les"])

    def test_init_algo_str(self):
        """Check no error the measure is a string."""
        fuzzy_ss = SimStringWrapper(
            measure="DICE", words=["paracetamol"], threshold=0.2
        )
        syns = list(fuzzy_ss.get_syns_of_word("paracetamol"))
        self.assertEqual(1, len(syns))

    def test_init_algo_str_lower(self):
        """Check no error the measure is lowercase."""
        fuzzy_ss = SimStringWrapper(
            measure="dice", words=["paracetamol"], threshold=0.2
        )
        syns = list(fuzzy_ss.get_syns_of_word("paracetamol"))
        self.assertEqual(1, len(syns))

    def test_init_algo_str_mispelled(self):
        with (self.assertRaises(KeyError)):
            SimStringWrapper(
                measure="cosin", words=["paracetamol"], threshold=0.2
            )

    def test_threshold_1(self):
        """Test threshold=1 is exact match"""
        fuzzy_ss = SimStringWrapper(words=["paracetamol"], threshold=1)
        syns = list(fuzzy_ss.get_syns_of_word("paracetomol"))
        self.assertEqual(0, len(syns))
        syns = list(fuzzy_ss.get_syns_of_word("paracetamol"))
        self.assertEqual(1, len(syns))

    def test_threshold_0_5(self):
        """Test synonyms returned depend on threshold."""
        fuzzy_ss = SimStringWrapper(words=["paracetamol"])
        syns = list(fuzzy_ss.get_syns_of_word("paracetomol"))
        self.assertEqual(1, len(syns))
        syns = list(fuzzy_ss.get_syns_of_word("para"))
        self.assertEqual(0, len(syns))

    def test_threshold_0_2(self):
        """Test synonyms returned depend on threshold."""
        fuzzy_ss = SimStringWrapper(
            words=["paracetamol"], threshold=0.2, min_nb_char=0
        )
        syns = list(fuzzy_ss.get_syns_of_word("paracetomol"))
        self.assertEqual(1, len(syns))
        syns = list(fuzzy_ss.get_syns_of_word("para"))
        self.assertEqual(1, len(syns))

    def test_measure_exact(self):
        """Test synonyms returned depend on threshold."""
        fuzzy_ss = SimStringWrapper(
            words=["paracetamol"], measure=ESimStringMeasure.EXACT
        )
        syns = list(fuzzy_ss.get_syns_of_word("paracetomol"))
        self.assertEqual(0, len(syns))
        syns = list(fuzzy_ss.get_syns_of_word("paracetamol"))
        self.assertEqual(1, len(syns))

    def test_other_measures(self):
        """Test other similarity measure ; check it returns a synonym."""
        for measure in ESimStringMeasure:
            if measure.value == "exact":
                continue
            fuzzy_ss = SimStringWrapper(words=["paracetamol"], measure=measure)
            syns = list(fuzzy_ss.get_syns_of_word("paracetomol"))
            self.assertEqual(1, len(syns))

    def test_matcher(self):
        """Test detection with a matcher"""
        fuzzy_ss = SimStringWrapper(words=self.matcher.get_keywords_unigrams())
        self.matcher.add_fuzzy_algo(fuzzy_algo=fuzzy_ss)
        annots = self.matcher.annot_text(text="le paractamol")
        self.assertEqual(1, len(annots))

    def test_cache_fuzzy_algos(self):
        """Test it can work with CacheFuzzyAlgos."""
        fuzzy_ss = SimStringWrapper(words=self.matcher.get_keywords_unigrams())
        cache = CacheFuzzyAlgos()
        cache.add_algo(algo=fuzzy_ss)
        self.matcher.add_fuzzy_algo(fuzzy_algo=cache)
        annots = self.matcher.annot_text(text="le paractamol")
        self.assertEqual(1, len(annots))

    def test_combine_multiple_algos(self):
        """Test we can add multiple simstring algorithms."""
        fuzzy_dice = SimStringWrapper(
            words=self.matcher.get_keywords_unigrams(),
            name="ss_dice",
            measure=ESimStringMeasure.DICE,
        )
        fuzzy_jaccard = SimStringWrapper(
            words=self.matcher.get_keywords_unigrams(),
            name="ss_jaccard",
            measure=ESimStringMeasure.JACCARD,
        )
        self.matcher.add_fuzzy_algo(fuzzy_algo=fuzzy_dice)
        self.matcher.add_fuzzy_algo(fuzzy_algo=fuzzy_jaccard)
        annots = self.matcher.annot_text(text="le paractamol")
        self.assertEqual(1, len(annots))
        annot = annots[0]
        algos_token_0 = annot.algos[0]
        self.assertEqual(["ss_dice", "ss_jaccard"], algos_token_0)

    def test_combine_multiple_algos_2(self):
        """Test the two simstring databases are independent to allow
        the user to customize different files."""
        fuzzy_dice = SimStringWrapper(
            words=self.matcher.get_keywords_unigrams(),
            name="ss_dice",
            measure=ESimStringMeasure.DICE,
        )
        fuzzy_jaccard = SimStringWrapper(
            words=["NothingInterestingHere"],
            name="ss_jaccard",
            measure=ESimStringMeasure.JACCARD,
        )
        self.matcher.add_fuzzy_algo(fuzzy_algo=fuzzy_dice)
        self.matcher.add_fuzzy_algo(fuzzy_algo=fuzzy_jaccard)
        annots = self.matcher.annot_text(text="le paractamol")
        self.assertEqual(1, len(annots))
        annot = annots[0]
        algos_token_0 = annot.algos[0]
        self.assertEqual(["ss_dice"], algos_token_0)
