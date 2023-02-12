import typing
import unittest

from typing import Iterable
from typing import List

from iamsystem.fuzzy.api import NormLabelAlgo
from iamsystem.fuzzy.api import SynType
from iamsystem.fuzzy.simstring import ESimStringMeasure
from iamsystem.fuzzy.spellwise import ESpellWiseAlgo
from iamsystem.fuzzy.util import SimpleWords2ignore
from iamsystem.keywords.collection import Terminology
from iamsystem.keywords.keywords import Entity
from iamsystem.keywords.keywords import Keyword
from iamsystem.matcher.annotation import Annotation
from iamsystem.matcher.annotation import replace_annots
from iamsystem.matcher.matcher import Matcher
from iamsystem.stopwords.negative import NegativeStopwords
from iamsystem.stopwords.simple import NoStopwords
from iamsystem.stopwords.simple import Stopwords
from iamsystem.tokenization.tokenize import english_tokenizer
from iamsystem.tokenization.tokenize import french_tokenizer
from iamsystem.tokenization.tokenize import split_find_iter_closure
from iamsystem.tokenization.tokenize import tokenize_and_order_decorator
from tests.utils import get_termino_ivg
from tests.utils_detector import TermSubClass
from tests.utils_detector import get_ent_sub_class_ivg


class MatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.terminoIVG = get_termino_ivg()
        self.stopwords = Stopwords()
        self.tokenizer = french_tokenizer()
        self.tokenizer.stopwords = self.stopwords
        self.matcher = Matcher(tokenizer=self.tokenizer)
        self.matcher.add_keywords(keywords=self.terminoIVG)
        self.terminoSubClass = get_ent_sub_class_ivg()
        self.detectorSubClass = Matcher(tokenizer=self.tokenizer)
        self.detectorSubClass.add_keywords(keywords=self.terminoSubClass)

    def test_tokenize(self):
        """split on the white space."""
        tokens = self.matcher.tokenize(text="insuffisance cardiaque")
        self.assertEqual(2, len(tokens))

    def test_is_stopwords(self):
        """Check it implements IStopword interface."""
        self.matcher.add_stopwords(words=["le", "à"])
        self.assertTrue(self.matcher.is_stopword("le"))
        self.assertTrue(not self.matcher.is_stopword("a"))
        self.assertTrue(self.matcher.is_stopword("à"))

    def test_change_default_stopwords(self):
        """NoStopwords doesn't implement 'add_stopwords' interface.
        A warning is raised to say it does nothing.
        """
        matcher = Matcher(stopwords=NoStopwords())
        with self.assertWarns(Warning):
            matcher.add_stopwords(words=["le"])

    def test_detect_exact_match(self):
        """Simple annotation by exact match algorithm."""
        annots: List[Annotation] = self.matcher.annot_text(
            text="insuffisance cardiaque"
        )
        self.assertEqual(1, len(annots))

    def test_add_keywords(self):
        """This function add keywords that can be detected."""
        matcher = Matcher()
        words = ["acute respiratory distress syndrome", "diarrrhea"]
        matcher.add_keywords(keywords=words)
        text = "Pt c/o acute respiratory distress syndrome and diarrrhea"
        annots = matcher.annot_text(text=text)
        self.assertEqual(2, len(annots))

    def test_keywords_attribute(self):
        """Keywords attribute stores the keyword.
        Two keywords were added in matcher.
        """
        self.assertEqual(2, len(self.matcher.keywords))

    def test_get_unigrams(self):
        """Returns a set of unigrams of added keywords.
        Add multiple times the keywords to check set is working fine.
        """
        self.matcher.add_keywords(keywords=self.terminoIVG)
        self.matcher.add_keywords(keywords=self.terminoIVG)
        self.assertEqual(3, len(self.matcher.get_keywords_unigrams()))
        self.assertTrue("insuffisance" in self.matcher.get_keywords_unigrams())
        self.assertTrue("cardiaque" in self.matcher.get_keywords_unigrams())
        self.assertTrue("gauche" in self.matcher.get_keywords_unigrams())

    def test_detect_overlap(self):
        """Default is to remove overlapping keywords detected"""
        annots: List[Annotation] = self.matcher.annot_text(
            text="insuffisance cardiaque gauche"
        )
        self.assertEqual(1, len(annots))
        self.matcher.remove_nested_annots = False
        annots: List[Annotation] = self.matcher.annot_text(
            text="insuffisance cardiaque gauche"
        )
        self.assertEqual(2, len(annots))

    def test_detect_with_last_span_is_stopword(self):
        """When a stopword follows a keyword,
        it's not part of the annotation."""
        ent = Entity("serpents", "C0037382")
        termino = Terminology()
        termino.add(ent)
        self.matcher = Matcher(tokenizer=self.tokenizer)
        self.matcher.add_keywords(keywords=termino)
        self.stopwords.add(words=["les"])
        annots: List[Annotation] = self.matcher.annot_text(
            text="serpents. Les vipères..."
        )
        self.assertEqual(1, len(annots))
        annot = annots[0]
        self.assertEqual(annot.label, "serpents")

    def test_detect_with_a_ent_sub_class(self):
        """Check no type error and attribute access of a Term subclass."""
        annots: List[Annotation] = self.detectorSubClass.annot_text(
            text="insuffisance cardiaque"
        )
        self.assertEqual(1, len(annots))
        self.assertTrue(isinstance(annots[0].keywords[0], TermSubClass))
        ent_sub_class: TermSubClass = typing.cast(
            TermSubClass, annots[0].keywords[0]
        )
        self.assertEqual("ICD-10", ent_sub_class.termino)

    def test_add_algo(self):
        """Create a fuzzy algorithm and add it, it must match anything."""
        fuzzy_algo = AnotherFuzzyAlgo(name="fuzzy")
        self.matcher.add_fuzzy_algo(fuzzy_algo)
        annots: List[Annotation] = self.matcher.annot_text(
            text="AnythingWouldWorks"
        )
        self.assertEqual(1, len(annots))

    def test_unordered_words_seq(self):
        """If the order of words doesn't matter, order alphabetically
        the tokens to search a keyword."""
        text = (
            "echographie ventriculaire gauche: FEVG diminuee signe "
            "d'une insuffisance moderee."
        )
        tokenizer = english_tokenizer()
        tokenizer.tokenize = tokenize_and_order_decorator(tokenizer.tokenize)
        matcher = Matcher(tokenizer=tokenizer)
        tokens = matcher.tokenize(text)
        self.assertEqual("d", tokens[0].norm_label)
        matcher.add_keywords(keywords=["insuffisance ventriculaire gauche"])
        matcher.w = 10
        annots = matcher.annot_text(text=text)
        self.assertEqual(1, len(annots))
        self.assertEqual(
            "insuffisance ventriculaire gauche", annots[0].keywords[0].label
        )

    def test_replace_annots(self):
        """Replace keyword by new labels in a document.
        Warning: it doesn't handle overlapping annotations. WIP to implement.
        """
        text = "insuffisance cardiaque gauche"
        annots: List[Annotation] = self.matcher.annot_text(text=text)
        new_labels = [annot.keywords[0].kb_id for annot in annots]
        new_text = replace_annots(
            text=text, annots=annots, new_labels=new_labels
        )
        self.assertEqual("I50.1", new_text)

    def test_negative_stopwords(self):
        """Matcher with a negative stopword.
        Ignore all the words excepts those in the keywords."""
        text = "insuffisance à l'echographie ventriculaire du cote gauche."
        termino = Terminology()
        termino.add_keywords(
            keywords=[Keyword(label="insuffisance ventriculaire gauche")]
        )
        stopwords = NegativeStopwords()
        tokenizer = english_tokenizer()
        stopwords.add_words(
            words_to_keep=termino.get_unigrams(
                tokenizer=tokenizer, stopwords=NoStopwords()
            )
        )
        matcher = Matcher(tokenizer=tokenizer, stopwords=stopwords)
        matcher.add_keywords(keywords=termino)
        annots = matcher.annot_text(text=text)
        self.assertEqual(1, len(annots))

    def test_keywords_iterator(self):
        matcher = Matcher()
        termino = Terminology()
        ent = Entity(label="ulcères gastriques", kb_id="K25")
        termino.add(ent)
        keyword_iter = iter(termino)
        matcher.add_keywords(keywords=keyword_iter)
        annots = matcher.annot_text(text="ulcères gastriques")
        self.assertEqual(1, len(annots))


class AnotherFuzzyAlgo(NormLabelAlgo):
    """A fuzzy algorithm that returns always the same sequence of tokens."""

    def __init__(self, name: str):
        super().__init__(name)

    def get_syns_of_word(self, word: str) -> Iterable[SynType]:
        """words used in tests"""
        return [self.words_seq_to_syn(words=["insuffisance", "cardiaque"])]


class MatcherBuild(unittest.TestCase):
    def test_no_keyword(self):
        """Check no error raised if no keywords."""
        matcher = Matcher.build(keywords=[])
        anns = matcher.annot_text(text="nothing detected")
        self.assertEqual(len(anns), 0)

    def test_abbreviations(self):
        """Test adding abbreviations works."""
        matcher = Matcher.build(
            keywords=get_termino_ivg(),
            abbreviations=[("ic", "insuffisance cardiaque")],
        )
        annots = matcher.annot_text(text="ic gauche")
        self.assertEqual(1, len(annots))

    def test_remove_nested_annots(self):
        """Test remove_nested_annots works."""
        matcher = Matcher.build(
            keywords=get_termino_ivg(), remove_nested_annots=False
        )
        annots = matcher.annot_text(text="insuffisance cardiaque gauche")
        self.assertEqual(2, len(annots))

    def test_w(self):
        """Test window works."""
        text = "insuffisance à cardiaque à gauche"
        matcher = Matcher.build(keywords=get_termino_ivg(), w=1)
        annots = matcher.annot_text(text=text)
        self.assertEqual(0, len(annots))

        matcher = Matcher.build(keywords=get_termino_ivg(), w=2)
        annots = matcher.annot_text(text=text)
        self.assertEqual(1, len(annots))

    def test_document_stopwords(self):
        """Test stopwords in document works."""
        text = "insuffisance à cardiaque à gauche"
        matcher = Matcher.build(keywords=get_termino_ivg())
        annots = matcher.annot_text(text=text)
        self.assertEqual(0, len(annots))

        matcher = Matcher.build(keywords=get_termino_ivg(), stopwords=["à"])
        annots = matcher.annot_text(text=text)
        self.assertEqual(1, len(annots))

    def test_keywords_stopwords(self):
        """Test stopwords works in keywords."""
        text = "insuffisance de carburant"
        matcher = Matcher.build(keywords=get_termino_ivg())
        annots = matcher.annot_text(text=text)
        self.assertEqual(0, len(annots))

        matcher = Matcher.build(
            keywords=get_termino_ivg(), stopwords=["cardiaque", "gauche"]
        )
        annots = matcher.annot_text(text=text)
        self.assertEqual(1, len(annots))

    def test_order_tokens(self):
        """Test order_tokens works."""
        text = "cardiaque insuffisance à gauche"
        matcher = Matcher.build(keywords=get_termino_ivg())
        annots = matcher.annot_text(text=text)
        self.assertEqual(0, len(annots))

        matcher = Matcher.build(keywords=get_termino_ivg(), order_tokens=True)
        annots = matcher.annot_text(text=text)
        self.assertEqual(1, len(annots))

    def test_order_tokens_abbreviations(self):
        """Test order_tokens works."""
        text = "gauche : ic"
        matcher = Matcher.build(
            keywords=["insuffisance cardiaque gauche"],
            abbreviations=[("ic", "insuffisance cardiaque")],
        )
        annots = matcher.annot_text(text=text)
        self.assertEqual(0, len(annots))

        matcher = Matcher.build(
            keywords=get_termino_ivg(),
            abbreviations=[("ic", "insuffisance cardiaque")],
            order_tokens=True,
        )
        annots = matcher.annot_text(text=text)
        self.assertEqual(1, len(annots))

    def test_negative(self):
        """Test negative stopwords works."""
        text = "insuffisance à cardiaque à gauche"
        matcher = Matcher.build(keywords=get_termino_ivg())
        annots = matcher.annot_text(text=text)
        self.assertEqual(0, len(annots))

        matcher = Matcher.build(keywords=get_termino_ivg(), negative=True)
        annots = matcher.annot_text(text=text)
        self.assertEqual(1, len(annots))

    def test_spellwise(self):
        """Test levenshtein works."""
        text = "insuffisance cardiaqu gauche"
        matcher = Matcher.build(keywords=get_termino_ivg())
        annots = matcher.annot_text(text=text)
        self.assertEqual(0, len(annots))

        matcher = Matcher.build(
            keywords=get_termino_ivg(),
            spellwise=[
                dict(measure=ESpellWiseAlgo.LEVENSHTEIN, max_distance=1)
            ],
        )
        annots = matcher.annot_text(text=text)
        self.assertEqual(1, len(annots))

    def test_spellwise_param_order(self):
        """Test levenshtein works with wrong params order"""
        text = "insuffisance cardiaqu gauche"
        matcher = Matcher.build(
            keywords=get_termino_ivg(),
            spellwise=[
                dict(max_distance=1, measure=ESpellWiseAlgo.LEVENSHTEIN)
            ],
        )
        annots = matcher.annot_text(text=text)
        self.assertEqual(1, len(annots))

    def test_spellwise_wrong_param(self):
        """Test levenshtein works with wrong params order"""
        with self.assertRaises(TypeError):
            Matcher.build(
                keywords=get_termino_ivg(),
                spellwise=[
                    dict(max_distance=1, algo=ESpellWiseAlgo.LEVENSHTEIN)
                ],
            )

    def test_spellwise_params_overrides(self):
        """Even though string_distance_ignored_w is set,
        it doesn't override user's choice of words2ignore for this
        spellwise algorithm."""
        text = "insuffisance cardiaqu gauche"
        words2ignore = SimpleWords2ignore()
        matcher = Matcher.build(
            keywords=get_termino_ivg(),
            spellwise=[
                dict(
                    max_distance=1,
                    measure=ESpellWiseAlgo.LEVENSHTEIN,
                    words2ignore=words2ignore,
                )
            ],
            string_distance_ignored_w=["cardiaqu"],
        )
        annots = matcher.annot_text(text=text)
        self.assertEqual(1, len(annots))

        # string distance is not calculated for 'cardiaqu' :
        matcher = Matcher.build(
            keywords=get_termino_ivg(),
            spellwise=[
                dict(
                    max_distance=1,
                    measure=ESpellWiseAlgo.LEVENSHTEIN,
                    # words2ignore=words2ignore
                )
            ],
            string_distance_ignored_w=["cardiaqu"],
        )
        annots = matcher.annot_text(text=text)
        self.assertEqual(0, len(annots))

    def test_simstring_2_cosine(self):
        """Test adding 2 simstring cosine measure doesn't raise an error.
        Check detection works"""
        matcher = Matcher.build(
            keywords=get_termino_ivg(),
            simstring=[
                dict(measure=ESimStringMeasure.COSINE, threshold=0.7),
                dict(measure=ESimStringMeasure.COSINE, threshold=0.2),
            ],
        )
        self.assertEqual(2, len(list(matcher.fuzzy_algos)))  # exact + cache
        annots = matcher.annot_text(text="insuf cardia gauch")
        self.assertEqual(1, len(annots))

    def test_word_normalizer(self):
        """Test adding 2 simstring cosine measure doesn't raise an error.
        Check detection works"""
        from nltk.stem.snowball import FrenchStemmer

        stemmer = FrenchStemmer()
        matcher = Matcher.build(
            keywords=get_termino_ivg(),
            normalizers=[dict(name="french_stemmer", norm_fun=stemmer.stem)],
        )
        annots = matcher.annot_text(text="insuffisance cardiaqu gauch")
        self.assertEqual(1, len(annots))

    def test_fuzzy_regex(self):
        """Test fuzzy regex works"""
        tokenizer = english_tokenizer()
        tokenizer.split = split_find_iter_closure(pattern=r"(\w|\.|,)+")
        matcher = Matcher.build(
            keywords=["diabete en valueannees"],
            tokenizer=tokenizer,
            stopwords=["en"],
            fuzzy_regex=[
                dict(
                    name="detection_annee",
                    pattern=r"[(19|20)0-9{2}]",
                    pattern_name="valueannees",
                )
            ],
        )
        annots = matcher.annot_text(text="diabete en 2010")
        self.assertEqual(1, len(annots))


if __name__ == "__main__":
    unittest.main()
