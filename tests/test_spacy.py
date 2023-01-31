import unittest

from typing import Iterable
from typing import List

import spacy

from spacy.lang.fr import French

from iamsystem import Abbreviations
from iamsystem import Annotation
from iamsystem import FuzzyAlgo
from iamsystem import IKeyword
from iamsystem import IStopwords
from iamsystem import Term
from iamsystem import Terminology
from iamsystem import french_tokenizer
from iamsystem.spacy.component import IAMsystemSpacy  # noqa
from iamsystem.spacy.stopwords import IsStopSpacy
from iamsystem.spacy.token import TokenSpacyAdapter


@spacy.registry.misc("umls_terms.v1")
def get_termino_umls() -> Iterable[IKeyword]:
    """An imaginary set of umls terms."""
    termino = Terminology()
    term1 = Term("Insuffisance Cardiaque", "I50.9")
    term2 = Term("Insuffisance Cardiaque Gauche", "I50.1")
    termino.add_keywords(keywords=[term1, term2])
    return termino


@spacy.registry.misc("fuzzy_algos_short_notes.v1")
def get_fuzzy_algos_short_notes() -> List[FuzzyAlgo]:
    """An imaginary set of fuzzy algorithms for medical short notes."""
    tokenizer = french_tokenizer()
    abbs = Abbreviations(name="French medical abbreviations")
    abbs.add(short_form="ins", long_form="insuffisance", tokenizer=tokenizer)
    abbs.add(
        short_form="ic",
        long_form="insuffisance cardiaque",
        tokenizer=tokenizer,
    )
    return [abbs]


@spacy.registry.misc("stopwords_spacy.v1")
def get_stopwords_short_notes() -> IStopwords[TokenSpacyAdapter]:
    """Use spaCy stopword list."""
    stopwords = IsStopSpacy()
    return stopwords


class SpacyCompTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nlp = French()
        cls.nlp.add_pipe(
            "iamsystem",
            name="iamsystem",
            last=True,
            config={
                "keywords": {"@misc": "umls_terms.v1"},
                "stopwords": {"@misc": "stopwords_spacy.v1"},
                "fuzzy_algos": {"@misc": "fuzzy_algos_short_notes.v1"},
            },
        )

    def test_comp_presence(self):
        self.assertTrue("iamsystem" in self.nlp.pipe_names)

    def test_set_span_extension(self):
        """check iamsystem was to span attribute."""
        doc = self.nlp("somethingThatReturnsNoAnnotation")
        self.assertTrue("iamsystem" in doc.spans)
        self.assertEqual(0, len(doc.spans["iamsystem"]))

    def test_detect(self):
        """Check detection works with exact match and annotation is added."""
        doc = self.nlp("insuffisance cardiaque gauche")
        self.assertEqual(1, len(doc.spans["iamsystem"]))
        span = doc.spans["iamsystem"][0]
        annot = span._.iamsystem
        self.assertTrue(isinstance(annot, Annotation))

    def test_detect_abb(self):
        """Check detection works with abbreviation."""
        doc = self.nlp("ic gauche")
        self.assertEqual(1, len(doc.spans["iamsystem"]))

    def test_change_default_config(self):
        """Change window and remove_nested default.
        Two annots: 'ic' and 'ic cardiaque'.
        """
        nlp = French()
        nlp.add_pipe(
            "iamsystem",
            name="iamsystem",
            last=True,
            config={
                "keywords": {"@misc": "umls_terms.v1"},
                "stopwords": {"@misc": "stopwords_spacy.v1"},
                "fuzzy_algos": {"@misc": "fuzzy_algos_short_notes.v1"},
                "w": 4,
                "remove_nested_annots": False,
            },
        )
        doc = nlp("ic: contraction du ventricule gauche faible")
        self.assertEqual(2, len(doc.spans["iamsystem"]))


if __name__ == "__main__":
    unittest.main()
