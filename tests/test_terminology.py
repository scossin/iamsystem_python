import unittest

from typing import List

from iamsystem.keywords.collection import Terminology
from iamsystem.keywords.keywords import Term
from iamsystem.stopwords.simple import Stopwords
from iamsystem.tokenization.tokenize import french_tokenizer
from tests.utils_detector import TermSubClass
from tests.utils_detector import get_term_sub_class_ivg


class TerminologyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.terminology = Terminology()
        self.term = Term("Insuffisance Cardiaque Gauche", "I50.1")
        self.terminology.add(keyword=self.term)
        self.stopwords = Stopwords()
        self.tokenizer = french_tokenizer()
        self.tokenizer.stopwords = self.stopwords

    def test_size(self):
        """Initial size is 0 : no keyword."""
        self.assertEqual(0, Terminology().size)
        self.assertEqual(1, self.terminology.size)

    def test_iterable(self):
        """Test we can iterate over the object."""
        count = 0
        for keyword in self.terminology:
            count += 1
        for keyword in self.terminology:
            count += 1
        self.assertEqual(2, count)

    def test_size_duplicated_term(self):
        """check adding twice the same term is ok"""
        self.terminology.add(keyword=self.term)
        self.assertEqual(2, self.terminology.size)

    def test_get_unigrams(self):
        """3 unigrams: insuffisance, cardiaque, gauche."""
        set_unigrams = self.terminology.get_unigrams(
            tokenizer=self.tokenizer, stopwords=self.stopwords
        )
        self.assertEqual(3, len(set_unigrams))

    def test_get_unigrams_with_a_stopword(self):
        """check stopword removal works : the function returns less unigrams"""
        self.stopwords.add(words=["insuffisance"])
        set_unigrams = self.terminology.get_unigrams(
            tokenizer=self.tokenizer, stopwords=self.stopwords
        )
        self.assertEqual(2, len(set_unigrams))
        self.assertTrue("insuffisance" not in set_unigrams)

    def test_add_keywords_subclass(self):
        """It works with a sublcass of Keyword."""
        term1 = TermSubClass("Insuffisance Cardiaque", "I50.9", "ICD-10")
        term2 = TermSubClass(
            "Insuffisance Cardiaque Gauche", "I50.1", "ICD-10"
        )
        termino = Terminology()
        termino.add_keywords(keywords=[term1, term2])
        self.assertEqual(2, termino.size)

    def test_term_sub_class(self):
        """test a keywords accepts a class that extends the Keyword class"""
        termino = get_term_sub_class_ivg()
        terms: List[TermSubClass] = list(termino.keywords)
        self.assertEqual(2, len(terms))
        one_term: TermSubClass = terms[0]
        self.assertEqual("ICD-10", one_term.termino)


class TermTest(unittest.TestCase):
    def setUp(self) -> None:
        self.term = Term("Insuffisance Cardiaque Gauche", "I50.1")

    def test__str__(self):
        """String representation."""
        self.assertEqual(
            "Insuffisance Cardiaque Gauche (I50.1)", self.term.__str__()
        )


if __name__ == "__main__":
    unittest.main()
