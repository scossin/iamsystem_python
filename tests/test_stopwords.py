import unittest

from iamsystem.stopwords.negative import NegativeStopwords
from iamsystem.stopwords.simple import NoStopwords
from iamsystem.stopwords.simple import Stopwords
from iamsystem.tokenization.api import TokenT
from iamsystem.tokenization.token import Token


class StopwordsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stopwords = Stopwords()

    def test_init_list(self):
        """init works with a list"""
        stopwords_list = ["le", "la"]
        stopwords = Stopwords(stopwords=stopwords_list)
        self.assertTrue(stopwords.is_stopword("le"))

    def test_init_dict(self):
        """init works with a dict"""
        stopwords_dict = {"le", "la"}
        stopwords = Stopwords(stopwords=stopwords_dict)
        self.assertTrue(stopwords.is_stopword("le"))

    def test_add(self):
        """Check it works before and after adding a stopword."""
        self.assertTrue(not self.stopwords.is_stopword("du"))
        self.stopwords.add(words=["du"])
        self.assertTrue(self.stopwords.is_stopword("du"))

    def test_add_multiple_words(self):
        """When adding multiple words, it works for every one of them."""
        self.stopwords.add(words=["le", "la"])
        self.assertTrue(self.stopwords.is_stopword("le"))
        self.assertTrue(self.stopwords.is_stopword("la"))

    def test_is_stopword_emptystring(self):
        """checks no error when empty string."""
        self.assertTrue(self.stopwords.is_stopword(" "))
        self.assertTrue(self.stopwords.is_stopword("\n"))
        self.assertTrue(self.stopwords.is_stopword(" \t "))

    def test_accent(self):
        """Accents are not removed."""
        self.stopwords.add(words=["à"])
        token_a_accent = Token(0, 1, label="à", norm_label="a")
        self.assertTrue(
            self.stopwords.is_token_a_stopword(token=token_a_accent)
        )
        token_a = Token(0, 1, label="a", norm_label="a")
        self.assertTrue(not self.stopwords.is_token_a_stopword(token=token_a))


class NegativeStopwordsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stopwords = NegativeStopwords()
        self.token = Token(0, 1, label="patient", norm_label="patient")

    def test_is_token_a_stopword(self):
        """It returns true because it's not a word to keep."""
        self.assertTrue(self.stopwords.is_token_a_stopword(token=self.token))

    def test_add_word_to_keep(self):
        """It returns False because it's a word to keep."""
        self.stopwords.add_words(words_to_keep=["patient"])
        self.assertTrue(
            not self.stopwords.is_token_a_stopword(token=self.token)
        )

    def test_fun_word_to_keep(self):
        """Returns false because it's written in lowercase
        and false is any uppercase letter.
        """

        def is_lower_case(token: TokenT) -> bool:
            """True if lowercase match."""
            return token.norm_label.islower()

        self.assertTrue(self.stopwords.is_token_a_stopword(token=self.token))
        self.stopwords.add_fun_is_a_word_to_keep(fun=is_lower_case)
        self.assertTrue(
            not self.stopwords.is_token_a_stopword(token=self.token)
        )


class NoStopwordsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stopwords = NoStopwords()
        self.token_du = Token(0, 2, label="du", norm_label="du")

    def test_is_stopword(self):
        """It returns always False."""
        self.assertTrue(not self.stopwords.is_token_a_stopword(self.token_du))

    def test_no_stopword_add(self):
        """'add_word' method is doesn't exist"""
        with self.assertRaises(AttributeError):
            self.stopwords.add("du")


if __name__ == "__main__":
    unittest.main()
