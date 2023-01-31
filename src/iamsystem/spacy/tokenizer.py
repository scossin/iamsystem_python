""" Tokenize keywords and documents using spaCy tokenizer. """


from typing import Sequence

from spacy import Language

from iamsystem.spacy.token import TokenSpacyAdapter
from iamsystem.tokenization.api import ITokenizer
from iamsystem.tokenization.normalize import normalizeFun


class SpacyTokenizer(ITokenizer[TokenSpacyAdapter]):
    """A class that wraps spaCy's tokenizer."""

    def __init__(self, nlp: Language, norm_fun: normalizeFun):
        """Create a tokenizer for iamsystem algorithm
        that uses spaCy's tokenizer.

        :param nlp: a spacy Language.
        :param norm_fun: a function that normalizes the 'norm_' attribute
            of a spaCy token, attribute used by iamsystem algorithm.
        """
        self.nlp = nlp
        self.norm_fun = norm_fun

    def tokenize(self, text: str) -> Sequence[TokenSpacyAdapter]:
        """Tokenize a text. This function is used only to tokenize the
        keywords by the matcher since this custom component receives
        from spaCy the document already tokenized.

        :param text: a string to tokenize with spaCy component.
        :return: an ordered sequence of tokens.
        """
        doc = self.nlp(text, disable=["iamsystem"])
        tokens = [
            TokenSpacyAdapter(spacy_token=token, norm_fun=self.norm_fun)
            for token in doc
        ]
        return tokens
