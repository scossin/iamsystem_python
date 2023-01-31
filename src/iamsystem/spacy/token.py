from spacy.tokens import Token as SpacyToken

from iamsystem.tokenization.api import IToken
from iamsystem.tokenization.normalize import lower_no_accents
from iamsystem.tokenization.normalize import normalizeFun


class TokenSpacyAdapter(IToken):
    """A custom Token that wraps spaCy's Token and implements the
    iamsystem's IToken interface."""

    def __init__(
        self,
        spacy_token: SpacyToken,
        norm_fun: normalizeFun = lower_no_accents,
    ):
        """Create a iamsystem's token from a spaCy token.

        :param spacy_token: a spacy.tokens instance.
        :param norm_fun: a function that normalizes the 'norm_' attribute
            of a spaCy token, attribute used by iamsystem.
        """
        self.spacy_token = spacy_token
        self.start = self.spacy_token.idx
        self.end = self.start + len(self.spacy_token.text)
        self.label = self.spacy_token.text
        self.normalize = norm_fun
        self.norm_label = self.normalize(self.spacy_token.norm_)
