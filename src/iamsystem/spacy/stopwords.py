""" Custom spaCy stopwords implementations."""

from iamsystem.spacy.token import TokenSpacyAdapter
from iamsystem.stopwords.api import IStopwords


class IsStopSpacy(IStopwords[TokenSpacyAdapter]):
    """Stopwords that uses spaCy's 'is_stop' function."""

    def is_token_a_stopword(self, token: TokenSpacyAdapter) -> bool:
        """Return spaCy's token attribute 'is_stop'."""
        return token.spacy_token.is_stop
