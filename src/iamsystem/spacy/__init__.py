""" Spacy components are not in iamsystem's __init__ to don't depend on
this library. """

__all__ = [
    "IAMsystemSpacy",
    "IsStopSpacy",
    "TokenSpacyAdapter",
    "SpacyTokenizer",
    "IAMsystemBuildSpacy",
]

from iamsystem.spacy.component import IAMsystemBuildSpacy
from iamsystem.spacy.component import IAMsystemSpacy
from iamsystem.spacy.stopwords import IsStopSpacy
from iamsystem.spacy.token import TokenSpacyAdapter
from iamsystem.spacy.tokenizer import SpacyTokenizer
