from typing import List

from iamsystem.keywords.collection import Terminology
from iamsystem.keywords.keywords import Keyword
from iamsystem.keywords.keywords import Term
from iamsystem.matcher.matcher import Matcher
from iamsystem.tokenization.token import Token
from iamsystem.tokenization.tokenize import french_tokenizer


def get_termino_ivg():
    """Returns 2 nested terms."""
    term1 = Term("Insuffisance Cardiaque", "I50.9")
    term2 = Term("Insuffisance Cardiaque Gauche", "I50.1")
    termino = Terminology()
    termino.add(term1)
    termino.add(term2)
    return termino


def get_termino_ulceres():
    """Returns 2 keywords."""
    term1 = Keyword("Ulcère gastrique")
    term2 = Keyword("Ulcère duodénale")
    termino = Terminology()
    termino.add(term1)
    termino.add(term2)
    return termino


def get_termino_irc():
    """Terms with ambiguity: IRC is ambiguous."""
    term1 = Term("Insuffisance Respiratoire Chronique", "J96.1")
    term2 = Term("Insuffisance Rénale Chronique", "N18")
    term3 = Term("Insuffisance Rénale Chronique Dialysée", "N18.X")
    termino = Terminology()
    termino.add(term1)
    termino.add(term2)
    termino.add(term3)
    return termino


def get_norm_tokens_ulceres():
    """Return tokens not normalized."""

    def no_normalizer(string: str):
        return string

    text = "Ulcères gastriques compliqués d'une hémorragie digestive"
    toknorm = french_tokenizer()
    toknorm.normalize = no_normalizer
    tokens: List[Token] = list(toknorm.tokenize(text=text))
    print(tokens)
    return text, tokens


def get_annot_tokens_ulceres():
    """Return text and annotations of a not important sentence."""
    matcher = Matcher(tokenizer=french_tokenizer())
    matcher.add_labels(labels=["ulceres gastriques compliques"])
    text = "Ulcères gastriques compliqués d'une hémorragie digestive"
    annots = matcher.annot_text(text=text, w=1)
    return text, annots
