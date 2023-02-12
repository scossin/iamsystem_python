from typing import List

from iamsystem.keywords.collection import Terminology
from iamsystem.keywords.keywords import Entity
from iamsystem.keywords.keywords import Keyword
from iamsystem.matcher.matcher import Matcher
from iamsystem.tokenization.token import Token
from iamsystem.tokenization.tokenize import french_tokenizer


def get_termino_ivg():
    """Returns 2 nested ents."""
    ent1 = Entity("Insuffisance Cardiaque", "I50.9")
    ent2 = Entity("Insuffisance Cardiaque Gauche", "I50.1")
    termino = Terminology()
    termino.add(ent1)
    termino.add(ent2)
    return termino


def get_termino_ulceres():
    """Returns 2 keywords."""
    ent1 = Keyword("Ulcère gastrique")
    ent2 = Keyword("Ulcère duodénale")
    termino = Terminology()
    termino.add(ent1)
    termino.add(ent2)
    return termino


def get_termino_irc():
    """Terms with ambiguity: IRC is ambiguous."""
    ent1 = Entity("Insuffisance Respiratoire Chronique", "J96.1")
    ent2 = Entity("Insuffisance Rénale Chronique", "N18")
    ent3 = Entity("Insuffisance Rénale Chronique Dialysée", "N18.X")
    termino = Terminology()
    termino.add(ent1)
    termino.add(ent2)
    termino.add(ent3)
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
