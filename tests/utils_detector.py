from typing import Tuple

from iamsystem.fuzzy.abbreviations import Abbreviations
from iamsystem.keywords.collection import Terminology
from iamsystem.keywords.keywords import Term
from iamsystem.matcher.util import StartState
from iamsystem.matcher.util import TransitionState
from iamsystem.tokenization.token import Token
from iamsystem.tokenization.tokenize import french_tokenizer
from iamsystem.tree.nodes import Node
from iamsystem.tree.trie import Trie


def get_abbs_ic() -> Abbreviations:
    """Two abbreviations: ic and ins"""
    abbs = Abbreviations(name="abbs")
    abbs.add_tokenized_long_form("ic", tuple(["insuffisance", "cardiaque"]))
    abbs.add_tokenized_long_form("ins", tuple(["insuffisance"]))
    return abbs


def get_abbs_irc() -> Abbreviations:
    """IRC: One ambiguous short form."""
    tokenizer = french_tokenizer()
    abbs = Abbreviations(name="abbs")
    long_form = "insuffisance renale chronique"
    abbs.add("irc", long_form, tokenizer)
    long_form = "insuffisance respiratoire chronique"
    abbs.add("irc", long_form, tokenizer)
    return abbs


def get_gauche_el_in_ivg() -> Tuple[Node, TransitionState]:
    """Return a transition state."""
    # root_node
    trie = Trie()
    root_node = trie.get_initial_state()
    root_el = StartState(node=trie.root_node)
    # insuffisance
    ins_node = Node(token="insuffisance", node_num=1, parent_node=root_node)
    ins_span = Token(
        label="Insuffisance", norm_label="insuffisance", start=0, end=12
    )
    ins_el: TransitionState = TransitionState(
        node=ins_node, token=ins_span, parent=root_el, algos=["exact"]
    )
    # ventriculaire
    vent_node = Node(token="ventriculaire", node_num=2, parent_node=ins_node)
    vent_span = Token(
        label="Ventriculaire", norm_label="ventriculaire", start=13, end=26
    )
    vent_el: TransitionState = TransitionState(
        node=vent_node, token=vent_span, parent=ins_el, algos=["exact"]
    )
    # gauche
    gauche_node = Node(token="gauche", node_num=3, parent_node=vent_node)
    gauche_span = Token(label="Gauche", norm_label="gauche", start=28, end=34)
    gauche_el: TransitionState = TransitionState(
        node=gauche_node, token=gauche_span, parent=vent_el, algos=["exact"]
    )
    return gauche_node, gauche_el


class TermSubClass(Term):
    """Add a termino attribute to a Term."""

    def __init__(self, label, code, termino: str):
        super().__init__(label, code)
        self.termino = termino


def get_term_sub_class_ivg():
    """Returns terms of a custom Term subclass."""
    term1 = TermSubClass("Insuffisance Cardiaque", "I50.9", "ICD-10")
    term2 = TermSubClass("Insuffisance Cardiaque Gauche", "I50.1", "ICD-10")
    termino = Terminology()
    termino.add(term1)
    termino.add(term2)
    return termino
