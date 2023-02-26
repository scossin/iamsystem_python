from typing import Tuple

from iamsystem.fuzzy.abbreviations import Abbreviations
from iamsystem.keywords.collection import Terminology
from iamsystem.keywords.keywords import Entity
from iamsystem.matcher.util import LinkedState
from iamsystem.matcher.util import create_start_state
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


def get_gauche_el_in_ivg() -> Tuple[Node, LinkedState]:
    """Return a transition state."""
    # root_node
    trie = Trie()
    root_node = trie.get_initial_state()
    root_el = create_start_state(initial_state=trie.root_node)
    # insuffisance
    ins_node = Node(token="insuffisance", node_num=1, parent_node=root_node)
    ins_span = Token(
        label="Insuffisance", norm_label="insuffisance", start=0, end=12, i=0
    )
    ins_el: LinkedState = LinkedState(
        node=ins_node,
        token=ins_span,
        parent=root_el,
        algos=["exact"],
        w_bucket=0,
    )
    # ventriculaire
    vent_node = Node(token="ventriculaire", node_num=2, parent_node=ins_node)
    vent_span = Token(
        label="Ventriculaire",
        norm_label="ventriculaire",
        start=13,
        end=26,
        i=1,
    )
    vent_el: LinkedState = LinkedState(
        node=vent_node,
        token=vent_span,
        parent=ins_el,
        algos=["exact"],
        w_bucket=0,
    )
    # gauche
    gauche_node = Node(token="gauche", node_num=3, parent_node=vent_node)
    gauche_span = Token(
        label="Gauche", norm_label="gauche", start=28, end=34, i=2
    )
    gauche_el: LinkedState = LinkedState(
        node=gauche_node,
        token=gauche_span,
        parent=vent_el,
        algos=["exact"],
        w_bucket=0,
    )
    return gauche_node, gauche_el


class TermSubClass(Entity):
    """Add a termino attribute to a Term."""

    def __init__(self, label, kb_id, termino: str):
        super().__init__(label, kb_id)
        self.termino = termino


def get_ent_sub_class_ivg():
    """Returns ents of a custom Term subclass."""
    ent1 = TermSubClass("Insuffisance Cardiaque", "I50.9", "ICD-10")
    ent2 = TermSubClass("Insuffisance Cardiaque Gauche", "I50.1", "ICD-10")
    termino = Terminology()
    termino.add(ent1)
    termino.add(ent2)
    return termino
