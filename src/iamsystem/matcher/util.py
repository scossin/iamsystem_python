""" Utility classes and functions to store states transition during
annotation."""
from __future__ import annotations

from typing import Generic
from typing import List
from typing import Optional

from iamsystem.tokenization.api import TokenT
from iamsystem.tokenization.token import Token
from iamsystem.tree.nodes import INode


class StateTransition(Generic[TokenT]):
    """Keep track of the sequence of tokens in a document that matched
    the sequence of tokens of a keyword. A state transition occurs between
    two states (nodes in a trie) with a word w. This object is a linked list,
    the first element is a start_state (root node in general)."""

    def __init__(
        self,
        previous_trans: Optional[StateTransition],
        node: INode,
        token: TokenT,
        algos: List[str],
        count_not_stopword: int,
    ):
        """

        :param previous_trans: the previous transition.
        :param node: the current state.
        :param algos: the algorthim(s) that matched that matched the token
            to the node.
        :param token: token from a document, a generic type that implements
            :class:`~iamsystem.IToken` protocol.
        :param count_not_stopword: the ith not stopword token. This value is
            used to check if a transition becomes out-of-reach and
            should be removed.
        """
        self.node = node
        self.token = token
        self.previous_trans = previous_trans
        self.algos = algos
        self.w_bucket = count_not_stopword
        self.id = node.node_num

    def is_obsolete(self, count_not_stopword: int, w: int) -> bool:
        """Check if a state transition is obsolete given a window size."""
        distance_2_current_token = count_not_stopword - self.w_bucket
        return (
            w - distance_2_current_token < 0
        ) and not StateTransition.is_first_trans(self)
        # Start state is never obsolete. It allows starting a new transition
        # sequence at any token (except stopword).

    def __eq__(self, other):
        """Two nodes are equal if they have the same number."""
        # No type checking to speed up the algorithm.
        return self.id == other.id

    def __hash__(self):
        """Use the node number as a unique identifier."""
        return self.id

    @classmethod
    def is_first_trans(cls, trans: StateTransition):
        """Check a transition is the first one."""
        return trans.previous_trans is None

    @classmethod
    def create_first_trans(cls, initial_state: INode):
        """Create the first transition with the initial state."""
        return StateTransition(
            previous_trans=None,
            node=initial_state,
            token=Token(
                start=-1,
                end=-1,
                norm_label="START_TOKEN",
                i=-1,
                label="START_TOKEN",
            ),
            algos=[],
            count_not_stopword=-1,
        )
