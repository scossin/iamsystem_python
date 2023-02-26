""" Utility classes and functions to store states transition during
annotation."""
from __future__ import annotations

from typing import Generic
from typing import List
from typing import Optional

from iamsystem.tokenization.api import TokenT
from iamsystem.tokenization.token import Token
from iamsystem.tree.nodes import INode


class LinkedState(Generic[TokenT]):
    """Keep track of the sequence of tokens in a document that matched
    the sequence of tokens of a keyword. This object is a linked list,
    the first element the create_start_state, others are transition_state."""

    def __init__(
        self,
        parent: Optional[LinkedState],
        node: INode,
        token: TokenT,
        algos: List[str],
        w_bucket: int,
    ):
        """

        :param parent: the previous state.
        :param node: the current state.
        :param token: token from a document, a generic type that implements
            :class:`~iamsystem.IToken` protocol.
        :param algos: the algorthim(s) that matched this state's token
            (from a keyword) to the document's token.
        :param w_bucket: the window bucket used to test if this instance
            becomes out-of-reach and should be remove.
        """
        self.node = node
        self.token = token
        self.parent = parent
        self.algos = algos
        self.w_bucket = w_bucket

    def __eq__(self, other):
        """Two nodes are equal if they have the same number."""
        # I removed these verifications to speed up the algorithm.
        # if self is other:
        #     return True
        # if isinstance(other, int):
        #     return self.node.node_num == other
        # if not isinstance(other, LinkedState):
        #     return False
        # other_state: LinkedState = other
        return self.node.node_num == other.node.node_num

    def __hash__(self):
        """Uses the node number as a unique identifier."""
        return self.node.node_num


def create_start_state(initial_state: INode):
    return LinkedState(
        parent=None,
        node=initial_state,
        token=Token(
            start=-1,
            end=-1,
            norm_label="START_TOKEN",
            i=-1,
            label="START_TOKEN",
        ),
        algos=[],
        w_bucket=-1,
    )
