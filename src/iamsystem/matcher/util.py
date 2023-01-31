""" Utility classes and functions to store states transition during
annotation."""
from __future__ import annotations

from typing import Generic
from typing import List

from typing_extensions import Protocol

from iamsystem.tokenization.api import TokenT
from iamsystem.tree.nodes import INode


class IState(Protocol):
    """A class that keeps track of a state, stored as a node in a tree."""

    node: INode


class StartState(IState):
    """A class to store the starting/initial state."""

    def __init__(self, node: INode):
        """Stores the initial_state.

        :param node: the initial state/node, i.e. a root_node.
        """
        self.node = node


class TransitionState(IState, Generic[TokenT]):
    """Keep track of the sequence of tokens in a document that matched
    the sequence of tokens of a keyword. This object is a linked list,
    the first element the start_state, others are transition_state."""

    def __init__(
        self, parent: IState, node: INode, token: TokenT, algos: List[str]
    ):
        """

        :param parent: the previous state.
        :param node: the current state.
        :param token: token from a document, a generic type that implements
            :class:`~iamsystem.IToken` protocol.
        :param algos: the algorthim(s) that matched this state's token
            (from a keyword) to the document's token.
        """
        self.node = node
        self.token = token
        self.parent = parent
        self.algos = algos
