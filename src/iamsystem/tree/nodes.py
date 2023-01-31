""" Store all keywords' tokens in nodes to build a trie datastructure. """

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence

from iamsystem.keywords.api import IKeyword


class INode(ABC):
    """A node in a tree. Each node represents also a state of a final state
    automata. Special nodes are:
    * root_node: the initial state.
    * empty_node: an 'exit' node to go when no transition is found.
    * node:
        * transition node that stores a token but not a keyword.
        * final state that stores a token and a keyword. When the algorithm
        moves to a final state, an annotation is created.
    """

    def __init__(self, node_num: int, parent_node: INode):
        """
        :param node_num: A unique node number.
        :param parent_node: Another node.
        """
        self.node_num = node_num
        self.parent_node = parent_node

    @abstractmethod
    def add_child_node(self, node: INode) -> None:
        """Add a child node."""
        raise NotImplementedError

    @abstractmethod
    def get_child_nodes(self) -> Iterable[INode]:
        """Retrive child nodes."""
        raise NotImplementedError

    @abstractmethod
    def has_transition_to(self, token: str):
        """true if the child node matches this token."""
        raise NotImplementedError

    @abstractmethod
    def goto_node(self, token: str) -> "INode":
        """From this node try to go to a child node with this token."""
        raise NotImplementedError

    @abstractmethod
    def jump_to_node(self, str_seq: Sequence[str]) -> "INode":
        """From this node try to go to a child node with this sequence of
        tokens."""
        raise NotImplementedError

    @abstractmethod
    def is_a_final_state(self) -> bool:
        """A node is a final state iff he stores a keyword."""
        raise NotImplementedError

    @abstractmethod
    def get_ancestors(self) -> Sequence[INode]:
        """Retrieve all parent nodes."""
        raise NotImplementedError

    @abstractmethod
    def get_keywords(self) -> Optional[Sequence[IKeyword]]:
        """None if not a finale state, one to many otherwise."""
        raise NotImplementedError

    @abstractmethod
    def get_token(self) -> str:
        """the (normalized) token stored by this node."""
        raise NotImplementedError


class EmptyNode(INode):
    """An 'exit' node to go when no transition is found."""

    def __init__(self):
        """Build a unique empty node."""
        super().__init__(node_num=-1, parent_node=self)

    def goto_node(self, token: str) -> INode:
        """Returns itself."""
        return self

    def has_transition_to(self, token: str):
        """No transition whatsoever."""
        return False

    def jump_to_node(self, str_seq: Sequence[str]) -> INode:
        """Returns itself."""
        return EMPTY_NODE

    def is_a_final_state(self):
        """Never ever."""
        return False

    def add_child_node(self, node: INode) -> None:
        """This method shouldn't be called."""
        raise NotImplementedError

    def get_child_nodes(self) -> Iterable[INode]:
        """This method shouldn't be called."""
        raise NotImplementedError

    def get_ancestors(self) -> Sequence[INode]:
        """This method shouldn't be called."""
        raise NotImplementedError

    def get_keywords(self) -> Optional[Sequence[IKeyword]]:
        """This method shouldn't be called."""
        raise NotImplementedError

    def get_token(self) -> str:
        """This method shouldn't be called."""
        raise NotImplementedError


EMPTY_NODE = EmptyNode()


class Node(INode, ABC):
    """A node that stores a keyword's token."""

    node_root_number = 0
    "Default root node number."

    @classmethod
    def is_root_node(cls, node: INode) -> bool:
        """Checks if a node is the root node."""
        return node.node_num == Node.node_root_number

    def __init__(
        self, node_num: int, token: str, parent_node: INode = EMPTY_NODE
    ):
        """

        :param token: a keyword's token.
        :param node_num: A unique node number.
        :param parent_node: The node storing the preceding token.
        """
        super().__init__(node_num, parent_node)
        self.token = token
        self.keywords: Optional[List[IKeyword]] = None
        self.childNodes: Dict[str, INode] = {}
        if parent_node is not EMPTY_NODE:
            self.parent_node = parent_node
            self.parent_node.add_child_node(self)

    def has_transition_to(self, token: str):
        """Check if a child node stores this token."""
        return token in self.childNodes.keys()

    def goto_node(self, token: str) -> INode:
        """Move to another state or go to empty_node if no transition is
        found."""
        return self.childNodes.get(token, EMPTY_NODE)

    def jump_to_node(self, str_seq: Sequence[str]) -> INode:
        """Move to another state or go to empty_node if no transition is
        found."""
        if len(str_seq) == 0:
            return EMPTY_NODE
        node: INode = self
        for token in str_seq:
            node = node.goto_node(token)
        return node

    def add_child_node(self, node: INode) -> None:
        """Add a node that stores the next keyword's token."""
        token = node.get_token()
        self.childNodes[token] = node

    def is_a_final_state(self) -> bool:
        """This node is a final state iff it stores a keyword."""
        return self.keywords is not None

    def add_keyword(self, keyword: IKeyword) -> None:
        """Add a keyword to this node."""
        if self.keywords is None:
            self.keywords = []
        self.keywords.append(keyword)

    def get_ancestors(self) -> Sequence[INode]:
        """Return all ancestors of this node.
        Last element of the sequence is the root_node."""
        ancestors = []
        ancest = self.parent_node
        while ancest.node_num != 0:
            ancestors.append(ancest)
            ancest = ancest.parent_node
        return ancestors

    def get_child_nodes(self) -> Iterable[INode]:
        """Return all the child of this node that correspond to all
        possible state transitions."""
        return self.childNodes.values().__iter__()

    def get_keywords(self) -> Optional[Sequence[IKeyword]]:
        """Return the keywords associated to this node if it's a final
        state, None otherwise."""
        return self.keywords

    def get_token(self) -> str:
        """Return the token associated to this node."""
        return self.token

    def __eq__(self, other):
        """Two nodes are equal if they have the same number."""
        if self is other:
            return True
        if not isinstance(other, Node):
            return False
        other_node: Node = other
        return self.node_num == other_node.node_num

    def __hash__(self):
        """Uses the node number as a unique identifier."""
        return self.node_num


def _create_a_root_node() -> Node:
    """Create a new root_node."""
    return Node(token="START_TOKEN", node_num=Node.node_root_number)
