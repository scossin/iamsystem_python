from typing_extensions import Protocol

from iamsystem.tree.nodes import INode


class IInitialState(Protocol):
    """A class that keeps an initial state."""

    def get_initial_state(self) -> INode:
        """Return an initial state."""
        raise NotImplementedError
