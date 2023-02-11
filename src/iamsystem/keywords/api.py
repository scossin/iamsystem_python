""" Declare Keyword Interface;"""
from typing import Collection

from typing_extensions import Protocol
from typing_extensions import runtime_checkable


@runtime_checkable
class IKeyword(Protocol):
    """A string to search in a document (ex: "heart failure")."""

    label: str


@runtime_checkable
class IEntity(IKeyword, Protocol):
    """An entity of a knowledge base."""

    kb_id: str


@runtime_checkable
class IStoreKeywords(Protocol):
    @property
    def keywords(self) -> Collection[IKeyword]:
        """Get the keywords."""
        raise NotImplementedError

    def add_keyword(self, keyword: IKeyword) -> None:
        """Add keywords to the store."""
        raise NotImplementedError
