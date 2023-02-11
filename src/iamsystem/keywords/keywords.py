""" Keywords implementation."""
from dataclasses import asdict
from dataclasses import dataclass

from iamsystem.keywords.api import IEntity
from iamsystem.keywords.api import IKeyword


# I considered TypedDict to store keywords but type checking is complicated.
# See https://dev.to/meeshkan/typeddict-vs-dataclasses-in-python-epic-typing-battle-onb # noqa


@dataclass
class Keyword(IKeyword):
    """Base class to search keywords in a document.

    Args:
        label (str): The string to search.
    """

    label: str

    def asdict(self):
        """Returns the fields of the dataclass instance."""
        return asdict(
            self,
        )

    def __str__(self):
        """Return a string representation."""
        return f"{self.label}"


@dataclass
class Entity(Keyword, IEntity):
    """An entity of a knowledge base.

    Args:
        kb_id (str): The entity id in the knowledge base.
    """

    kb_id: str

    def __str__(self):
        """An opinionated string representation."""
        return f"{self.label} ({self.kb_id})"
