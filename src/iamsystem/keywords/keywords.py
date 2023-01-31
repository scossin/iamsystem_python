""" Keywords implementation."""
from iamsystem.keywords.api import IKeyword


class Keyword(IKeyword):
    """Base class to search keywords in a document."""

    def __init__(self, label: str):
        """Create a keyword.

        :param label: a string to search in a document (ex: "heart failure").
        """
        self.label = label

    def get_kb_id(self):
        """Get the knowledge base id of this keyword.
        It returns the label if this method is not overriden in the subclass.

        :return: A unique identifier.
        """
        return self.label

    def __str__(self):
        """Return the only attribute."""
        return self.label


class Term(Keyword):
    """This class represents a term in a particular domain where each
    keyword is associated to a unique identifier called a code."""

    def __init__(self, label: str, code: str):
        """Create a term.

        :param label: a string to search in a document (ex: "heart failure").
        :param code: the code associated to this keyword.
        """
        super().__init__(label)
        self.code = code

    def get_kb_id(self):
        """returns the code of this term."""
        return self.code

    def __str__(self):
        """An opinionated string representation."""
        return f"{self.label} ({self.code})"
