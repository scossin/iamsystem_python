""" String normalization functions. """
from typing import Callable

from anyascii import anyascii  # type: ignore


normalizeFun = Callable[[str], str]


def lower_no_accents(string: str) -> str:
    """Remove accents and lowercase the string."""
    return _remove_accents(string).lower()


def _remove_accents(string: str) -> str:
    """Remove accents with anyascii library."""
    unaccented_string: str = anyascii(string.replace("μ", "u"))
    return unaccented_string
