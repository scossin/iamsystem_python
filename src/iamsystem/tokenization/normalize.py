""" String normalization functions. """
from typing import Callable

from unidecode import unidecode_expect_ascii  # type: ignore


normalizeFun = Callable[[str], str]


def lower_no_accents(string: str) -> str:
    """Remove accents and lowercase the string."""
    return _remove_accents(string).lower()


def _remove_accents(string: str) -> str:
    """Remove accents with unidecode library."""
    unaccented_string: str = unidecode_expect_ascii(string)
    return unaccented_string
