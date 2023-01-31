""" Keyword iterables."""
from typing import Collection
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Set

from iamsystem.keywords.api import IKeyword
from iamsystem.keywords.api import IStoreKeywords
from iamsystem.keywords.util import get_unigrams
from iamsystem.stopwords.api import IStopwords
from iamsystem.tokenization.api import ITokenizer


class Terminology(IStoreKeywords):
    """A utility class to store a set of keywords."""

    def __init__(self):
        self._keywords: List[IKeyword] = []
        # add an alias to shorten function name:
        self.add = self.add_keyword

    def add_keyword(self, keyword: IKeyword) -> None:
        """Add a keyword.

        :param keyword: a :class:`~iamsystem.IKeyword` or a subclass.
        :return: None
        """
        self._keywords.append(keyword)

    def add_keywords(self, keywords: Iterable[IKeyword]) -> None:
        """Add multiple keywords.

        :param keywords: a :class:`~iamsystem.IKeyword` or a subclass.
        :return: None
        """
        for keyword in keywords:
            self.add_keyword(keyword)

    @property
    def size(self) -> int:
        """Get the number of keywords."""
        return len(self._keywords)

    @property
    def keywords(self) -> Collection[IKeyword]:
        """Get the collection of keywords."""
        return self._keywords

    def get_unigrams(
        self, tokenizer: ITokenizer, stopwords: IStopwords
    ) -> Set[str]:
        """Get all the unigrams (single words excluding stopwords)
        in the keywords."""
        return get_unigrams(self, tokenizer=tokenizer, stopwords=stopwords)

    def __iter__(self) -> Iterator[IKeyword]:
        """Return a keyword iterator."""
        return iter(self._keywords)
