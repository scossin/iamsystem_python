""" Tokenizer implementations. """

import re

from typing import Callable
from typing import Iterable
from typing import List
from typing import Sequence

from iamsystem.stopwords.api import IStopwords
from iamsystem.tokenization.api import IOffsets
from iamsystem.tokenization.api import ITokenizer
from iamsystem.tokenization.api import TokenT
from iamsystem.tokenization.normalize import lower_no_accents
from iamsystem.tokenization.normalize import normalizeFun
from iamsystem.tokenization.token import Offsets
from iamsystem.tokenization.token import Token


splitFun = Callable[[str], Iterable[IOffsets]]


def split_find_iter_closure(pattern: str) -> splitFun:
    """Build a split function that maps a document to (start, end) tuples.

    :param pattern: a regex to split sentence characters.
    :return: a split function.
    """
    pattern = pattern
    r = re.compile(pattern)

    def split(text: str) -> Iterable[IOffsets]:
        """Split the text with a regular expression"""
        match_iter = r.finditer(text)
        tokens = map(
            lambda match: Offsets(start=match.start(), end=match.end()),
            match_iter,
        )
        return tokens

    return split


split_alpha_num = split_find_iter_closure(pattern=r"\w+")


class TokenizerImp(ITokenizer[Token]):
    """A :class:`~iamsystem.ITokenizer` implementation.
    Class responsible for the tokenization, normalization of tokens.
    See also :func:`~iamsystem.french_tokenizer`,
    :func:`~iamsystem.english_tokenizer`.
    """

    def __init__(self, split: splitFun, normalize: normalizeFun):
        """Create a custom tokenizer that splits and normalizes a string.

        :param split: a function that split a text into (start,end) tuples.
          This function must return an iterable of :class:`~iamsystem.IOffsets`
          . See also :func:`~iamsystem.split_find_iter_closure`.
        :param normalize: a function that normalizes a string.
            This function must return a string.
        """
        self.split = split
        self.normalize = normalize

    def tokenize(self, text: str) -> Sequence[Token]:
        """Split the text into a sequence of :class:`~iamsystem.Token`."""
        offsets: Iterable[IOffsets] = self.split(text)
        tokens: List[Token] = [
            Token(
                start=offset.start,
                end=offset.end,
                label=text[offset.start : offset.end],  # noqa
                norm_label=self.normalize(
                    text[offset.start : offset.end]  # noqa
                ),
            )
            for offset in offsets
        ]
        return tokens


def remove_stopwords(
    tokens: Sequence[TokenT], stopwords=IStopwords[TokenT]
) -> Sequence[TokenT]:
    """Utility function to filter stopwords from a sequence of tokens"""
    tokens_wo_stop: Sequence[TokenT] = tuple(
        filter(lambda token: not stopwords.is_token_a_stopword(token), tokens)
    )
    return tokens_wo_stop


def french_tokenizer() -> TokenizerImp:
    """An opinionated French tokenizer.
     | It splits the text by 'word' character.
     | It normalizes by lowercasing and unicode normalization form.

    :return: a :class:`~iamsystem.TokenizerImp` implementation.
    """
    return TokenizerImp(split=split_alpha_num, normalize=lower_no_accents)


def english_tokenizer() -> TokenizerImp:
    """An opinionated English tokenizer.
     | It splits the text by 'word' character.
     | It normalizes by lowercasing.

    :return: a :class:`~iamsystem.TokenizerImp` implementation.
    """
    return TokenizerImp(split=split_alpha_num, normalize=lambda s: s.lower())


tokenize_fun = Callable[[str], Sequence[TokenT]]


def tokenize_and_order_decorator(tokenize: tokenize_fun) -> tokenize_fun:
    """Decorate a tokenize function: the tokens are sorted alphabetically
    by their label.

    :param tokenize: a tokenize function to decorate.
    :return: the decorated tokenize function.
    """

    def tokenize_and_order(text: str) -> List[TokenT]:
        """tokenize and sort."""
        tokens = list(tokenize(text))
        tokens.sort(key=lambda x: x.norm_label, reverse=False)
        return tokens

    return tokenize_and_order
