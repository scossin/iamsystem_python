""" Fuzzy algorithm that needs to handle a regular expression."""
import re

from copy import deepcopy
from typing import Iterable

from iamsystem.fuzzy.api import ContextFreeAlgo
from iamsystem.fuzzy.api import FuzzyAlgo
from iamsystem.fuzzy.api import INormLabelAlgo
from iamsystem.fuzzy.api import SynType
from iamsystem.keywords.api import IKeyword
from iamsystem.tokenization.api import ITokenizer
from iamsystem.tokenization.api import TokenT
from iamsystem.tokenization.util import replace_offsets_by_new_str


class FuzzyRegex(ContextFreeAlgo, INormLabelAlgo):
    """A :class:`~iamsystem.FuzzyAlgo` to handle regular expressions.
    Useful when one or multiple tokens of a keyword need to be matched
    to a regular expression.
    """

    def __init__(self, name: str, pattern: str, pattern_name: str):
        """Create a FuzzyRegex instance.

        :param name: a name given to this algorithm.
        :param pattern: a regular expression.
        :param pattern_name: a name given to this pattern (ex: 'numval')
            that is also a token of a :class:`~iamsystem.IKeyword`.
        """
        super().__init__(name=name)
        self.pattern_name = pattern_name
        self.r = re.compile(pattern)  # r"^\d*[.,]?\d*$"

    def token_matches_pattern(self, token: TokenT) -> bool:
        """Return True if this token matches this instance's pattern."""
        return bool(self.r.match(token.label))

    def replace_pattern_in_keyword(
        self, keyword: IKeyword, tokenizer: ITokenizer
    ) -> IKeyword:
        """Utility function to replace keyword's tokens that match the pattern
        by the pattern name."""
        clone_keyword = deepcopy(keyword)
        tokens = tokenizer.tokenize(text=clone_keyword.label)
        tokens_matched = [
            token for token in tokens if self.r.match(token.label)
        ]
        tokens_match_new_str = [
            (token, self.pattern_name) for token in tokens_matched
        ]
        new_label = replace_offsets_by_new_str(
            text=clone_keyword.label, offsets_new_str=tokens_match_new_str
        )
        clone_keyword.label = new_label
        return clone_keyword

    def get_syns_of_token(self, token: TokenT) -> Iterable[SynType]:
        """Return the pattern_name if this token matches the regular
        expression."""
        if self.token_matches_pattern(token=token):
            return [self.word_to_syn(word=self.pattern_name)]
        else:
            return FuzzyAlgo.NO_SYN

    def get_syns_of_word(self, word: str) -> Iterable[SynType]:
        """Return the pattern_name if this word matches it."""
        if self.r.match(word):
            return [self.word_to_syn(word=self.pattern_name)]
        else:
            return FuzzyAlgo.NO_SYN
