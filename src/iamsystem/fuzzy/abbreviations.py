""" Handle abbreviations. """
from collections import defaultdict
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Sequence

from iamsystem.fuzzy.api import ContextFreeAlgo
from iamsystem.fuzzy.api import FuzzyAlgo
from iamsystem.fuzzy.api import INormLabelAlgo
from iamsystem.fuzzy.api import SynType
from iamsystem.tokenization.api import IToken
from iamsystem.tokenization.api import ITokenizer
from iamsystem.tokenization.api import TokenT


token_is_an_abb = Callable[[TokenT], bool]


def token_is_upper_case(token: TokenT) -> bool:
    """Return True if all token's characters are uppercase."""
    return token.label.isupper()


class Abbreviations(ContextFreeAlgo[TokenT], INormLabelAlgo):
    """A :class:`~iamsystem.FuzzyAlgo` to handle abbreviations.
    This class doesn't take into account the context of a document to return
    a long form.
    """

    def __init__(
        self,
        name: str,
        token_is_an_abbreviation: token_is_an_abb = lambda token: True,
    ):
        """Create an instance to store abbreviations.

        :param name: a name given to this algorithm. (ex: 'medical abbs')
        :param token_is_an_abbreviation: a function that verify if a token is
         an abbreviation (ex: checks all letters are uppercase).
         The function is called before the dictionary look-up
         is performed to retrieve long forms.
         Default: no checks performed, the function returns always true.
        """
        super().__init__(name)
        self.is_token_an_abbreviation = token_is_an_abbreviation
        self.abbs: Dict[str, List[SynType]] = defaultdict(list)

    def add(
        self, short_form: str, long_form: str, tokenizer: ITokenizer
    ) -> None:
        """Add an abbreviation.

        :param short_form: an abbreviation short form (ex: CHF).
        :param long_form: an abbreviation long form.
            (ex: congestive heart failure).
        :param tokenizer: a :class:`~iamsystem.ITokenizer` to tokenize
            the long form. It is recommanded to use your
            :class:`~iamsystem.Matcher` tokenizer.
        :return: None.
        """
        # the short_form is lower cased to be stored in a dictionary. The
        # responsability to check case sensitivity depends on the
        # 'token_is_an_abbreviation' function.
        short_form = short_form.lower()
        tokens: Sequence[IToken] = tokenizer.tokenize(long_form)
        syn = self.words_seq_to_syn(
            words=[token.norm_label for token in tokens]
        )
        self.abbs[short_form].append(syn)

    def add_tokenized_long_form(
        self, short_form, long_form: Sequence[str]
    ) -> None:
        """Add an abbreviation already tokenized."""
        syn = self.words_seq_to_syn(words=long_form)
        self.abbs[short_form].append(syn)

    def get_syns_of_token(self, token: TokenT) -> Iterable[SynType]:
        """Return the abbreviation long form(s)."""
        if not self.is_token_an_abbreviation(token):
            return FuzzyAlgo.NO_SYN
        return self.get_syns_of_word(token.norm_label)

    def get_syns_of_word(self, word: str) -> Iterable[SynType]:
        """Return the abbreviation long form(s)."""
        return self.abbs.get(word, FuzzyAlgo.NO_SYN)
