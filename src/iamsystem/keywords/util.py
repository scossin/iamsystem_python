from typing import Iterable
from typing import Sequence
from typing import Set

from iamsystem.keywords.api import IKeyword
from iamsystem.stopwords.api import IStopwords
from iamsystem.tokenization.api import IToken
from iamsystem.tokenization.api import ITokenizer
from iamsystem.tokenization.tokenize import remove_stopwords


def get_unigrams(
    keywords: Iterable[IKeyword], tokenizer: ITokenizer, stopwords: IStopwords
) -> Set[str]:
    """Get all the unigrams (single words excluding stopwords) in keywords."""
    unigram_set: Set[str] = set()
    for keyword in keywords:
        tokens: Sequence[IToken] = tokenizer.tokenize(keyword.label)
        tokens_wo_stop: Sequence[IToken] = remove_stopwords(
            tokens=tokens, stopwords=stopwords
        )
        for token in tokens_wo_stop:
            unigram_set.add(token.norm_label)
    return unigram_set
