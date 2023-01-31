""" Main public API of the package, it annotates a document with keywords."""
from __future__ import annotations

import warnings

from collections import defaultdict
from typing import Callable
from typing import Collection
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Set

from iamsystem import Terminology
from iamsystem.fuzzy.api import FuzzyAlgo
from iamsystem.fuzzy.api import ISynsProvider
from iamsystem.fuzzy.api import SynAlgos
from iamsystem.fuzzy.exact import ExactMatch
from iamsystem.keywords.api import IKeyword
from iamsystem.keywords.api import IStoreKeywords
from iamsystem.keywords.keywords import Keyword
from iamsystem.keywords.util import get_unigrams
from iamsystem.matcher.annotation import Annotation
from iamsystem.matcher.annotation import create_annot
from iamsystem.matcher.annotation import rm_nested_annots
from iamsystem.matcher.annotation import sort_annot
from iamsystem.matcher.api import IMatcher
from iamsystem.matcher.util import IState
from iamsystem.matcher.util import StartState
from iamsystem.matcher.util import TransitionState
from iamsystem.stopwords.api import ISimpleStopwords
from iamsystem.stopwords.api import IStopwords
from iamsystem.stopwords.api import IStoreStopwords
from iamsystem.stopwords.simple import Stopwords
from iamsystem.tokenization.api import ITokenizer
from iamsystem.tokenization.api import TokenT
from iamsystem.tokenization.tokenize import french_tokenizer
from iamsystem.tree.nodes import EMPTY_NODE
from iamsystem.tree.nodes import INode
from iamsystem.tree.trie import Trie


filter_fun = Callable[[Annotation[TokenT]], Annotation[TokenT]]


class Matcher(IMatcher[TokenT]):
    """Main public API to perform semantic annotation (aka entity linking)
    with iamsystem algorithm."""

    def __init__(
        self,
        tokenizer: ITokenizer = french_tokenizer(),
        stopwords: IStopwords[TokenT] = None,
    ):
        """Create an IAMsystem matcher to annotate documents.

        :param tokenizer: default :func:`~iamsystem.french_tokenizer`.
            A :class:`~iamsystem.ITokenizer` instance responsible for
            tokenizing and normalizing.
        :param stopwords: provide a :class:`~iamsystem.IStopwords`.
            If None, default to :class:`~iamsystem.Stopwords`.
        """
        self._tokenizer = tokenizer
        self._fuzzy_algos: List[FuzzyAlgo[TokenT]] = [ExactMatch()]
        self._trie: Trie = Trie()
        self._termino: IStoreKeywords = Terminology()
        self._remove_nested_annots = True
        if stopwords is not None:
            self._stopwords = stopwords
        else:
            self._stopwords = Stopwords()

    @property
    def remove_nested_annots(self) -> bool:
        """Matcher config: whether to remove nested annotations.
        Default to True."""
        return self._remove_nested_annots

    @remove_nested_annots.setter
    def remove_nested_annots(self, remove_nested_annots: bool) -> None:
        """Set remove nested annotation value. Default to True.

        :param remove_nested_annots: if two annotations overlap,
            remove the shorter one. Default to True since
            longest annotations are often more specific than shorter ones.
        :return: None
        """
        self._remove_nested_annots = remove_nested_annots

    @property
    def fuzzy_algos(self) -> Iterable[FuzzyAlgo[TokenT]]:
        """The fuzzy algorithms used by the algorithm.

        :return: :class:`~iamsystem.FuzzyAlgo` instances responsible for
            finding possible synonyms for each token of a document.
        """
        return self._fuzzy_algos

    def is_token_a_stopword(self, token: TokenT) -> bool:
        """Check if a token is a stopword.

        :param token: a generic token that implements
          :class:`~iamsystem.IToken`.
        :return: True if the token is a stopword.
        """
        return self._stopwords.is_token_a_stopword(token=token)

    def is_stopword(self, word: str) -> bool:
        """Return True if word is a stopword."""
        if isinstance(self._stopwords, ISimpleStopwords):
            return self._stopwords.is_stopword(word=word)
        else:
            warnings.warn(
                f"{self._stopwords.__class__} does not implement this method."
            )
            return False

    def tokenize(self, text: str) -> Sequence[TokenT]:
        """Tokenize a text with the tokenizer's instance.

        :param text: a document or a keyword.
        :return: A sequence of tokens, the type depends on the tokenizer but
            must implement :class:`~iamsystem.IToken` protocol.
        """
        return self._tokenizer.tokenize(text=text)

    def add_labels(self, labels: Iterable[str]) -> None:
        """Utility function to call 'add_keywords' by providing a list of
        labels, :class:`~iamsystem.IKeyword` instances are created and added.

        :param labels: the labels (keywords) to be searched in the document.
        :return: None.
        """
        keywords = [Keyword(label=label) for label in labels]
        self.add_keywords(keywords=keywords)

    def add_keyword(self, keyword: IKeyword) -> None:
        """Add a keyword to find in a document.

        :param keyword: :class:`~iamsystem.IKeyword` to search in a document.
        :return: None.
        """
        self._termino.add_keyword(keyword=keyword)
        self._trie.add_keyword(
            keyword=keyword,
            tokenizer=self,
            stopwords=self,
        )

    def add_keywords(self, keywords: Iterable[IKeyword]) -> None:
        """Utility function to add multiple keywords.

        :param keywords: :class:`~iamsystem.IKeyword` to search in a document.
        :return: None.
        """
        for keyword in keywords:
            self.add_keyword(keyword=keyword)

    @property
    def keywords(self) -> Collection[IKeyword]:
        """Return the keywords added."""
        return self._termino.keywords

    def get_keywords_unigrams(self) -> Set[str]:
        """Get all the unigrams (single words excluding stopwords)
        in the keywords."""
        return get_unigrams(
            keywords=self.keywords,
            tokenizer=self,
            stopwords=self,
        )

    def add_stopwords(self, words: Iterable[str]) -> None:
        """Add words (tokens) to be ignored in :class:`~iamsystem.IKeyword`
        and in documents.

        :param words: a list of words to ignore.
        :return: None.
        """
        if isinstance(self._stopwords, IStoreStopwords):
            self._stopwords.add(words=words)
        else:
            warnings.warn(
                f"Adding stopwords have no effect on class "
                f"{self._stopwords.__class__}"
            )

    def add_fuzzy_algo(self, fuzzy_algo: FuzzyAlgo[TokenT]) -> None:
        """Add a fuzzy algorithms to provide synonym(s) that helps matching
            a token of a document and a token of a keyword.

        :param fuzzy_algo: a :class:`~iamsystem.FuzzyAlgo` instance.
        :return: None.
        """
        self._fuzzy_algos.append(fuzzy_algo)

    def get_synonyms(
        self, tokens: Sequence[TokenT], i: int, w_states: List[List[IState]]
    ) -> Iterable[SynAlgos]:
        """Get synonyms of a token with configured fuzzy algorithms.

        :param tokens: document's tokens.
        :param i: the ith token for which synonyms are expected.
        :param w_states: algorithm's states.
        :return: tuples of synonyms and fuzzy algorithm's names.
        """
        syns_collector = defaultdict(list)
        for algo in self.fuzzy_algos:
            for syn, algo_name in algo.get_synonyms(
                tokens=tokens, i=i, w_states=w_states
            ):
                syns_collector[syn].append(algo_name)
        synonyms: List[SynAlgos] = list(syns_collector.items())
        return synonyms

    def annot_text(self, text: str, w: int = 1) -> List[Annotation[TokenT]]:
        """Annotate a document.

        :param text: the document to annotate.
        :param w: Window. How much discontinuous keyword's tokens to find
            can be. By default, w=1 means the sequence must be continuous.
            w=2 means each token can be separated by another token.
        :return: a list of :class:`~iamsystem.Annotation`.
        """
        tokens: Sequence[TokenT] = self.tokenize(text)
        return self.annot_tokens(tokens=tokens, w=w)

    def annot_tokens(
        self, tokens: Sequence[TokenT], w: int
    ) -> List[Annotation[TokenT]]:
        """Annotate a sequence of tokens.

        :param tokens: an ordered or unordered sequence of tokens.
        :param w: Window. How much discontinuous keyword's tokens
            to find can be. By default, w=1 means the sequence must be
            continuous. w=2 means each token can be separated by another token.
        :param remove_nested_annots: if two annotations overlap,
            remove the shorter one.
        :return: a list of :class:`~iamsystem.Annotation`.
        """
        annots = detect(
            tokens=tokens,
            w=w,
            initial_state=self._trie.root_node,
            syns_provider=self,
            stopwords=self,
        )
        if self._remove_nested_annots:
            annots = rm_nested_annots(annots=annots, keep_ancestors=False)
        return annots


def detect(
    tokens: Sequence[TokenT],
    w: int,
    initial_state: INode,
    syns_provider: ISynsProvider,
    stopwords: IStopwords,
) -> List[Annotation[TokenT]]:
    """Main internal function that implements iamsystem's algorithm.
    Algorithm formalized in https://ceur-ws.org/Vol-3202/livingner-paper11.pdf

    :param tokens: a sequence of :class:`~iamsystem.IToken`.
    :param w: window, how many previous tokens can the algorithm look at.
    :param initial_state: a node/state in the trie, i.e. the root node.
    :param syns_provider: a class that provides synonyms for each token.
    :param stopwords: an instance of :class:`~iamsystem.IStopwords`
    that checks if a token is a stopword.
    :return: A list of :class:`~iamsystem.Annotation`.
    """
    annots: List[Annotation] = []
    # +1 to insert the start_state.
    w_states: List[List[IState]] = [[]] * (w + 1)
    start_state = StartState(node=initial_state)
    # [w] element stores only the start_state. This element is not replaced.
    w_states[w] = [start_state]
    # different from i for a stopword-independent window size.
    count_not_stopword = 0
    for i, token in enumerate(tokens):
        if stopwords.is_token_a_stopword(token):
            continue
        count_not_stopword += 1
        syns_algos: Iterable[SynAlgos] = syns_provider.get_synonyms(
            tokens=tokens, i=i, w_states=w_states
        )
        # stores matches between document's tokens and keywords'tokens.
        tokens_states: List[TransitionState] = []

        # 1 to many synonyms depending on fuzzy_algos configuration.
        for syn, algos in syns_algos:
            # 0 to many states for [0] to [w-1] ; [w] only the start state.
            for states in w_states:
                for state in states:
                    new_state = state.node.jump_to_node(syn)
                    # when no path is found, EMPTY_NODE is returned.
                    if new_state is EMPTY_NODE:
                        continue
                    token_state = TransitionState(
                        parent=state, node=new_state, token=token, algos=algos
                    )
                    tokens_states.append(token_state)
                    if new_state.is_a_final_state():
                        annot = create_annot(last_el=token_state)
                        annots.append(annot)
        # function 'count_not_stopword % w' has range [0 ; w-1]
        w_states[count_not_stopword % w].clear()
        w_states[count_not_stopword % w] = tokens_states
        # Mypy: Incompatible types in assignment (expression has type
        # "List[TokenState[Any]]", target has type "List[State]")
        # but TokenState is a sublcass of State.
    sort_annot(annots)  # mutate the list like annots.sort()
    return annots
