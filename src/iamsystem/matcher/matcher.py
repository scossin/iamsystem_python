""" Main public API of the package."""
from __future__ import annotations

import typing
import warnings

from collections import defaultdict
from typing import Any
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Union

from iamsystem.fuzzy.abbreviations import Abbreviations
from iamsystem.fuzzy.api import FuzzyAlgo
from iamsystem.fuzzy.api import INormLabelAlgo
from iamsystem.fuzzy.api import ISynsProvider
from iamsystem.fuzzy.api import SynAlgos
from iamsystem.fuzzy.cache import CacheFuzzyAlgos
from iamsystem.fuzzy.exact import ExactMatch
from iamsystem.fuzzy.norm_fun import WordNormalizer
from iamsystem.fuzzy.regex import FuzzyRegex
from iamsystem.fuzzy.simstring import SimStringWrapper
from iamsystem.fuzzy.spellwise import SpellWiseWrapper
from iamsystem.fuzzy.util import SimpleWords2ignore
from iamsystem.keywords.api import IKeyword
from iamsystem.keywords.api import IStoreKeywords
from iamsystem.keywords.collection import Terminology
from iamsystem.keywords.keywords import Keyword
from iamsystem.keywords.util import get_unigrams
from iamsystem.matcher.annotation import Annotation
from iamsystem.matcher.annotation import create_annot
from iamsystem.matcher.annotation import rm_nested_annots
from iamsystem.matcher.annotation import sort_annot
from iamsystem.matcher.api import IMatcher
from iamsystem.matcher.util import LinkedState
from iamsystem.matcher.util import create_start_state
from iamsystem.stopwords.api import ISimpleStopwords
from iamsystem.stopwords.api import IStopwords
from iamsystem.stopwords.api import IStoreStopwords
from iamsystem.stopwords.negative import NegativeStopwords
from iamsystem.stopwords.simple import NoStopwords
from iamsystem.stopwords.simple import Stopwords
from iamsystem.tokenization.api import ITokenizer
from iamsystem.tokenization.api import TokenT
from iamsystem.tokenization.tokenize import french_tokenizer
from iamsystem.tokenization.tokenize import tokenize_and_order_decorator
from iamsystem.tree.nodes import EMPTY_NODE
from iamsystem.tree.nodes import INode
from iamsystem.tree.trie import Trie


class Matcher(IMatcher[TokenT]):
    """Main public API to perform semantic annotation (aka entity linking)
    with iamsystem algorithm."""

    def __init__(
        self,
        tokenizer: ITokenizer = french_tokenizer(),
        stopwords: IStopwords[TokenT] = None,
    ):
        """Create an IAMsystem matcher to annotate documents.
        Prefer :py:meth:`~iamsystem.Matcher.build` method to create a matcher.

        :param tokenizer: default :func:`~iamsystem.french_tokenizer`.
            A :class:`~iamsystem.ITokenizer` instance responsible for
            tokenizing and normalizing.
        :param stopwords: a :class:`~iamsystem.IStopwords` to ignore empty
            words in keywords and documents.
            If None, default to :class:`~iamsystem.Stopwords`.
        """
        self._w = 1
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
    def stopwords(self) -> IStopwords[TokenT]:
        """Return the :class:`~iamsystem.IStopwords` used by the matcher."""
        return self._stopwords

    @stopwords.setter
    def stopwords(self, stopwords: IStopwords[TokenT]) -> None:
        """Set the stopwords.
        Note that keywords already added will not be modified.

        :param stopwords: a :class:`~iamsystem.IStopwords` to ignore empty
            words in keywords and documents.
        :return: None
        """
        self._stopwords = stopwords

    @property
    def tokenizer(self) -> ITokenizer[TokenT]:
        """Return the :class:`~iamsystem.ITokenizer` used by the matcher."""
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: ITokenizer[TokenT]) -> None:
        """Change the tokenizer.
        Note that keywords already added will not be modified.

        :param tokenizer: A :class:`~iamsystem.ITokenizer` instance
            responsible for tokenizing and normalizing.
        :return: None
        """
        self._tokenizer = tokenizer

    @property
    def remove_nested_annots(self) -> bool:
        """Whether to remove nested annotations. Default to True."""
        return self._remove_nested_annots

    @remove_nested_annots.setter
    def remove_nested_annots(self, remove_nested_annots: bool) -> None:
        """Set remove_nested_annots value. Default to True.

        :param remove_nested_annots: if two annotations overlap,
            remove the shorter one. Default to True since
            longest annotations are often more specific than shorter ones.
        :return: None
        """
        self._remove_nested_annots = remove_nested_annots

    @property
    def w(self) -> int:
        """Return the window parameter of this matcher."""
        return self._w

    @w.setter
    def w(self, value: int) -> None:
        """Set the window parameter. Default to 1.

        :param value:  How much discontinuous keyword's tokens
            to find can be. By default, w=1 means the sequence must be
            continuous. w=2 means each token can be separated by another token.
        :return: None
        """
        self._w = value

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
                f"{self._stopwords.__class__.__name__} "
                f"does not implement this method."
            )
            return False

    def tokenize(self, text: str) -> Sequence[TokenT]:
        """Tokenize a text with the tokenizer's instance.

        :param text: a document or a keyword.
        :return: A sequence of tokens, the type depends on the tokenizer but
            must implement :class:`~iamsystem.IToken` protocol.
        """
        return self._tokenizer.tokenize(text=text)

    def add_keywords(
        self, keywords: Iterable[Union[str, IKeyword, Dict[Any]]]
    ) -> None:
        """Utility function to add multiple keywords.

        :param keywords: an iterable of string (labels) or
            :class:`~iamsystem.IKeyword` to search in a document.
        :return: None.
        """
        for kw in keywords:
            if isinstance(kw, str):
                kw = Keyword(label=kw)
            self.add_keyword(keyword=kw)

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
                f"{self._stopwords.__class__.__name__}"
            )

    def add_fuzzy_algo(self, fuzzy_algo: FuzzyAlgo[TokenT]) -> None:
        """Add a fuzzy algorithms to provide synonym(s) that helps matching
            a token of a document and a token of a keyword.

        :param fuzzy_algo: a :class:`~iamsystem.FuzzyAlgo` instance.
        :return: None.
        """
        self._fuzzy_algos.append(fuzzy_algo)

    def get_initial_state(self) -> INode:
        """Return the initial state from which iamsystem algorithm will start
        searching for a sequence of keywords'tokens."""
        return self._trie.get_initial_state()

    def get_synonyms(
        self,
        tokens: Sequence[TokenT],
        token: TokenT,
        states: Set[LinkedState],
    ) -> List[SynAlgos]:
        """Get synonyms of a token with configured fuzzy algorithms.

        :param tokens: document's tokens.
        :param token: the token for which synonyms are expected.
        :param states: algorithm's states.
        :return: tuples of synonyms and fuzzy algorithm's names.
        """
        syns_collector = defaultdict(list)
        for algo in self.fuzzy_algos:
            for syn, algo_name in algo.get_synonyms(
                tokens=tokens, token=token, states=states
            ):
                syns_collector[syn].append(algo_name)
        synonyms: List[SynAlgos] = list(syns_collector.items())
        return synonyms

    def annot_text(self, text: str) -> List[Annotation[TokenT]]:
        """Annotate a document.

        :param text: the document to annotate.
        :return: a list of :class:`~iamsystem.Annotation`.
        """
        tokens: Sequence[TokenT] = self.tokenize(text)
        return self.annot_tokens(tokens=tokens)

    def annot_tokens(
        self, tokens: Sequence[TokenT]
    ) -> List[Annotation[TokenT]]:
        """Annotate a sequence of tokens.

        :param tokens: an ordered or unordered sequence of tokens.
        :return: a list of :class:`~iamsystem.Annotation`.
        """
        annots = detect(
            tokens=tokens,
            w=self.w,
            initial_state=self.get_initial_state(),
            syns_provider=self,
            stopwords=self,
        )
        if self._remove_nested_annots:
            annots = rm_nested_annots(annots=annots, keep_ancestors=False)
        return annots

    @classmethod
    def build(
        cls,
        keywords: Iterable[Union[str, IKeyword]],
        tokenizer: ITokenizer = None,
        stopwords: Union[IStopwords[TokenT], Iterable[str]] = NoStopwords(),
        w=1,
        order_tokens=False,
        negative=False,
        remove_nested_annots=True,
        string_distance_ignored_w: Optional[Iterable[str]] = None,
        abbreviations: Optional[Iterable[Tuple[str, str]]] = None,
        spellwise: Optional[List[Dict[Any, Any]]] = None,
        simstring: Optional[List[Dict[Any, Any]]] = None,
        normalizers: Optional[List[Dict[Any, Any]]] = None,
        fuzzy_regex: Optional[List[Dict[Any, Any]]] = None,
    ) -> Matcher[TokenT]:
        """
        Create an IAMsystem matcher to annotate documents.

        :param keywords: an iterable of keywords string or
            :class:`~iamsystem.IKeyword` instances.
        :param tokenizer: default :func:`~iamsystem.french_tokenizer`.
            A :class:`~iamsystem.ITokenizer` instance responsible for
            tokenizing and normalizing.
        :param stopwords: provide a :class:`~iamsystem.IStopwords`.
            If None, default to :class:`~iamsystem.NoStopwords`.
        :param w: Window. How much discontinuous keyword's tokens
            to find can be. By default, w=1 means the sequence must be
            continuous. w=2 means each token can be separated by another token.
        :param order_tokens: order tokens alphabetically if order doesn't
            matter in the matching strategy.
        :param negative: every unigram not in the keywords is a stopword.
            Default to False. If stopwords are also passed, they will be
            removed from keywords' tokens and so still be stopwords.
        :param remove_nested_annots: if two annotations overlap,
            remove the shorter one. Default to True.
        :param string_distance_ignored_w: words ignored by string distance
            algorithms to avoid false positives matched.
        :param abbreviations: an iterable of tuples (short_form, long_form).
        :param spellwise: an iterable of :class:`~iamsystem.SpellWiseWrapper`
            init parameters. if 'string_distance_ignored_w' is set, these words
            are passed to SpellWiseWrapper init function.
        :param simstring: an iterable of :class:`~iamsystem.SimStringWrapper`
            init parameters. if 'string_distance_ignored_w' is set, these words
            are passed to SimStringWrapper init function.
        :param normalizers: an iterable of :class:`~iamsystem.WordNormalizer`
            init parameters.
        :param fuzzy_regex: an iterable of :class:`~iamsystem.FuzzyRegex`
            init parameters.
        """

        # Tokenizer configuration
        if tokenizer is None:
            tokenizer = french_tokenizer()

        # Decorate tokenize function to order alphabetically
        if order_tokens:
            tokenizer.tokenize = tokenize_and_order_decorator(
                tokenizer.tokenize
            )

        # Start building and configuring the matcher
        matcher: Matcher[TokenT] = Matcher(tokenizer=tokenizer)

        # Configure stopwords
        if isinstance(stopwords, Iterable):
            matcher.add_stopwords(words=stopwords)
        elif isinstance(stopwords, IStopwords):
            matcher.stopwords = stopwords

        # Configure annot_text function
        matcher.w = w
        matcher.remove_nested_annots = remove_nested_annots

        # Add the keywords
        matcher.add_keywords(keywords=keywords)

        # add negative stopwords after stopwords and keywords are added
        # since this class needs keywords'unigrams without stopwords.
        if negative:
            matcher.stopwords = NegativeStopwords(
                words_to_keep=matcher.get_keywords_unigrams()
            )

        # fuzzy algorithms parameterization

        def _add_algo_in_cache_closure(
            cache: CacheFuzzyAlgos, matcher: Matcher
        ):
            """Internal function to add cache_fuzzy algorithm to cache"""

            def add_algo_in_cache(algo=INormLabelAlgo):
                """Add an algorithm in cache."""
                if cache not in matcher.fuzzy_algos:
                    matcher.add_fuzzy_algo(fuzzy_algo=cache)
                cache.add_algo(algo=algo)

            return add_algo_in_cache

        cache = CacheFuzzyAlgos()
        add_algo_in_cache = _add_algo_in_cache_closure(
            cache=cache, matcher=matcher
        )

        # Abbreviations
        if abbreviations is not None:
            _abbreviations = Abbreviations(name="abbs")
            matcher.add_fuzzy_algo(fuzzy_algo=_abbreviations)
            for abb in abbreviations:
                short_form, long_form = abb
                _abbreviations.add(
                    short_form=short_form,
                    long_form=long_form,
                    tokenizer=matcher.tokenizer,
                )

        # WordNormalizer
        if normalizers is not None:
            for params in normalizers:
                word_normalizer = WordNormalizer(**params)
                word_normalizer.add_words(
                    words=matcher.get_keywords_unigrams()
                )
                add_algo_in_cache(algo=word_normalizer)

        # FuzzyRegex
        if fuzzy_regex is not None:
            for params in fuzzy_regex:
                fuzzy = FuzzyRegex(**params)
                add_algo_in_cache(algo=fuzzy)
                if negative:
                    negative_stopwords = typing.cast(
                        NegativeStopwords,
                        matcher.stopwords,
                    )
                    negative_stopwords.add_fun_is_a_word_to_keep(
                        fuzzy.token_matches_pattern
                    )

        # String Distances
        # words ignored by string distance algorithms
        words2ignore = None
        if string_distance_ignored_w is not None:
            words2ignore = SimpleWords2ignore(words=string_distance_ignored_w)

        # Parameterize spellwise
        if spellwise is not None:
            for params in spellwise:
                # don't override user's 'words2ignore':
                if "words2ignore" not in params:
                    params["words2ignore"] = words2ignore
                algo = SpellWiseWrapper(**params)
                algo.add_words(words=matcher.get_keywords_unigrams())
                add_algo_in_cache(algo=algo)

        # Parameterize simstring
        if simstring is not None:
            for params in simstring:
                # don't override user's 'words2ignore':
                if "words2ignore" not in params:
                    params["words2ignore"] = words2ignore
                ss_algo = SimStringWrapper(
                    words=matcher.get_keywords_unigrams(),
                    **params,
                )
                add_algo_in_cache(algo=ss_algo)

        return matcher


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
    # states stores linkedstate instance that keeps track of a tree path
    # and document's tokens that matched.
    states: Set[LinkedState] = set()
    start_state = create_start_state(initial_state=initial_state)
    states.add(start_state)
    # count_not_stopword allows a stopword-independent window size.
    count_not_stopword = 0
    stop_tokens: List[TokenT] = []
    new_states: List[LinkedState] = []
    # states2remove store states that will be out-of-reach
    # at next iteration.
    states2remove: List[LinkedState] = []
    for i, token in enumerate(tokens):
        if stopwords.is_token_a_stopword(token):
            stop_tokens.append(token)
            continue
        # w_bucket stores when a state will be out-of-reach given window size
        # 'count_not_stopword % w' has range [0 ; w-1]
        w_bucket = count_not_stopword % w
        new_states.clear()
        states2remove.clear()
        count_not_stopword += 1
        # syns: 1 to many synonyms depending on fuzzy_algos configuration.
        syns_algos: List[SynAlgos] = syns_provider.get_synonyms(
            tokens=tokens, token=token, states=states
        )

        for state in states:
            if state.w_bucket == w_bucket:
                states2remove.append(state)
            # 0 to many states for [0] to [w-1] ; [w] only the start state.
            for syn, algos in syns_algos:
                node = state.node.jump_to_node(syn)
                # when no path is found, EMPTY_NODE is returned.
                if node is EMPTY_NODE:
                    continue
                new_state = LinkedState(
                    parent=state,
                    node=node,
                    token=token,
                    algos=algos,
                    w_bucket=w_bucket,
                )
                new_states.append(new_state)
                # Why 'new_state not in states':
                # if node_num is already in the states set,
                # it means an annotation was already created for this state.
                # For example 'cancer cancer', if an annotation was created
                # for the first 'cancer' then we don't want to create
                # a new one for the second 'cancer'.
                if node.is_a_final_state() and new_state not in states:
                    annot = create_annot(
                        last_el=new_state, stop_tokens=stop_tokens
                    )
                    annots.append(annot)
        # Prepare next iteration: first loop remove out-of-reach states.
        # Second iteration add new states.
        for state in states2remove:
            states.remove(state)
        for state in new_states:
            # this condition happens in the 'cancer cancer' example.
            # the effect is replacing a previous token by a new one.
            if state in states:
                states.remove(state)
            states.add(state)
    sort_annot(annots)  # mutate the list like annots.sort()
    return annots
