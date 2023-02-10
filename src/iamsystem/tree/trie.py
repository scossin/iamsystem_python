""" A trie implementation to store the keywords: each node stores the token of
each keyword, an edge links two tokens. It provides a fast sequence look-up
to check if a keyword exists or not.
"""
import warnings

from typing import Iterable
from typing import List
from typing import Sequence

from iamsystem.keywords.api import IKeyword
from iamsystem.stopwords.api import IStopwords
from iamsystem.tokenization.api import IToken
from iamsystem.tokenization.api import ITokenizer
from iamsystem.tokenization.tokenize import remove_stopwords
from iamsystem.tree.api import IInitialState
from iamsystem.tree.nodes import Node
from iamsystem.tree.nodes import _create_a_root_node


class Trie(IInitialState):
    """A trie datastructure to store keywords."""

    def __init__(self):
        """Build a trie datastructure."""
        self.root_node = _create_a_root_node()
        self._node_number = 1  # next node number

    def add_keyword(
        self, keyword: IKeyword, tokenizer: ITokenizer, stopwords: IStopwords
    ) -> None:
        """

        :param keyword: a :class:`~iamsystem.IKeyword` to add to the trie.
        :param tokenizer: a :class:`~iamsystem.FuzzyAlgo` to
         tokenize keyword's label.
        :param stopwords: a :class:`~iamsystem.IStopwords` to ignore
         stopwords in keywords' label.
        :return: None.
        """
        tokens: Sequence[IToken] = tokenizer.tokenize(keyword.label)
        tokens_wo_stop: Sequence[IToken] = remove_stopwords(
            tokens=tokens, stopwords=stopwords
        )
        norm_labels = [token.norm_label for token in tokens_wo_stop]
        if len(norm_labels) == 0:
            warnings.warn(
                f"Keyword '{keyword}' was not added to the dictionary "
                f"because no token remain after tokenization"
            )
        self.add_keyword_with_tokens(keyword, norm_labels)

    def add_keywords(
        self,
        keywords: Iterable[IKeyword],
        tokenizer: ITokenizer,
        stopwords: IStopwords,
    ) -> None:
        """Add several keywords or a keywords.

        :param keywords: an iterable :class:`~iamsystem.IKeyword`.
        :param tokenizer: a :class:`~iamsystem.FuzzyAlgo` to
         tokenize keyword's label.
        :param stopwords: a :class:`~iamsystem.IStopwords` to ignore
         stopwords in keywords' label.
        :return: None.
        """
        for keyword in keywords:
            self.add_keyword(keyword, tokenizer, stopwords)

    def add_keyword_with_tokens(
        self, keyword: IKeyword, tokens: List[str]
    ) -> None:
        """Add a keyword already tokenized.

        :param keyword: a :class:`~iamsystem.IKeyword`.
        :param tokens: keyword's tokens.
        :return: None.
        """
        current_node = self.root_node
        for token in tokens:
            if current_node.has_transition_to(token):
                current_node = current_node.goto_node(token)
            else:
                current_node = Node(
                    token=token,
                    parent_node=current_node,
                    node_num=self._node_number,
                )
                self._node_number += 1
        current_node.add_keyword(keyword)

    def get_initial_state(self) -> Node:
        """The initial state of a trie is its root node."""
        return self.root_node

    def get_number_of_nodes(self) -> int:
        """Number of :class:`~iamsystem.INode` instances stored."""
        return self._node_number
