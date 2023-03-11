""" Main API output."""
import functools

from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import iamsystem

from iamsystem.brat.formatter import EBratFormatters
from iamsystem.keywords.api import IEntity
from iamsystem.keywords.api import IKeyword
from iamsystem.matcher.api import IAnnotation
from iamsystem.matcher.api import IBratFormatter
from iamsystem.matcher.printannot import PrintAnnot
from iamsystem.matcher.util import StateTransition
from iamsystem.tokenization.api import TokenT
from iamsystem.tokenization.span import Span
from iamsystem.tokenization.span import is_shorter_span_of
from iamsystem.tokenization.util import itoken_to_dict
from iamsystem.tokenization.util import min_start_or_end
from iamsystem.tokenization.util import offsets_overlap
from iamsystem.tokenization.util import replace_offsets_by_new_str
from iamsystem.tree.nodes import INode


class Annotation(Span[TokenT], IAnnotation[TokenT]):
    """Ouput class of :class:`~iamsystem.Matcher` storing information on the
    detected entities."""

    def __init__(
        self,
        tokens: List[TokenT],
        algos: List[List[str]],
        node: INode,
        stop_tokens: List[TokenT],
        text: Optional[str] = None,
    ):
        """Create an annotation.

        :param tokens: a sequence of TokenT, a generic type that implements
            :class:`~iamsystem.IToken` protocol.
        :param algos: the list of fuzzy algorithms that matched the tokens.
            One to several algorithms per token.
        :param node: a final state of iamsystem algorithm containing the
            keyword that matched this sequence of tokens.
        :param stop_tokens: the list of stopwords tokens of the document.
        :param text: the annotated text/document.
        """
        super().__init__(tokens)
        self._algos = algos
        self._node = node
        self._stop_tokens = stop_tokens
        self._text = text

    @property
    def text(self) -> Optional[str]:
        """Return the annotated text."""
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        """Set the annotated text."""
        self._text = value

    @property
    def algos(self) -> List[List[str]]:
        return self._algos

    @property
    def label(self):
        """@Deprecated. An annotation label. Return 'tokens_label' attribute"""
        return self.tokens_label

    @property
    def stop_tokens(self) -> List[TokenT]:
        """The list of stopwords tokens inside the annotation detected by
        the Matcher stopwords instance."""
        # Note that _stop_tokens are stopwords of the document. The reason to
        # filter now and not before is that, when order_tokens = T, stopwords
        # inside an annotation may not have been seen.
        stop_tokens_in_annot = [
            token
            for token in self._stop_tokens
            if self.start_i < token.i < self.end_i
        ]
        stop_tokens_in_annot.sort(key=lambda token: token.i)
        return stop_tokens_in_annot

    @property
    def keywords(self) -> Sequence[IKeyword]:
        """The linked entities, :class:`~iamsystem.IKeyword` instances that
        matched a document's tokens."""
        return self._node.get_keywords()  # type: ignore

    def get_tokens_algos(self) -> Iterable[Tuple[TokenT, List[str]]]:
        """Get each token and the list of fuzzy algorithms that matched it.

        :return: an iterable of tuples (token0, ['algo1',...]) where token0 is
            a token and ['algo1',...] a list of fuzzy algorithms.
        """
        return zip(self._tokens, self.algos)

    def to_dict(self, text: str = None) -> Dict[str, Any]:
        """Return a dictionary representation of this object.

        :param text: the document from which this annotation comes from.
         Default to None.
        :return: A dictionary of relevant attributes.
        """
        dic = {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "norm_label": self.tokens_norm_label,
            "tokens": [itoken_to_dict(token) for token in self.tokens],
            "algos": self.algos,
            "kb_ids": [
                keyword.kb_id
                for keyword in self.keywords
                if isinstance(keyword, IEntity)
            ],
            "kw_labels": [keyword.label for keyword in self.keywords],
            "version": iamsystem.__annot_version__,
        }
        if text is not None:
            text_substring = text[self.start : self.end]  # noqa
            dic["substring"] = text_substring
        return dic

    def __str__(self) -> str:
        """Annotation string representation with Brat offsets format."""
        return f"{self.to_string()}"

    def to_string(self, text=False, debug=False) -> str:
        """Get a default string representation of this object.

        :param text: the document from which this annotation comes from.
            Default to None. If set, add the document substring:
            text[first-token-start-offset : last-token-end-offset].
        :param debug: default to False. If True, add the sequence of tokens
            and fuzzyalgo names.
        :return: a concatenated string
        """
        columns = [Annotation.annot_to_str(annot=self)]
        if text:
            text_substring = self.text[self.start : self.end]  # noqa
            columns.append(text_substring)
        if debug:
            token_annots_str = self._get_norm_label_algos_str()
            columns.append(token_annots_str)
        return "\t".join(columns).replace("\n", "\\n")

    def _get_norm_label_algos_str(self):
        """Get a string representation of tokens and algorithms."""
        return ";".join(
            [
                f"{token.norm_label}({','.join(algos)})"
                for token, algos in self.get_tokens_algos()
            ]
        )

    annot_to_str: Callable[[IAnnotation], str] = PrintAnnot().annot_to_str
    " A class function that generates a string representation of an annotation."  # noqa

    @classmethod
    def set_brat_formatter(
        cls, brat_formatter: Union[EBratFormatters, IBratFormatter]
    ):
        """Change Brat Formatter to change text-span and offsets.

        :param brat_formatter: A Brat formatter to produce
            a different Brat annotation. If None, default to
            :class:`~iamsystem.ContSeqFormatter`.
        :return: None
        """
        if isinstance(brat_formatter, EBratFormatters):
            brat_formatter = brat_formatter.value
        cls.annot_to_str = PrintAnnot(
            brat_formatter=brat_formatter
        ).annot_to_str


def is_ancestor_annot_of(a: Annotation, b: Annotation) -> bool:
    """True if a is an ancestor of b."""
    if a is b:
        return False
    if a.start != b.start or a.end > b.end:
        return False
    ancestors = b._node.get_ancestors()
    return a._node in ancestors


def sort_annot(annots: List[Annotation]) -> None:
    """Custom sort function by 1) start value 2) end value."""
    annots.sort(key=functools.cmp_to_key(min_start_or_end))


def rm_nested_annots(annots: List[Annotation], keep_ancestors=False):
    """In case of two nested annotations, remove the shorter one.
    For example, if we have "prostate" and "prostate cancer" annnotations,
    "prostate" annotation is removed.

    :param annots: a list of annotations.
    :param keep_ancestors: Default to False. Whether to keep the nested
      annotations that are ancestors and remove only other cases.
    :return: a filtered list of annotations.
    """
    # Assuming annotations are already sorted by start and end values,
    # an ancestor will always occur before its childs. For example, ancestor
    # "insuffisance" will alway occur before "insuffisance cardiaque". the
    # algorithm below check if each annotation is an ancestor by searching
    # childs to the right. Although the algorithm has two nested loops,
    # its complexity is not O(n²) since the 'break' keyword is quickly
    # executed.
    ancest_indices = set()
    short_indices = set()
    # count = 0
    for i, annot in enumerate(annots):
        for _y, other in enumerate(annots[(i + 1) :]):  # noqa
            y = _y + i + 1  # y is the indice of other in annots list.
            if not offsets_overlap(annot, other):
                break
            if is_shorter_span_of(annot, other):
                short_indices.add(i)
                # because ancestor is a special case of nested annot.
                if is_ancestor_annot_of(annot, other):
                    ancest_indices.add(i)
            if is_shorter_span_of(other, annot):
                short_indices.add(y)
            # count += 1
    # print(f"count:{count}")
    if keep_ancestors:
        indices_2_remove = set(
            [i for i in short_indices if i not in ancest_indices]
        )
    else:
        indices_2_remove = short_indices
    indices_2_keep = [
        i for i in range(len(annots)) if i not in indices_2_remove
    ]
    annots_filt = [annots[i] for i in indices_2_keep]
    return annots_filt


def create_annot(
    last_trans: StateTransition, stop_tokens: List[TokenT]
) -> Annotation:
    """last_trans contains all the state transitions and sequence of tokens in
    text. The last_trans's node is a final state which means it is associated
    with one or many keywords."""
    if not last_trans.node.is_a_final_state():
        raise ValueError("StateTransition's node is not a final state.")
    node = last_trans.node
    trans_states = _linkedlist_to_list(last_trans)
    # order by token indice (important if tokens were ordered alphabetically).
    # Note that node might not be the last anymore.
    trans_states.sort(key=lambda x: x.token.i)
    tokens: List[TokenT] = [t.token for t in trans_states]
    algos = [t.algos for t in trans_states]
    # Note that the annotations are created during iterating over the
    # document's tokens. If tokens are ordered alphabetically,
    # the list of stopwords inside an annotation are not known at this step.
    # Thus, all the stopwords detected are passed to each annotation:
    # it's not possible to filter them here, at the moment of creating an
    # annotation.
    annot = Annotation(
        tokens=tokens,
        algos=algos,
        node=node,
        stop_tokens=stop_tokens,
    )
    return annot


def _linkedlist_to_list(last_el: StateTransition) -> List[StateTransition]:
    """Convert a linked list to a list."""
    transitions: List[StateTransition] = [last_el]
    previous_trans = last_el.previous_trans
    while not StateTransition.is_first_trans(previous_trans):
        transitions.append(previous_trans)
        previous_trans = previous_trans.previous_trans
    transitions.reverse()
    return transitions


def replace_annots(
    text: str, annots: Sequence[Annotation], new_labels: Sequence[str]
):
    """Replace each annotation in a document (text parameter) by a new label.
    Warning: an annotation is ignored if overlapped by another one.

    :param text: the document from which the annotations come from.
    :param annots: an ordered sequence of annotation.
    :param new_labels: one new label per annotation, same length as annots
      expected.
    :return: a new document.
    """
    if len(annots) != len(new_labels):
        raise ValueError(
            "annots and new_labels parameters don't have the same length."
        )
    return replace_offsets_by_new_str(
        text=text, offsets_new_str=zip(annots, new_labels)
    )
