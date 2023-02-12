""" Main API output."""
import functools

from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Tuple

from iamsystem.keywords.api import IEntity
from iamsystem.keywords.api import IKeyword
from iamsystem.matcher.util import TransitionState
from iamsystem.tokenization.api import TokenT
from iamsystem.tokenization.span import Span
from iamsystem.tokenization.span import is_shorter_span_of
from iamsystem.tokenization.util import itoken_to_dict
from iamsystem.tokenization.util import min_start_or_end
from iamsystem.tokenization.util import offsets_overlap
from iamsystem.tokenization.util import replace_offsets_by_new_str


class Annotation(Span[TokenT]):
    """Ouput class of :class:`~iamsystem.Matcher` storing information on the
    detected entities."""

    def __init__(self, tokens_states: Sequence[TransitionState[TokenT]]):
        tokens: List[TokenT] = [tok_state.token for tok_state in tokens_states]
        tokens.sort(key=functools.cmp_to_key(min_start_or_end))
        super().__init__(tokens)
        self.algos = [token_state.algos for token_state in tokens_states]
        self._last_state = tokens_states[-1].node

    @property
    def keywords(self) -> Sequence[IKeyword]:
        """The linked entities, :class:`~iamsystem.IKeyword` instances that
        matched a document's tokens."""
        return self._last_state.get_keywords()  # type: ignore

    def get_tokens_algos(self) -> Iterable[Tuple[TokenT, List[str]]]:
        """Get each token and the list of fuzzy algorithms that matched it.

        :return: an itera  of tuples (token0, ['algo1',...]) where token0 is
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
            "offsets": self.to_brat_format(),
            "label": self.label,
            "norm_label": self.norm_label,
            "tokens": [itoken_to_dict(token) for token in self.tokens],
            "algos": self.algos,
            "kb_ids": [
                keyword.kb_id
                for keyword in self.keywords
                if isinstance(keyword, IEntity)
            ],
            "kw_labels": [keyword.label for keyword in self.keywords],
        }
        if text is not None:
            text_substring = text[self.start : self.end]  # noqa
            dic["substring"] = text_substring
        return dic

    def to_string(self, text: str = None, debug=False) -> str:
        """Get a default string representation of this object.

        :param text: the document from which this annotation comes from.
         Default to None. If set, add the document substring: text[
         first-token-start-offset : last-token-end-offset].
        :param debug:
         default to False. If True, add the sequence of tokens and fuzzyalgo
         names.
        :return: a concatenated string of 'keywords'\t'start'
         'end'\t'substring'?\t'debug_info'?
        """
        columns = self._to_list()
        if text is not None:
            text_substring = text[self.start : self.end]  # noqa
            columns.append(text_substring)
        if debug:
            token_annots_str = self._get_norm_label_algos_str()
            columns.append(token_annots_str)
        return "\t".join(columns)

    def _get_norm_label_algos_str(self):
        """Get a string representation of tokens and algorithms."""
        return ";".join(
            [
                f"{token.norm_label}({','.join(algos)})"
                for token, algos in self.get_tokens_algos()
            ]
        )

    def _to_list(self) -> List[Any]:
        """A list of attributes."""
        columns = [
            self.label,
            self.to_brat_format(),
            self._keywords_to_string(),
        ]
        return columns

    def _keywords_to_string(self):
        """Merge the keywords."""
        keywords_str = [str(keyword) for keyword in self.keywords]
        return ";".join(keywords_str)

    def __str__(self) -> str:
        """Annotation string representation with Brat offsets format."""
        return f"{self.to_string()}"


def is_ancestor_annot_of(a: Annotation, b: Annotation) -> bool:
    """True if a is an ancestor of b."""
    if a is b:
        return False
    if a.start != b.start or a.end > b.end:
        return False
    ancestors = b._last_state.get_ancestors()
    return a._last_state in ancestors


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


def create_annot(last_el: TransitionState) -> Annotation:
    """last_el contains a sequence of tokens in text and a final state (a
    matcher keyword)."""
    if not last_el.node.is_a_final_state():
        raise ValueError("LinkedToken is not a final state.")
    tokens_states = tokens_states_to_list(last_el)
    annot = Annotation(tokens_states=tokens_states)
    return annot


def tokens_states_to_list(last_el: TransitionState) -> List[TransitionState]:
    """Convert a linked list to a list."""
    tokens_states: List[TransitionState] = [last_el]
    parent = last_el.parent
    while isinstance(parent, TransitionState):
        tokens_states.append(parent)
        parent = parent.parent
    tokens_states.reverse()
    return tokens_states


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
