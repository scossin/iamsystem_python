""" Utility functions """
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Tuple

from iamsystem.tokenization.api import IOffsets
from iamsystem.tokenization.api import IToken


def offsets_overlap(a: IOffsets, b: IOffsets) -> bool:
    """True if a and b have indices in common in their range (start:end)."""
    if a is b:
        return False
    return (b.start <= a.start <= b.end) or (a.start <= b.start <= a.end)


def get_tokens_text_substring(offsets_seq: Sequence[IOffsets], text: str):
    """Get each offsets' substring in the document and concatenate them.

    :param offsets_seq: a sequence of start,end positions in the document.
    :param text: the document from which these offsets come from.
    :return: a concatenation of substring of text.
    """
    tokens_text = [
        text[offsets.start : offsets.end] for offsets in offsets_seq  # noqa
    ]
    return " ".join(tokens_text)


def get_span_seq_id(offsets_seq: Sequence[IOffsets]):
    """Create a unique id for a sequence of offsets."""
    offsets_seq_id = [
        f"({offsets.start},{offsets.end})" for offsets in offsets_seq
    ]
    span_seq_id = ";".join(offsets_seq_id)
    return span_seq_id


def get_min_start_offset(offsets_seq: Sequence[IOffsets]) -> int:
    """Return the minimum start value in a sequence of offsets."""
    return min(offsets.start for offsets in offsets_seq)


def get_max_end_offset(offsets_seq: Sequence[IOffsets]) -> int:
    """Return the max end value in a sequence of offsets."""
    return max(offsets.end for offsets in offsets_seq)


def concat_tokens_norm_label(tokens: Sequence[IToken]) -> str:
    """Concatenate the normalized label of each token in the sequence."""
    labels: List[str] = [token.norm_label for token in tokens]
    return " ".join(labels)


def concat_tokens_label(tokens: Sequence[IToken]) -> str:
    """Concatenate the label of each token in the sequence."""
    labels: List[str] = [token.label for token in tokens]
    return " ".join(labels)


def replace_offsets_by_new_str(
    text: str, offsets_new_str: Iterable[Tuple[IOffsets, str]]
) -> str:
    """Replace multiple text substring delimited by offsets

    :param text: the initial document.
    :param offsets_new_str: an ordered sequence of (offsets, new_str) tuples
      where offsets is an instance that implements :class:`~iamsystem.IOffsets`
      (ex: :class:`~iamsystem.IToken`, :class:`~iamsystem.Annotation`) protocol
      and new_str the new string.
    :return: a new string.
    """
    new_string: List[str] = []
    i = 0
    for offsets, new_token_str in offsets_new_str:
        if offsets.start < i:
            continue
        new_string.append(text[i : offsets.start] + new_token_str)  # noqa
        i = offsets.end
    new_string.append(text[i:])
    return "".join(new_string)


def min_start_or_end(a: IOffsets, b: IOffsets) -> int:
    """Comparator function to order 2 offsets by their start and end values"""
    diff_start = a.start - b.start
    if diff_start == 0:
        return a.end - b.end
    else:
        return diff_start


def itoken_to_dict(token: IToken):
    """Return a dictionary representation of a token."""
    return {
        "start": token.start,
        "end": token.end,
        "label": token.label,
        "norm_label": token.norm_label,
    }
