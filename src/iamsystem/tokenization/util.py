""" Utility functions """
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Tuple

from iamsystem.tokenization.api import IOffsets
from iamsystem.tokenization.api import IToken
from iamsystem.tokenization.token import Offsets


def merge_offsets(offsets_seq: Sequence[IOffsets]) -> Sequence[IOffsets]:
    """Merge 2 or more offsets to a single offsets when continuous.
    Ex: 2 4;5;20 => 2;20
    """
    # Why it is complicated to build a Brat format ? Brat handles
    # discontinuous word. An example is: 0 5;16 23 where ";" indicates the
    # presence of discontinuous tokens. With a sequence of offsets
    # there is no easy way to know if the tokens are continuous are not.
    # Hence, there are several solutions: 1) For
    # each token, get its Brat format and concatenate them. The output may
    # look like this : 0 5;6:20;21:23 This is the method implemented by the
    # 'get_span_seq_id' function. This is similar to annotating token by
    # token although the tokens are continuous in the text. 2) if the
    # token's end offset + 1 matches the next token's start then 'merge'
    # them. However, if there is an extraspace in the text,
    # this method fails and returns an output similar to 1)
    # 3) With the document in input, check if characters between token's end
    # offset and next token's start are all empty characters.
    # It solved problem of 2) but this mehtod fails if any stopword is removed.

    # I choose to implement solution 2) which should work in most of the cases.
    if len(offsets_seq) == 0:
        raise ValueError("empty tokens list")
    offsets: List[IOffsets] = [offsets_seq[0]]
    for token in offsets_seq[1:]:
        last_offset: IOffsets = offsets[-1]
        if (
            token.start == last_offset.end
            or token.start == last_offset.end + 1
        ):
            merged_offset = Offsets(start=last_offset.start, end=token.end)
            offsets[-1] = merged_offset  # offsets replacement.
        else:
            offsets.append(token)
    return offsets


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
