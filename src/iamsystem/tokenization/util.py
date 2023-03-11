""" Utility functions """
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Tuple

from iamsystem.brat.util import get_brat_format_seq
from iamsystem.matcher.api import IAnnotation
from iamsystem.tokenization.api import IOffsets
from iamsystem.tokenization.api import IToken
from iamsystem.tokenization.tokenize import Offsets


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


def group_continuous_seq(tokens: List[IToken]) -> List[List[IToken]]:
    """Group continuous sequences.
    From a sequence of tokens, group tokens that follow each other by
    their indice. Ex: [1,2,3,5,6] => [[1,2,3], [5,6]]"""
    if len(tokens) == 0:
        return []
    tokens.sort(key=lambda t: t.i)
    # 1) Split the sequence by missing indices
    seq: List[IToken] = [tokens[0]]
    sequences: List[List[IToken]] = [seq]
    for token in tokens[1:]:
        last_token = seq[-1]
        if last_token.i + 1 == token.i:  # if continuous
            seq.append(token)
        else:  # discontinuous, create a new sequence
            seq = [token]
            sequences.append(seq)
    return sequences


def remove_trailing_stopwords(
    sequences: List[List[IToken]], stop_i=List[int]
) -> List[List[IToken]]:
    """In each continuous sequence, we want to remove trailing stopwords.
    Ex: [['North', 'and'], ['America']] -> [['North'], ['America']]

    :param sequences: multiple continuous sequences
    :param stop_i: stopwords indices
    :return: sequences without trailing stopwords.
    """
    out_seq = []
    for seq in sequences:
        i_not_stop = [token.i for token in seq if token.i not in stop_i]
        if len(i_not_stop) == 0:  # stopwords only in the sequence
            continue
        last_i = i_not_stop[-1]
        out_seq.append(seq[: last_i + 1])
    return out_seq


def get_text_span(text: str, offsets: IOffsets) -> str:
    """Return the text substring of an offsets."""
    return text[offsets.start : offsets.end]  # noqa


def get_text_and_offsets_of_sequences(
    sequences: List[List[IToken]], annot: IAnnotation
) -> Tuple[str, str]:
    """Return text of brat offsets from multiple sequences."""
    offsets: List[IOffsets] = multiple_seq_to_offsets(sequences=sequences)
    seq_offsets = get_brat_format_seq(offsets)
    seq_label = " ".join(
        [get_text_span(annot.text, one_offsets) for one_offsets in offsets]
    )
    return seq_label, seq_offsets


def multiple_seq_to_offsets(sequences: List[List[IToken]]) -> List[IOffsets]:
    """Create an Offsets for each continuous sequence, start being the
        start offset of the first token in the sequence and
        end being the end offset of the last token in the sequence.

    :param sequences: multiple continuous sequences
    :return: a list of offsets.
    """
    offsets: List[IOffsets] = [
        Offsets(start=seq[0].start, end=seq[-1].end) for seq in sequences
    ]
    return offsets
