""" Brat utility functions. """
from typing import Sequence

from iamsystem.tokenization.api import IOffsets
from iamsystem.tokenization.util import merge_offsets


def get_brat_format(offsets: IOffsets) -> str:
    """Brat offsets format."""
    return f"{offsets.start} {offsets.end}"


def get_brat_format_seq(offsets_seq: Sequence[IOffsets]) -> str:
    """Brat format of a discontinuous offsets sequence."""
    brat_offsets_list = [get_brat_format(offsets) for offsets in offsets_seq]
    brat_offsets = ";".join(brat_offsets_list)
    return brat_offsets


def merge_offsets_and_get_brat_format(offsets_seq: Sequence[IOffsets]) -> str:
    """Merge offsets when possible and return Brat format."""
    offsets_seq_merged = merge_offsets(offsets_seq)
    return get_brat_format_seq(offsets_seq=offsets_seq_merged)
