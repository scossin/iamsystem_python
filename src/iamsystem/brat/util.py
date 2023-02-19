""" Brat utility functions. """
from typing import Sequence

from iamsystem.tokenization.api import IOffsets


def get_brat_format(offsets: IOffsets) -> str:
    """Brat offsets format."""
    return f"{offsets.start} {offsets.end}"


def get_brat_format_seq(offsets_seq: Sequence[IOffsets]) -> str:
    """Brat format of a discontinuous offsets sequence."""
    brat_offsets_list = [get_brat_format(offsets) for offsets in offsets_seq]
    brat_offsets = ";".join(brat_offsets_list)
    return brat_offsets
