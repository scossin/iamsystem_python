""" Brat connectors. """
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Sequence

from iamsystem.brat.util import get_brat_format_seq
from iamsystem.matcher.annotation import Annotation
from iamsystem.tokenization.api import IOffsets
from iamsystem.tokenization.util import get_tokens_text_substring
from iamsystem.tokenization.util import merge_offsets


class BratEntity:
    """Class representing a Brat Entity. https://brat.nlplab.org/standoff.html:
    'Each entity annotation has a unique ID and is defined by type
    (e.g. Person or Organization). and the span of characters containing
    the entity mention (represented as a "start end" offset pair).'

    Format: ID \t TYPE START END[;START END]* \t TEXT.
    """

    def __init__(
        self,
        entity_id: str,
        brat_type: str,
        offsets: Sequence[IOffsets],
        text: str,
    ):
        """Create a Brat Entity.

        :param entity_id: a unique ID (^T[0-9]+$).
        :param brat_type: A Brat entity type (see Brat documentation).
        :param offsets: (start,end) annotation offsets.
          See :class:`~iamsystem.IOffsets`.
        :param text: document substring using (start,end) offsets.
        """
        self.entity_id = self._check_entity_id(entity_id)
        self.brat_type = brat_type
        self.offsets = offsets
        self.text = text

    @staticmethod
    def _check_entity_id(entity_id: str) -> str:
        """Check id is valid."""
        if len(entity_id) == 0 or entity_id[0] != "T":
            raise ValueError("Brat entity ID must start by a T")
        return entity_id

    def __str__(self):
        """Return the Brat string format."""
        return (
            f"{self.entity_id}\t"
            f"{self.brat_type} "
            f"{get_brat_format_seq(self.offsets)}\t"
            f"{self.text}"
        )


class BratNote:
    """Class representing a Brat Note. https://brat.nlplab.org/standoff.html
    Brat notes are used to store additionnal information on a detected entity.
    Format: #ID \t TYPE REFID \t NOTE
    """

    TYPE = "IAMSYSTEM"
    """BratNote type. Replace by 'AnnotatorNotes' to be human writable
    in Brat interface"""

    def __init__(self, note_id: str, ref_id: str, note: str):
        """Create a Brat Note.

        :param note_id: a unique ID (^#[0-9]+$)
        :param ref_id: a unique ID. For a BratEntity, the format is (^T[0-9]+$)
        :param note: any string comment.
        """
        self.note_id = self._check_note_id(note_id)
        self.ref_id = self._check_ref_id(ref_id)
        self.note = note

    @staticmethod
    def _check_note_id(note_id: str) -> str:
        """Check id is correct."""
        if len(note_id) == 0 or note_id[0] != "#":
            raise ValueError("Brat note ID must start by a #")
        return note_id

    @staticmethod
    def _check_ref_id(ref_id: str) -> str:
        """Check ref_id is valid."""
        if len(ref_id) == 0 or ref_id[0] != "T":
            raise ValueError("Brat ref ID must start by a T")
        return ref_id

    def __str__(self):
        """Return the Brat string format."""
        return (
            f"{self.note_id}\t"
            f"{BratNote.TYPE} "
            f"{self.ref_id}\t"
            f"{self.note}"
        )


get_note_fun = Callable[[Annotation], str]


def get_note_keyword_label(annot: Annotation) -> str:
    """Return the string representation of the first keyword
    of the annotation."""
    return str(annot.keywords[0])


class BratDocument:
    """Class representing a Brat Document containing Brat's annotations,
    namely Brat Entity and Brat Note in this package.
    A BratDocument should be linked to a single text document.
    Entities and notes can be serialized in a text file with 'ann' extension,
    one per line. See https://brat.nlplab.org/standoff.html
    """

    def __init__(self):
        self.brat_entities: List[BratEntity] = []
        self.brat_notes: List[BratNote] = []
        self.get_note: get_note_fun = get_note_keyword_label

    def add_annots(
        self,
        annots: List[Annotation],
        text: str,
        keyword_attr: str = None,
        brat_type: str = None,
    ) -> None:
        """Add iamsystem annotations to convert them to Brat format.

        :param annots: a list of :class:`~iamsystem.Annotation`,
          :class:`~iamsystem.Matcher` output.
        :param text: the document from which these annotations comes from.
        :param keyword_attr: the attribute name of a
            :class:`~iamsystem.IKeyword` that stores brat_type.
            Default to None. If None, brat_type parameter must be used.
        :param brat_type: A string, the Brat entity type for all these
            annotations. Default to None. If None, keyword_attr parameter
            must be used.
        :return: None
        """
        if keyword_attr is None and brat_type is None:
            raise ValueError("keyword_attr or brat_type argument must be set.")
        for annot in annots:
            b_type = ""
            if keyword_attr is not None:
                b_type = annot.keywords[0].__getattribute__(keyword_attr)
            elif brat_type is not None:
                b_type = brat_type
            brat_entity = BratEntity(
                entity_id=self._get_entity_id(),
                brat_type=b_type,
                offsets=merge_offsets(annot.tokens),
                text=get_tokens_text_substring(annot.tokens, text=text),
            )
            self.brat_entities.append(brat_entity)

            brat_note = BratNote(
                note_id=self._get_note_id(),
                ref_id=brat_entity.entity_id,
                note=self.get_note(annot),
            )
            self.brat_notes.append(brat_note)

    def add_entity(
        self, brat_type: str, offsets: List[IOffsets], text: str
    ) -> None:
        """Add a Brat Entity.

        :param brat_type: A Brat entity type (see Brat documentation).
        :param offsets: a list of (start,end) annotation offsets.
            See :class:`~iamsystem.IOffsets`.
            A list is expected since the tokens can be discontinuous.
        :param text: document substring using (start,end) offsets
            (not the document itself).
        :return: None
        """
        brat_entity = BratEntity(
            entity_id=self._get_entity_id(),
            brat_type=brat_type,
            offsets=offsets,
            text=text,
        )
        self.brat_entities.append(brat_entity)

    def get_entities(self) -> Iterable[BratEntity]:
        """An iterable of Brat entities."""
        return iter(self.brat_entities)

    def get_notes(self) -> Iterable[BratNote]:
        """An iterable of Brat notes."""
        return iter(self.brat_notes)

    def entities_to_string(self) -> str:
        """Brat entities in the Brat format ready to be serialized
        to '.ann' text file."""
        brat_entities = [
            str(brat_entity) for brat_entity in self.brat_entities
        ]
        return "\n".join(brat_entities)

    def notes_to_string(self) -> str:
        """Brat notes in the Brat format ready to be serialized
        to '.ann' text file."""
        brat_notes = [str(brat_note) for brat_note in self.brat_notes]
        return "\n".join(brat_notes)

    def __str__(self):
        """Return the Brat string format all of entities."""
        return (
            f"{self.entities_to_string()}\n"
            f"{self.notes_to_string()}".strip()
        )

    def _get_entity_id(self) -> str:
        """Build an entity id."""
        return f"T{len(self.brat_entities) + 1}"

    def _get_note_id(self) -> str:
        """Build an node id."""
        return f"#{len(self.brat_notes) + 1}"


class BratWriter:
    """Utility class to write IAMsystem annotations in Brat format
    to a text file."""

    @classmethod
    def saveEntities(
        cls, brat_entities: Iterable[BratEntity], write: Callable[[str], Any]
    ) -> None:
        """Write Brat entities.

        :param brat_entities: an iterable of Brat entities.
        :param write: a write function (ex: f.write from
            'with(open(filename, 'w')) as f:')
        :return: None
        """
        for brat_entity in brat_entities:
            write(str(brat_entity))
            write("\n")

    @classmethod
    def saveNotes(
        cls, brat_notes: Iterable[BratNote], write: Callable[[str], Any]
    ) -> None:
        """Write Brat notes.

        :param brat_notes: an iterable of Brat notes.
        :param write: a write function
            ex: f.write from 'with(open(filename, 'w')) as f:
        :return: None
        """
        for brat_note in brat_notes:
            write(str(brat_note))
            write("\n")
