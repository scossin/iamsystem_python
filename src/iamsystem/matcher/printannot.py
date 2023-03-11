""" An utility class to build multiple "print annotation" strategies
depending on the chosen BratFormatter."""
from iamsystem.brat.formatter import ContSeqFormatter
from iamsystem.brat.formatter import IBratFormatter
from iamsystem.brat.formatter import TokenFormatter
from iamsystem.matcher.api import IAnnotation


class PrintAnnot:
    def __init__(self, brat_formatter: IBratFormatter = None):
        """Create a PrintAnnot instance to change annotation's to_string
            method behavior.

        :param brat_formatter: A Brat formatter to produce
            a different Brat annotation. If None, default to
            :class:`~iamsystem.ContSeqFormatter`.
        """
        self._brat_formatter = brat_formatter or ContSeqFormatter()

    def annot_to_str(self, annot: IAnnotation):
        """Return a string representation of this annotation.

        :param annot: An annotation.
        :return: A concatenation of annotation 'text-span', 'offsets' and
            'keywords' separated by a tabulation.
            text-span and offsets are generated by the BratFormatter.
        """
        brat_formatter = self._brat_formatter
        # IndividualTokenFormatter doesn't use annotation 'text'. Without text
        # It can produce valid Brat offsets.
        # Other formatter must not be used if text.
        if annot.text is None:
            brat_formatter = TokenFormatter()
        text_span, offsets = brat_formatter.get_text_and_offsets(annot=annot)
        keywords_str = [str(keyword) for keyword in annot.keywords]
        columns = [text_span, offsets, ";".join(keywords_str)]
        return "\t".join(columns)
