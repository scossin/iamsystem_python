import unittest

from typing import List

from iamsystem.tokenization.api import IToken
from iamsystem.tokenization.normalize import lower_no_accents
from iamsystem.tokenization.token import Offsets
from iamsystem.tokenization.token import Token
from iamsystem.tokenization.tokenize import french_tokenizer
from iamsystem.tokenization.tokenize import split_alpha_num
from iamsystem.tokenization.util import concat_tokens_label
from iamsystem.tokenization.util import concat_tokens_norm_label
from iamsystem.tokenization.util import get_span_seq_id
from iamsystem.tokenization.util import group_continuous_seq
from iamsystem.tokenization.util import multiple_seq_to_offsets
from iamsystem.tokenization.util import remove_trailing_stopwords
from iamsystem.tokenization.util import replace_offsets_by_new_str


class SplitAlphaNumTest(unittest.TestCase):
    def test_split_alpha_num_start_end(self):
        """Offsets values are correct."""
        offsets: List[Offsets] = list(split_alpha_num("one two"))
        self.assertEqual(2, len(offsets))
        two = offsets[1]
        self.assertEqual(7, two.end)
        self.assertEqual(4, two.start)

    def test_split_alpha_num_emptystring(self):
        """Empty string returns nothing."""
        offsets: List[Offsets] = list(split_alpha_num(""))
        self.assertEqual(0, len(offsets))
        offsets: List[Token] = list(split_alpha_num(" "))
        self.assertEqual(0, len(offsets))

    def test_split_alpha_num_punctuation(self):
        """It splits with a punctuation mark."""
        text = "one,two"
        tokens: List[Offsets] = list(split_alpha_num(text))
        self.assertEqual(2, len(tokens))
        self.assertEqual(7, tokens[1].end)
        one = tokens[0]
        self.assertEqual("one", text[one.start : one.end])  # noqa

    def test_split_alpha_num_quotes(self):
        """It splits with simple quotes."""
        tokens: List[Offsets] = list(split_alpha_num("L'ulcères"))
        self.assertEqual(2, len(tokens))

    def test_split_alpha_num_dash(self):
        """It splits with a dash."""
        tokens: List[Offsets] = list(split_alpha_num("meningo-encéphalite"))
        self.assertEqual(2, len(tokens))


class FrenchTokenizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = french_tokenizer()
        self.tokens: List[IToken] = list(
            self.tokenizer.tokenize("Meningo-encéphalite")
        )

    def test_get_span_seq_id(self):
        """Id of a sequence of tokens."""
        span_id = get_span_seq_id(self.tokens)
        self.assertEqual("(0,7);(8,19)", span_id)

    def test_concat_tokens_norm_label(self):
        """Concatenante the labels by a blank " "."""
        self.assertEqual(
            "meningo encephalite", concat_tokens_norm_label(self.tokens)
        )

    def test_concat_tokens_label(self):
        """Concatenante the label by a blank " "."""
        self.assertEqual(
            "Meningo encéphalite", concat_tokens_label(self.tokens)
        )

    def test_replace_offsets_by_new_str(self):
        """Replace each token by a new label."""
        text = "insuffisance -> ventriculaire -> gauche"
        tokens = self.tokenizer.tokenize(text=text)
        self.assertEqual(3, len(tokens))
        new_labels = ["ins", "vent", "g."]
        tokens_new_str = zip(tokens, new_labels)
        new_string = replace_offsets_by_new_str(
            text=text, offsets_new_str=tokens_new_str
        )
        self.assertEqual(new_string, "ins -> vent -> g.")


class DiscontinuousSeqTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = french_tokenizer()
        self.tokens = self.tokenizer.tokenize(
            "a sentence with multiple tokens to gen num tokens, number 10"
        )

    def test_group_continuous_seq(self):
        tokens = [self.tokens[i] for i in [0, 1, 2, 5, 6, 8, 10]]
        continuous_seq = group_continuous_seq(tokens=tokens)
        len_seq = [len(seq) for seq in continuous_seq]
        self.assertEqual(len_seq, [3, 2, 1, 1])
        self.assertEqual(3, len(continuous_seq[0]))

    def test_remove_trailing_stopwords(self):
        tokens = [self.tokens[i] for i in [0, 1, 2, 5, 6, 8, 10]]
        sequences = group_continuous_seq(tokens=tokens)
        # 2 is trailing in [0,1,2]
        out_seq = remove_trailing_stopwords(sequences=sequences, stop_i=[2])
        self.assertEqual(2, len(out_seq[0]))
        # 1 is not trailing in [0,1,2]
        out_seq = remove_trailing_stopwords(sequences=sequences, stop_i=[1])
        self.assertEqual(3, len(out_seq[0]))
        # [5,6] and [10] are removed. 2 sequences are removed:
        self.assertEqual(4, len(out_seq))
        out_seq = remove_trailing_stopwords(
            sequences=sequences, stop_i=[5, 6, 10]
        )
        self.assertEqual(2, len(out_seq))

    def test_discont_seq_to_offsets(self):
        tokens = [self.tokens[i] for i in [0, 1, 2, 5, 6, 8, 10]]
        sequences = group_continuous_seq(tokens=tokens)
        offsets = multiple_seq_to_offsets(sequences=sequences)
        self.assertEqual(4, len(offsets))


class LowerNoAccentsTest(unittest.TestCase):
    def test_uppercase(self):
        """It removes uppercase."""
        norm_str = lower_no_accents("One Two")
        self.assertEqual("one two", norm_str)

    def test_accents_removal(self):
        """It removes accents."""
        norm_str = lower_no_accents(" ulcères ")
        self.assertEqual(" ulceres ", norm_str)

    def test_microgramme_symbol(self):
        """Test it normalises μg to ug."""
        norm_str = lower_no_accents("μg")
        self.assertEqual("ug", norm_str)


if __name__ == "__main__":
    unittest.main()
