import unittest

from abc import ABC
from typing import Iterable
from typing import List
from typing import Sequence
from typing import cast

from iamsystem.fuzzy.api import FuzzyAlgo
from iamsystem.fuzzy.api import ISynsProvider
from iamsystem.fuzzy.api import SynAlgo
from iamsystem.fuzzy.spellwise import ESpellWiseAlgo
from iamsystem.fuzzy.spellwise import SpellWiseWrapper
from iamsystem.matcher.annotation import rm_nested_annots
from iamsystem.matcher.matcher import Matcher
from iamsystem.matcher.matcher import detect
from iamsystem.matcher.util import IState
from iamsystem.stopwords.api import IStopwords
from iamsystem.stopwords.simple import Stopwords
from iamsystem.tokenization.api import IToken
from iamsystem.tokenization.api import TokenT
from iamsystem.tokenization.token import Token
from iamsystem.tokenization.tokenize import french_tokenizer
from iamsystem.tree.trie import Trie
from tests.utils import get_termino_irc
from tests.utils import get_termino_ivg
from tests.utils import get_termino_ulceres
from tests.utils_detector import get_abbs_ic
from tests.utils_detector import get_abbs_irc


def build_detect(
    trie: Trie, syns_provider: ISynsProvider, stopwords: IStopwords
):
    """Wrap trie, syns_provider and stopwords that don't change."""

    def _detect(tokens: Sequence[TokenT], w: int):
        """Return a custom detect function."""
        return detect(
            tokens=tokens,
            w=w,
            initial_state=trie.root_node,
            syns_provider=syns_provider,
            stopwords=stopwords,
        )

    return _detect


class DetectTest(unittest.TestCase):
    """Test main function API to annotate a document."""

    def setUp(self) -> None:
        """Create a custom detect_ivg function."""
        self.stopwords = Stopwords()
        self.tokenizer = french_tokenizer()
        self.matcher = Matcher()
        termino_ivg = get_termino_ivg()
        self.trieIVG = Trie()
        self.trieIVG.add_keywords(
            keywords=termino_ivg,
            tokenizer=self.tokenizer,
            stopwords=self.stopwords,
        )
        self.detect_ivg = build_detect(
            trie=self.trieIVG,
            syns_provider=self.matcher,
            stopwords=self.stopwords,
        )

    def test_detection(self):
        """Two annotation are detected:
        * insuffisance cardiaque
        * insuffisance cardiaque gauche
        """
        text = "Signes d'insuffisance cardiaque gauche"
        tokens = self.tokenizer.tokenize(text)
        annots = self.detect_ivg(tokens=tokens, w=1)
        self.assertEqual(2, len(annots))
        self.assertEqual(1, len(annots[0].keywords))
        self.assertEqual("I50.9", annots[0].keywords[0].kb_id)
        self.assertEqual("I50.1", annots[1].keywords[0].kb_id)

    def test_rm_ancestors(self):
        """Calling this function remove 'insuffisance cardiaque'
        annotation that is an ancestor.
        """
        text = "Signes d'insuffisance cardiaque gauche"
        tokens = self.tokenizer.tokenize(text)
        annots = self.detect_ivg(tokens=tokens, w=1)
        self.assertEqual(2, len(annots))
        annots_filt = rm_nested_annots(annots=annots, keep_ancestors=False)
        self.assertEqual(1, len(annots_filt))
        self.assertEqual(
            annots_filt[0].norm_label, "insuffisance cardiaque gauche"
        )

    def test_overlap_but_not_ancestors(self):
        """span of 'ulcere duodenal' overlaps 'ulcere gastrique' but
        'ulcere gastrique' is not an ancestor, so it shouldn't be removed."""
        text = "ulcere gastrique et duodenale"
        termino_ulceres = get_termino_ulceres()
        trie_ulceres = Trie()
        trie_ulceres.add_keywords(
            keywords=termino_ulceres,
            tokenizer=self.tokenizer,
            stopwords=self.stopwords,
        )
        detect_ulceres = build_detect(
            trie=trie_ulceres,
            syns_provider=self.matcher,
            stopwords=self.stopwords,
        )
        self.stopwords.add(words=["et"])
        tokens = self.tokenizer.tokenize(text)
        annots = detect_ulceres(tokens=tokens, w=2)
        annots = rm_nested_annots(annots=annots, keep_ancestors=False)
        self.assertEqual(2, len(annots))
        self.assertEqual(0, annots[1].start)
        self.assertEqual(len(text), annots[1].end)

    def test_abbreviations(self):
        """Add the abbreviation 'IC' for 'insuffisance cardique'."""
        self.matcher.add_fuzzy_algo(get_abbs_ic())
        text = "Le patient présente des signes d'IC gauche"
        tokens = self.tokenizer.tokenize(text)
        annots = self.detect_ivg(tokens=tokens, w=1)
        self.assertEqual(2, len(annots))

    def test_window(self):
        """Different w values result in different annotations."""
        text = "insuffisance de la fonction cardiaque"
        # w = 1
        tokens = self.tokenizer.tokenize(text)
        annots = self.detect_ivg(tokens=tokens, w=1)
        self.assertEqual(0, len(annots))

        # w = 3 (insuffisance is 4 words away, no annot
        tokens = self.tokenizer.tokenize(text)
        annots = self.detect_ivg(tokens=tokens, w=3)
        self.assertEqual(0, len(annots))

        # w = 4
        tokens = self.tokenizer.tokenize(text)
        annots = self.detect_ivg(tokens=tokens, w=4)
        self.assertEqual(1, len(annots))

    def test_window_with_stopwords(self):
        """Since stopwords are ignored, words can be much closer with stopwords
        In this example, although 'cardiaque' is separated from 'insuffisance'
        by 3 words, the ent is detected with a window of 1.
        """
        text = "insuffisance de la fonction cardiaque"
        self.stopwords.add(words=["de", "la", "fonction"])
        tokens = self.tokenizer.tokenize(text)
        annots = self.detect_ivg(tokens=tokens, w=1)
        self.assertEqual(1, len(annots))

    def test_empty_string(self):
        """An empty string returns no error and no annotation."""
        tokens = self.tokenizer.tokenize("")
        annots = self.detect_ivg(tokens=tokens, w=1)
        self.assertEqual(0, len(annots))

    def test_multiple_fuzzy_algos(self):
        """Each token contains information about which fuzzy algorithm matched
        the label of the token. Check the number of algorithm matched each
        token.
        """
        abbs = get_abbs_ic()
        leven: SpellWiseWrapper = SpellWiseWrapper(
            ESpellWiseAlgo.LEVENSHTEIN, max_distance=1
        )
        leven.add_words(words=["insuffisance", "cardiaque"])
        self.matcher.add_fuzzy_algo(abbs)
        self.matcher.add_fuzzy_algo(leven)
        text = "Ins Cardiaqu"
        tokens = self.tokenizer.tokenize(text)
        annots = self.detect_ivg(tokens=tokens, w=1)
        self.assertEqual(1, len(annots))
        # IC was matched by the 'Abbreviation' algorithm
        annotation = annots[0]
        self.assertEqual(2, len(annotation._tokens))
        ins_token, algos = list(annotation.get_tokens_algos())[0]
        self.assertEqual(
            1, len(algos)
        )  # only one algorithm matched this token
        self.assertTrue(abbs.name in algos)  # it was the abbreviation

        ins_token, algos = list(annotation.get_tokens_algos())[1]
        self.assertEqual(
            1, len(algos)
        )  # only one algorithm matched this token
        self.assertTrue(leven.name in algos)  # it was the abbreviation

    def test_token_type(self):
        """A custom token type with a POS property."""
        token_ins = TokenPOS(
            start=0, end=4, label="ins", norm_label="ins", pos="NOUN"
        )
        token_card = TokenPOS(
            start=0,
            end=4,
            label="cardiaque",
            norm_label="cardiaque",
            pos="ADJ",
        )
        tokens = [token_ins, token_card]
        fuzzy_lemma = FuzzyAlgoPos()
        self.matcher.add_fuzzy_algo(fuzzy_lemma)
        annots = self.detect_ivg(tokens=tokens, w=1)
        ins_token: TokenPOS = cast(TokenPOS, annots[0]._tokens[0])
        self.assertEqual("NOUN", ins_token.pos)
        self.assertEqual(1, len(annots))
        token_ins.pos = "PUNCT"
        annots = self.detect_ivg(tokens=tokens, w=1)
        self.assertEqual(0, len(annots))


class DetectAmbiguityTest(unittest.TestCase):
    def setUp(self) -> None:
        # IRC
        self.tokenizer = french_tokenizer()
        self.stopwords = Stopwords()
        self.matcher = Matcher()
        self.trieIRC = Trie()
        self.trieIRC.add_keywords(
            keywords=get_termino_irc(),
            tokenizer=self.tokenizer,
            stopwords=self.stopwords,
        )
        self.detect_irc = build_detect(
            trie=self.trieIRC,
            syns_provider=self.matcher,
            stopwords=self.stopwords,
        )

    def test_ambiguity(self):
        """When two paths are available, the 'annot_text' algorithm
        can return multiple annotations for the same token. In this example,
        the short form 'IRC' is ambiguous and the function returns 2 annots.
        """
        self.matcher.add_fuzzy_algo(get_abbs_irc())
        text = "antécédent d'IRC"
        tokens = self.tokenizer.tokenize(text)
        annots = self.detect_irc(tokens=tokens, w=1)
        annots = rm_nested_annots(annots=annots, keep_ancestors=False)
        self.assertEqual(2, len(annots))  # two long forms / keywords detected

    def test_ambiguity_rm_nested(self):
        """rm_nested_annots remove the nested annotations.
        In case of an ambiguity, if the two ambiguous annotations have
        the same length then no annotation is removed.
        """
        self.matcher.add_fuzzy_algo(get_abbs_irc())
        text = "antécédent d'IRC"
        tokens = self.tokenizer.tokenize(text)
        annots = self.detect_irc(tokens=tokens, w=1)
        self.assertEqual(2, len(annots))  # two long forms / keywords detected
        annots = rm_nested_annots(annots=annots, keep_ancestors=False)
        self.assertEqual(2, len(annots))  # no ent removed

    def test_ambiguity_rm_nested_2(self):
        """An ambiguity can be removed thanks to the context: here the word
        'dialysee' follows 'IRC' so it gives context to remove the ambiguity
        of the 'IRC' ent.
        """
        self.matcher.add_fuzzy_algo(get_abbs_irc())
        text = "antécédent d'IRC dialysée"
        tokens = self.tokenizer.tokenize(text)
        annots = self.detect_irc(tokens=tokens, w=1)
        self.assertEqual(3, len(annots))  # two long forms / keywords detected
        annots_filt = rm_nested_annots(annots=annots, keep_ancestors=False)
        self.assertEqual(
            1, len(annots_filt)
        )  # word 'dialysée' provides the context to remove the ambiguity
        annots_filt = rm_nested_annots(annots=annots, keep_ancestors=True)
        self.assertEqual(2, len(annots_filt))


class TokenPOS(Token, IToken):
    """A custom token type to implement a custom FuzzyAlgo."""

    def __init__(
        self, start: int, end: int, label: str, norm_label: str, pos: str
    ):
        super().__init__(start, end, label, norm_label)
        self.pos = pos


class FuzzyAlgoPos(FuzzyAlgo, ABC):
    """A custom fuzzy algo working with TokenPOS"""

    def __init__(self, name: str = "test_pos"):
        super().__init__(name)

    def get_synonyms(
        self,
        tokens: Sequence[TokenPOS],
        i: int,
        w_states: List[List[IState]],
    ) -> Iterable[SynAlgo]:
        """Returns only if POS is NOUN"""
        token = tokens[i]
        if token.pos == "NOUN":
            yield self.word_to_syn("insuffisance"), self.name
        else:
            yield tuple(), "NO_SYN"


if __name__ == "__main__":
    unittest.main()
