import unittest

from iamsystem.keywords.keywords import Entity
from iamsystem.stopwords.simple import Stopwords
from iamsystem.tokenization.normalize import lower_no_accents
from iamsystem.tokenization.tokenize import TokenizerImp
from iamsystem.tokenization.tokenize import split_alpha_num
from iamsystem.tree.nodes import EMPTY_NODE
from iamsystem.tree.nodes import Node
from iamsystem.tree.trie import Trie
from tests.utils import get_termino_ivg


class TrieTest(unittest.TestCase):
    def setUp(self) -> None:
        self.terminoIVG = get_termino_ivg()
        self.stopwords = Stopwords()
        self.tokenizer = TokenizerImp(
            split=split_alpha_num, normalize=lower_no_accents
        )

    def test_get_number_of_nodes(self):
        """Number of tokens in keywords."""
        trie = Trie()
        self.assertEqual(1, trie.get_number_of_nodes())
        # first ent
        ent = Entity("Insuffisance Cardiaque", "I50.9")
        tokens = ["insuffisance", "cardiaque"]
        trie.add_keyword_with_tokens(keyword=ent, tokens=tokens)
        self.assertEqual(3, trie.get_number_of_nodes())  # one per token
        # second ent
        ent = Entity("Insuffisance Cardiaque Gauche", "I50.1")
        tokens = ["insuffisance", "cardiaque", "gauche"]
        trie.add_keyword_with_tokens(keyword=ent, tokens=tokens)
        self.assertEqual(3 + 1, trie.get_number_of_nodes())  # new node gauche

    def test_get_initial_state(self):
        """The initial state is the root node of the trie."""
        trie = Trie()
        self.assertTrue(Node.is_root_node(trie.get_initial_state()))

    def test_build_trie(self):
        """'insuffisance ventriculaire gauche'
        START_TOKEN -> 'insuffisance'"""
        trie = Trie()
        trie.add_keywords(
            keywords=self.terminoIVG,
            tokenizer=self.tokenizer,
            stopwords=self.stopwords,
        )
        self.assertTrue(
            trie.get_initial_state().has_transition_to("insuffisance")
        )

    def test_build_trie_with_stopword(self):
        """Check the tries doesn't create a node if the ent is a stopword"""
        trie = Trie()
        trie.add_keywords(
            keywords=self.terminoIVG,
            tokenizer=self.tokenizer,
            stopwords=self.stopwords,
        )
        # 4 nodes: root, insuffisance, cardiaque, gauche.
        self.assertEqual(trie.get_number_of_nodes(), 4)
        self.stopwords.add(words=["insuffisance"])
        trie = Trie()
        trie.add_keywords(
            keywords=self.terminoIVG,
            tokenizer=self.tokenizer,
            stopwords=self.stopwords,
        )
        self.assertEqual(trie.get_number_of_nodes(), 3)  # 4 - 1 nodes
        # insuffisance is ignored and is not a node anymore :
        self.assertTrue(
            not trie.get_initial_state().has_transition_to("insuffisance")
        )
        self.assertTrue(
            trie.get_initial_state().has_transition_to("cardiaque")
        )

    def test_build_trie_no_normalizer(self):
        """Check the tries stores the token unormalized when changing the
        normalize function"""

        def no_normalizer(string: str) -> str:
            return string

        self.tokenizer.normalize = no_normalizer
        trie = Trie()
        trie.add_keywords(
            keywords=self.terminoIVG,
            tokenizer=self.tokenizer,
            stopwords=self.stopwords,
        )
        self.assertEqual(trie.get_number_of_nodes(), 4)
        self.assertTrue(
            not trie.get_initial_state().has_transition_to("insuffisance")
        )
        # no normalization : Insuffisance is not normalized to insuffisance
        self.assertTrue(
            trie.get_initial_state().has_transition_to("Insuffisance")
        )

    def test_build_trie_warning_ent_not_added(self):
        """Stopwords are removed before added to the trie.
        If no tokens remain, a warning is generated.
        """
        stopwords = Stopwords()
        stopwords.add(["insuffisance", "cardiaque"])
        tokenizer = TokenizerImp(
            split=split_alpha_num, normalize=lower_no_accents
        )
        termino = get_termino_ivg()
        trie = Trie()
        with self.assertWarns(Warning):
            trie.add_keywords(
                keywords=termino, tokenizer=tokenizer, stopwords=stopwords
            )


class NodeTest(unittest.TestCase):
    def setUp(self) -> None:
        trie = Trie()
        root_node = trie.get_initial_state()
        self.ins_node = Node(
            token="insuffisance", node_num=1, parent_node=root_node
        )
        self.card_node = Node(
            token="cardiaque", node_num=2, parent_node=self.ins_node
        )
        self.node_gauche = Node(
            token="gauche", node_num=3, parent_node=self.card_node
        )

    def test_keyword_not_overriden(self):
        """Adding the same ent to a node doesn't override it."""
        ent = Entity("Insuffisance Cardiaque Gauche", "XXX")
        node = Node(node_num=3, token="gauche")
        node.add_keyword(ent)
        node.add_keyword(ent)
        self.assertEqual(2, len(list(node.get_keywords())))

    def test_node_equality(self):
        """Two nodes are equals if they have the same 'node_num' value"""
        first_node = Node(token="insuffisance", node_num=1)
        self.assertEqual(first_node, first_node)
        second_node = Node(token="cardiaque", node_num=1)
        self.assertEqual(first_node, second_node)
        third_node = Node(token="insuffisance", node_num=99)
        self.assertNotEqual(first_node, third_node)

    def test_has_transition_to(self):
        """insuffisance(ins) -> cardiaque -> gauche."""
        self.assertTrue(self.ins_node.has_transition_to("cardiaque"))
        self.assertTrue(not self.ins_node.has_transition_to("gauche"))
        self.assertTrue(self.ins_node.has_transition_to("cardiaque"))

    def test_get_ancestors(self):
        """insuffisance <- cardiaque <- gauche"""
        ancestors = self.node_gauche.get_ancestors()
        self.assertTrue(self.card_node in ancestors)
        self.assertTrue(self.ins_node in ancestors)

    def test_get_child_nodes(self):
        """insuffisance -> cardiaque"""
        ins_node = Node(token="insuffisance", node_num=1)
        card_node = Node(token="cardiaque", node_num=2, parent_node=ins_node)
        self.assertEqual(1, len(list(ins_node.get_child_nodes())))
        self.assertEqual(0, len(list(card_node.get_child_nodes())))

    def test_goto_node(self):
        """insuffisance -> cardiaque."""
        ins_node = Node(token="insuffisance", node_num=1)
        card_node = Node(token="cardiaque", node_num=2, parent_node=ins_node)
        node = ins_node.goto_node("cardiaque")
        self.assertEqual(node, card_node)

    def test_goto_node_dead_end(self):
        """insuffisance -> None ; so EMPTY_NODE is expected."""
        ins_node = Node(token="insuffisance", node_num=1)
        node = ins_node.goto_node("cardiaque")
        self.assertEqual(node, EMPTY_NODE)

    def test_get_token(self):
        """Simple attribute access."""
        ins_node = Node(token="insuffisance", node_num=1)
        self.assertEqual(ins_node.token, "insuffisance")

    def test_jumpToNode(self):
        """insuffisance -> cardiaque -> gauche.
        From insuffisance we can jump to gauche.
        """
        ins_node = Node(token="insuffisance", node_num=1)
        card_node = Node(token="cardiaque", node_num=2, parent_node=ins_node)
        gauche_node = Node(token="gauche", node_num=3, parent_node=card_node)
        self.assertEqual(ins_node.jump_to_node(["cardiaque"]), card_node)
        self.assertEqual(
            ins_node.jump_to_node(["cardiaque", "gauche"]), gauche_node
        )
        self.assertEqual(ins_node.jump_to_node(["gauche"]), EMPTY_NODE)

    def test_is_a_final_state(self):
        """iff a keyword is associated to a Node."""
        ins_node = Node(token="insuffisance", node_num=1)
        self.assertFalse(ins_node.is_a_final_state())
        ent = Entity("Insuffisance", "XXX")
        ins_node.add_keyword(ent)
        self.assertTrue(ins_node.is_a_final_state())

    def test_get_keywords(self):
        """Add and retrieve an added keyword."""
        ins_node = Node(token="insuffisance", node_num=1)
        ent = Entity("Insuffisance", "XXX")
        ins_node.add_keyword(ent)
        self.assertTrue(ent in ins_node.get_keywords())


class EmptyNodeTest(unittest.TestCase):
    def test_goto_node(self):
        """Goes always to itself."""
        node = EMPTY_NODE.goto_node("insuffisance")
        self.assertEqual(node, EMPTY_NODE)

    def test_jump_to_node(self):
        """jumps always to itself."""
        node = EMPTY_NODE.jump_to_node(["insuffisance", "cardiaque"])
        self.assertEqual(node, EMPTY_NODE)

    def test_is_a_final_state(self):
        """Never ever."""
        self.assertTrue(not EMPTY_NODE.is_a_final_state())

    def test_has_transition_to(self):
        """Returns always false."""
        self.assertTrue(not EMPTY_NODE.has_transition_to("any token"))

    def test_get_node_number(self):
        """Constant."""
        self.assertTrue(EMPTY_NODE.node_num == -1)

    def test_get_parent_node(self):
        """Itself."""
        self.assertEqual(EMPTY_NODE, EMPTY_NODE.parent_node)


if __name__ == "__main__":
    unittest.main()
