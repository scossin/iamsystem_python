""" IAMsystem matching strategies."""
from collections import defaultdict
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set

from iamsystem.fuzzy.api import ISynsProvider
from iamsystem.fuzzy.api import SynAlgos
from iamsystem.matcher.annotation import Annotation
from iamsystem.matcher.annotation import create_annot
from iamsystem.matcher.annotation import sort_annot
from iamsystem.matcher.api import IAnnotation
from iamsystem.matcher.api import IMatchingStrategy
from iamsystem.matcher.util import StateTransition
from iamsystem.stopwords.api import IStopwords
from iamsystem.tokenization.api import TokenT
from iamsystem.tokenization.tokenize import Token
from iamsystem.tree.nodes import EMPTY_NODE
from iamsystem.tree.nodes import INode


class WindowMatching(IMatchingStrategy):
    """Default matching strategy.
    It keeps track of all states within a window range and can be produce
    overlapping/nested annotations.
    If you want to use a large window with a large dictionary, it is
    recommended to use 'LargeWindowMatching' instead.
    """

    def detect(
        self,
        tokens: Sequence[TokenT],
        w: int,
        initial_state: INode,
        syns_provider: ISynsProvider,
        stopwords: IStopwords,
    ) -> List[IAnnotation[TokenT]]:
        """Overrides."""
        annots: List[Annotation] = []
        transitions: Set[StateTransition] = set()
        first_trans = StateTransition.create_first_trans(
            initial_state=initial_state
        )
        transitions.add(first_trans)
        # count_not_stopword allows a stopword-independent window size.
        count_not_stopword = 0
        stop_tokens: List[TokenT] = []
        new_trans: List[StateTransition] = []
        trans2remove: List[StateTransition] = []
        for i, token in enumerate(tokens):
            if stopwords.is_token_a_stopword(token):
                stop_tokens.append(token)
                continue
            new_trans.clear()
            trans2remove.clear()
            count_not_stopword += 1
            syns_algos: List[SynAlgos] = syns_provider.get_synonyms(
                tokens=tokens, token=token, transitions=transitions
            )
            for trans in transitions:
                if trans.is_obsolete(
                    count_not_stopword=count_not_stopword, w=w
                ):
                    trans2remove.append(trans)
                    continue
                for syn, algos in syns_algos:
                    next_node = trans.node.jump_to_node(syn)
                    # when no path is found, EMPTY_NODE is returned.
                    if next_node is EMPTY_NODE:
                        continue
                    next_trans = StateTransition(
                        previous_trans=trans,
                        node=next_node,
                        token=token,
                        algos=algos,
                        count_not_stopword=count_not_stopword,
                    )
                    new_trans.append(next_trans)
                    if next_node.is_a_final_state():
                        annot = create_annot(
                            last_trans=next_trans, stop_tokens=stop_tokens
                        )
                        annots.append(annot)
            # Prepare next iteration: first loop remove out-of-reach states.
            # Second iteration add new states.
            for trans in trans2remove:
                transitions.remove(trans)
            for trans in new_trans:
                # this condition happens in the 'cancer cancer' example above.
                # the effect is replacing a previous transition by a new,
                # more recent one.
                if trans in transitions:
                    transitions.remove(trans)
                transitions.add(trans)
        sort_annot(annots)  # mutate the list like annots.sort()
        return annots


class LargeWindowMatching(IMatchingStrategy):
    """A large window strategy suited for a large window (ex: w=1000) and
    a large dictionary. This strategy is faster than the Window strategy if
    the dictionary is large, otherwise it's slower. It trades space for
    time complexity (space memory increases but matching speed increases).
    """

    def __init__(self):
        self.initial_state: Optional[INode] = None
        self.transitions = {}
        self.avaible_trans = {}

    def detect(
        self,
        tokens: Sequence[TokenT],
        w: int,
        initial_state: INode,
        syns_provider: ISynsProvider,
        stopwords: IStopwords,
    ) -> List[IAnnotation[TokenT]]:
        """Overrides."""
        # create a cache for the initial state:
        if self.initial_state is None or self.initial_state != initial_state:
            self._initialize(initial_state=initial_state)
        annots: List[Annotation] = []
        transitions: Dict[int, StateTransition] = self.transitions.copy()
        # Avaible_trans stores which transition has a transition with word w.
        # This hash table filters transitions and avoid a loop over
        # all transitions to check if a transition is possible.
        avaible_trans: Dict[str, Set[int]] = self.avaible_trans.copy()
        count_not_stopword = 0
        stop_tokens: List[TokenT] = []
        new_trans: Set[StateTransition] = set()
        emptylist = []
        for i, token in enumerate(tokens):
            if stopwords.is_token_a_stopword(token):
                stop_tokens.append(token)
                continue
            new_trans.clear()
            count_not_stopword += 1
            syns_algos: List[SynAlgos] = syns_provider.get_synonyms(
                tokens=tokens,
                token=token,
                transitions=iter(transitions.values()),
            )
            for syn, algos in syns_algos:
                transitions_id = avaible_trans.get(syn[0], emptylist)
                for trans_id in transitions_id.copy():
                    trans: StateTransition = transitions.get(trans_id, None)
                    # case a transition was obsolete and removed:
                    if trans is None:
                        transitions_id.remove(trans_id)
                        continue
                    # case a transition is obsolete, we remove it now:
                    if trans.is_obsolete(
                        count_not_stopword=count_not_stopword, w=w
                    ):
                        del transitions[trans_id]
                        transitions_id.remove(trans_id)
                        continue
                    node = trans.node.jump_to_node(syn)
                    # when no path is found, EMPTY_NODE is returned.
                    if node is EMPTY_NODE:
                        # I could raise a valueError here since it should be
                        # impossible: all the states have a transition to
                        # the current synonym.
                        continue
                    next_trans = StateTransition(
                        previous_trans=trans,
                        node=node,
                        token=token,
                        algos=algos,
                        count_not_stopword=count_not_stopword,
                    )
                    new_trans.add(next_trans)
            for trans in new_trans:
                if trans.node.is_a_final_state():
                    annot = create_annot(
                        last_trans=trans, stop_tokens=stop_tokens
                    )
                    annots.append(annot)
                for nexttoken in trans.node.get_children_tokens():
                    avaible_trans[nexttoken].add(trans.id)
                transitions[trans.id] = trans
        sort_annot(annots)  # mutate the list like annots.sort()
        return annots

    def _initialize(self, initial_state: INode) -> None:
        """Initialize hashtable to avoid repeating this operation multiple
        times.

        :param initial_state: the initial state (eg. root node).
        :return: None.
        """
        self.initial_state = initial_state
        self.transitions: Dict[int, StateTransition] = {}
        self.avaible_trans: Dict[str, Set[int]] = defaultdict(set)
        first_trans = StateTransition.create_first_trans(
            initial_state=initial_state
        )
        self.transitions[first_trans.id] = first_trans
        for token in first_trans.node.get_children_tokens():
            self.avaible_trans[token].add(first_trans.id)


class NoOverlapMatching(IMatchingStrategy):
    """The old matching strategy that was in used till 2022.
    The 'w' parameter has no effect.
    It annotates the longest path and outputs no overlapping annotation
    except in case of ambiguity.
    Algorithm formalized in https://ceur-ws.org/Vol-3202/livingner-paper11.pdf # noqa
    """

    END_TOKEN = Token(
        start=-1,
        end=-1,
        label="IAMSYSTEM_END_TOKEN",
        norm_label="IAMSYSTEM_END_TOKEN",
        i=-1,
    )

    def detect(
        self,
        tokens: Sequence[TokenT],
        w: int,
        initial_state: INode,
        syns_provider: ISynsProvider,
        stopwords: IStopwords,
    ) -> List[IAnnotation[TokenT]]:
        """Overrides.
        Note that w parameter is ignored and so has no effect."""
        annots: List[Annotation] = []
        transitions: Set[StateTransition] = set()
        first_trans = StateTransition.create_first_trans(
            initial_state=initial_state
        )
        transitions.add(first_trans)
        stop_tokens: List[TokenT] = []
        # i stores the position of the current token.
        i = 0
        # started_at is used for back-tracking, it stores the 'i' value
        # from which a state transition started. If the search founds nothing
        # i is reset to last i value + 1.
        started_at = 0
        while i < len(tokens) + 1:
            # The goal of this 'END_TOKEN' is to generate a dead end
            # for the last transitions (end of the document).
            if i == len(tokens):
                token = NoOverlapMatching.END_TOKEN
            else:
                token = tokens[i]
            if stopwords.is_token_a_stopword(token):
                stop_tokens.append(token)
                i += 1
                started_at += 1
                continue
            new_trans: Set[StateTransition] = set()
            syns_algos: List[SynAlgos] = syns_provider.get_synonyms(
                tokens=tokens, token=token, transitions=transitions
            )
            for trans in transitions:
                for syn, algos in syns_algos:
                    node = trans.node.jump_to_node(syn)
                    # when no path is found, EMPTY_NODE is returned.
                    if node is EMPTY_NODE:
                        continue
                    next_trans = StateTransition(
                        previous_trans=trans,
                        node=node,
                        token=token,
                        algos=algos,
                        count_not_stopword=-1,
                    )
                    new_trans.add(next_trans)
            # 1) Case the algorithm is exploring a path:
            if len(new_trans) != 0:
                transitions = new_trans
                i += 1
                # don't do 'started_at += 1' to allow backtracking later.
            # 2) Case the algorithm has finished exploring a path:
            else:
                # the algorithm has gone nowhere from initial state:
                if len(transitions) == 1 and first_trans in transitions:
                    i += 1
                    started_at += 1
                    continue
                # the algorithm has gone somewhere. Save annotations and
                # restart at last annotation ith token + 1.
                last_i = self._add_annots(
                    annots=annots,
                    transitions=transitions,
                    started_at=started_at,
                    stop_tokens=stop_tokens,
                )
                i = last_i + 1
                started_at = started_at + 1
                transitions.clear()
                transitions.add(first_trans)
        sort_annot(annots)  # mutate the list like annots.sort()
        return annots

    @staticmethod
    def _add_annots(
        annots: List[Annotation],
        transitions: Set[StateTransition],
        started_at: int,
        stop_tokens: List[TokenT],
    ) -> int:
        """Create annotations and mutate annots list.

        :param annots: the list of annotations.
        :param transitions: the current algorithm's states.
        :param started_at: the 'i' token at whcih the algorithm started a
            search.
        :param stop_tokens: stopwords
        :return: the last annotation 'i' value or started_at if no annotation
            generated.
        """
        last_annot_i = -1
        for trans in transitions:
            current_trans = trans
            # back track till the first state that is a final state
            # ex: 'cancer de la', backtrack to 'cancer'.
            while not current_trans.node.is_a_final_state():
                current_trans = current_trans.previous_trans
                if StateTransition.is_first_trans(current_trans):
                    break
            if current_trans.node.is_a_final_state():
                annot = create_annot(
                    last_trans=current_trans, stop_tokens=stop_tokens
                )
                last_annot_i = max(annot.end_i, last_annot_i)
                annots.append(annot)
        return max(started_at, last_annot_i)


class EMatchingStrategy(Enum):
    """Enumeration of matching strategies."""

    WINDOW = WindowMatching()
    " Default matching strategy. "
    LARGE_WINDOW = LargeWindowMatching()
    " Same annotations as Window but faster than window is large. "
    NO_OVERLAP = NoOverlapMatching()
    " No overlap/nested annotations, fastest strategies."
