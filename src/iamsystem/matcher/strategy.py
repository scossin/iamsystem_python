""" IAMsystem matching strategies."""
from collections import defaultdict
from enum import Enum
from typing import Dict
from typing import List
from typing import Sequence
from typing import Set

from iamsystem.fuzzy.api import ISynsProvider
from iamsystem.fuzzy.api import SynAlgos
from iamsystem.matcher.annotation import Annotation
from iamsystem.matcher.annotation import create_annot
from iamsystem.matcher.annotation import sort_annot
from iamsystem.matcher.api import IAnnotation
from iamsystem.matcher.api import IMatchingStrategy
from iamsystem.matcher.util import LinkedState
from iamsystem.matcher.util import create_start_state
from iamsystem.stopwords.api import IStopwords
from iamsystem.tokenization.api import TokenT
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
        # states stores linkedstate instance that keeps track of a tree path
        # and document's tokens that matched.
        states: Set[LinkedState] = set()
        start_state = create_start_state(initial_state=initial_state)
        states.add(start_state)
        # count_not_stopword allows a stopword-independent window size.
        count_not_stopword = 0
        stop_tokens: List[TokenT] = []
        new_states: List[LinkedState] = []
        # states2remove store states that will be out-of-reach
        # at next iteration.
        states2remove: List[LinkedState] = []
        for i, token in enumerate(tokens):
            if stopwords.is_token_a_stopword(token):
                stop_tokens.append(token)
                continue
            # w_bucket stores when a state will be out-of-reach
            # 'count_not_stopword % w' has range [0 ; w-1]
            w_bucket = count_not_stopword % w
            new_states.clear()
            states2remove.clear()
            count_not_stopword += 1
            # syns: 1 to many synonyms depending on fuzzy_algos configuration.
            syns_algos: List[SynAlgos] = syns_provider.get_synonyms(
                tokens=tokens, token=token, states=states
            )

            for state in states:
                if state.w_bucket == w_bucket:
                    states2remove.append(state)
                # 0 to many states for [0] to [w-1] ; [w] only the start state.
                for syn, algos in syns_algos:
                    node = state.node.jump_to_node(syn)
                    # when no path is found, EMPTY_NODE is returned.
                    if node is EMPTY_NODE:
                        continue
                    new_state = LinkedState(
                        parent=state,
                        node=node,
                        token=token,
                        algos=algos,
                        w_bucket=w_bucket,
                    )
                    new_states.append(new_state)
                    # Why 'new_state not in states':
                    # if node_num is already in the states set,it means
                    # an annotation was already created for this state.
                    # For example 'cancer cancer', if an annotation was created
                    # for the first 'cancer' then we don't want to create
                    # a new one for the second 'cancer'.
                    if node.is_a_final_state() and new_state not in states:
                        annot = create_annot(
                            last_el=new_state, stop_tokens=stop_tokens
                        )
                        annots.append(annot)
            # Prepare next iteration: first loop remove out-of-reach states.
            # Second iteration add new states.
            for state in states2remove:
                states.remove(state)
            for state in new_states:
                # this condition happens in the 'cancer cancer' example.
                # the effect is replacing a previous token by a new one.
                if state in states:
                    states.remove(state)
                states.add(state)
        sort_annot(annots)  # mutate the list like annots.sort()
        return annots


class LargeWindowMatching(IMatchingStrategy):
    """A large window strategy suited for a large window (ex: w=1000) and
    a large dictionary. This strategy is faster than the Window strategy if
    the dictionary is large, otherwise it's slower. It trades space for
    time complexity (space memory increases but matching speed increases).
    """

    def __init__(self):
        self.states = {}
        self.avaible_trans = {}
        self.is_initialized = False

    def detect(
        self,
        tokens: Sequence[TokenT],
        w: int,
        initial_state: INode,
        syns_provider: ISynsProvider,
        stopwords: IStopwords,
    ) -> List[IAnnotation[TokenT]]:
        """Overrides."""
        if not self.is_initialized:
            self._initialize(initial_state=initial_state)
            self.is_initialized = True
        annots: List[Annotation] = []
        states: Dict[int, LinkedState] = self.states.copy()
        # avaible_trans stores which state have a transition to a synonym.
        avaible_trans: Dict[str, Set[int]] = self.avaible_trans.copy()
        count_not_stopword = 0
        stop_tokens: List[TokenT] = []
        new_states: Set[LinkedState] = set()
        emptylist = []
        for i, token in enumerate(tokens):
            if stopwords.is_token_a_stopword(token):
                stop_tokens.append(token)
                continue
            new_states.clear()
            count_not_stopword += 1
            # syns: 1 to many synonyms depending on fuzzy_algos configuration.
            syns_algos: List[SynAlgos] = syns_provider.get_synonyms(
                tokens=tokens, token=token, states=iter(states.values())
            )
            for syn, algos in syns_algos:
                states_id = avaible_trans.get(syn[0], emptylist)
                for state_id in states_id.copy():
                    state: LinkedState = states.get(state_id, None)
                    # case a state was obsolete and removed:
                    if state is None:
                        states_id.remove(state_id)
                        continue
                    # case a state is obsolete and removed here:
                    if state.is_obsolete(
                        count_stop_word=count_not_stopword, w=w
                    ):
                        del states[state_id]
                        states_id.remove(state_id)
                        continue
                    node = state.node.jump_to_node(syn)
                    # when no path is found, EMPTY_NODE is returned.
                    if node is EMPTY_NODE:
                        # I could raise a valueError here since it should be
                        # impossible: all the states have a transition to
                        # the current synonym.
                        continue
                    new_state = LinkedState(
                        parent=state,
                        node=node,
                        token=token,
                        algos=algos,
                        w_bucket=count_not_stopword,
                    )
                    new_states.add(new_state)
            for state in new_states:
                # create an annotation if:
                # 1) node is a final state
                # 2) an annotation wasn't created yet for this state:
                # 2.1 there is no previous 'none-obsolete state'.
                if state.node.is_a_final_state():
                    old_state = states.get(state.id, None)
                    if old_state is None or old_state.is_obsolete(
                        count_stop_word=count_not_stopword, w=w
                    ):
                        annot = create_annot(
                            last_el=state, stop_tokens=stop_tokens
                        )
                        annots.append(annot)
                for nexttoken in state.node.get_child_tokens():
                    avaible_trans[nexttoken].add(state.id)
                states[state.id] = state
        sort_annot(annots)  # mutate the list like annots.sort()
        return annots

    def _initialize(self, initial_state: INode) -> None:
        """Initialize hashtable to avoid repeating this operation multiple
        times.

        :param initial_state: the initial state (eg. root node).
        :return: None.
        """
        self.states: Dict[int, LinkedState] = {}
        self.avaible_trans: Dict[str, Set[int]] = defaultdict(set)
        start_state = create_start_state(initial_state=initial_state)
        self.states[start_state.id] = start_state
        for token in start_state.node.get_child_tokens():
            self.avaible_trans[token].add(start_state.id)


class NoOverlapMatching(IMatchingStrategy):
    """The old matching strategy that was in used till 2022.
    The 'w' parameter has no effect.
    It annotates the longest path and outputs no overlapping annotation
    except in case of ambiguity. It's the fastest strategy.
    Algorithm formalized in https://ceur-ws.org/Vol-3202/livingner-paper11.pdf # noqa
    """

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
        # states stores linkedstate instance that keeps track of a tree path
        # and document's tokens that matched.
        states: Set[LinkedState] = set()
        start_state = create_start_state(initial_state=initial_state)
        states.add(start_state)
        stop_tokens: List[TokenT] = []
        # i stores the position of the current token.
        i = 0
        # started_at is used for back-tracking, it stores the initial 'i'
        # from which the initial state started.
        started_at = 0
        while i < len(tokens):
            token = tokens[i]
            if stopwords.is_token_a_stopword(token):
                stop_tokens.append(token)
                i += 1
                started_at += 1
                continue
            new_states: Set[LinkedState] = set()
            # syns: 1 to many synonyms depending on fuzzy_algos configuration.
            syns_algos: List[SynAlgos] = syns_provider.get_synonyms(
                tokens=tokens, token=token, states=states
            )
            for state in states:
                for syn, algos in syns_algos:
                    node = state.node.jump_to_node(syn)
                    # when no path is found, EMPTY_NODE is returned.
                    if node is EMPTY_NODE:
                        continue
                    new_state = LinkedState(
                        parent=state,
                        node=node,
                        token=token,
                        algos=algos,
                        w_bucket=-1,
                    )
                    new_states.add(new_state)
            # Case the algorithm is exploring a path:
            if len(new_states) != 0:
                states = new_states
                i += 1
                # don't 'started_at += 1' to allow backtracking later.
            # Case the algorithm has finished exploring a path:
            else:
                # the algorithm has gone nowhere from initial state:
                if len(states) == 1 and start_state in states:
                    i += 1
                    started_at += 1
                    continue
                # the algorithm has gone somewhere. Save annotations and
                # restart at last annotation ith token + 1 (no overlap).
                last_i = self._add_annots(
                    annots=annots,
                    states=states,
                    started_at=started_at,
                    stop_tokens=stop_tokens,
                )
                i = last_i + 1
                started_at = started_at + 1
                states.clear()
                states.add(start_state)
        # All tokens have been seen. Create last annotations if any states:
        self._add_annots(
            annots=annots,
            states=states,
            started_at=started_at,
            stop_tokens=stop_tokens,
        )
        sort_annot(annots)  # mutate the list like annots.sort()
        return annots

    @staticmethod
    def _add_annots(
        annots: List[Annotation],
        states: Set[LinkedState],
        started_at: int,
        stop_tokens: List[TokenT],
    ) -> int:
        """Create annotations and mutate annots list.

        :param annots: the list of annotations.
        :param states: the current algorithm's states.
        :param started_at: the 'i' token at whcih the algorithm started a
            search.
        :param stop_tokens: stopwords
        :return: the last annotation 'i' value or started_at if no annotation
            generated.
        """
        last_annot_i = -1
        for state in states:
            current_state = state
            # back track till the first state that is a final state
            # ex: 'cancer de la', backtrack to 'cancer'.
            while (
                not current_state.node.is_a_final_state()
                and current_state.parent is not None
            ):
                current_state = current_state.parent
            if current_state.node.is_a_final_state():
                annot = create_annot(
                    last_el=current_state, stop_tokens=stop_tokens
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
