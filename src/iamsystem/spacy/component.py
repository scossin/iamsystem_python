""" Custom spaCy components to add iamsystem algorithm in a spaCy pipeline."""

import importlib

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List

from spacy import Language
from spacy.tokens import Doc
from spacy.tokens import Span

from iamsystem.fuzzy.api import FuzzyAlgo
from iamsystem.keywords.api import IEntity
from iamsystem.keywords.api import IKeyword
from iamsystem.matcher.annotation import Annotation
from iamsystem.matcher.api import IMatcher
from iamsystem.matcher.matcher import Matcher
from iamsystem.spacy.token import TokenSpacyAdapter
from iamsystem.spacy.tokenizer import SpacyTokenizer
from iamsystem.stopwords.api import IStopwords
from iamsystem.tokenization.api import ITokenizer
from iamsystem.tokenization.normalize import lower_no_accents
from iamsystem.tokenization.normalize import normalizeFun


class BaseCustomComp(ABC):
    """Base class to build a custom spaCy component."""

    def __init__(self, nlp: Language, name: str, attr: str = "iamsystem"):
        """Create a custom spaCy component.

        :param nlp: a spacy Language.
        :param name: the name of this spaCy component.
        :param attr: the attribute that stores iamsystem's annotations
            in a spaCy span instance.
        """

        self.name = name
        self.nlp = nlp
        self.attr = attr
        self._set_extensions()

    @property
    @abstractmethod
    def matcher(self) -> IMatcher[TokenSpacyAdapter]:
        """A :class:`~iamsystem.Matcher`.
        This method is abstract to allow several ways to make a dependency
        injection of a matcher in this class.
        """
        raise NotImplementedError

    def _set_extensions(self) -> None:
        """Add an extension to the spaCy Span class to store iamsystem's
        annotations."""
        if not Span.has_extension(self.attr):
            Span.set_extension(self.attr, default=None)

    def __call__(self, doc: Doc) -> Doc:
        """Function called by spaCy to execute this component."""
        spans = self.process(doc)
        if self.attr not in doc.spans:
            doc.spans[self.attr] = spans
        return doc

    def process(self, doc) -> List[Span]:
        """Annotate a document. Call IAMsystem algorithm."""
        tokens = self.matcher.tokenize(text=doc)
        anns: List[Annotation[TokenSpacyAdapter]] = self.matcher.annot_tokens(
            tokens=tokens
        )
        spacy_spans = []
        for ann in anns:
            start_i = ann.tokens[0].spacy_token.i
            end_i = ann.tokens[-1].spacy_token.i
            # 'A label to attach to the span, e.g. for named entities.'
            labels = [str(keyword) for keyword in ann.keywords]
            label = ";".join(labels)
            # 'A knowledge base ID to attach to the span for named entities.'
            entities = [kw for kw in ann.keywords if isinstance(kw, IEntity)]
            if len(entities) != 0:
                kbids = [entity.kb_id for entity in entities]
                kb_id = ";".join(kbids)
                span = Span(
                    doc=doc, start=start_i, end=end_i, label=label, kb_id=kb_id
                )
            else:
                span = Span(doc=doc, start=start_i, end=end_i, label=label)
            span._.set(self.attr, ann)
            spacy_spans.append(span)
        return spacy_spans


@Language.factory("iamsystem")
class IAMsystemSpacy(BaseCustomComp):
    """A stateful component.
    'Component factories are callables that take settings and return a pipeline
    component function. This is useful if your component is stateful and if
    you need to customize their creation'.
    See: https://spacy.io/usage/processing-pipelines#custom-components
    """

    def __init__(
        self,
        nlp: Language,
        name: str,
        keywords: Iterable[IKeyword],
        fuzzy_algos: Iterable[FuzzyAlgo],
        w: int = 1,
        remove_nested_annots: bool = True,
        stopwords: IStopwords[TokenSpacyAdapter] = None,
        norm_fun: normalizeFun = lower_no_accents,
        attr: str = "iamsystem",
    ):
        """
        Create a custom spaCy component.
        :class:`~iamsystem.Matcher` uses spaCy tokenizer to tokenize the
        documents and the keywords.

        :param nlp: a spacy Language.
        :param name: the name of this spaCy component.
        :param keywords: a list of :class:`~iamsystem.IKeywords` to detect
          in a document.
        :param fuzzy_algos: a list of :class:`~iamsystem.FuzzyAlgo`.
        :param w: :class:`~iamsystem.Matcher`'s window parameter.
        :param remove_nested_annots: whether to remove nested annotations.
        :param stopwords: :class:`~iamsystem.IStopwords` instance.
        :param norm_fun: a function that normalizes the 'norm_' attribute
            of a spaCy token, attribute used by iamsystem.
        :param attr: the attribute to store iamsystem's annotation in a spaCy
            span instance.
        """

        super().__init__(
            nlp=nlp,
            name=name,
            attr=attr,
        )
        tokenizer: ITokenizer[TokenSpacyAdapter] = SpacyTokenizer(
            nlp=nlp, norm_fun=norm_fun
        )
        self._matcher = Matcher(tokenizer=tokenizer, stopwords=stopwords)
        self._matcher.remove_nested_annots = remove_nested_annots
        self._matcher.w = w
        for algo in fuzzy_algos:
            self._matcher.add_fuzzy_algo(fuzzy_algo=algo)
        self._matcher.add_keywords(keywords=keywords)

    @property
    def matcher(self) -> IMatcher[TokenSpacyAdapter]:
        """A matcher that uses spaCy tokenizer."""
        return self._matcher


@Language.factory("iamsystem_matcher")
class IAMsystemBuildSpacy(BaseCustomComp):
    """A serializable custom component."""

    def __init__(
        self,
        nlp: Language,
        name: str,
        build_params: Dict[Any, Any],
        serialized_kw: Dict[Any, Any] = None,
        attr: str = "iamsystem",
        norm_fun: normalizeFun = None,
    ):
        """
        Create a custom spaCy component.
        :class:`~iamsystem.Matcher` uses spaCy tokenizer to tokenize the
        documents and the keywords.

        :param nlp: a spacy Language.
        :param name: the name of this spaCy component.
        :param attr: the attribute to store iamsystem's annotation in a spaCy
            span instance.
        :param serialized_kw: a way to import serialized keywords.
            A dictionary containing 3 fields:
            - 'module': module name of the class to import. ex: 'iamsystem'.
            - 'class_name': the Keyword class to import.
            - 'kw': an iterable of dict created with the asdict()
                function.
            If None, keywords are expected in 'build_params'. You will need
            a registered function to import the keywords.
        :param norm_fun: a function that normalizes the 'norm_' attribute
            of a spaCy token, attribute used by iamsystem. Default to lower
            case and remove accents.
        :param build_params: `~iamsystem.Matcher.build` parameters, the spacy
            tokenizer will be used whatever the tokenizer value.
        """

        super().__init__(
            nlp=nlp,
            name=name,
            attr=attr,
        )
        if norm_fun is None:
            norm_fun = lower_no_accents
        tokenizer: ITokenizer[TokenSpacyAdapter] = SpacyTokenizer(
            nlp=nlp, norm_fun=norm_fun
        )
        build_params["tokenizer"] = tokenizer
        # import serializable keywords:
        if serialized_kw is not None:
            KeywordClass = getattr(
                importlib.import_module(serialized_kw["module"]),
                serialized_kw["class_name"],
            )
            kws = [KeywordClass(**params) for params in serialized_kw["kws"]]
            build_params["keywords"] = kws
        self._matcher = Matcher.build(**build_params)

    @property
    def matcher(self) -> IMatcher[TokenSpacyAdapter]:
        """A matcher that uses spaCy tokenizer."""
        return self._matcher
