
spaCy
-----

.. _spaCy: https://spacy.io/

This package provides a stateful spaCy component to add iamsystem algorithm in a spaCy pipeline.
Since a :ref:`api_doc:Matcher` configuration is not JSON serializable, matcher's parameters
are passed in registered functions:


.. code-block:: python

        from typing import Iterable
        from typing import List

        import spacy

        from spacy.lang.fr import French

        from iamsystem import Abbreviations
        from iamsystem import FuzzyAlgo
        from iamsystem.spacy import IAMsystemSpacy  # noqa
        from iamsystem.spacy import IsStopSpacy
        from iamsystem.spacy import TokenSpacyAdapter
        from iamsystem import IKeyword
        from iamsystem import IStopwords
        from iamsystem import Term
        from iamsystem import Terminology
        from iamsystem import french_tokenizer

        @spacy.registry.misc("umls_terms.v1")
        def get_termino_umls() -> Iterable[IKeyword]:
            """An imaginary set of umls terms."""
            termino = Terminology()
            term1 = Term("Insuffisance Cardiaque", "I50.9")
            term2 = Term("Insuffisance Cardiaque Gauche", "I50.1")
            termino.add_keywords(keywords=[term1, term2])
            return termino

        @spacy.registry.misc("fuzzy_algos_short_notes.v1")
        def get_fuzzy_algos_short_notes() -> List[FuzzyAlgo]:
            """An imaginary set of fuzzy algorithms for medical short notes."""
            tokenizer = french_tokenizer()
            abbs = Abbreviations(name="French medical abbreviations")
            abbs.add(short_form="ins", long_form="insuffisance",
                     tokenizer=tokenizer)
            abbs.add(
                short_form="ic",
                long_form="insuffisance cardiaque",
                tokenizer=tokenizer,
            )
            return [abbs]

        @spacy.registry.misc("stopwords_spacy.v1")
        def get_stopwords_short_notes() -> IStopwords[TokenSpacyAdapter]:
            """Use spaCy stopword list."""
            stopwords = IsStopSpacy()
            return stopwords

        nlp = French()
        nlp.add_pipe(
            "iamsystem",
            name="iamsystem",
            last=True,
            config={
                "keywords": {"@misc": "umls_terms.v1"},
                "stopwords": {"@misc": "stopwords_spacy.v1"},
                "fuzzy_algos": {"@misc": "fuzzy_algos_short_notes.v1"},
                "w": 1,
                "remove_nested_annots": True,
            },
        )
        doc = nlp("ic gauche")
        spans = doc.spans["iamsystem"]
        for span in spans:
            print(span._.iamsystem)
        # ic gauche	0 9	Insuffisance Cardiaque Gauche (I50.1)

See :ref:`api_doc:IAMsystemSpacy` to configure this component.
