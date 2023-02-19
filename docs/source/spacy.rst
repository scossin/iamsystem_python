
spaCy
-----

.. _spaCy: https://spacy.io/

With a list of words
^^^^^^^^^^^^^^^^^^^

This package provides a spaCy component to add iamsystem algorithm in a spaCy pipeline.

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :start-after: # start_test_spacy_readme_example
    :end-before: # end_test_spacy_readme_example


The *build_params* expects serializable :ref:`api_doc:Matcher build` parameters.
See :ref:`api_doc:IAMsystemBuildSpacy` to configure this component.


With a list of keywords
^^^^^^^^^^^^^^^^^^^^^^^

Since :ref:`api_doc:Keyword` implementation is not JSON serializable, you will have an error passing keywords instance
to the keywords parameter. You have three options:

- Create your own :ref:`api_doc:IKeyword` implementation that is JSON serializable.
- Pass a registered function:

.. code-block:: python

    @spacy.registry.misc("umls_ents.v1")
    def get_termino_umls() -> Iterable[IKeyword]:
        """An imaginary set of umls ents."""
        termino = Terminology()
        ent1 = Entity("Insuffisance Cardiaque", "I50.9")
        ent2 = Entity("Insuffisance Cardiaque Gauche", "I50.1")
        termino.add_keywords(keywords=[ent1, ent2])
        return termino

    "build_params": {
        "keywords": {"@misc": "umls_ents.v1"},
    }

Note that if you call *nlp.to_disk* your keywords will **not** be serialized.

- Pass the :ref:`api_doc:Keyword` as a dictionary with *asdict()* function and pass the module and classname of the Keyword dataclass:

.. code-block:: python

    config={
        "serialized_kw": {
            "module": "iamsystem",
            "class_name": "Keyword",
            "kws": [Keyword(label="insuffisance cardiaque").asdict()],
        },
        "build_params": {"w": 1},
    },

Note that if you call *nlp.to_disk* your keywords will be serialized.
