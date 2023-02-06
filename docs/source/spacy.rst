
spaCy
-----

.. _spaCy: https://spacy.io/

This package provides a stateful spaCy component to add iamsystem algorithm in a spaCy pipeline.
Since a :ref:`api_doc:Matcher` configuration is not JSON serializable, matcher's parameters
are passed in registered functions:


.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :start-after: # start_test_component
    :end-before: # end_test_component

See :ref:`api_doc:IAMsystemSpacy` to configure this component.
