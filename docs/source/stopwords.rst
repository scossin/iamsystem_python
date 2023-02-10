
Stopwords
---------

Default Stopwords
^^^^^^^^^^^^^^^^^
It can be useful to remove stopwords, i.e. words that are not relevant to find a match.
For example, the words 'unspecified' or 'NOS' (Not Otherwise Specified) is frequently used in medical terminologies
to denote an entity that has been incompletely characterized.

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :emphasize-lines: 9
    :start-after: # start_test_add_stopword
    :end-before: # end_test_add_stopword


NegativeStopwords
^^^^^^^^^^^^^^^^^
Sometimes it's useful to ignore all the words but those of the keywords.
For example, we want to find the label *"calcium blood"* whatever the words between calcium and blood
as long as the order is kept.
One solution would be to change the :ref:`matcher:Context window (w)`.
Another solution is to use :ref:`api_doc:NegativeStopwords` to ignore all words except
those that the user wants to keep:

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :start-after: # start_test_negative_stopword
    :end-before: # end_test_negative_stopword
