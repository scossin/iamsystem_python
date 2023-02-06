
Matcher
------
The simplest example is to search a list of keywords in a document.
By default, the :ref:`api_doc:Matcher` performs exact match only.



With a list of words (keywords)
^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :start-after: # start_test_exact_match_keywords
    :end-before: # end_test_exact_match_keywords

The matcher outputs a list of :ref:`annotation:Annotation`.
To add attributes, create a :ref:`api_doc:Keyword` subclass.
The :ref:`api_doc:Term` class shown below associates a unique identifier to each label.

With a list of terms
^^^^^^^^^^^^^^^^^^^^
Often, keywords are derived from a knowledge graph that associates a label with a unique identifier.
The :ref:`api_doc:Term` has a *code* attribute to store an identifier.

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :start-after: # start_test_exact_match_terms
    :end-before: # end_test_exact_match_terms


Context window (w)
^^^^^^^^^^^^^^^^^^
iamsystem algorithm tries to match a sequence of tokens in a document to a sequence of tokens in a keyword/term.
The **w** parameter determines how much discontinuous the sequence of tokens can be.
By default, *w=1* means that the sequence must be continuous.

Let's say we want to detect the keyword *"calcium level"* in a document.
With *w=1*, the matcher wouldn't find the keyword in *"calcium blood level"*
since the sequence of tokens in the document is discontinuous.
One solution would be to add *"blood"* to the :ref:`stopwords:Stopwords` list,
however if *"blood"* is used by another keyword it would be a bad solution.
Another solution is to set *w=2* that lets the algorithm searches 2 words after token *"calcium"*.

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :emphasize-lines: 6
    :start-after: # start_test_window
    :end-before: # end_test_window

The semicolon indicates that the sequence is discontinuous.
The first token "calcium" starts at character 0 and ends at character 6 (7-1).
The second token "level" starts at character 14 and ends at character 18 (19-1).

Unidirectional detection
^^^^^^^^^^^^^^^^^^^^^^^^

Word order is important.
When the sequence of words in the document is not the same as the words sequence of the keyword,
the algorithm fails to detect it. For example:

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :start-after: # start_test_fail_order
    :end-before: # end_test_fail_order

This problem can be solved by changing the order of the tokens in a sentence
which is the responsibility of the tokenizer.
See Tokenizer section on :ref:`tokenizer:Change tokens order`.
