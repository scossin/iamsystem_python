
Matcher
-------
The simplest example is to search a list of words in a document.
To do so, :ref:`api_doc:Matcher` is the main public API of this package.
I recommend to use the :ref:`api_doc:Matcher build` method to simplify its construction:

With a list of words (keywords)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :start-after: # start_test_exact_match_keywords
    :end-before: # end_test_exact_match_keywords

The matcher outputs a list of :ref:`annotation:Annotation`.
By default, it performs exact match only.
A limitation of passing words to the matcher is that no attributes are associated.

With a list of entities
^^^^^^^^^^^^^^^^^^^^^^^
Often, keywords are derived from a knowledge graph that associates a label with a unique identifier.
The :ref:`api_doc:Entity` class has a *kb_id* attribute to store an identifier.

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :start-after: # start_test_exact_match_ents
    :end-before: # end_test_exact_match_ents

With a custom of keyword subclass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you need to add other attributes to a keyword, you can create your own :ref:`api_doc:IKeyword` implementation.

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :start-after: # start_test_exact_match_custom_keyword
    :end-before: # end_test_exact_match_custom_keyword

Note you can add different keyword types.

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
