
Matcher
------
The simplest example is to search a list of keywords in a document.
By default, the :ref:`api_doc:Matcher` performs exact match only.



With a list of words (keywords)
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

        from iamsystem import Matcher
        labels = ["acute respiratory distress syndrome", "diarrrhea"]
        text = "Pt c/o Acute Respiratory Distress Syndrome and diarrrhea"
        matcher = Matcher()
        matcher.add_labels(labels=labels)
        annots = matcher.annot_text(text=text)
        for annot in annots:
            print(annot)


.. code-block:: pycon

        # Acute Respiratory Distress Syndrome	7 42	acute respiratory distress syndrome
        # diarrrhea	47 56	diarrrhea

Have a look at :ref:`annotation:Annotation's format`.
To add attributes to words, create a :ref:`api_doc:Keyword` subclass.
The :ref:`api_doc:Term` class shown below associates a unique identifier with each word.

With a list of terms
^^^^^^^^^^^^^^^^^^^^
Often, keywords are derived from a knowledge graph that associates a label with a unique identifier.
The :ref:`api_doc:Term` has a *code* attribute to store an identifier.

.. code-block:: python

        from iamsystem import Matcher, Term
        term1 = Term(label="acute respiratory distress syndrome", code="J80")
        term2 = Term(label="diarrrhea", code="R19.7")
        text = "Pt c/o acute respiratory distress syndrome and diarrrhea"
        matcher = Matcher()
        matcher.add_keywords(keywords=[term1, term2])
        annots = matcher.annot_text(text=text)
        for annot in annots:
            print(annot)

.. code-block:: pycon

        # acute respiratory distress syndrome	7 42	acute respiratory distress syndrome (J80)
        # diarrrhea (R19.7)	47	56

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

.. code-block:: python
    :linenos:
    :emphasize-lines: 5

        from iamsystem import Matcher
        labels = ["calcium level"]
        matcher = Matcher()
        matcher.add_labels(labels=labels)
        annots = matcher.annot_text(text="calcium blood level", w=2)
        for annot in annots:
            print(annot)

.. code-block:: pycon

        # calcium level	0 7;14 19	calcium level

The semicolon indicates that the sequence is discontinuous.
The first token "calcium" starts at character 0 and ends at character 6 (7-1).
The second token "level" starts at character 14 and ends at character 18 (19-1).

Unidirectional detection
^^^^^^^^^^^^^^^^^^^^^^^^

Word order is important.
When the sequence of words in the document is not the same as the words sequence of the keyword,
the algorithm fails to detect it. For example:

.. code-block:: python

        from iamsystem import Matcher
        labels = ["calcium level"]
        matcher = Matcher()
        matcher.add_labels(labels=labels)
        annots = matcher.annot_text(text="level calcium", w=1)
        print(len(annots)) # 0

This problem can be solved by changing the order of the tokens in a sentence
which is the responsibility of the tokenizer.
See Tokenizer section on :ref:`tokenizer:Change tokens order`.
