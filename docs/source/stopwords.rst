
Stopwords
---------

Default Stopwords
^^^^^^^^^^^^^^^^^
It can be useful to remove stopwords, i.e. words that are not relevant to find a match.
For example, the words 'unspecified' or 'NOS' (Not Otherwise Specified) is frequently used in medical terminologies
to denote an entity that has been incompletely characterized.


.. code-block:: python
    :linenos:
    :emphasize-lines: 4

        from iamsystem import Matcher, Term, english_tokenizer
        tokenizer = english_tokenizer()
        matcher = Matcher(tokenizer=tokenizer)
        matcher.add_stopwords(words=["unspecified"])
        term = Term(label="Essential hypertension, unspecified", code="I10.9")
        matcher.add_keywords(keywords=[term])
        text = "Medical history: essential hypertension"
        annots = matcher.annot_text(text=text)
        for annot in annots:
            print(annot)

.. code-block:: pycon

        # essential hypertension	17 39	Essential hypertension, unspecified (I10.9)

NegativeStopwords
^^^^^^^^^^^^^^^^^
Sometimes it's useful to ignore all the words but those of the keywords.
For example, we want to find the label *"calcium blood"* whatever the words between calcium and blood
are as long as the order is kept.
One solution would be to change the :ref:`matcher:Context window (w)`.
Another solution is to use :ref:`api_doc:NegativeStopwords` to ignore all words except
those that the user wants to keep:

.. code-block:: python
    :linenos:
    :emphasize-lines: 7

        from iamsystem import Matcher, Terminology, NegativeStopwords, english_tokenizer, Keyword, NoStopwords
        text = "the level of calcium can be measured in the blood."
        termino = Terminology()
        termino.add_keywords(keywords=[Keyword(label="calcium blood")])
        neg_stopwords = NegativeStopwords()
        tokenizer = english_tokenizer()
        neg_stopwords.add_words(words_to_keep=termino.get_unigrams(tokenizer=tokenizer, stopwords=NoStopwords()))
        matcher = Matcher(tokenizer=tokenizer, stopwords=neg_stopwords)
        matcher.add_keywords(keywords=termino)
        annots = matcher.annot_text(text=text, w=1)
        for annot in annots:
            print(annot)

.. code-block:: pycon

        # calcium blood	13 20;44 49	calcium blood

Note that you can use the :ref:`api_doc:Terminology` class to retrieve all the unigrams of your keywords.
