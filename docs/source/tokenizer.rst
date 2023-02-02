Tokenizer
-----------
The iamsystem matcher is highly dependent on how documents and keywords are tokenized and normalized.
The :ref:`api_doc:ITokenizer` is responsible for turning text into tokens.
To do so, the :ref:`api_doc:TokenizerImp` class performs tokenization with two inner functions:

- split the text into (start,end) offsets
- normalize each token

.. _adapter: https://en.wikipedia.org/wiki/Adapter_pattern

The :ref:`api_doc:english_tokenizer` and :ref:`api_doc:french_tokenizer` are concrete implementations.
To use another library to perform the tokenization you can build an `adapter`_
by creating a new implementation of the a :ref:`api_doc:ITokenizer` class.
For example, this package provides a :ref:`spacy:spaCy` custom component that consumes spaCy's tokenizer.

Default split function
^^^^^^^^^^^^^^^^^^^^^^^^
By default, the :ref:`api_doc:Matcher` class calls the :ref:`api_doc:french_tokenizer` that splits a document
by word character *(a letter or digit or underbar [a-zA-Z0-9_])*.

I recommend that you check the generated tokens to verify it matches your needs.
For example:

.. code-block:: python

        from iamsystem import english_tokenizer
        tokenizer = english_tokenizer()
        tokens = tokenizer.tokenize("SARS-CoV+")
        for token in tokens:
            print(token)

.. code-block:: pycon

        # Token(label='SARS', norm_label='sars', start=0, end=4)
        # Token(label='CoV', norm_label='cov', start=5, end=8)

The '+' sign is ignored even though it is important.
The *split* function can be modified as follow :

.. code-block:: python

        from iamsystem import english_tokenizer, split_find_iter_closure
        tokenizer = english_tokenizer()
        tokenizer.split = split_find_iter_closure(pattern=r"(\w+|\+)")
        tokens = tokenizer.tokenize("SARS-CoV+")
        for token in tokens:
            print(token)

.. code-block:: pycon

        # Token(label='SARS', norm_label='sars', start=0, end=4)
        # Token(label='CoV', norm_label='cov', start=5, end=8)
        # Token(label='+', norm_label='+', start=8, end=9)

Change default Tokenizer
^^^^^^^^^^^^^^^^^^^^^^^^
To change Matcher's default tokenizer, pass it to the constructor.

.. code-block:: python
    :linenos:
    :emphasize-lines: 6

        from iamsystem import Matcher, Term, split_find_iter_closure, english_tokenizer
        term1 = Term(label="SARS-CoV+", code="95209-3")
        text = "Pt c/o acute respiratory distress syndrome. RT-PCR sars-cov+"
        tokenizer = english_tokenizer()
        tokenizer.split = split_find_iter_closure(pattern=r"(\w+|\+)")
        matcher = Matcher(tokenizer=tokenizer)
        matcher.add_keywords(keywords=[term1])
        annots = matcher.annot_text(text=text)
        for annot in annots:
            print(annot)

.. code-block:: pycon

        # sars cov +	51 60	SARS-CoV+ (95209-3)


Default normalize function
^^^^^^^^^^^^^^^^^^^^^^^^^^
You can override the *normalize* function of a tokenizer to suit your needs.
The :ref:`api_doc:english_tokenizer` normalizes each token by doing lowercasing.
The :ref:`api_doc:french_tokenizer` performs lowercasing and remove accents.
The only difference between the french_tokenizer and the english_tokenizer is the removal of diacritics
done with the unidecode library that tries to transform the label in ASCII characters.
Using the french_tokenizer for english documents adds very little overhead.


Change tokens order
^^^^^^^^^^^^^^^^^^^
Word order is important for iamsystem.
In the example below, the keyword *"blood calcium level "* is mentioned but the tokens
are discontinuous and not in the right order.
One solution is to order the tokens alphabetically.
By doing this, the tokens of the document and the keyword are in the same order.
Given a wide window, the keyword can be found.

.. code-block:: python
    :linenos:
    :emphasize-lines: 4,8

        from iamsystem import Matcher, english_tokenizer, tokenize_and_order_decorator
        text = "the level of calcium can measured in the blood."
        tokenizer = english_tokenizer()
        tokenizer.tokenize = tokenize_and_order_decorator(tokenizer.tokenize)
        matcher = Matcher(tokenizer=tokenizer)
        matcher.add_labels(labels=["blood calcium level"])
        tokens = matcher.tokenize(text=text)
        annots = matcher.annot_tokens(tokens=tokens, w=len(tokens))
        for annot in annots:
            print(annot)
        # level calcium blood	4 9;13 20;41 46	blood calcium level

Note that the window size is calculated with the number of tokens.
This approach is not suitable if the document is very long or the number of keywords is big.
