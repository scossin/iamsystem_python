Tokenizer
-----------
The iamsystem matcher is highly dependent on how documents and keywords are tokenized and normalized.
The :ref:`api_doc:ITokenizer` is responsible for turning text into tokens.
To do so, the :ref:`api_doc:TokenizerImp` class performs alphanumeric tokenization with two inner functions:

- split the text into (start,end) offsets
- normalize each token

The :ref:`api_doc:english_tokenizer` and :ref:`api_doc:french_tokenizer` are concrete implementations.

.. _adapter: https://en.wikipedia.org/wiki/Adapter_pattern

Other libraries offer more elaborate tokenizers, I recommend you use them.
To use the tokenizer of another library you can build an `adapter`_
by creating a new implementation of the a :ref:`api_doc:ITokenizer` interface.
For example, this package provides a :ref:`spacy:spaCy` custom component that consumes spaCy's tokenizer.

Default split function
^^^^^^^^^^^^^^^^^^^^^^^^
By default, the :ref:`api_doc:Matcher` class calls the :ref:`api_doc:french_tokenizer` that splits a document
by word character *(a letter or digit or underbar [a-zA-Z0-9_])*.

I recommend that you check the generated tokens to verify it matches your needs.
For example:

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :start-after: # start_test_tokenizer
    :end-before: # end_test_tokenizer

The '+' sign is ignored even though it is important.
The *split* function can be modified as follow :

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :emphasize-lines: 5
    :start-after: # start_test_custom_tokenizer
    :end-before: # end_test_custom_tokenizer

Change default Tokenizer
^^^^^^^^^^^^^^^^^^^^^^^^
To change Matcher's default tokenizer, pass it to the constructor.

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :emphasize-lines: 10
    :start-after: # start_test_matcher_with_custom_tokenizer
    :end-before: # end_test_matcher_with_custom_tokenizer


Default normalize function
^^^^^^^^^^^^^^^^^^^^^^^^^^
You can override the *normalize* function of a tokenizer to suit your needs.
The :ref:`api_doc:english_tokenizer` normalizes each token by doing lowercasing.
The :ref:`api_doc:french_tokenizer` performs lowercasing and remove accents.
The only difference between the french_tokenizer and the english_tokenizer is the removal of diacritics
done with the `anyascii` library that tries to transform the label in ASCII characters.
Using the french_tokenizer for english documents adds very little overhead.


Change tokens order
^^^^^^^^^^^^^^^^^^^
Word order is important for iamsystem.
In the example below, the keyword *"blood calcium level "* is mentioned but the tokens
are discontinuous and not in the right order.
One solution is to order the tokens alphabetically.
By doing this, the tokens of the document and the keyword are in the same order.
Given a wide window, the keyword can be found.

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :emphasize-lines: 9
    :start-after: # start_test_unordered_words_seq
    :end-before: # end_test_unordered_words_seq

*order_tokens* parameter changes iamsystem's matching strategy but it doesn't change the document's tokens order.
This approach is not suitable if the document is very long or the number of keywords is large.
