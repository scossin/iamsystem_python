Fuzzy Algorithms
----------------

Introduction
^^^^^^^^^^^^

iamsystem algorithm tries to match a sequence of tokens in a document to a sequence of tokens in a keyword.
The default fuzzy algorithm of the :ref:`api_doc:Matcher` class is the exact match algorithm.
In general, in entity linking tasks, exact matching has high precision but low recall since a single
character difference in a token can lead to a miss.

In this package, a *fuzzy algorithm* is an algorithm that is a called for each token in a document
and can return one or more *synonym*, i.e. another string with the same meaning.
The combination of several fuzzy algorithms offers great flexibility in the matching strategy,
it increases recall but can also decrease precision.

This package doesn't contain any implementation of approximate string matching algorithms,
it relies on and wraps external libraries to do so.
Some external libraries are not in the requirement file of this package,
so you will need to install them manually depending on the fuzzy algorithm you wish to add.

Which fuzzy algorithm to choose
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The set of fuzzy algorithms is configured by the user.
Which one to add depends heavily on your documents and the keywords you want to detect.

If your documents contain a lot of **typos**, :ref:`fuzzy:String Distance` algorithms can help.
If your documents contain a lot of **abbreviations**, it's useful to have a sense inventory
and add abbreviations to the :ref:`fuzzy:Abbreviations` class.
If your documents and keywords contain **inflected forms** (singular, plurial, conjugated form),
it is useful to add a normalization method (lemmatization, stemming) with the :ref:`fuzzy:WordNormalizer` class.
If your keywords contain **regular expressions**, the :ref:`fuzzy:FuzzyRegex` class takes care of that.

Remember that for each token in the document, all fuzzy algorithms added to the :ref:`matcher:Matcher`
will be called, so the more algorithms you add, the slower iamsystem.
However, algorithms that are context independant can be cached to avoid calling them multiple times.

Abbreviations
^^^^^^^^^^^^^
The :ref:`api_doc:Abbreviations` class allows you to provide a sense inventory of abbreviations
to the matcher.

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :start-after: # start_test_abbreviations
    :end-before: # end_test_abbreviations

Note the following:

- The first word "Pt" is associated with a single annotation.

Since "hospitalized" comes after the abbreviation and since the matcher removes nested annotation
by default (See :ref:`annotation:Full overlapping`), the ambiguity is removed.

- The last word "PT" has two annotations

The :ref:`api_doc:Abbreviations` is context independent and cannot resolve the ambiguity here.
To solve this problem, the annotations need to be post-processed (rules, language models...) to identify the most likely long form.

In the case where two abbreviations have different string cases
(Pt stands only for patient and PT for physiotherapy), the :ref:`api_doc:Abbreviations` class
can be configured to be case sensitive.
The :ref:`api_doc:Abbreviations` class can be configured with a method that
checks if the document's token is an abbreviation or not:

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :start-after: # start_test_uppercase
    :end-before: # end_test_uppercase

Notice that TokenT is a generic token type, so
if you use a custom tokenizer (i.e. from an external library like spaCy) you can access custom attributes.


String Distance
^^^^^^^^^^^^^^^
.. _spellwise: https://github.com/chinnichaitanya/spellwise
.. _pysimstring: https://github.com/percevalw/pysimstring


This package utilizes the `spellwise`_ and `pysimstring`_ libraries to access string distance algorithms.

Spellwise
"""""""""

In the example below, iamsystem is configured with two spellwise algorithms:
Levenshtein distance which measures the number of edits needed to transform one word into another,
and Soundex which is a phonetic algorithm.

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :start-after: # start_test_spellwise
    :end-before: # end_test_spellwise

The *spellwise* parameter of the *build* function expects an iterable of dictionary.
The key-value pairs of a dictionary are passed to the :ref:`api_doc:SpellWiseWrapper` init function.
Since a string distance algorithm is context independent, the Matcher *build* function placed them in
a :ref:`fuzzy:CacheFuzzyAlgos` to avoid calling them multiple times.
For a list of available Spellwise algorithms, see :ref:`api_doc:ESpellWiseAlgo`.

String distance algorithms are often used to detect typos in a document.
False positives are common since two words could have a short string distance.
To avoid calling a string distance algorithm on common words of a language, you can set
*string_distance_ignored_w* parameter:

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :start-after: # start_test_string_distance_ignored_w
    :end-before: # end_test_string_distance_ignored_w

Since *poils* is one substitution from *poids*, the algorithm returns a false positive.
By adding *poils* to *string_distance_ignored_w*, the string distance algorithm is not called.

I recommend to pass all common words of a language to *string_distance_ignored_w* parameter,
it will make iamsystem faster since all string distance algorithms will be called only for unknown words
and this will reduce false positives.

SimString
"""""""""
.. _simstring: http://chokkan.org/software/simstring/

The `pysimstring`_ library provides an API to the fast `simstring`_ algorithm implemented in C++.
The *simstring* parameter of the Matcher *build* function expects an iterable of dictionary.
The key-value pairs of a dictionary are passed to the :ref:`api_doc:SimStringWrapper` init function.
Since a string distance algorithm is context independent, the build function placed them in a :ref:`fuzzy:CacheFuzzyAlgos`
to avoid calling them multiple times.

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :start-after: # start_test_simstring
    :end-before: # end_test_simstring

Using the cosine similarity and a threshold of 0.7,
the tokens *respiratori* matched to *respiratory* and *disstress* matched to *distress*.

CacheFuzzyAlgos
^^^^^^^^^^^^^^^

Fuzzy algorithms that are not context depend can be cached to avoid calling them multiple times.
The :ref:`api_doc:CacheFuzzyAlgos` stores fuzzy algorithms, calls them once and then stores
their results.

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :emphasize-lines:  17, 20, 22
    :start-after: # start_test_cache_fuzzy_algos
    :end-before: # end_test_cache_fuzzy_algos

Note that although we could have put the Abbreviations instance in the cache, it's not necessary
to do so since this algorithm is as fast as the cache.
If you use the Matcher *build* function, string distance algorithms are automatically cached.


FuzzyRegex
^^^^^^^^^^^

Regular expressions are very useful and can be used with iamsystem.
For example, if you want to detect blood test results in electronic health records,
such as calcium levels in blood, you can have a regular expression in your
keyword: *"calcium (^\d*[.,]?\d*$) mmol/L"*.
The *fuzzy_regex* parameter expects an iterable of dictionary. Key-value pairs of the dictionary correspond to
:ref:`api_doc:FuzzyRegex` init function parameters.

The regular expression *(^\d*[.,]?\d*$)* is placed in the FuzzyRegex instance,
with a patter name (ex: *numval*), and the pattern name is placed in the keyword
(*"calcium numval mmol/L"*).

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :start-after: # start_test_fuzzy_regex
    :end-before: # end_test_fuzzy_regex

Note that the :ref:`tokenizer:Default split function` must be modified to detect decimal values.
Also note that the label of the keyword *"calcium numval mmol/L"* (line 7) contains the same pattern name *numval*.
When the fuzzy algorithm receives the token value *2.1*, it finds that it matches its regular expression
and returns the pattern name *numval*.

In the example above, stopwords have been added, otherwise the algorithm wouldn't have found
the keyword with a context window of 1.
It's often the case that intermediate words are not known in avance, so this method wouldn't work.
Another way to do exactly the same annotation is to use the :ref:`stopwords:NegativeStopwords` class
which ignores all unigrams that are not in the keywords:

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :start-after: # start_test_fuzzy_regex_negative_stopwords
    :end-before: # end_test_fuzzy_regex_negative_stopwords

WordNormalizer
^^^^^^^^^^^^^^

Word normalization is a common pre-processing step in NLP.
The idea is to group words that have the same normalized form;
for example *"eating"*, *"eats"*... have the same canonical form *"eat"*.

The :ref:`api_doc:WordNormalizer` offers the possibility to add a normalization function.
A token in a document will match a token in a keyword if they have the same normalized form.

.. _nltk: https://www.nltk.org/

In the example below, `nltk`_ is used to access a French stemmer.
The stemming function is given to the :ref:`api_doc:WordNormalizer` class:

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :start-after: # start_test_word_normalization
    :end-before: # end_test_word_normalization


Abstract Base classes
^^^^^^^^^^^^^^^^^^^^^
You might be interested in the fuzzy algorithms abstract base classes
if you want to create a new custom fuzzy algorithm.
The hierarchy is the following:

- :ref:`api_doc:FuzzyAlgo`

Implements this class to create a context dependent algorithm.
For each token for which a synonym is expected, the context words
and the algorithm's states are available.

- :ref:`api_doc:ContextFreeAlgo`

Implements this class to create a context-free algorithm that depends only on the
current token. The class has access to the generic token for which a synonym is expected.
Examples of such algorithms: :ref:`fuzzy:FuzzyRegex`, :ref:`fuzzy:Abbreviations`.

- :ref:`api_doc:NormLabelAlgo`

Implements this class to create a context-free algorithm that depends only on the normalized
form of the token. The class has access to the normalized label of the token for which a synonym is expected.
These algorithms can be cached with :ref:`fuzzy:CacheFuzzyAlgos`.
Examples of such algorithms: :ref:`fuzzy:String Distance`, :ref:`fuzzy:WordNormalizer`.
