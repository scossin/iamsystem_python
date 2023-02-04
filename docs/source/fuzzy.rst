Fuzzy Algorithms
----------------

Introduction
^^^^^^^^^^^^

iamsystem algorithm tries to match a sequence of tokens in a document to a sequence of tokens in a keyword.
The default fuzzy algorithm of the :ref:`matcher:Matcher` class is the exact match algorithm.
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
See :ref:`fuzzy:CacheFuzzyAlgos`.

Abbreviations
^^^^^^^^^^^^^
The :ref:`api_doc:Abbreviations` class allows you to provide a sense inventory of abbreviations
to the matcher.

.. code-block:: python
    :linenos:
    :emphasize-lines: 3,4,5

        from iamsystem import Matcher, Abbreviations, english_tokenizer, Term
        tokenizer = english_tokenizer()
        abbs = Abbreviations(name="abbs")
        abbs.add(short_form="Pt", long_form="patient", tokenizer=tokenizer)
        abbs.add(short_form="PT", long_form="physiotherapy", tokenizer=tokenizer)
        abbs.add(short_form="ARD", long_form="Acute Respiratory Distress", tokenizer=tokenizer)
        matcher = Matcher(tokenizer=tokenizer)
        term1 = Term(label="acute respiratory distress", code="J80")
        term2 = Term(label="patient", code="D007290")
        term3 = Term(label="patient hospitalized", code="D007297")
        term4 = Term(label="physiotherapy", code="D007297")
        matcher.add_keywords(keywords=[term1, term2, term3, term4])
        matcher.add_fuzzy_algo(fuzzy_algo=abbs)
        annots = matcher.annot_text(text="Pt hospitalized with ARD. Treament: PT")
        for annot in annots:
            print(annot.to_string(debug=True))


.. code-block:: pycon

        # Pt hospitalized	0 15	patient hospitalized (D007297)	pt(abbs);hospitalized(exact)
        # ARD	21 24	acute respiratory distress (J80)	ard(abbs)
        # PT	36 38	patient (D007290)	pt(abbs)
        # PT	36 38	physiotherapy (D007297)	pt(abbs)

Note the following:

- The first word "Pt" is associated with a single annotation.

Since "hospitalized" comes after the abbreviation and since the matcher removes nested keywords
by default (See :ref:`annotation:Full overlapping`), the ambiguity is removed.

- The last word "PT" has two annotations

The :ref:`api_doc:Abbreviations` is context independent and cannot resolve the ambiguity here.
To solve this problem, the annotations could be post-processed to identify the correct long form.
A second solution would be to create a custom :ref:`api_doc:FuzzyAlgo` instance which
would be context dependent and which would return the most likely long.


In the case where two abbreviations have different string cases
(Pt stands only for patient and PT for physiotherapy), the :ref:`api_doc:Abbreviations` class
can be configured to be case sensitive.


The :ref:`api_doc:Abbreviations` class can be configured with a method that
checks if the document's token is an abbreviation or not:

.. code-block:: python
    :linenos:
    :emphasize-lines: 12,15


        from iamsystem import Matcher, Abbreviations, english_tokenizer, Term, TokenT

        def upper_case_only(token: TokenT) -> bool:
            """ Return True if all token's characters are uppercase."""
            return token.label.isupper()

        def first_letter_capitalized(token: TokenT) -> bool:
            """ Return True if the first letter is uppercase."""
            return token.label[0].isupper() and not token.label.isupper()

        tokenizer = english_tokenizer()
        abbs_upper = Abbreviations(name="upper case abbs", token_is_an_abbreviation=upper_case_only)
        abbs_upper.add(short_form="PT", long_form="physiotherapy", tokenizer=tokenizer)
        abbs_upper.add(short_form="ARD", long_form="Acute Respiratory Distress", tokenizer=tokenizer)
        abbs_capitalized = Abbreviations(name="capitalized abbs", token_is_an_abbreviation=first_letter_capitalized)
        abbs_capitalized.add(short_form="Pt", long_form="patient", tokenizer=tokenizer)
        matcher = Matcher(tokenizer=tokenizer)
        term1 = Term(label="acute respiratory distress", code="J80")
        term2 = Term(label="patient", code="D007290")
        term3 = Term(label="patient hospitalized", code="D007297")
        term4 = Term(label="physiotherapy", code="D007297")
        matcher.add_keywords(keywords=[term1, term2, term3, term4])
        matcher.add_fuzzy_algo(fuzzy_algo=abbs_upper)
        matcher.add_fuzzy_algo(fuzzy_algo=abbs_capitalized)
        annots = matcher.annot_text(text="Pt hospitalized with ARD. Treament: PT")
        for annot in annots:
            print(annot.to_string(debug=True))

.. code-block:: pycon

        # Pt hospitalized	0 15	patient hospitalized (D007297)	pt(capitalized abbs);hospitalized(exact)
        # ARD	21 24	acute respiratory distress (J80)	ard(upper case abbs)
        # PT	36 38	physiotherapy (D007297)	pt(upper case abbs)

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

.. code-block:: python
    :linenos:
    :emphasize-lines: 6,7,8,9

        from iamsystem import ESpellWiseAlgo
        from iamsystem import Matcher
        from iamsystem import SpellWiseWrapper
        from iamsystem import Term

        levenshtein = SpellWiseWrapper(
            ESpellWiseAlgo.LEVENSHTEIN, max_distance=1, min_nb_char=5
        )
        soundex = SpellWiseWrapper(ESpellWiseAlgo.SOUNDEX, max_distance=1)
        term1 = Term(label="acute respiratory distress", code="J80")
        matcher = Matcher()
        matcher.add_keywords(keywords=[term1])
        for algo in [levenshtein, soundex]:
            algo.add_words(words=matcher.get_keywords_unigrams())
            matcher.add_fuzzy_algo(algo)
        annots = matcher.annot_text(text="acute resiratory distresssss")
        for annot in annots:
            print(annot.to_string(debug=True))

.. code-block:: pycon

           # acute resiratory distresssss	0 28	acute respiratory distress (J80)	acute(exact,LEVENSHTEIN,SOUNDEX);resiratory(LEVENSHTEIN);distresssss(SOUNDEX)

The *get_unigrams* function retrieve all the single words (excluding stopwords) form the keywords.
Spellwise algorithms need to get the keywords'words to return a suggestion.
For a list of available Spellwise algorithms, see :ref:`api_doc:ESpellWiseAlgo`.
See also :ref:`api_doc:SpellWiseWrapper` for configuration.

When the number of keywords is large, these algorithms can be slow.
Since their output doesn't depend on the context,
I recommend using the :ref:`fuzzy:CacheFuzzyAlgos` class to store them.

SimString
"""""""""
.. _simstring: http://chokkan.org/software/simstring/

The `pysimstring`_ library provides an API to the fast `simstring`_ algorithm implemented in C++.

In the example below, all the unigrams of the keywords are indexed by simstring.
Then, for each token in the document, simstring is called to return the closest matches.

.. code-block:: python

        from iamsystem.fuzzy.simstring import (
            SimStringWrapper,
            ESimStringMeasure,
        )
        from iamsystem import Term, Matcher

        term1 = Term(label="acute respiratory distress", code="J80")
        matcher = Matcher()
        matcher.add_keywords(keywords=[term1])
        fuzzy_ss = SimStringWrapper(
            words=matcher.get_keywords_unigrams(),
            measure=ESimStringMeasure.COSINE,
            threshold=0.7,
        )
        matcher.add_fuzzy_algo(fuzzy_algo=fuzzy_ss)
        annots = matcher.annot_text(text="acute respiratori disstress")
        for annot in annots:
            print(annot)
        # acute respiratori disstress	0 27	acute respiratory distress (J80)

Using the cosine similarity and a threshold of 0.7,
the tokens *respiratori* matched to *respiratory* and *disstress* matched to *distress*.

CacheFuzzyAlgos
^^^^^^^^^^^^^^^

Fuzzy algorithms that are not context depend can be cached to avoid calling them multiple times.
The :ref:`api_doc:CacheFuzzyAlgos` stores fuzzy algorithms, calls them once and then stores
their results.

.. code-block:: python
    :linenos:
    :emphasize-lines: 17, 20, 22

        from iamsystem import Abbreviations
        from iamsystem import CacheFuzzyAlgos
        from iamsystem import ESpellWiseAlgo
        from iamsystem import Matcher
        from iamsystem import SpellWiseWrapper
        from iamsystem import Term

        matcher = Matcher()
        abbs = Abbreviations(name="abbs")
        abbs.add(short_form="a", long_form="acute", tokenizer=matcher)
        levenshtein = SpellWiseWrapper(
            ESpellWiseAlgo.LEVENSHTEIN, max_distance=1, min_nb_char=5
        )
        soundex = SpellWiseWrapper(ESpellWiseAlgo.SOUNDEX, max_distance=1)
        term1 = Term(label="acute respiratory distress", code="J80")
        matcher.add_keywords(keywords=[term1])
        cache = CacheFuzzyAlgos()
        for algo in [levenshtein, soundex]:
            algo.add_words(words=matcher.get_keywords_unigrams())
            cache.add_algo(algo=algo)
        # cache.add_algo(algo=abbs)  ## no need to be this one in cache
        matcher.add_fuzzy_algo(fuzzy_algo=cache)
        matcher.add_fuzzy_algo(fuzzy_algo=abbs)
        annots = matcher.annot_text(text="a resiratory distresssss")
        for annot in annots:
            print(annot.to_string(debug=True))

.. code-block:: pycon

        # acute resiratory distresssss	0 28	acute respiratory distress (J80)	acute(exact,LEVENSHTEIN,SOUNDEX);resiratory(LEVENSHTEIN);distresssss(SOUNDEX)

Note that although we could have put the Abbreviations instance in the cache, it's not necessary
to do so since this algorithm is a fast as the cache because it stores the abbreviations in a dictionary.


FuzzyRegex
^^^^^^^^^^^

Regular expressions are very useful and can be used with iamsystem.
For example, if you want to detect blood test results in electronic health records,
such as calcium levels in blood, you can have a regular expression in your
keyword: *"calcium (^\d*[.,]?\d*$) mmol/L"*.
The class :ref:`api_doc:FuzzyRegex` allows you to do this.
The regular expression *(^\d*[.,]?\d*$)* is placed in the FuzzyRegex instance,
with a patter name (ex: *numval*), and the pattern name is placed in your keyword
(*"calcium numval mmol/L"*).

.. code-block:: python
    :linenos:
    :emphasize-lines: 2,3,7

        from iamsystem import Matcher, FuzzyRegex, split_find_iter_closure, english_tokenizer
        fuzzy = FuzzyRegex(algo_name="regex_num", pattern=r"^\d*[.,]?\d*$", pattern_name="numval")
        split = split_find_iter_closure(pattern=r"(\w|\.|,)+")
        tokenizer = english_tokenizer()
        tokenizer.split = split
        detector = Matcher(tokenizer=tokenizer)
        detector.add_labels(labels=["calcium numval mmol/L"])
        detector.add_stopwords(words=["level", "is", "normal"])
        detector.add_fuzzy_algo(fuzzy_algo=fuzzy)
        annots = detector.annot_text(text="the blood calcium level is normal: 2.1 mmol/L", w=1)
        for annot in annots:
            print(annot)
        # calcium 2.1 mmol L	10 17;35 45	calcium numval mmol/L
        self.assertEqual(1, len(annots))

.. code-block:: pycon

        # calcium 2.1 mmol L	10 17;35 45	calcium numval mmol/L

Note that the :ref:`tokenizer:Default split function` must be modified to detect decimal values.
Also note that the label of the keyword *"calcium numval mmol/L"* (line 7) contains the same pattern name *numval*.
When the fuzzy algorithm receives the token value *2.1*, it finds that it matches its regular expression
and returns the pattern name *numval*.

In the example above, stopwords have been added, otherwise the algorithm wouldn't have found
the keyword with a context window of 1.
It's often the case that intermediate words are not known in avance, so this method wouldn't work.
Another way to do exactly the same annotation is to use the :ref:`stopwords:NegativeStopwords` class
which ignores all unigrams that are not in the keywords:

.. code-block:: python

        from iamsystem import FuzzyRegex
        from iamsystem import Keyword
        from iamsystem import Matcher
        from iamsystem import NegativeStopwords
        from iamsystem import NoStopwords
        from iamsystem import Terminology
        from iamsystem import english_tokenizer
        from iamsystem import split_find_iter_closure

        fuzzy = FuzzyRegex(
            algo_name="regex_num",
            pattern=r"^\d*[.,]?\d*$",
            pattern_name="numval",
        )
        split = split_find_iter_closure(pattern=r"(\w|\.|,)+")
        tokenizer = english_tokenizer()
        tokenizer.split = split
        keyword = Keyword(label="calcium numval mmol/L")
        termino = Terminology()
        termino.add_keywords(keywords=[keyword])
        stopwords = NegativeStopwords(
            words_to_keep=termino.get_unigrams(
                tokenizer=tokenizer, stopwords=NoStopwords()
            )
        )
        stopwords.add_fun_is_a_word_to_keep(fuzzy.token_matches_pattern)
        matcher = Matcher(tokenizer=tokenizer, stopwords=stopwords)
        matcher.add_keywords(keywords=termino)
        matcher.add_fuzzy_algo(fuzzy_algo=fuzzy)
        annots = matcher.annot_text(
            text="the blood calcium level is normal: 2.1 mmol/L", w=1
        )
        for annot in annots:
            print(annot)

.. code-block:: pycon

        # calcium 2.1 mmol L	10 17;35 45	calcium numval mmol/L

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

.. code-block:: python
    :linenos:
    :emphasize-lines: 12,13,14

        from nltk.stem.snowball import FrenchStemmer

        from iamsystem import Matcher
        from iamsystem import Term
        from iamsystem import WordNormalizer
        from iamsystem import french_tokenizer

        tokenizer = french_tokenizer()
        matcher = Matcher(tokenizer=tokenizer)
        matcher.add_stopwords(words=["de", "la"])
        stemmer = FrenchStemmer()
        fuzzy_stemmer = WordNormalizer(
            name="french_stemmer", norm_fun=stemmer.stem
        )
        term1 = Term(label="cancer de la prostate", code="C61")
        matcher.add_keywords(keywords=[term1])
        fuzzy_stemmer.add_words(words=matcher.get_keywords_unigrams())
        matcher.add_fuzzy_algo(fuzzy_stemmer)
        annots = matcher.annot_text(text="cancer prostatique")
        for annot in annots:
            print(annot)

.. code-block:: pycon

         # cancer prostatique	0 18	cancer de la prostate (C72)


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
