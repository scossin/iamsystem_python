API Documentation
================

Documentation of classes and methods.

Matcher
-------

.. autoclass:: iamsystem.Matcher
   :members:
   :undoc-members: build
   :show-inheritance:

   .. automethod:: __init__

Matcher build
^^^^^^^^^^^^^

.. autoclass:: iamsystem.Matcher
   :members: build
   :noindex:

EMatchingStrategy
^^^^^^^^^^^^^^^^^
.. autoclass:: iamsystem.EMatchingStrategy
   :members:
   :noindex:

Span
----
.. autoclass:: iamsystem.matcher.annotation.Span
   :inherited-members:
   :members:
   :undoc-members:
   :show-inheritance:

Annotation
----------
.. autoclass:: iamsystem.Annotation
   :inherited-members:
   :members:
   :undoc-members:
   :show-inheritance:

rm_nested_annots
^^^^^^^^^^^^^^^^
.. autofunction:: iamsystem.rm_nested_annots

replace_annots
^^^^^^^^^^^^^^
.. autofunction:: iamsystem.replace_annots

Keyword and subclasses
----------------------

IKeyword
^^^^^^^^
.. autoclass:: iamsystem.IKeyword
   :members:
   :undoc-members:
   :show-inheritance:

IEntity
^^^^^^^^
.. autoclass:: iamsystem.IEntity
   :members:
   :undoc-members:
   :show-inheritance:


Keyword
^^^^^^^
.. autoclass:: iamsystem.Keyword
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Entity
^^^^^^
.. autoclass:: iamsystem.Entity
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Terminology
^^^^^^^^^^^
.. autoclass:: iamsystem.Terminology
   :members:
   :show-inheritance:

   .. automethod:: __init__

Tokenization
-------

IOffsets
^^^^^^^^
.. autoclass:: iamsystem.IOffsets
   :show-inheritance:
   :members:

Offsets
^^^^^^^
.. autoclass:: iamsystem.Offsets
   :show-inheritance:

   .. automethod:: __init__

IToken
^^^^^^
.. autoclass:: iamsystem.IToken
   :members:
   :show-inheritance:

Token
^^^^^
.. autoclass:: iamsystem.Token
   :show-inheritance:

   .. automethod:: __init__

ITokenizer
^^^^^^^^^^
.. autoclass:: iamsystem.ITokenizer
   :members: tokenize
   :undoc-members:
   :show-inheritance:

TokenizerImp
^^^^^^^^^^^^
.. autoclass:: iamsystem.TokenizerImp
   :members:
   :undoc-members: tokenize
   :show-inheritance:

   .. automethod:: __init__


english_tokenizer
^^^^^^^^^^^^^^^^^
.. autofunction:: iamsystem.english_tokenizer

french_tokenizer
^^^^^^^^^^^^^^^
.. autofunction:: iamsystem.french_tokenizer

Build a custom split function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: iamsystem.split_find_iter_closure

Order tokens
^^^^^^^^^^^^
.. autofunction:: iamsystem.tokenize_and_order_decorator


Stopwords classes
-----------------

IStopwords
^^^^^^^^^^
.. autoclass:: iamsystem.IStopwords
   :members:
   :undoc-members:
   :show-inheritance:

Stopwords
^^^^^^^^^^
.. autoclass:: iamsystem.Stopwords
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

NegativeStopwords
^^^^^^^^^^^^
.. autoclass:: iamsystem.NegativeStopwords
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

NoStopwords
^^^^^^^^^^
.. autoclass:: iamsystem.NoStopwords
   :members:
   :undoc-members:
   :show-inheritance:


Fuzzy algorithms
----------------

Abstract Base classes
^^^^^^^^^^^^^^^^^^^^^

FuzzyAlgo
"""""""""
.. autoclass:: iamsystem.FuzzyAlgo
   :members:
   :undoc-members:
   :show-inheritance:

ContextFreeAlgo
"""""""""""""""
.. autoclass:: iamsystem.ContextFreeAlgo
   :members:
   :undoc-members:
   :show-inheritance:

NormLabelAlgo
"""""""""""""
.. autoclass:: iamsystem.NormLabelAlgo
   :members:
   :undoc-members:
   :show-inheritance:

CacheFuzzyAlgos
^^^^^^^^^^^^^^^
.. autoclass:: iamsystem.CacheFuzzyAlgos
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Abbreviations
^^^^^^^^^^^^^

.. autoclass:: iamsystem.Abbreviations
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__


FuzzyRegex
^^^^^^^^^^
.. autoclass:: iamsystem.FuzzyRegex
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

WordNormalizer
^^^^^^^^^^^^^^
.. autoclass:: iamsystem.WordNormalizer
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

SpellWise
^^^^^^^^^

SpellWiseWrapper
""""""""""""""""
.. autoclass:: iamsystem.SpellWiseWrapper
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__


ESpellWiseAlgo
""""""""""""""
.. autoclass:: iamsystem.ESpellWiseAlgo
   :members:
   :undoc-members:
   :show-inheritance:

SimString
^^^^^^^^^

SimStringWrapper
""""""""""""""""
.. autoclass:: iamsystem.SimStringWrapper
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

ESimStringMeasure
"""""""""""""""""
.. autoclass:: iamsystem.ESimStringMeasure
   :members:
   :undoc-members:
   :show-inheritance:

Brat
----

Formatter
^^^^^^^^^

EBratFormatters
"""""""""""""""
.. autoclass:: iamsystem.EBratFormatters
   :members:
   :undoc-members:
   :show-inheritance:

ContSeqFormatter
""""""""""""""
.. autoclass:: iamsystem.ContSeqFormatter
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

ContSeqStopFormatter
""""""""""""""
.. autoclass:: iamsystem.ContSeqStopFormatter
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

TokenFormatter
""""""""""""""
.. autoclass:: iamsystem.TokenFormatter
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

SpanFormatter
""""""""""""""
.. autoclass:: iamsystem.SpanFormatter
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

BratDocument
^^^^^^^^^^^^
.. autoclass:: iamsystem.BratDocument
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

BratEntity
^^^^^^^^^^^^^
.. autoclass:: iamsystem.BratEntity
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

BratNote
^^^^^^^^^^^^^
.. autoclass:: iamsystem.BratNote
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

BratWriter
^^^^^^^^^^
.. autoclass:: iamsystem.BratWriter
   :members:
   :show-inheritance:

   .. automethod:: __init__

spaCy
-----

IAMsystemSpacy
^^^^^^^^^^^^^^
.. autoclass:: iamsystem.spacy.IAMsystemSpacy
   :members:
   :show-inheritance:

   .. automethod:: __init__

IAMsystemBuildSpacy
^^^^^^^^^^^^^^^^^^^
.. autoclass:: iamsystem.spacy.IAMsystemBuildSpacy
   :members:
   :show-inheritance:

   .. automethod:: __init__

TokenSpacyAdapter
^^^^^^^^^^^^^^^^^
.. autoclass:: iamsystem.spacy.TokenSpacyAdapter
   :members:
   :show-inheritance:

   .. automethod:: __init__

IsStopSpacy
^^^^^^^^^^^
.. autoclass:: iamsystem.spacy.IsStopSpacy
   :members:
   :show-inheritance:

   .. automethod:: __init__

SpacyTokenizer
^^^^^^^^^^^^^^
.. autoclass:: iamsystem.spacy.SpacyTokenizer
   :members:
   :show-inheritance:

   .. automethod:: __init__
