Annotation
----------
A :ref:`api_doc:Matcher` outputs instances of :ref:`api_doc:Annotation`.
iamsystem algorithm tries to match a sequence of tokens in a document to a sequence of tokens in a keyword/term.
An :ref:`api_doc:Annotation` instance stores the sequence of tokens of a document matched to one or multiple keywords.
Also, the name of the fuzzy algorithm that matched a token in a document is stored for machine learning
or debugging purposes.

Annotation's format
^^^^^^^^^^^^^^^^^^^

The *to_string* method returns a string representation containing three tabulated fields:

- A concatenation of tokens label as they appear in the document.
- The start-end offsets in the Brat format (start and end are separated by a space,
  a semicolon is used to separate offsets of discontinuous tokens).
- A string representation of detected Keywords.

For example:

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :start-after: # start_test_annotation_format
    :end-before: # end_test_annotation_format

Passing the document to the *to_string* function adds the document substring
that begins at the first token start offset and ends at the last token end offset.
If debug equals True, it adds each token's normalized label and the name(s) of the fuzzy algorithm(s) that detected
it.

The method *to_dict* returns a dictionary representation of an annotation.

Multiple keywords per annotation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An :ref:`api_doc:Annotation` has multiple keywords if and only if these keywords have the same
tokenization output, i.e. the same sequence of tokens.
This happens if two terms have the same label but
also if the normalization process removes punctuation or if stopwords are ignored.
In the example below, only one annotation is produced and it has 3 keywords:

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :start-after: # start_test_annotation_multiple_keywords
    :end-before: # end_test_annotation_multiple_keywords

Overlapping and ancestors
^^^^^^^^^^^^^^^^^^^^^^^^^
In a knowledge base, labels can share a same prefix.
For example keywords *"lung"* and *"lung cancer"* have the same prefix *"lung"*.
*"lung"* is called an **ancestor** of *"lung cancer"* because iamsystem algorithm constructs a
graph representation of keywords. Note that ancestor is not defined by a binary
relation (e.g. subsomption) that could exist in the knowledge base but only when
two keywords have a common prefix.

Full overlapping
""""""""""""""""

Definition: let a1 and a2 two annotations. If a1.start <= a2.start and a1.end > a2.end
then we say that a1 **fully overlaps** a2.
Furthermore, if a1 has all the tokens of a2 then a2 is called a **nested annotation**.
By default, the matcher removes nested annotation.
For example:

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :start-after: # start_test_annotation_overlapping_ancestors
    :end-before: # end_test_annotation_overlapping_ancestors


Another example where the first annotation fully overlaps the second but the latter is not
a nested annotation:

.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :start-after: # start_test_annotation_overlapping_not_ancestors
    :end-before: # end_test_annotation_overlapping_not_ancestors

The first annotation, starting at offset 0 and ending at offset 23, fully overlaps the second.
However, it doesn't have all the tokens of the second annotation,
thus the second annotation is not a nested annotation and it's not removed.
The brat format shows that *North America* keyword is a discontinuous sequence of tokens in the document.

Under the hood, the :ref:`api_doc:rm_nested_annots` function is called to remove nested annotations.
Ancestors are a frequent cause of nested annotations but not the only one.
This function allows to remove nested annotations but to keep ancestors.
Removing or keeping ancestors depends on your use case.
In a semantic annotation task, only the longest terms must be kept
so the ancestors need to be removed.
In an information retrieval task, ancestors could be kept in the index.

Partial overlapping
""""""""""""""""

Definition: let a1 and a2 two annotations. If a1.start < a2.start and a2.start < a1.end
then we say that a1 **partially overlaps** a2.


.. literalinclude:: ../../tests/test_doc.py
    :language: python
    :dedent:
    :linenos:
    :start-after: # start_test_annotation_partial_overlap
    :end-before: # end_test_annotation_partial_overlap

The first annotation partially overlaps the second because it ends after the second starts.
In this example, both annotations share the *"cancer"* token.

Similarly the :ref:`api_doc:rm_nested_annots` function has no effect here.
