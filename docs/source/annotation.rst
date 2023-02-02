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

.. code-block:: python
    :linenos:
    :emphasize-lines: 11,12,13

        from iamsystem import Matcher, Abbreviations, Term
        matcher = Matcher()
        abb = Abbreviations(name="abbs")
        abb.add(short_form="infect", long_form="infectious", tokenizer=matcher)
        matcher.add_fuzzy_algo(abb)
        term = Term(label="infectious disease", code="D007239")
        matcher.add_keywords(keywords=[term])
        text = "Infect mononucleosis disease"
        annots = matcher.annot_text(text=text, w=2)
        for annot in annots:
            print(annot)
            print(annot.to_string(text=text))
            print(annot.to_string(text=text, debug=True))


.. code-block:: pycon

        # Infect disease	0 6;21 28	infectious disease (D007239)
        # Infect disease	0 6;21 28	infectious disease (D007239)	Infect mononucleosis disease
        # Infect disease	0 6;21 28	infectious disease (D007239)	Infect mononucleosis disease	infect(abbs);disease(exact)

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

.. code-block:: python

        from iamsystem import Matcher, english_tokenizer, Term
        term1 = Term(label="Infectious Disease", code="J80")
        term2 = Term(label="infectious disease", code="C0042029")
        term3 = Term(label="infectious disease, unspecified", code="C0042029")
        tokenizer = english_tokenizer()
        matcher = Matcher(tokenizer=tokenizer)
        matcher.add_stopwords(words=["unspecified"])
        matcher.add_keywords(keywords=[term1, term2, term3])
        text = "History of infectious disease"
        annots = matcher.annot_text(text=text)
        annot = annots[0]
        for keyword in annot.keywords:
            print(keyword)

.. code-block:: pycon

        # Infectious Disease (J80)
        # infectious disease (C0042029)
        # infectious disease, unspecified (C0042029)

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
By default, the :ref:`matcher:Matcher` removes nested annotation.
For example:

.. code-block:: python
    :linenos:
    :emphasize-lines: 5, 10

        from iamsystem import Matcher
        matcher = Matcher()
        matcher.add_labels(labels=["lung", "lung cancer"])
        text = "Presence of a lung cancer"
        annots = matcher.annot_text(text=text, w=1)
        for annot in annots:
            print(annot)
        # lung cancer	14 25	lung cancer
        self.assertEqual("lung cancer	14 25	lung cancer", str(annots[0]))
        matcher.remove_nested_annots = False
        annots = matcher.annot_text(text=text, w=1)
        for annot in annots:
            print(annot)
        # lung	14 18	lung
        # lung cancer	14 25	lung cancer

Another example where the first annotation fully overlaps the second but the latter is not
a nested annotation:

.. code-block:: python

        from iamsystem import Matcher
        matcher = Matcher()
        matcher.add_labels(labels=["North America", "South America"])
        text = "North and South America"
        annots = matcher.annot_text(text=text, w=3)
        for annot in annots:
            print(annot)

.. code-block:: pycon

        # North America	0 5;16 23	North America
        # South America	10 23	South America

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
In an information retrieval task, ancestors should be kept.

Partial overlapping
""""""""""""""""

Definition: let a1 and a2 two annotations. If a1.start < a2.start and a2.start < a1.end
then we say that a1 **partially overlaps** a2.

.. code-block:: python

        from iamsystem import Matcher
        matcher = Matcher()
        matcher.add_labels(labels=["lung cancer", "cancer prognosis"])
        annots = matcher.annot_text(text="lung cancer prognosis")
        for annot in annots:
            print(annot)

.. code-block:: pycon

        # lung cancer	0 11	lung cancer
        # cancer prognosis	5 21	cancer prognosis

The first annotation partially overlaps the second because it ends after the second starts.
In this example, both annotations share the *"cancer"* token.

Similarly the the :ref:`api_doc:rm_nested_annots` function has no effect here.
