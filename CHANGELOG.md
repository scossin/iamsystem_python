# ChangeLog

## Version 0.5.0 (beta)
- Fix issue 18: create multiple annotations when a keyword is repeated in the same window.

## Version 0.4.0 (beta)

### Breaking changes
- IAnnotation: remove 'brat_formatter' instance getter/setter.
'set_brat_formatter' becomes a class method to change the BratFormatter.
- Rename BratFormatters classes.
- Span (super class of Annotation): remove 'to_brat_format' method.
- Remove "offsets" attribute from the dictionary produced by
the 'to_dict' method of an annotation.
- FuzzyAlgo and ISynsProvider, get_synonyms method: change parameter name 'states' to 'transitions'.

### Enhancement
- Bug fixes: 11 to 16.
- Add the 'NoOverlap' matching strategy.
- Add a versioning number to the dictionary representation of an annotation.


## Version 0.3.0 (beta)

### Breaking changes

- IToken interface: add a 'i' attribute, the token index.
- FuzzyAlgo: replace 'i', the index of the token for which synonyms are
expected by the token: since token has now a 'i' attribute, it's possible to
retrieve its context tokens.
- BratDocument: remove "text" parameter. This parameter is now handled by the
annotation's brat_formatter.

### Enhancement

- BratFormatter: the user can set how to create a Brat annotation.
- IAnnotation interface added.
- The list of stopwords are added to each annotation instance.
It allows to keep information on the matching strategy and to know the importance of stopwords in the detection.


## Version 0.2.0 (beta)

### Breaking changes

- IMatcher interface: the window size parameter *w* was removed from the *annot_text* function.
In the Matcher class, this parameter becomes an attribute.
- Matcher: remove *add_labels* function, *add_keywords* function supports an Iterable of string labels.
- SpellWiseWrapper init: 'spellwise_algo' argument renamed 'measure' to be consistent with string distance algorithm.
- SpellWiseWrapper *add_words_to_ignore* is deprecated, moved to the init function.
- Fuzzyregex init: 'algo_name' argument renamed 'name'.
- IKeyword: remove *get_kb_id* function, a keyword doesn't have a kb_id.
An IEntity was created to store a kb_id attribute.
- Term class is renamed 'Entity', code attribute is replace by 'kb_id'.

### Enhancement

- Added support for pysimstring library (string distance fuzzy algorithm).
- Created StringDistance, parent class of pysimstring and spellwise wrapper.
- Created IWords2ignore for StringDistance algorithms to ignore common words.
This speeds up these algorithms and reduces the false positive rate.
- Added a Matcher.build() function that greatly simplifies the construction of the Matcher.
The documentation has been updated accordingly.
- Added an *IBaseMatcher* interface to be the main interface of any IAMsystem matcher (currently only one)
which should not be changed in the future.

## Version 0.1.1 (alpha)

### Initial release
