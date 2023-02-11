# ChangeLog

## Version 0.2.0 (beta)

### Breaking changes

- IMatcher interface: the window size parameter *w* was removed from the *annot_text* function.
In the Matcher class, this parameter becomes an attribute.
- Matcher: remove *add_labels* function, *add_keywords* function supports an Iterable of string labels.
- SpellWiseWrapper init: 'spellwise_algo' argument renamed 'measure' to be consistent with string distance algorithm.
- SpellWiseWrapper *add_words_to_ignore* is deprecated, moved to the init function.
- Fuzzyregex init: 'algo_name' argument renamed 'name'.


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
