# iamsystem
[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg) [![macOS](https://svgshare.com/i/ZjP.svg)](https://svgshare.com/i/ZjP.svg)
[![Windows](https://svgshare.com/i/ZhY.svg)](https://svgshare.com/i/ZhY.svg) [![PyPI version fury.io](https://badge.fury.io/py/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)

[![PyPI version fury.io](https://badge.fury.io/py/ansicolortags.svg)](https://pypi.python.org/pypi/iamsystem/)

![test](https://github.com/scossin/iamsystem_python/actions/workflows/test.yml/badge.svg)

A python implementation of IAMsystem algorithm, a fast dictionary-based approach for semantic annotation, a.k.a entity linking.


## Installation

```bash
pip install iamsystem
```

## Usage
You provide a list of keywords you want to detect in a document,
you can add and combine abbreviations, normalization methods (lemmatization, stemming) and approximate string matching algorithms,
IAMsystem algorithm performs the semantic annotation.

### Quick example

```python
from iamsystem import Matcher, Abbreviations, SpellWiseWrapper,\
    ESpellWiseAlgo
matcher = Matcher()
# add a list of words to detect
matcher.add_labels(labels=["North America", "South America"])
matcher.add_stopwords(words=["and"])
# add a list of abbreviations (optional)
abbs = Abbreviations(name="common abbreviations")
abbs.add(short_form="amer", long_form="America", tokenizer=matcher)
matcher.add_fuzzy_algo(fuzzy_algo=abbs)
# add a string distance algorithm (optional)
levenshtein = SpellWiseWrapper(
    ESpellWiseAlgo.LEVENSHTEIN, max_distance=1
)
levenshtein.add_words(words=matcher.get_keywords_unigrams())
matcher.add_fuzzy_algo(fuzzy_algo=levenshtein)
# perform semantic annotation:
annots = matcher.annot_text(text="Northh and south Amer.", w=2)
for annot in annots:
    print(annot)
# Northh Amer	0 6;17 21	North America
# south Amer	11 21	South America
```


## Algorithm
The algorithm was developed in the context of a [PhD thesis](https://theses.hal.science/tel-03857962/).
It proposes a solution to quickly annotate documents using a large dictionary (> 300K keywords) and fuzzy matching algorithms.
No string distance algorithm is implemented in this package, it imports and leverages external libraries like [spellwise](https://github.com/chinnichaitanya/spellwise)
and [nltk](https://github.com/nltk/nltk).
Its algorithmic complexity is O(n(log(m))) with n the number of tokens in a document and m the size of the dictionary.
The formalization of the algorithm is available in this [paper](https://ceur-ws.org/Vol-3202/livingner-paper11.pdf).

The algorithm was initially developed in Java (https://github.com/scossin/IAMsystem) and
has participated in several semantic annotation competitions in the medical domain where it has obtained very satisfactory results.

### Citation
```
@article{cossin_iam_2018,
	title = {{IAM} at {CLEF} {eHealth} 2018: {Concept} {Annotation} and {Coding} in {French} {Death} {Certificates}},
	shorttitle = {{IAM} at {CLEF} {eHealth} 2018},
	url = {http://arxiv.org/abs/1807.03674},
	urldate = {2018-07-11},
	journal = {arXiv:1807.03674 [cs]},
	author = {Cossin, Sébastien and Jouhet, Vianney and Mougin, Fleur and Diallo, Gayo and Thiessard, Frantz},
	month = jul,
	year = {2018},
	note = {arXiv: 1807.03674},
	keywords = {Computer Science - Computation and Language},
}
```

## Changelog

**0.1.0**
* Initial release
