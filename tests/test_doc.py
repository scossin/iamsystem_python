""" Examples found in the documentation."""
import builtins
import sys
import unittest


def print(o) -> None:
    """Utility function that print in pycharm IDE and
    print nothing in command line."""
    if sys.stdin and sys.stdin.isatty():
        return None
    else:
        builtins.print(f"# {o}")


class MatcherDocTest(unittest.TestCase):
    def test_readme_example(self):
        from iamsystem import Abbreviations
        from iamsystem import ESpellWiseAlgo
        from iamsystem import Matcher
        from iamsystem import SpellWiseWrapper

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
        self.assertEqual(2, len(annots))

    def test_exact_match_keywords(self):
        """Matcher with a list of words."""
        from iamsystem import Matcher

        labels = ["acute respiratory distress syndrome", "diarrrhea"]
        text = "Pt c/o Acute Respiratory Distress Syndrome and diarrrhea"
        matcher = Matcher()
        matcher.add_labels(labels=labels)
        annots = matcher.annot_text(text=text)
        for annot in annots:
            print(annot)
        # Acute Respiratory Distress Syndrome	7 42	acute respiratory distress syndrome # noqa
        # diarrrhea	47 56	diarrrhea
        self.assertEqual(
            "Acute Respiratory Distress Syndrome	7 42	acute respiratory "
            "distress syndrome",
            str(annots[0]),
        )
        print(str(annots[1]))
        self.assertEqual("diarrrhea	47 56	diarrrhea", str(annots[1]))

    def test_exact_match_terms(self):
        """Matcher with Term class."""
        from iamsystem import Matcher
        from iamsystem import Term

        term1 = Term(label="acute respiratory distress syndrome", code="J80")
        term2 = Term(label="diarrrhea", code="R19.7")
        text = "Pt c/o acute respiratory distress syndrome and diarrrhea"
        matcher = Matcher()
        matcher.add_keywords(keywords=[term1, term2])
        annots = matcher.annot_text(text=text)
        for annot in annots:
            print(annot)
        # acute respiratory distress syndrome	7 42	acute respiratory distress syndrome (J80) # noqa
        # diarrrhea (R19.7)	47	56
        self.assertEqual(
            "acute respiratory distress syndrome	7 42	acute respiratory "
            "distress syndrome (J80)",
            str(annots[0]),
        )
        self.assertEqual("diarrrhea	47 56	diarrrhea (R19.7)", str(annots[1]))

    def test_window(self):
        """Matcher with a window different than 1."""
        from iamsystem import Matcher

        labels = ["calcium level"]
        matcher = Matcher()
        matcher.add_labels(labels=labels)
        annots = matcher.annot_text(text="calcium blood level", w=2)
        for annot in annots:
            print(annot)
        # calcium level	0 7;14 19	calcium level
        self.assertEqual(
            "calcium level	0 7;14 19	calcium level", str(annots[0])
        )

    def test_fail_order(self):
        """Matcher fails to detect when tokens order is not the same in
        keywords and document."""
        from iamsystem import Matcher

        labels = ["calcium level"]
        matcher = Matcher()
        matcher.add_labels(labels=labels)
        annots = matcher.annot_text(text="level calcium", w=2)
        print(len(annots))
        # 0
        self.assertEqual(0, len(annots))


class TokenizerDocTest(unittest.TestCase):
    def test_tokenizer(self):
        """Alphanumeric tokenizer limits : '+' sign is not a token."""
        from iamsystem import english_tokenizer

        tokenizer = english_tokenizer()
        tokens = tokenizer.tokenize("SARS-CoV+")
        for token in tokens:
            print(token)
        # Token(label='SARS', norm_label='sars', start=0, end=4)
        # Token(label='CoV', norm_label='cov', start=5, end=8)
        self.assertEqual(
            "Token(label='SARS', norm_label='sars', start=0, end=4)",
            str(tokens[0]),
        )

    def test_custom_tokenizer(self):
        """Change tokenizer's split function."""
        from iamsystem import english_tokenizer
        from iamsystem import split_find_iter_closure

        tokenizer = english_tokenizer()
        tokenizer.split = split_find_iter_closure(pattern=r"(\w+|\+)")
        tokens = tokenizer.tokenize("SARS-CoV+")
        for token in tokens:
            print(token)
        # Token(label='SARS', norm_label='sars', start=0, end=4)
        # Token(label='CoV', norm_label='cov', start=5, end=8)
        # Token(label='+', norm_label='+', start=8, end=9)
        self.assertEqual(
            "Token(label='+', norm_label='+', start=8, end=9)", str(tokens[2])
        )

    def test_matcher_with_custom_tokenizer(self):
        """Matcher with a custom tokenizer."""
        from iamsystem import Matcher
        from iamsystem import Term
        from iamsystem import english_tokenizer
        from iamsystem import split_find_iter_closure

        term1 = Term(label="SARS-CoV+", code="95209-3")
        text = "Pt c/o acute respiratory distress syndrome. RT-PCR sars-cov+"
        tokenizer = english_tokenizer()
        tokenizer.split = split_find_iter_closure(pattern=r"(\w+|\+)")
        matcher = Matcher(tokenizer=tokenizer)
        matcher.add_keywords(keywords=[term1])
        annots = matcher.annot_text(text=text)
        for annot in annots:
            print(annot)
        # sars cov +	51 60	SARS-CoV+ (95209-3)
        self.assertEqual(
            "sars cov +	51 60	SARS-CoV+ (95209-3)", str(annots[0])
        )

    def test_unordered_words_seq(self):
        """Tokenizer orders the tokens to have a match when the order of
        tokens is not the same in document and keywords."""
        from iamsystem import Matcher
        from iamsystem import english_tokenizer
        from iamsystem import tokenize_and_order_decorator

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
        self.assertEqual(1, len(annots))


class StopwordsTest(unittest.TestCase):
    def test_add_stopword(self):
        """Adding stopwords to have a match."""
        from iamsystem import Matcher
        from iamsystem import Term
        from iamsystem import english_tokenizer

        tokenizer = english_tokenizer()
        matcher = Matcher(tokenizer=tokenizer)
        matcher.add_stopwords(words=["unspecified"])
        term = Term(label="Essential hypertension, unspecified", code="I10.9")
        matcher.add_keywords(keywords=[term])
        text = "Medical history: essential hypertension"
        annots = matcher.annot_text(text=text)
        for annot in annots:
            print(annot)
        # essential hypertension	17 39	Essential hypertension, unspecified (I10.9) # noqa
        self.assertEqual(
            "essential hypertension	17 39	Essential hypertension, "
            "unspecified (I10.9)",
            str(annots[0]),
        )

    def test_negative_stopword(self):
        """Matcher with negatives stopwords."""
        from iamsystem import Keyword
        from iamsystem import Matcher
        from iamsystem import NegativeStopwords
        from iamsystem import NoStopwords
        from iamsystem import Terminology
        from iamsystem import english_tokenizer

        text = "the level of calcium can be measured in the blood."
        termino = Terminology()
        termino.add_keywords(keywords=[Keyword(label="calcium blood")])
        stopwords = NegativeStopwords()
        tokenizer = english_tokenizer()
        stopwords.add_words(
            words_to_keep=termino.get_unigrams(
                tokenizer=tokenizer, stopwords=NoStopwords()
            )
        )
        matcher = Matcher(tokenizer=tokenizer, stopwords=stopwords)
        matcher.add_keywords(keywords=termino)
        annots = matcher.annot_text(text=text, w=1)
        for annot in annots:
            print(annot)
        # calcium blood	13 20;44 49	calcium blood
        self.assertEqual(1, len(annots))


class AnnotationDocTest(unittest.TestCase):
    def test_annotation_format(self):
        """String representation of annotation."""
        from iamsystem import Abbreviations
        from iamsystem import Matcher
        from iamsystem import Term

        matcher = Matcher()
        abb = Abbreviations(name="abbs")
        abb.add(
            short_form="infect",
            long_form="infectious",
            tokenizer=matcher,
        )
        matcher.add_fuzzy_algo(abb)
        term = Term(label="infectious disease", code="D007239")
        matcher.add_keywords(keywords=[term])
        text = "Infect mononucleosis disease"
        annots = matcher.annot_text(text=text, w=2)
        for annot in annots:
            print(annot)
            print(annot.to_string(text=text))
            print(annot.to_string(text=text, debug=True))
        # Infect disease	0 6;21 28	infectious disease (D007239) # noqa
        # Infect disease	0 6;21 28	infectious disease (D007239)	Infect mononucleosis disease # noqa
        # Infect disease	0 6;21 28	infectious disease (D007239)	Infect mononucleosis disease	infect(abbs);disease(exact) # noqa
        self.assertEqual(
            "Infect disease	0 6;21 28	infectious disease (D007239)",
            str(annots[0]),
        )
        self.assertEqual(
            "Infect disease	0 6;21 28	infectious disease (D007239)	"
            "Infect mononucleosis disease",
            str(annots[0].to_string(text=text)),
        )
        self.assertEqual(
            "Infect disease	0 6;21 28	infectious disease (D007239)	"
            "Infect mononucleosis disease	infect(abbs);disease(exact)",
            str(annots[0].to_string(text=text, debug=True)),
        )

    def test_annotation_multiple_keywords(self):
        """One annotation can have multiple keywords."""
        from iamsystem import Matcher
        from iamsystem import Term
        from iamsystem import english_tokenizer

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
        # Infectious Disease (J80)
        # infectious disease (C0042029)
        # infectious disease, unspecified (C0042029)
        keyword_str = [str(keyword) for keyword in annot.keywords]
        self.assertTrue("Infectious Disease (J80)" in keyword_str)
        self.assertTrue("infectious disease (C0042029)" in keyword_str)
        self.assertTrue(
            "infectious disease, unspecified (C0042029)" in keyword_str
        )

    def test_annotation_overlapping_ancestors(self):
        """Remove or keep ancestors."""
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
        self.assertEqual("lung	14 18	lung", str(annots[0]))

    def test_annotation_overlapping_not_ancestors(self):
        """Case of overlapping but not an ancestor."""
        from iamsystem import Matcher

        matcher = Matcher()
        matcher.add_labels(labels=["North America", "South America"])
        text = "North and South America"
        annots = matcher.annot_text(text=text, w=3)
        for annot in annots:
            print(annot)
        # North America	0 5;16 23	North America
        # South America	10 23	South America
        self.assertEqual(
            "North America	0 5;16 23	North America", str(annots[0])
        )
        self.assertEqual("South America	10 23	South America", str(annots[1]))

    def test_annotation_partial_overlap(self):
        """Case of partial annotation overlapping that have a word
        in common."""
        from iamsystem import Matcher

        matcher = Matcher()
        matcher.add_labels(labels=["lung cancer", "cancer prognosis"])
        annots = matcher.annot_text(text="lung cancer prognosis")
        for annot in annots:
            print(annot)
        # lung cancer	0 11	lung cancer
        # cancer prognosis	5 21	cancer prognosis
        self.assertEqual("lung cancer	0 11	lung cancer", str(annots[0]))
        self.assertEqual(
            "cancer prognosis	5 21	cancer prognosis", str(annots[1])
        )

    # TODO: how to handle overlapping annotations ?
    # def test_replace_annots(self):
    #     from iamsystem import Matcher, Term, Annotation, replace_annots
    #     matcher = Matcher()
    #     term1 = Term(label="North America", code="NA")
    #     term2 = Term(label="South America", code="SA")
    #     matcher.add_keywords(keywords=[term1, term2])
    #     text = "North and South America"
    #     annots = matcher.annot_text(text=text, w=3)
    #     new_labels = [annot.keywords[0].get_kb_id() for annot in annots]
    #     new_text = replace_annots(text=text, annots=annots,
    #     new_labels=new_labels)
    #     print(new_text)
    #     self.assertEqual("NA and SA", new_text)


class BratDocTest(unittest.TestCase):
    def test_brat_document(self):
        """Brat document example."""
        from iamsystem import BratDocument
        from iamsystem import Matcher
        from iamsystem import Term

        matcher = Matcher()
        term1 = Term(label="North America", code="NA")
        matcher.add_keywords(keywords=[term1])
        text = "North and South America"
        annots = matcher.annot_text(text=text, w=3)
        brat_document = BratDocument()
        brat_document.add_annots(
            annots, text=text, brat_type="CONTINENT", keyword_attr=None
        )
        print(str(brat_document))
        # T1	CONTINENT 0 5;16 23	North America
        # #1	IAMSYSTEM T1	North America (NA)
        self.assertEqual(
            "T1\tCONTINENT 0 5;16 23\tNorth America\n#1\tIAMSYSTEM T1\tNorth "
            "America (NA)",
            str(brat_document),
        )

    def test_brat_doc_keyword(self):
        """Brat document example with a custom Keyword that stores
        brat_type."""
        from iamsystem import Term

        class Entity(Term):
            def __init__(self, label: str, code: str, brat_type: str):
                super().__init__(label, code)
                self.brat_type = brat_type

        from iamsystem import BratDocument
        from iamsystem import Matcher

        matcher = Matcher()
        term1 = Entity(label="North America", code="NA", brat_type="CONTINENT")
        matcher.add_keywords(keywords=[term1])
        text = "North and South America"
        annots = matcher.annot_text(text=text, w=3)
        brat_document = BratDocument()
        brat_document.add_annots(
            annots=annots, text=text, keyword_attr="brat_type"
        )
        print(str(brat_document))
        # T1	CONTINENT 0 5;16 23	North America
        # #1	IAMSYSTEM T1	North America (NA)
        self.assertEqual(
            "T1\tCONTINENT 0 5;16 23\tNorth America\n#1\tIAMSYSTEM T1\tNorth "
            "America (NA)",
            str(brat_document),
        )

    def test_brat_writer(self):
        """BratWriter example."""
        from iamsystem import BratDocument
        from iamsystem import BratWriter
        from iamsystem import Matcher
        from iamsystem import Term

        matcher = Matcher()
        term1 = Term(label="North America", code="NA")
        matcher.add_keywords(keywords=[term1])
        text = "North and South America"
        annots = matcher.annot_text(text=text, w=3)
        brat_document = BratDocument()
        brat_document.add_annots(annots, text=text, brat_type="CONTINENT")
        # filename = "./doc.ann"
        # with(open(filename, 'w')) as f:
        #     BratWriter.saveEntities(brat_entities=
        #     brat_document.get_entities(), write=f.write)
        #     BratWriter.saveNotes(brat_notes=
        #     brat_document.get_notes(), write=f.write)
        BratWriter.saveEntities(
            brat_entities=brat_document.get_entities(), write=lambda x: None
        )
        BratWriter.saveNotes(
            brat_notes=brat_document.get_notes(), write=lambda x: None
        )


class FuzzyDocTest(unittest.TestCase):
    def test_abbreviations(self):
        """Abbreviations without customization."""
        from iamsystem import Abbreviations
        from iamsystem import Matcher
        from iamsystem import Term
        from iamsystem import english_tokenizer

        tokenizer = english_tokenizer()
        abbs = Abbreviations(name="abbs")
        abbs.add(short_form="Pt", long_form="patient", tokenizer=tokenizer)
        abbs.add(
            short_form="PT", long_form="physiotherapy", tokenizer=tokenizer
        )
        abbs.add(
            short_form="ARD",
            long_form="Acute Respiratory Distress",
            tokenizer=tokenizer,
        )
        matcher = Matcher(tokenizer=tokenizer)
        term1 = Term(label="acute respiratory distress", code="J80")
        term2 = Term(label="patient", code="D007290")
        term3 = Term(label="patient hospitalized", code="D007297")
        term4 = Term(label="physiotherapy", code="D007297")
        matcher.add_keywords(keywords=[term1, term2, term3, term4])
        matcher.add_fuzzy_algo(fuzzy_algo=abbs)
        annots = matcher.annot_text(
            text="Pt hospitalized with ARD. Treament: PT"
        )
        for annot in annots:
            print(annot.to_string(debug=True))
        # Pt hospitalized	0 15	patient hospitalized (D007297)	pt(abbs);hospitalized(exact) # noqa
        # ARD	21 24	acute respiratory distress (J80)	ard(abbs)
        # PT	36 38	patient (D007290)	pt(abbs)
        # PT	36 38	physiotherapy (D007297)	pt(abbs)
        self.assertEqual(4, len(annots))

    def test_uppercase(self):
        """Abbreviations that check uppercase."""
        from iamsystem import Abbreviations
        from iamsystem import Matcher
        from iamsystem import Term
        from iamsystem import TokenT
        from iamsystem import english_tokenizer

        def upper_case_only(token: TokenT) -> bool:
            """Return True if all token's characters are uppercase."""
            return token.label.isupper()

        def first_letter_capitalized(token: TokenT) -> bool:
            """Return True if the first letter is uppercase."""
            return token.label[0].isupper() and not token.label.isupper()

        tokenizer = english_tokenizer()
        abbs_upper = Abbreviations(
            name="upper case abbs", token_is_an_abbreviation=upper_case_only
        )
        abbs_upper.add(
            short_form="PT", long_form="physiotherapy", tokenizer=tokenizer
        )
        abbs_upper.add(
            short_form="ARD",
            long_form="Acute Respiratory Distress",
            tokenizer=tokenizer,
        )
        abbs_capitalized = Abbreviations(
            name="capitalized abbs",
            token_is_an_abbreviation=first_letter_capitalized,
        )
        abbs_capitalized.add(
            short_form="Pt", long_form="patient", tokenizer=tokenizer
        )
        matcher = Matcher(tokenizer=tokenizer)
        term1 = Term(label="acute respiratory distress", code="J80")
        term2 = Term(label="patient", code="D007290")
        term3 = Term(label="patient hospitalized", code="D007297")
        term4 = Term(label="physiotherapy", code="D007297")
        matcher.add_keywords(keywords=[term1, term2, term3, term4])
        matcher.add_fuzzy_algo(fuzzy_algo=abbs_upper)
        matcher.add_fuzzy_algo(fuzzy_algo=abbs_capitalized)
        annots = matcher.annot_text(
            text="Pt hospitalized with ARD. Treament: PT"
        )
        for annot in annots:
            print(annot.to_string(debug=True))
        # Pt hospitalized	0 15	patient hospitalized (D007297)	pt(capitalized abbs);hospitalized(exact) # noqa
        # ARD	21 24	acute respiratory distress (J80)	ard(upper case abbs)
        # PT	36 38	physiotherapy (D007297)	pt(upper case abbs)
        self.assertEqual(3, len(annots))

    def test_spellwise(self):
        """Spellwise library examples."""
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
        # acute resiratory distresssss	0 28	acute respiratory distress (J80) acute(exact,LEVENSHTEIN,SOUNDEX);resiratory(LEVENSHTEIN);distresssss(SOUNDEX) # noqa
        self.assertEqual(1, len(annots))

    def test_cache_fuzzy_algos(self):
        """Cache example."""
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
        # a resiratory distresssss	0 24	acute respiratory distress (J80)	a(abbs);resiratory(LEVENSHTEIN);distresssss(SOUNDEX) # noqa
        self.assertEqual(1, len(annots))

    def test_fuzzy_regex(self):
        """FuzzyRegex example."""
        from iamsystem import FuzzyRegex
        from iamsystem import Matcher
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
        detector = Matcher(tokenizer=tokenizer)
        detector.add_labels(labels=["calcium numval mmol/L"])
        detector.add_stopwords(words=["level", "is", "normal"])
        detector.add_fuzzy_algo(fuzzy_algo=fuzzy)
        annots = detector.annot_text(
            text="the blood calcium level is normal: 2.1 mmol/L", w=1
        )
        for annot in annots:
            print(annot)
        # calcium 2.1 mmol L	10 17;35 45	calcium numval mmol/L
        self.assertEqual(1, len(annots))

    def test_fuzzy_regex_negative_stopwords(self):
        """combine NegativeStopwords with FuzzyRegex."""
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
        # calcium 2.1 mmol L	10 17;35 45	calcium numval mmol/L
        self.assertEqual(1, len(annots))

    def test_word_normalization(self):
        """Stemming example."""
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
        # cancer prostatique	0 18	cancer de la prostate (C72)
        self.assertEqual(1, len(annots))


class SpacyDocTest(unittest.TestCase):
    def test_component(self):
        """Test detection with component."""
        from typing import Iterable
        from typing import List

        import spacy

        from spacy.lang.fr import French

        from iamsystem import Abbreviations
        from iamsystem import FuzzyAlgo
        from iamsystem import IKeyword
        from iamsystem import IStopwords
        from iamsystem import Term
        from iamsystem import Terminology
        from iamsystem import french_tokenizer
        from iamsystem.spacy import IAMsystemSpacy  # noqa
        from iamsystem.spacy import IsStopSpacy
        from iamsystem.spacy import TokenSpacyAdapter

        @spacy.registry.misc("umls_terms.v1")
        def get_termino_umls() -> Iterable[IKeyword]:
            """An imaginary set of umls terms."""
            termino = Terminology()
            term1 = Term("Insuffisance Cardiaque", "I50.9")
            term2 = Term("Insuffisance Cardiaque Gauche", "I50.1")
            termino.add_keywords(keywords=[term1, term2])
            return termino

        @spacy.registry.misc("fuzzy_algos_short_notes.v1")
        def get_fuzzy_algos_short_notes() -> List[FuzzyAlgo]:
            """An imaginary set of fuzzy algorithms for medical short notes."""
            tokenizer = french_tokenizer()
            abbs = Abbreviations(name="French medical abbreviations")
            abbs.add(
                short_form="ins", long_form="insuffisance", tokenizer=tokenizer
            )
            abbs.add(
                short_form="ic",
                long_form="insuffisance cardiaque",
                tokenizer=tokenizer,
            )
            return [abbs]

        @spacy.registry.misc("stopwords_spacy.v1")
        def get_stopwords_short_notes() -> IStopwords[TokenSpacyAdapter]:
            """Use spaCy stopword list."""
            stopwords = IsStopSpacy()
            return stopwords

        nlp = French()
        nlp.add_pipe(
            "iamsystem",
            name="iamsystem",
            last=True,
            config={
                "keywords": {"@misc": "umls_terms.v1"},
                "stopwords": {"@misc": "stopwords_spacy.v1"},
                "fuzzy_algos": {"@misc": "fuzzy_algos_short_notes.v1"},
                "w": 1,
                "remove_nested_annots": True,
            },
        )
        doc = nlp("ic gauche")
        self.assertEqual(1, len(doc.spans["iamsystem"]))
        spans = doc.spans["iamsystem"]
        for span in spans:
            print(span._.iamsystem)
        # ic gauche	0 9	Insuffisance Cardiaque Gauche (I50.1)


if __name__ == "__main__":
    unittest.main()
