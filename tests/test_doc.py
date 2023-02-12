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
        from iamsystem import Matcher

        matcher = Matcher.build(
            keywords=["North America", "South America"],
            stopwords=["and"],
            abbreviations=[("amer", "America")],
            spellwise=[dict(measure="Levenshtein", max_distance=1)],
            w=2,
        )
        annots = matcher.annot_text(text="Northh and south Amer.")
        for annot in annots:
            print(annot)
        # Northh Amer	0 6;17 21	North America
        # south Amer	11 21	South America
        self.assertEqual(2, len(annots))

    def test_exact_match_keywords(self):
        """Matcher with a list of words."""
        # start_test_exact_match_keywords
        from iamsystem import Matcher

        matcher = Matcher.build(
            keywords=["acute respiratory distress syndrome", "diarrrhea"]
        )
        annots = matcher.annot_text(
            text="Pt c/o Acute Respiratory Distress " "Syndrome and diarrrhea"
        )
        for annot in annots:
            print(annot)
        # Acute Respiratory Distress Syndrome	7 42	acute respiratory distress syndrome # noqa
        # diarrrhea	47 56	diarrrhea
        # end_test_exact_match_keywords
        self.assertEqual(
            "Acute Respiratory Distress Syndrome	7 42	acute respiratory "
            "distress syndrome",
            str(annots[0]),
        )
        self.assertEqual("diarrrhea	47 56	diarrrhea", str(annots[1]))

    def test_exact_match_ents(self):
        """Matcher with Term class."""
        # start_test_exact_match_ents
        from iamsystem import Entity
        from iamsystem import Matcher

        ent1 = Entity(label="acute respiratory distress syndrome", kb_id="J80")
        ent2 = Entity(label="diarrrhea", kb_id="R19.7")
        text = "Pt c/o acute respiratory distress syndrome and diarrrhea"
        matcher = Matcher.build(keywords=[ent1, ent2])
        annots = matcher.annot_text(text=text)
        for annot in annots:
            print(annot)
        # acute respiratory distress syndrome	7 42	acute respiratory distress syndrome (J80) # noqa
        # diarrrhea (R19.7)	47	56
        # end_test_exact_match_ents
        self.assertEqual(
            "acute respiratory distress syndrome	7 42	acute respiratory "
            "distress syndrome (J80)",
            str(annots[0]),
        )
        self.assertEqual("diarrrhea	47 56	diarrrhea (R19.7)", str(annots[1]))

    def test_exact_match_custom_keyword(self):
        """Matcher with Term class."""
        # start_test_exact_match_custom_keyword
        from iamsystem import Entity
        from iamsystem import IEntity
        from iamsystem import Matcher

        class MyKeyword(IEntity):
            def __init__(
                self, label: str, category: str, kb_name: str, uri: str
            ):
                """label is the only mandatory attribute."""
                self.label = label
                self.kb_name = kb_name
                self.category = category
                self.kb_id = uri

            def __str__(self):
                """Called by print(annot)"""
                return f"{self.kb_id}"

        ent1 = MyKeyword(
            label="acute respiratory distress syndrome",
            category="disease",
            kb_name="wikipedia",
            uri="https://www.wikidata.org/wiki/Q344873",
        )
        ent2 = Entity(label="diarrrhea", kb_id="R19.7")
        text = "Pt c/o acute respiratory distress syndrome and diarrrhea"
        matcher = Matcher.build(keywords=[ent1, ent2])
        annots = matcher.annot_text(text=text)
        for annot in annots:
            print(annot)
        # acute respiratory distress syndrome	7 42	https://www.wikidata.org/wiki/Q344873 # noqa
        # diarrrhea	47 56	diarrrhea (R19.7)
        # end_test_exact_match_custom_keyword
        self.assertEqual(
            "acute respiratory distress syndrome	7 42	https://www.wikidata.org/wiki/Q344873",  # noqa
            str(annots[0]),
        )
        self.assertEqual("diarrrhea	47 56	diarrrhea (R19.7)", str(annots[1]))

    def test_window(self):
        """Matcher with a window different than 1."""
        # start_test_window
        from iamsystem import Matcher

        matcher = Matcher.build(keywords=["calcium level"], w=2)
        annots = matcher.annot_text(text="calcium blood level")
        for annot in annots:
            print(annot)
        # calcium level	0 7;14 19	calcium level
        # end_test_window
        self.assertEqual(
            "calcium level	0 7;14 19	calcium level", str(annots[0])
        )

    def test_fail_order(self):
        """Matcher fails to detect when tokens order is not the same in
        keywords and document."""
        # start_test_fail_order
        from iamsystem import Matcher

        matcher = Matcher.build(keywords=["calcium level"], w=2)
        annots = matcher.annot_text(text="level calcium")
        print(len(annots))  # 0
        # end_test_fail_order
        self.assertEqual(0, len(annots))


class TokenizerDocTest(unittest.TestCase):
    def test_tokenizer(self):
        """Alphanumeric tokenizer limits : '+' sign is not a token."""
        # start_test_tokenizer
        from iamsystem import english_tokenizer

        tokenizer = english_tokenizer()
        tokens = tokenizer.tokenize("SARS-CoV+")
        for token in tokens:
            print(token)
        # Token(label='SARS', norm_label='sars', start=0, end=4)
        # Token(label='CoV', norm_label='cov', start=5, end=8)
        # end_test_tokenizer
        self.assertEqual(
            "Token(label='SARS', norm_label='sars', start=0, end=4)",
            str(tokens[0]),
        )

    def test_custom_tokenizer(self):
        """Change tokenizer's split function."""
        # start_test_custom_tokenizer
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
        # end_test_custom_tokenizer
        self.assertEqual(
            "Token(label='+', norm_label='+', start=8, end=9)", str(tokens[2])
        )

    def test_matcher_with_custom_tokenizer(self):
        """Matcher with a custom tokenizer."""
        # start_test_matcher_with_custom_tokenizer
        from iamsystem import Entity
        from iamsystem import Matcher
        from iamsystem import english_tokenizer
        from iamsystem import split_find_iter_closure

        ent1 = Entity(label="SARS-CoV+", kb_id="95209-3")
        text = "Pt c/o acute respiratory distress syndrome. RT-PCR sars-cov+"
        tokenizer = english_tokenizer()
        tokenizer.split = split_find_iter_closure(pattern=r"(\w+|\+)")
        matcher = Matcher.build(keywords=[ent1], tokenizer=tokenizer)
        annots = matcher.annot_text(text=text)
        for annot in annots:
            print(annot)
        # sars cov +	51 60	SARS-CoV+ (95209-3)
        # end_test_matcher_with_custom_tokenizer
        self.assertEqual(
            "sars cov +	51 60	SARS-CoV+ (95209-3)", str(annots[0])
        )

    def test_unordered_words_seq(self):
        """Tokenizer orders the tokens to have a match when the order of
        tokens is not the same in document and keywords."""
        # start_test_unordered_words_seq
        from iamsystem import Matcher
        from iamsystem import english_tokenizer

        text = "the level of calcium can measured in the blood."
        tokenizer = english_tokenizer()
        matcher = Matcher.build(
            keywords=["blood calcium level"],
            tokenizer=tokenizer,
            order_tokens=True,
            w=5,
        )
        annots = matcher.annot_text(text=text)
        for annot in annots:
            print(annot)
        # level calcium blood	4 9;13 20;41 46	blood calcium level
        # end_test_unordered_words_seq
        self.assertEqual(1, len(annots))


class StopwordsTest(unittest.TestCase):
    def test_add_stopword(self):
        """Adding stopwords to have a match."""
        # start_test_add_stopword
        from iamsystem import Entity
        from iamsystem import Matcher
        from iamsystem import english_tokenizer

        ent = Entity(
            label="Essential hypertension, unspecified", kb_id="I10.9"
        )
        matcher = Matcher.build(
            keywords=[ent],
            tokenizer=english_tokenizer(),
            stopwords=["unspecified"],
        )
        text = "Medical history: essential hypertension"
        annots = matcher.annot_text(text=text)
        for annot in annots:
            print(annot)
        # essential hypertension	17 39	Essential hypertension, unspecified (I10.9) # noqa
        # end_test_add_stopword
        self.assertEqual(
            "essential hypertension	17 39	Essential hypertension, "
            "unspecified (I10.9)",
            str(annots[0]),
        )

    def test_negative_stopword(self):
        """Matcher with negatives stopwords."""
        # start_test_negative_stopword
        from iamsystem import Matcher

        text = "the level of calcium can be measured in the blood."
        matcher = Matcher.build(keywords=["calcium blood"], negative=True)
        annots = matcher.annot_text(text=text)
        for annot in annots:
            print(annot)
        # calcium blood	13 20;44 49	calcium blood
        # end_test_negative_stopword
        self.assertEqual(1, len(annots))


class AnnotationDocTest(unittest.TestCase):
    def test_annotation_format(self):
        """String representation of annotation."""
        # start_test_annotation_format
        from iamsystem import Entity
        from iamsystem import Matcher

        ent = Entity(label="infectious disease", kb_id="D007239")
        matcher = Matcher.build(
            keywords=[ent], abbreviations=[("infect", "infectious")], w=2
        )
        text = "Infect mononucleosis disease"
        annots = matcher.annot_text(text=text)
        for annot in annots:
            print(annot)
            print(annot.to_string(text=text))
            print(annot.to_string(text=text, debug=True))
        # Infect disease	0 6;21 28	infectious disease (D007239) # noqa
        # Infect disease	0 6;21 28	infectious disease (D007239)	Infect mononucleosis disease # noqa
        # Infect disease	0 6;21 28	infectious disease (D007239)	Infect mononucleosis disease	infect(abbs);disease(exact) # noqa
        # end_test_annotation_format
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
        # start_test_annotation_multiple_keywords
        from iamsystem import Entity
        from iamsystem import Matcher
        from iamsystem import english_tokenizer

        ent1 = Entity(label="Infectious Disease", kb_id="J80")
        ent2 = Entity(label="infectious disease", kb_id="C0042029")
        ent3 = Entity(
            label="infectious disease, unspecified", kb_id="C0042029"
        )
        matcher = Matcher.build(
            keywords=[ent1, ent2, ent3],
            tokenizer=english_tokenizer(),
            stopwords=["unspecified"],
        )
        text = "History of infectious disease"
        annots = matcher.annot_text(text=text)
        annot = annots[0]
        for keyword in annot.keywords:
            print(keyword)
        # Infectious Disease (J80)
        # infectious disease (C0042029)
        # infectious disease, unspecified (C0042029)
        # end_test_annotation_multiple_keywords
        keyword_str = [str(keyword) for keyword in annot.keywords]
        self.assertTrue("Infectious Disease (J80)" in keyword_str)
        self.assertTrue("infectious disease (C0042029)" in keyword_str)
        self.assertTrue(
            "infectious disease, unspecified (C0042029)" in keyword_str
        )

    def test_annotation_overlapping_ancestors(self):
        """Remove or keep ancestors."""
        # start_test_annotation_overlapping_ancestors
        from iamsystem import Matcher

        matcher = Matcher.build(keywords=["lung", "lung cancer"], w=1)
        text = "Presence of a lung cancer"
        annots = matcher.annot_text(text=text)
        for annot in annots:
            print(annot)
        # lung cancer	14 25	lung cancer
        matcher.remove_nested_annots = False
        annots_2 = matcher.annot_text(text=text)
        for annot in annots_2:
            print(annot)
        # lung	14 18	lung
        # lung cancer	14 25	lung cancer
        # end_test_annotation_overlapping_ancestors
        self.assertEqual("lung cancer	14 25	lung cancer", str(annots[0]))
        self.assertEqual("lung	14 18	lung", str(annots_2[0]))

    def test_annotation_overlapping_not_ancestors(self):
        """Case of overlapping but not an ancestor."""
        # start_test_annotation_overlapping_not_ancestors
        from iamsystem import Matcher

        matcher = Matcher.build(
            keywords=["North America", "South America"], w=3
        )
        text = "North and South America"
        annots = matcher.annot_text(text=text)
        for annot in annots:
            print(annot)
        # North America	0 5;16 23	North America
        # South America	10 23	South America
        # end_test_annotation_overlapping_not_ancestors
        self.assertEqual(
            "North America	0 5;16 23	North America", str(annots[0])
        )
        self.assertEqual("South America	10 23	South America", str(annots[1]))

    def test_annotation_partial_overlap(self):
        """Case of partial annotation overlapping that have a word
        in common."""
        # start_test_annotation_partial_overlap
        from iamsystem import Matcher

        matcher = Matcher.build(keywords=["lung cancer", "cancer prognosis"])
        annots = matcher.annot_text(text="lung cancer prognosis")
        for annot in annots:
            print(annot)
        # lung cancer	0 11	lung cancer
        # cancer prognosis	5 21	cancer prognosis
        # end_test_annotation_partial_overlap
        self.assertEqual("lung cancer	0 11	lung cancer", str(annots[0]))
        self.assertEqual(
            "cancer prognosis	5 21	cancer prognosis", str(annots[1])
        )

    # TODO: how to handle overlapping annotations ?
    # def test_replace_annots(self):
    #     from iamsystem import Matcher, Term, Annotation, replace_annots
    #     matcher = Matcher()
    #     ent1 = Term(label="North America", code="NA")
    #     ent2 = Term(label="South America", code="SA")
    #     matcher.add_keywords(keywords=[ent1, ent2])
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
        # start_test_brat_document
        from iamsystem import BratDocument
        from iamsystem import Entity
        from iamsystem import Matcher

        ent1 = Entity(label="North America", kb_id="NA")
        matcher = Matcher.build(keywords=[ent1], w=3)
        text = "North and South America"
        annots = matcher.annot_text(text=text)
        brat_document = BratDocument()
        brat_document.add_annots(
            annots, text=text, brat_type="CONTINENT", keyword_attr=None
        )
        print(str(brat_document))
        # T1	CONTINENT 0 5;16 23	North America
        # #1	IAMSYSTEM T1	North America (NA)
        # end_test_brat_document
        self.assertEqual(
            "T1\tCONTINENT 0 5;16 23\tNorth America\n#1\tIAMSYSTEM T1\tNorth "
            "America (NA)",
            str(brat_document),
        )

    def test_brat_doc_keyword(self):
        """Brat document example with a custom Keyword that stores
        brat_type."""
        # start_test_brat_doc_keyword
        from iamsystem import Entity

        class Entity(Entity):
            def __init__(self, label: str, code: str, brat_type: str):
                super().__init__(label, code)
                self.brat_type = brat_type

        from iamsystem import BratDocument
        from iamsystem import Matcher

        ent1 = Entity(label="North America", code="NA", brat_type="CONTINENT")
        matcher = Matcher.build(keywords=[ent1], w=3)
        text = "North and South America"
        annots = matcher.annot_text(text=text)
        brat_document = BratDocument()
        brat_document.add_annots(
            annots=annots, text=text, keyword_attr="brat_type"
        )
        print(str(brat_document))
        # T1	CONTINENT 0 5;16 23	North America
        # #1	IAMSYSTEM T1	North America (NA)
        # end_test_brat_doc_keyword
        self.assertEqual(
            "T1\tCONTINENT 0 5;16 23\tNorth America\n#1\tIAMSYSTEM T1\tNorth "
            "America (NA)",
            str(brat_document),
        )

    def test_brat_writer(self):
        """BratWriter example."""
        # start_test_brat_writer
        import os
        import tempfile

        from iamsystem import BratDocument
        from iamsystem import BratWriter
        from iamsystem import Entity
        from iamsystem import Matcher

        ent1 = Entity(label="North America", kb_id="NA")
        matcher = Matcher.build(keywords=[ent1], w=3)
        text = "North and South America"
        annots = matcher.annot_text(text=text)
        doc = BratDocument()
        doc.add_annots(annots, text=text, brat_type="CONTINENT")
        temp_path = tempfile.mkdtemp()
        os.makedirs(temp_path, exist_ok=True)
        filename = os.path.join(temp_path, "docs.ann")
        with (open(filename, "w")) as f:
            BratWriter.saveEntities(
                brat_entities=doc.get_entities(), write=f.write
            )
            BratWriter.saveNotes(brat_notes=doc.get_notes(), write=f.write)
        # end_test_brat_writer
        with (open(filename, "r")) as f:
            lines = f.readlines()
            self.assertEqual(
                lines[0], "T1	CONTINENT 0 5;16 23	North America\n"
            )
            self.assertEqual(lines[1], "#1	IAMSYSTEM T1	North America (NA)\n")


class FuzzyDocTest(unittest.TestCase):
    def test_abbreviations(self):
        """Abbreviations without customization."""
        # start_test_abbreviations
        from iamsystem import Entity
        from iamsystem import Matcher

        ent1 = Entity(label="acute respiratory distress", kb_id="J80")
        ent2 = Entity(label="patient", kb_id="D007290")
        ent3 = Entity(label="patient hospitalized", kb_id="D007297")
        ent4 = Entity(label="physiotherapy", kb_id="D007297")
        matcher = Matcher.build(
            keywords=[ent1, ent2, ent3, ent4],
            abbreviations=[
                ("Pt", "patient"),
                ("PT", "physiotherapy"),
                ("ARD", "Acute Respiratory Distress"),
            ],
        )
        annots = matcher.annot_text(
            text="Pt hospitalized with ARD. Treament: PT"
        )
        for annot in annots:
            print(annot.to_string(debug=True))
        # Pt hospitalized	0 15	patient hospitalized (D007297)	pt(abbs);hospitalized(exact) # noqa
        # ARD	21 24	acute respiratory distress (J80)	ard(abbs)
        # PT	36 38	patient (D007290)	pt(abbs)
        # PT	36 38	physiotherapy (D007297)	pt(abbs)
        # end_test_abbreviations
        self.assertEqual(4, len(annots))

    def test_uppercase(self):
        """Abbreviations that check uppercase."""
        # start_test_uppercase
        from iamsystem import Abbreviations
        from iamsystem import Entity
        from iamsystem import Matcher
        from iamsystem import TokenT
        from iamsystem import english_tokenizer

        def upper_case_only(token: TokenT) -> bool:
            """Return True if all token's characters are uppercase."""
            return token.label.isupper()

        def first_letter_capitalized(token: TokenT) -> bool:
            """Return True if the first letter is uppercase."""
            return token.label[0].isupper() and not token.label.isupper()

        tokenizer = english_tokenizer()
        ent1 = Entity(label="acute respiratory distress", kb_id="J80")
        ent2 = Entity(label="patient", kb_id="D007290")
        ent3 = Entity(label="patient hospitalized", kb_id="D007297")
        ent4 = Entity(label="physiotherapy", kb_id="D007297")
        matcher = Matcher.build(
            keywords=[ent1, ent2, ent3, ent4], tokenizer=tokenizer
        )

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
        # end_test_uppercase
        self.assertEqual(3, len(annots))

    def test_spellwise(self):
        """Spellwise library examples."""
        # start_test_spellwise
        from iamsystem import Entity
        from iamsystem import ESpellWiseAlgo
        from iamsystem import Matcher

        ent1 = Entity(label="acute respiratory distress", kb_id="J80")
        matcher = Matcher.build(
            keywords=[ent1],
            spellwise=[
                dict(
                    measure=ESpellWiseAlgo.LEVENSHTEIN,
                    max_distance=1,
                    min_nb_char=5,
                ),
                dict(measure=ESpellWiseAlgo.SOUNDEX, max_distance=1),
            ],
        )
        annots = matcher.annot_text(text="acute resiratory distresssss")
        for annot in annots:
            print(annot.to_string(debug=True))
        # acute resiratory distresssss	0 28	acute respiratory distress (J80) acute(exact,LEVENSHTEIN,SOUNDEX);resiratory(LEVENSHTEIN);distresssss(SOUNDEX) # noqa
        # end_test_spellwise
        self.assertEqual(1, len(annots))

    def test_string_distance_ignored_w(self):
        """Spellwise library examples."""
        # start_test_string_distance_ignored_w
        from iamsystem import ESpellWiseAlgo
        from iamsystem import Matcher

        matcher = Matcher.build(
            keywords=["poids"],
            spellwise=[
                dict(
                    measure=ESpellWiseAlgo.LEVENSHTEIN,
                    max_distance=1,
                    min_nb_char=4,
                )
            ],
        )
        annots = matcher.annot_text(text="Absence de poils.")
        for annot in annots:
            print(annot)
        matcher = Matcher.build(
            keywords=["poids"],
            spellwise=[
                dict(
                    measure=ESpellWiseAlgo.LEVENSHTEIN,
                    max_distance=1,
                    min_nb_char=4,
                )
            ],
            string_distance_ignored_w=["poils"],
        )
        # poils	11 16	poids
        annots_2 = matcher.annot_text(text="Absence de poils.")
        for annot in annots_2:
            print(annot)  # 0
        # end_test_string_distance_ignored_w
        self.assertEqual(1, len(annots))
        self.assertEqual(0, len(annots_2))

    def test_simstring(self):
        """Simstring example."""
        # start_test_simstring
        from iamsystem import Entity
        from iamsystem import Matcher
        from iamsystem.fuzzy.simstring import ESimStringMeasure

        ent1 = Entity(label="acute respiratory distress", kb_id="J80")
        matcher = Matcher.build(
            keywords=[ent1],
            simstring=[dict(measure=ESimStringMeasure.COSINE, threshold=0.7)],
        )
        annots = matcher.annot_text(text="acute respiratori disstress")
        for annot in annots:
            print(annot)
        # acute respiratori disstress	0 27	acute respiratory distress (J80)
        # end_test_simstring
        self.assertEqual(1, len(annots))

    def test_cache_fuzzy_algos(self):
        """Cache example."""
        # start_test_cache_fuzzy_algos
        from iamsystem import Abbreviations
        from iamsystem import CacheFuzzyAlgos
        from iamsystem import Entity
        from iamsystem import ESpellWiseAlgo
        from iamsystem import Matcher
        from iamsystem import SpellWiseWrapper

        ent1 = Entity(label="acute respiratory distress", kb_id="J80")
        matcher = Matcher.build(keywords=[ent1])
        abbs = Abbreviations(name="abbs")
        abbs.add(short_form="a", long_form="acute", tokenizer=matcher)
        test = dict(
            measure=ESpellWiseAlgo.LEVENSHTEIN, max_distance=1, min_nb_char=5
        )
        levenshtein = SpellWiseWrapper(**test)
        soundex = SpellWiseWrapper(ESpellWiseAlgo.SOUNDEX, max_distance=1)
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
        # end_test_cache_fuzzy_algos
        self.assertEqual(1, len(annots))

    def test_fuzzy_regex(self):
        """FuzzyRegex example."""
        # start_test_fuzzy_regex
        from iamsystem import Matcher
        from iamsystem import english_tokenizer
        from iamsystem import split_find_iter_closure

        tokenizer = english_tokenizer()
        tokenizer.split = split_find_iter_closure(pattern=r"(\w|\.|,)+")
        matcher = Matcher.build(
            keywords=["calcium numval mmol/L"],
            tokenizer=tokenizer,
            stopwords=["level", "is", "normal"],
            fuzzy_regex=[
                dict(
                    name="regex_num",
                    pattern=r"^\d*[.,]?\d*$",
                    pattern_name="numval",
                )
            ],
        )
        annots = matcher.annot_text(
            text="the blood calcium level is normal: 2.1 mmol/L"
        )
        for annot in annots:
            print(annot)
        # calcium 2.1 mmol L	10 17;35 45	calcium numval mmol/L
        # end_test_fuzzy_regex
        self.assertEqual(1, len(annots))

    def test_fuzzy_regex_negative_stopwords(self):
        """combine NegativeStopwords with FuzzyRegex."""
        # start_test_fuzzy_regex_negative_stopwords
        from iamsystem import Matcher
        from iamsystem import english_tokenizer
        from iamsystem import split_find_iter_closure

        tokenizer = english_tokenizer()
        tokenizer.split = split_find_iter_closure(pattern=r"(\w|\.|,)+")
        matcher = Matcher.build(
            keywords=["calcium numval mmol/L"],
            tokenizer=tokenizer,
            negative=True,
            fuzzy_regex=[
                dict(
                    name="regex_num",
                    pattern=r"^\d*[.,]?\d*$",
                    pattern_name="numval",
                )
            ],
        )
        annots = matcher.annot_text(
            text="the blood calcium level is normal: 2.1 mmol/L"
        )
        for annot in annots:
            print(annot)
        # calcium 2.1 mmol L	10 17;35 45	calcium numval mmol/L
        # end_test_fuzzy_regex_negative_stopwords
        self.assertEqual(1, len(annots))

    def test_word_normalization(self):
        """Stemming example."""
        # start_test_word_normalization
        from nltk.stem.snowball import FrenchStemmer

        from iamsystem import Entity
        from iamsystem import Matcher
        from iamsystem import french_tokenizer

        ent1 = Entity(label="cancer de la prostate", kb_id="C61")
        stemmer = FrenchStemmer()
        matcher = Matcher.build(
            keywords=[ent1],
            tokenizer=french_tokenizer(),
            stopwords=["de", "la"],
            normalizers=[dict(name="french_stemmer", norm_fun=stemmer.stem)],
        )
        annots = matcher.annot_text(text="cancer prostatique")
        for annot in annots:
            print(annot)
        # cancer prostatique	0 18	cancer de la prostate (C72)
        # end_test_word_normalization
        self.assertEqual(1, len(annots))


class SpacyDocTest(unittest.TestCase):
    def test_component(self):
        """Test detection with component."""
        # start_test_component
        from typing import Iterable
        from typing import List

        import spacy

        from spacy.lang.fr import French

        from iamsystem import Abbreviations
        from iamsystem import Entity
        from iamsystem import FuzzyAlgo
        from iamsystem import IKeyword
        from iamsystem import IStopwords
        from iamsystem import Terminology
        from iamsystem import french_tokenizer
        from iamsystem.spacy import IAMsystemSpacy  # noqa
        from iamsystem.spacy import IsStopSpacy
        from iamsystem.spacy import TokenSpacyAdapter

        @spacy.registry.misc("umls_ents.v1")
        def get_termino_umls() -> Iterable[IKeyword]:
            """An imaginary set of umls ents."""
            termino = Terminology()
            ent1 = Entity("Insuffisance Cardiaque", "I50.9")
            ent2 = Entity("Insuffisance Cardiaque Gauche", "I50.1")
            termino.add_keywords(keywords=[ent1, ent2])
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
                "keywords": {"@misc": "umls_ents.v1"},
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
        # end_test_component


if __name__ == "__main__":
    unittest.main()
