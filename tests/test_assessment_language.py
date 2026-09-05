from researchbridge.assessment.language import is_likely_non_latin_script


def test_english_idea_not_flagged():
    assert not is_likely_non_latin_script(
        "A method for using large language models to automatically generate unit tests."
    )


def test_short_text_not_flagged_even_if_non_latin():
    assert not is_likely_non_latin_script("ذكاء")


def test_arabic_idea_flagged():
    assert is_likely_non_latin_script(
        "طريقة جديدة لاستخدام الشبكات العصبية في تحليل النصوص العربية للكشف عن الأخبار الكاذبة"
    )


def test_mixed_latin_and_technical_terms_not_flagged():
    assert not is_likely_non_latin_script(
        "Using GPT-4 and BERT embeddings for transformer-based classification of clinical notes."
    )
