from researchbridge.assessment.language import is_likely_non_english, is_likely_non_latin_script


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


def test_french_idea_is_flagged_as_likely_non_english() -> None:
    """The script heuristic misses Latin-script non-English entirely, so
    French/Spanish/German ideas got no reliability caveat. Found live
    2026-09-09: a French dermoscopy idea retrieved Spanish-language papers
    and the dimension extractor invented the non-word "Entrenchement",
    exactly the degraded retrieval the caveat exists to warn about."""
    text = (
        "Nous proposons un systeme d'apprentissage profond pour la detection automatique "
        "des lesions cutanees a partir d'images dermoscopiques, en utilisant des reseaux "
        "de neurones convolutifs entraines sur des donnees multicentriques."
    )

    assert is_likely_non_english(text) is True


def test_spanish_idea_is_flagged_as_likely_non_english() -> None:
    text = (
        "Proponemos un sistema de aprendizaje profundo para la deteccion automatica de "
        "lesiones en la piel a partir de imagenes dermoscopicas, utilizando redes "
        "neuronales convolucionales entrenadas con datos de varios centros."
    )

    assert is_likely_non_english(text) is True


def test_english_idea_is_not_flagged() -> None:
    text = (
        "We propose a deep learning system for the automatic detection of skin lesions "
        "from dermoscopic images, using convolutional neural networks trained on "
        "multicentre data from several hospitals."
    )

    assert is_likely_non_english(text) is False


def test_english_idea_dense_with_technical_terms_is_not_flagged() -> None:
    text = (
        "A machine-learning interatomic potential trained on density functional theory "
        "calculations of lithium-argyrodite solid electrolytes, to predict grain-boundary "
        "ionic conductivity at scales inaccessible to ab initio molecular dynamics."
    )

    assert is_likely_non_english(text) is False


def test_non_latin_script_is_also_reported_as_non_english() -> None:
    text = "نقترح نظام تعلم عميق للكشف التلقائي عن أمراض النباتات من صور الأوراق باستخدام الشبكات العصبية"

    assert is_likely_non_english(text) is True


def test_short_input_is_not_guessed_at() -> None:
    assert is_likely_non_english("Nous proposons") is False
