from __future__ import annotations

from researchbridge.extraction.validation import VALIDATABLE_CLAIM_TYPES, validate_claim_type


def test_results_sentence_mislabeled_as_research_gap_is_rejected() -> None:
    # The reported bug: a benchmark-results sentence surfaced under
    # "research_gap" in a real ResearchAssessment report.
    text = "Our model achieves 94.2% AUC and an F1 score of 0.89 on the benchmark."
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is False
    assert result.reason


def test_results_sentence_with_boilerplate_future_work_mention_is_rejected() -> None:
    # A real production example: a pure results sentence ending in
    # boilerplate ("...paving the way for future research") must not pass
    # as a research gap just because "future research" appears in it.
    text = (
        "Our best model achieves a MAE of 1087 steps, 65% lower than the state "
        "of the art, proving the feasibility of the task and paving the way for future research."
    )
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is False


def test_results_sentence_with_literal_word_gap_is_rejected() -> None:
    # Another real production example: "closes X% of the gap between A and
    # B" is a performance-gap result, not a stated research gap - the
    # literal word "gap" must not be enough on its own when a metric is
    # also present.
    text = "The operator closes 15.4% of the gap between flagging nothing and a perfect filter (p=0.0010)."
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is False


def test_genuine_research_gap_sentence_is_accepted() -> None:
    text = "Extending this approach to multilingual settings remains an open problem for future work."
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is True


def test_method_sentence_mislabeled_as_applications_is_rejected() -> None:
    text = "We propose a transformer-based architecture trained on paired image-text data."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False
    assert result.reason


def test_results_sentence_mislabeled_as_applications_is_rejected() -> None:
    text = "The model outperforms all baselines, achieving state-of-the-art accuracy of 97.1%."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_genuine_application_sentence_is_accepted() -> None:
    text = "This method can be applied to real-time fraud detection in financial transactions."
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


def test_generic_usefulness_statement_is_not_enough_for_applications() -> None:
    # Sec 5's failure mode: a generic "this is useful" sentence with no
    # actual application/use-context named isn't an application either.
    text = "These findings could be useful for future studies in general."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_task_restatement_dressed_as_usefulness_is_rejected() -> None:
    # the reported bug: no actor, no institution, no downstream action -
    # just the paper's own predictive task restated as a gerund
    text = (
        "In this paper will deliberate various techniques of data mining "
        "which are useful for predicting performance level of students."
    )
    result = validate_claim_type("applications", text)

    assert result.is_valid is False
    assert result.reason


def test_software_applications_noun_sense_is_rejected() -> None:
    # the reported bug: "applications" meaning computer programs, not
    # "applications of this research" - a lexical homonym, not a weak signal
    text = (
        "These are aided by the automation of many procedures involved in "
        "typical student activities, which manage huge amounts of information "
        "collected through software applications for technology-oriented learning."
    )
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_application_naming_an_actor_is_accepted() -> None:
    text = "Deployed by banks to flag suspicious transactions for manual review by compliance teams."
    result = validate_claim_type("applications", text)

    assert result.is_valid is True
    assert result.tier == "strong"


def test_application_naming_a_downstream_action_is_accepted() -> None:
    text = "Can be used by dermatologists to prioritize which patients need urgent biopsy."
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


def test_application_naming_an_external_setting_is_accepted() -> None:
    # no actor/action keyword, but a genuine "in X" domain qualifier beyond
    # the bare task object - must still be accepted
    text = "Can be applied to accelerate drug discovery pipelines in pharmaceutical R&D."
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


def test_bare_predictive_task_with_no_qualifier_is_rejected() -> None:
    # same shape as the reported bug but phrased with "applied to" instead
    # of "useful for" - must still be rejected, since the complement names
    # nothing beyond the task's own object
    text = "This approach can be applied to predicting customer churn."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_method_language_with_used_to_is_still_rejected() -> None:
    # "used" is a recognized deployment verb, but "used to train" is method
    # language, not a deployment claim - the complement must still clear
    # the actor/action/qualifying-context requirement
    text = "The dataset was used to train the model on benchmark image data."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_vague_qualifier_still_rejected_even_with_a_trailing_in_phrase() -> None:
    # "in general" must not be mistaken for a genuine domain qualifier
    text = "These findings could be useful for future studies in general."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_results_claim_without_metric_language_is_rejected() -> None:
    text = "This paper studies the problem of efficient graph coloring."
    result = validate_claim_type("results", text)

    assert result.is_valid is False


def test_results_claim_with_achievement_language_is_accepted() -> None:
    text = "Experimental results show our method outperforms the strongest baseline."
    result = validate_claim_type("results", text)

    assert result.is_valid is True


def test_limitations_claim_needs_limitation_language() -> None:
    text = "We propose a graph-based method for X."
    result = validate_claim_type("limitations", text)

    assert result.is_valid is False


def test_genuine_limitation_is_accepted() -> None:
    text = "However, the approach does not scale to graphs with more than a million nodes."
    result = validate_claim_type("limitations", text)

    assert result.is_valid is True


def test_method_claim_needs_method_language() -> None:
    text = "Coordination across regions is expensive and error-prone."
    result = validate_claim_type("method", text)

    assert result.is_valid is False


def test_genuine_method_claim_is_accepted() -> None:
    text = "We propose a lock-free queue for high throughput systems."
    result = validate_claim_type("method", text)

    assert result.is_valid is True


def test_problem_statement_with_results_language_is_rejected() -> None:
    text = "Our model achieves 99% accuracy on the held-out test set."
    result = validate_claim_type("problem", text)

    assert result.is_valid is False


def test_genuine_problem_statement_is_accepted() -> None:
    text = "Coordination across regions is expensive and error-prone."
    result = validate_claim_type("problem", text)

    assert result.is_valid is True


def test_unvalidated_claim_type_is_always_accepted() -> None:
    # dataset/research_question/main_contribution/the stub extractor's
    # "contribution" have no lexicon here (Sec 28's minimum-scope list) -
    # they're passed through unvalidated, not silently "validated ok".
    text = "Our model achieves 94.2% AUC and an F1 score of 0.89 on the benchmark."
    result = validate_claim_type("dataset", text)

    assert result.is_valid is True


def test_validatable_claim_types_covers_the_minimum_scope_fields() -> None:
    assert VALIDATABLE_CLAIM_TYPES == {
        "problem",
        "method",
        "results",
        "limitations",
        "research_gap",
        "applications",
    }


def test_strong_gap_language_sets_strong_tier() -> None:
    text = "Extending this approach to multilingual settings remains an open problem for future work."
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is True
    assert result.tier == "strong"


def test_weak_gap_language_sets_weak_tier() -> None:
    text = "This finding suggests a promising direction for future research in the area."
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is True
    assert result.tier == "weak"


def test_non_research_gap_types_have_no_tier() -> None:
    text = "However, the approach does not scale to graphs with more than a million nodes."
    result = validate_claim_type("limitations", text)

    assert result.is_valid is True
    assert result.tier is None


def test_rejected_claim_has_no_tier() -> None:
    text = "Our model achieves 94.2% AUC and an F1 score of 0.89 on the benchmark."
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is False
    assert result.tier is None
