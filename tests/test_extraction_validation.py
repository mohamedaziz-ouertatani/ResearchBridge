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


def test_computed_technical_gap_is_rejected_even_with_no_competing_metric() -> None:
    # real production example (2026-09-03 investigation, found reviewing an
    # actual exported ResearchAssessment): "computes the gap between
    # Average Daily Demand and Instantaneous Demand" is the paper's own
    # method - a numeric difference it calculates, not a stated gap in the
    # literature. Distinct from test_results_sentence_with_literal_word_gap
    # _is_rejected above: there the giveaway was a competing metric/percent
    # (_has_result_signal), here there is none at all - the giveaway is the
    # governing verb ("computes ... the gap") describing a calculation.
    text = (
        "This paper presents a novel solution that is based on an Artificial Intelligence Agent "
        "that continuously computes the gap between “Average Daily Demand” and "
        "“Instantaneous Demand” of a consumer, and allows the Battery Banks to discharge "
        "just enough to fill the gaps and eliminate kinks in the energy usage graph."
    )
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is False


def test_gap_between_is_still_accepted_when_not_a_computation_verb() -> None:
    # regression guard: the new technical-gap-metric check must not reject
    # every "gap between" sentence via the weak tier - only ones governed
    # by a computation verb (computes/calculates/measures/determines)
    # immediately before it. No strong-tier language here either, so this
    # exercises the weak-tier path specifically.
    text = "There is a significant gap between the accuracy achievable in theory and what current systems reach in practice."
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is True
    assert result.tier == "weak"


def test_open_question_answering_task_name_is_not_a_research_gap() -> None:
    # real production example (2026-09-04 investigation, found live-testing
    # the assessment pipeline with real ideas): "open (question|problem)"
    # matched "six existing open question answering datasets" - a standard
    # NLP task name ("open question answering"), not a stated gap. The
    # claim actually describes the paper's own benchmark contribution.
    # Verified against the whole corpus before narrowing: every other
    # "open question"/"open problem" occurrence sampled was genuine gap
    # language - only "answering" immediately after was a false positive.
    text = (
        "To address this, we present MultiMedQA, a benchmark combining six existing open "
        "question answering datasets spanning professional medical exams, research, and "
        "consumer queries; and HealthSearchQA, a new free-response dataset of medical "
        "questions searched online."
    )
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is False


def test_open_question_is_still_accepted_when_not_a_task_name() -> None:
    # regression guard: the new lookahead must not reject every "open
    # question"/"open problem" - only the "... answering" task-name shape
    for text in [
        "Whether such capabilities generalize to unseen domains remains an open question.",
        "This is an open problem in the field, and future work should address it.",
        "An open question in previous studies is how to filter out common memorization.",
    ]:
        result = validate_claim_type("research_gap", text)
        assert result.is_valid is True, text
        assert result.tier == "strong", text


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


def test_application_informing_decisions_is_accepted_as_a_downstream_action() -> None:
    # real corpus case found investigating applications' 97% corpus-wide
    # rejection rate (2026-09-04): "zLend is deployed in production,
    # informing real lending decisions via third-party API integrations"
    # is unambiguous deployment language, but the curated phrase "decision
    # support" doesn't literally appear - only its meaning ("informing...
    # decisions"). Matches the shape, not a fixed vocabulary.
    text = "zLend is deployed in production, informing real lending decisions via third-party API integrations."
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


def test_deployed_through_names_a_real_collaboration() -> None:
    # real corpus case (Fix D, 2026-09-04): "deployed" only recognized
    # (in|to|for|by) as its preposition - "deployed THROUGH collaboration
    # between WeBank and Extreme Vision" never matched the verb clause at
    # all, so the named-companies check never ran.
    text = (
        "The platform has been deployed through collaboration between WeBank and Extreme Vision "
        "to help customers develop computer vision-based safety monitoring solutions in smart city applications."
    )
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


def test_deployed_via_names_a_real_platform() -> None:
    # real corpus case (Fix D): same missing-preposition gap with "via"
    text = (
        "AI-HEALS is deployed via a WeChat Mini Program and features automated health-education "
        "content delivery, digital diaries for data logging, and intelligent Q&A functions."
    )
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


def test_deployed_via_a_bare_task_restatement_is_still_rejected() -> None:
    # regression guard: adding "via"/"through" must not reopen the
    # original bare-task-restatement bug - "via" doesn't naturally
    # introduce a gerund describing the system's own task the way "to"
    # does, but verify the full validator still rejects it if it tried
    text = "This approach can be deployed via predicting customer churn."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_application_naming_an_external_setting_is_accepted() -> None:
    # no actor/action keyword, but a genuine "in X" domain qualifier beyond
    # the bare task object - must still be accepted
    text = "Can be applied to accelerate drug discovery pipelines in pharmaceutical R&D."
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


def test_qualifying_context_fallback_only_acceptance_is_tier_weak() -> None:
    # the documented "in QPE" vs "in NHS" collision: both are accepted (Gate
    # 1 has no paper context to tell a paper's own acronym from a real
    # institution), but the acceptance path is the weakest of the four
    # (no named actor/institution/downstream-action/enumeration), so it's
    # tier="weak" - Gate 2 (assessment/applications.py, which HAS paper
    # context) uses this to apply extra scrutiny. Same code path either way.
    for text in [
        "The method is useful for improving forecasts, particularly for QPE.",
        "The method is useful for improving healthcare outcomes, particularly for NHS.",
    ]:
        result = validate_claim_type("applications", text)
        assert result.is_valid is True, text
        assert result.tier == "weak", text


def test_named_actor_acceptance_is_still_tier_strong() -> None:
    # regression guard: the weak-tier addition must not downgrade genuine
    # actor/institution/downstream-action/enumeration acceptances
    text = "Deployed by banks to flag suspicious transactions for manual review by compliance teams."
    result = validate_claim_type("applications", text)

    assert result.is_valid is True
    assert result.tier == "strong"


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


def test_a_later_qualifying_occurrence_is_still_found_after_an_earlier_failed_one() -> None:
    # real corpus case: the first "used in X" clause names nothing
    # external, but a later "used as X in Y" clause in the same sentence
    # does - matching must not stop at the first (failing) occurrence
    text = (
        "Simulating self-heating effects in Fin field-effect transistors used in modern "
        "integrated circuits; the improvements are expected to benefit devices used as "
        "solid-state synapses in neuromorphic computing circuits."
    )
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


def test_enumeration_is_accepted_without_a_curated_actor_word() -> None:
    # real corpus case: a multi-item "applications such as" list is
    # unambiguous evidence on its own, even though none of "object
    # search"/"robot navigation"/"augmented reality" is a curated actor
    text = (
        "Visual grounding has widespread applications such as object search, video "
        "analysis, automation, robot navigation, and augmented reality."
    )
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


def test_enumeration_with_a_comma_before_such_as_is_accepted() -> None:
    # Fix C (2026-09-04, found building a claim-revalidation backfill): the
    # original enumeration alternative required "such as"/"including"
    # immediately after "application(s)" with no comma - ordinary written
    # English ("...numerous real-world applications, such as robotics,
    # autonomous vehicles...") was silently rejected as having no
    # deployment context at all. Real corpus case.
    text = (
        "Instance segmentation is an important pre-processing task in numerous "
        "real-world applications, such as robotics, autonomous vehicles, and "
        "human-computer interaction."
    )
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


def test_enumeration_with_an_intervening_phrase_before_such_as_is_accepted() -> None:
    # real corpus case: "applications OF KRL, such as..." - a short
    # prepositional phrase (of/for/in + noun) between "application(s)" and
    # the enumeration trigger, previously unmatched
    text = (
        "We also review the real-world applications of KRL, such as language "
        "modeling, question answering, information retrieval, and recommender systems."
    )
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


def test_method_enumeration_is_still_rejected_even_though_it_mentions_applications() -> None:
    # regression guard: Fix C's widened "applications ... such as" pattern
    # must not swallow a METHOD/tool enumeration merely because the word
    # "application(s)" or "applied" appears elsewhere in the same
    # sentence - the trigger must still be anchored to "application(s)"
    # itself, immediately or via a short of/for/in phrase, not any word
    # in the sentence
    for text in [
        "Parallel processing infrastructure, such as Hadoop, and programming "
        "models, such as MapReduce, are being used to promptly process that "
        "amount of data.",
        "Machine learning classifiers, including Support Vector Machines (SVM), "
        "Extreme Gradient Boosting (XGBoost), and Random Forest (RF), were "
        "applied to enhance performance.",
        "eXplainable AI (XAI) techniques, such as SHAP and LIME, are applied to "
        "analyze predictions and identify key linguistic features.",
    ]:
        result = validate_claim_type("applications", text)
        assert result.is_valid is False, text


def test_abbreviation_period_does_not_truncate_the_qualifying_context() -> None:
    # real corpus case: "e.g." mid-sentence must not cut off "organization"
    # (an actor) that appears later in the same clause
    text = (
        "Small-sized domain-specific pre-trained data can be especially useful in "
        "practice when models need to be retrained, e.g. due to data that is "
        "confidential to an organization."
    )
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


def test_generic_given_user_qualifier_is_rejected() -> None:
    # Fix C: real residual false positive noted in assessment/applications
    # .py's OWN_TASK_OVERLAP_THRESHOLD comment - "for a given user" reads
    # as "for" + a following word, satisfying _QUALIFYING_CONTEXT_RE's
    # fallback, but names no actor/institution/setting beyond the system's
    # own subject - a bare task restatement, same failure mode as the
    # reported "useful for predicting performance level of students" bug
    text = "This model is useful for predicting future trip patterns for a given user."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_generic_individual_users_qualifier_is_rejected() -> None:
    text = "The system is applicable to personalizing recommendations for individual users."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_a_named_actor_after_a_generic_user_phrase_is_still_accepted() -> None:
    # the vague-user exclusion must not blind the checker to a genuine
    # named actor appearing elsewhere in the same clause
    text = "Can be used by clinicians to support a given user in adjusting their treatment plan."
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


# --- Traffic-congestion-idea investigation regression set (2026-09-03) ---
# Gate 1 behavior for all 10 cases from that investigation, cases 7/8 being
# Fix B (structural, not vocabulary, additions - see module docstring's
# _DEPLOYMENT_CLAUSE_RE/_ACTOR_SETTING_RE comments for what changed and why).
# Case 1 (the enumeration false positive) is Gate 2's job - see
# test_assessment_applications.py - Gate 1 is intentionally permissive of
# enumeration on its own, since it has no paper context to know an
# enumerated item restates the paper's own task.


def test_traffic_case2_bare_task_restatement_is_rejected() -> None:
    text = "The proposed model is useful for predicting student performance."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_traffic_case3_software_applications_noun_sense_is_rejected() -> None:
    text = "software applications for technology-oriented learning"
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_traffic_case4_generic_future_research_usefulness_is_rejected() -> None:
    text = "This approach could be useful for future research."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_traffic_case5_generic_domain_statement_is_rejected() -> None:
    text = "This method has applications in machine learning and artificial intelligence."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_traffic_case6_research_task_restatement_is_rejected() -> None:
    # "the model" + "applied to" + a bare task complement with no actor,
    # institution, or qualifying context beyond the task's own object -
    # same shape as case2, phrased with a different verb
    text = "The model can be applied to traffic congestion prediction."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_traffic_case7_deployment_by_a_named_authority_is_accepted() -> None:
    # Fix B: "authorities" added to the actor/institution lexicon -
    # structurally the same category as the already-present "agencies" /
    # "regulators" / "government", not a new kind of signal
    text = "The system can be deployed by city traffic authorities to monitor congestion and optimize traffic flow."
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


def test_traffic_case8_support_actor_in_downstream_action_is_accepted() -> None:
    # Fix B: new structural "support <actor> in <downstream action>-ing"
    # pattern, matched by word-class not a curated actor/action vocabulary
    # - "commuters" and "selecting alternative routes" are never named
    # explicitly in the regex, only the grammatical shape is
    text = "The predictions can support commuters in selecting alternative routes during periods of heavy congestion."
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


def test_traffic_case9_named_deployment_domain_is_accepted() -> None:
    text = "The method can be applied in intelligent transportation systems for real-time traffic management."
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


def test_traffic_case10_own_contribution_disguised_as_application_is_rejected() -> None:
    text = "We develop an application that predicts traffic congestion using LSTM."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_qpe_bare_policy_evaluation_restating_its_own_rl_algorithm_is_rejected() -> None:
    # Root cause of the original QPE bug report: QPE is this paper's OWN
    # algorithm name (Quantum Policy Evaluation, an RL algorithm - see its
    # own "problem" claim), and "policy evaluations" here means the RL
    # sense, not governance. Before the negative lookahead on "policy" in
    # _DOWNSTREAM_ACTION_RE, bare "policy" matched unconditionally and this
    # claim was wrongly tagged "strong", skipping Gate 2's weak-tier
    # scrutiny entirely. Verified against the whole corpus: "policy
    # evaluation(s)" is standalone-RL terminology in all 23 real claims
    # containing that exact phrase, never governance.
    text = "The learned quantum environment is then applied in QPE to also compute policy evaluations on quantum hardware."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_genuine_policy_making_downstream_action_is_still_accepted() -> None:
    # The QPE fix's negative lookahead only excludes "policy evaluation(s)"
    # - "policy making" (governance, no RL ambiguity) must keep matching as
    # a downstream action.
    text = "Further, it offers a theoretical framework that could be useful for a further study in policy making on the issue."
    result = validate_claim_type("applications", text)

    assert result.is_valid is True
    assert result.tier == "strong"


def test_support_verb_alone_without_downstream_gerund_is_still_rejected() -> None:
    # Fix B's new "support X in Y-ing" pattern must not become a bare
    # "support" bypass - no "in <gerund>" complement, so no signal fires
    text = "This library provides broad support for many use cases."
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
    # research_question/main_contribution/the stub extractor's
    # "contribution" have no lexicon here (Sec 28's minimum-scope list) -
    # they're passed through unvalidated, not silently "validated ok".
    text = "This text contains no relevant language for any validated claim type at all."
    result = validate_claim_type("research_question", text)

    assert result.is_valid is True


def test_validatable_claim_types_covers_the_minimum_scope_fields() -> None:
    # "dataset" joined 2026-09-04 - see validation.py's module docstring
    assert VALIDATABLE_CLAIM_TYPES == {
        "problem",
        "method",
        "results",
        "limitations",
        "research_gap",
        "applications",
        "dataset",
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


def test_bare_future_work_boilerplate_is_weak_not_strong() -> None:
    # real production example (2026-09-04 investigation, found reviewing a
    # real exported ResearchAssessment): this sentence names no actual
    # unresolved problem - it's a closing-section "here's what we'll do
    # next" statement, indistinguishable in kind from the already-weak
    # "future research" example above (test_weak_gap_language_sets_weak_
    # tier). It was tier "strong" (literal "future work" match) before
    # this fix, and was the deciding signal behind a real "confidence:
    # high" verdict driven by boilerplate - exactly the failure mode
    # is_strongly_stated exists to catch, see gap.py's own docstring.
    text = (
        "Future work will focus on integrating blockchain-based audit trails and federated "
        "learning for enhanced privacy and cross-institutional fraud intelligence sharing."
    )
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is True
    assert result.tier == "weak"


def test_future_work_naming_a_real_limitation_is_still_strong() -> None:
    # "future work" isn't disqualified outright - a sentence that both
    # says "future work" AND independently states a genuine unresolved
    # problem (here: "we leave") still reads as strong, same as before
    text = "We leave the extension to multi-GPU settings for future work."
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is True
    assert result.tier == "strong"


def test_meta_discourse_gap_boilerplate_is_rejected() -> None:
    # real production example (2026-09-05 investigation, a federated-
    # learning healthcare assessment): this sentence names zero actual
    # content - it's a table-of-contents-style self-reference ("here's the
    # section where we talk about this"), not a stated unresolved problem.
    # Matched _WEAK_GAP_LANGUAGE_RE on "future work" alone before this fix.
    text = "We conclude the paper by discussing the existing gaps and future work in an e-healthcare system"
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is False


def test_future_work_is_discussed_passive_boilerplate_is_rejected() -> None:
    text = "Future work is discussed in Section 6."
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is False


def test_the_paper_discusses_existing_gaps_boilerplate_is_rejected() -> None:
    text = "The paper discusses existing gaps and future research."
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is False


def test_genuine_weak_tier_future_research_with_substantive_content_is_accepted() -> None:
    # regression guard: the meta-discourse guard must not reject an ordinary
    # weak-tier claim just because it mentions "future research" - only the
    # self-referential "the paper discusses gaps/future work" shape.
    text = "This limitation remains unresolved and motivates future research into cross-domain generalization."
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is True


def test_strong_tier_sentence_mentioning_discuss_is_still_accepted() -> None:
    # regression guard: the meta-discourse guard only applies to the
    # weak-tier path (per validation.py's existing "strong tier is trusted
    # either way" stance for other weak-tier-only guards) - a sentence with
    # unambiguous strong-tier gap language must still pass even if it
    # happens to use the word "discuss".
    text = "We discuss why this remains an open problem."
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is True
    assert result.tier == "strong"


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


def test_application_is_weakly_grounded_matches_the_persisted_tier() -> None:
    # the public re-derivation Gate 2 calls must agree with what
    # validate_claim_type itself persists as validation_tier="weak"
    from researchbridge.extraction.validation import application_is_weakly_grounded

    weak_text = "The method is useful for improving forecasts, particularly for QPE."
    strong_text = "Deployed by banks to flag suspicious transactions for manual review by compliance teams."

    assert application_is_weakly_grounded(weak_text) is True
    assert application_is_weakly_grounded(strong_text) is False


# --- dataset claim validation (added 2026-09-04) - see validation.py's own
# module docstring for the real false positive this fixes and the
# calibration sample it was checked against.


def test_generic_field_level_prose_mislabeled_as_dataset_is_rejected() -> None:
    # the actual reported bug: this sentence, extracted as claim_type
    # "dataset", alone drove a "technical_feasibility_level: high" verdict
    # for an unrelated, vague idea via assessment/feasibility.py
    text = (
        "By harnessing machine learning algorithms, natural language processing, and "
        "computer vision, AI enables the analysis of complex medical data."
    )
    result = validate_claim_type("dataset", text)

    assert result.is_valid is False
    assert result.reason


def test_named_dataset_is_accepted() -> None:
    text = (
        "We conducted recognition experiments on three subsets of the CASIA-OLHWDB1.1 "
        "dataset: digits, English upper letters, and Chinese radicals."
    )
    result = validate_claim_type("dataset", text)

    assert result.is_valid is True


def test_named_database_without_the_word_dataset_is_accepted() -> None:
    text = "Database creation was enabled by the search engine tool for DIII-D data, TokSearch."
    result = validate_claim_type("dataset", text)

    assert result.is_valid is True


def test_concrete_sample_count_is_accepted() -> None:
    text = "After data preparation, the final analytical dataset contains 140,053 observations."
    result = validate_claim_type("dataset", text)

    assert result.is_valid is True


def test_concrete_image_count_without_the_word_dataset_is_accepted() -> None:
    text = "Results: the model was trained on 596,980 images, including 426,674 SB images."
    result = validate_claim_type("dataset", text)

    assert result.is_valid is True


def test_named_benchmark_is_accepted() -> None:
    text = (
        "Empirical evaluations on standard benchmarks, including Split-CIFAR-10, "
        "Split-CIFAR-100, and Split-TinyImageNet, demonstrate that FedQCL outperforms "
        "state-of-the-art baselines."
    )
    result = validate_claim_type("dataset", text)

    assert result.is_valid is True


def test_method_description_mislabeled_as_dataset_is_rejected() -> None:
    text = (
        "In this paper, we propose an energy-aware low-rank student network construction "
        "framework based on truncated singular value decomposition and knowledge distillation."
    )
    result = validate_claim_type("dataset", text)

    assert result.is_valid is False


def test_motivation_sentence_mislabeled_as_dataset_is_rejected() -> None:
    text = (
        "Due to the communication bottleneck in distributed and federated learning "
        "applications, algorithms using communication compression have attracted "
        "significant attention and are widely used in practice."
    )
    result = validate_claim_type("dataset", text)

    assert result.is_valid is False


def test_bare_data_mention_without_quantity_or_name_is_rejected() -> None:
    # "data"/"information" alone is too generic - must name the dataset or
    # give a concrete quantity, see the module docstring
    text = "This model introduces a flexible structure that can deal with missing data."
    result = validate_claim_type("dataset", text)

    assert result.is_valid is False


def test_dataset_claim_has_no_tier() -> None:
    text = "After data preparation, the final analytical dataset contains 140,053 observations."
    result = validate_claim_type("dataset", text)

    assert result.is_valid is True
    assert result.tier is None
