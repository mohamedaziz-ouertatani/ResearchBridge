"""Calibrate the out-of-corpus detection threshold against the live corpus.

Motivation (2026-09-09): assessment/build.py needs to know when the corpus
has nothing relevant to say about an idea, so it can suppress the
paper-grounded narrative fields instead of answering from whatever
happened to be nearest. The first version of that guard reused
novelty.FAR_DISTANCE (0.65) against the NEAREST retrieved paper, and live
measurement showed that signal cannot do the job: a 13th-century
manuscript idea retrieved its nearest paper at 0.403, and even word salad
reached only 0.640, so the guard essentially never fired.

This script measures three candidate signals over a labelled set of real
ideas and reports which ones separate in-domain from out-of-domain. The
answer it produced on 2026-09-09 (78,796-paper corpus, all-MiniLM-L6-v2):

    signal      in-domain max   out-of-domain min   separable
    nearest         0.426             0.415         no (overlap)
    mean of 5       0.440             0.451         yes, margin 0.011
    mean of 10      0.463             0.491         yes, margin 0.028

Mean distance over the retrieved set is the usable signal, and top-10 has
the wider margin - which is why OUT_OF_CORPUS_MEAN_DISTANCE sits at 0.48,
roughly the midpoint of that gap. Nearest-distance is a statement about
one lucky match; the mean is a statement about whether the neighbourhood
as a whole is on topic, which is the question actually being asked.

Re-run this after any corpus growth or embedder change - the threshold is
calibrated to a specific corpus and model, not a universal constant. It is
deliberately a SEPARATE constant from novelty.FAR_DISTANCE, which gates
per-paper relevance in five modules; retuning that would change every
field's behaviour at once.

Read-only. Prints a table; writes nothing.

    python scripts/calibrate_out_of_corpus.py
"""

from __future__ import annotations

import statistics
import sys

from researchbridge.api.deps import get_embedder, get_session_factory
from researchbridge.config import load_config
from researchbridge.embedding.search import search_by_text

# Genuine research ideas this corpus should be able to speak to.
IN_DOMAIN = [
    "A retrieval-augmented generation system for legal contract review that flags unusual indemnity clauses with citations to source judgments.",
    "A graph neural network predicting drug-target binding affinity from protein contact maps and molecular graphs.",
    "Federated learning across hospitals with per-institution differential privacy calibrated to cohort size.",
    "Detecting pneumonia in chest radiographs with a convolutional network evaluated against radiologist baselines.",
    "Machine-learning interatomic potentials for lithium-argyrodite solid electrolytes trained on DFT calculations.",
    "Measuring deceptive alignment in multi-agent language model systems under reinforcement learning from human feedback.",
    "A transformer architecture for long-horizon multivariate time series forecasting with sparse attention.",
    "Self-supervised pretraining for 3D medical image segmentation using masked autoencoders.",
    "Quantifying gender bias in machine translation output across morphologically rich target languages.",
    "An energy-efficient spiking neural network accelerator for always-on keyword spotting on edge devices.",
    "Conformal prediction intervals for deep ensembles under covariate shift in clinical risk models.",
    "Neural architecture search under hardware latency constraints for mobile vision backbones.",
    "Contrastive learning of speech representations for low-resource African language ASR.",
    "Differentially private synthetic tabular data generation evaluated on downstream utility.",
    "A benchmark for evaluating code-generation models on repository-level refactoring tasks.",
    "Causal effect estimation from observational electronic health records using targeted learning.",
    "Reinforcement learning for adaptive traffic signal control in a microsimulation of an urban grid.",
    "Explainability methods for gradient-boosted decision trees in credit scoring under regulatory audit.",
]

# Genuine research ideas from fields with no computational angle at all.
OUT_OF_DOMAIN = [
    "Pigment analysis of the Winchester Bible to determine whether the Master of the Leaping Figures trained in Byzantium.",
    "A stratigraphic reassessment of Late Bronze Age destruction layers at Tell Tayinat using ceramic seriation.",
    "The prosody of enclitic pronouns in Homeric hexameter and its bearing on oral-formulaic composition theory.",
    "A field study of cleaner wrasse mutualism on degraded reefs following the 2016 bleaching event.",
    "Comparing single-incision versus conventional laparoscopic cholecystectomy on postoperative adhesion formation.",
    "A palladium-catalysed asymmetric route to the taxane core using a chiral phosphine ligand.",
    "Harmonic ambiguity in the late motets of Josquin des Prez and its implications for modal theory.",
    "Rereading the Confessions of Augustine through fourth-century North African rhetorical education.",
    "Ethnographic fieldwork on kinship obligation among transhumant pastoralists in the eastern Rif.",
    "Soil organic carbon dynamics under no-till rotation in semi-arid Mediterranean cereal systems.",
    "The effect of eccentric hamstring loading on sprint kinematics in elite adolescent footballers.",
    "Enamel demineralisation around orthodontic brackets bonded with resin-modified glass ionomer.",
    "Post-Ottoman waqf property registration and its influence on nineteenth-century Damascene urban form.",
    "Serological surveillance of bluetongue virus in Sardinian sheep flocks across three transmission seasons.",
    "Narrative unreliability in the late fiction of Machado de Assis and Brazilian abolitionist discourse.",
    "Reconstructing Holocene sea-level change from salt-marsh foraminifera in the Bay of Fundy.",
    "The role of guild statutes in regulating apprenticeship in fifteenth-century Florentine wool production.",
    "Nurse-led debriefing after in-hospital cardiac arrest and its effect on team psychological safety.",
]

TOP_K = 10


def _signals(distances: list[float]) -> dict[str, float]:
    return {
        "nearest": distances[0],
        "mean5": statistics.fmean(distances[:5]),
        "mean10": statistics.fmean(distances),
    }


def main() -> int:
    load_config()
    embedder = get_embedder()
    session = get_session_factory()()
    try:
        measured: dict[str, list[dict[str, float]]] = {"in": [], "out": []}
        for label, texts in (("in", IN_DOMAIN), ("out", OUT_OF_DOMAIN)):
            for text in texts:
                hits = search_by_text(session, text, embedder, TOP_K)
                if not hits:
                    print(f"no retrieval for {text[:50]!r} - is the corpus empty?")
                    return 1
                measured[label].append(_signals([distance for _paper, distance in hits]))
    finally:
        session.close()

    print(f"in-domain n={len(measured['in'])}  out-of-domain n={len(measured['out'])}  top_k={TOP_K}\n")
    print(f"{'signal':<10} {'in.max':>8} {'out.min':>9} {'margin':>8}  verdict")
    print("-" * 56)
    for signal in ("nearest", "mean5", "mean10"):
        in_max = max(row[signal] for row in measured["in"])
        out_min = min(row[signal] for row in measured["out"])
        margin = out_min - in_max
        verdict = (
            f"separable, midpoint {(in_max + out_min) / 2:.3f}" if margin > 0 else "OVERLAP - unusable"
        )
        print(f"{signal:<10} {in_max:>8.3f} {out_min:>9.3f} {margin:>8.3f}  {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
