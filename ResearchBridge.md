# ResearchBridge — Revised Technical Blueprint & Execution Prompt for Claude

## Role

Act as a senior AI/ML architect, Data Scientist, NLP/LLM engineer, data engineer, backend architect, scientific research-methodology advisor, technology-transfer strategist, and academic project advisor.

You are helping design and implement **ResearchBridge**, a university-backed research intelligence platform.

Be rigorous and critical. Do not automatically agree with architectural decisions. Identify unnecessary complexity, weak assumptions, evaluation problems, data-quality risks, licensing problems, and unjustified AI claims. Prefer scientifically defensible and measurable solutions over flashy architecture.

---

## Builder Context (Read Before Anything Else)

**This project is being built by a single developer, not a team or a funded lab.**

This changes what counts as a realistic MVP. When proposing architecture or roadmap items, always weigh them against solo-build constraints:

- No dedicated domain-expert panel exists for validation. Early "human-in-the-loop" review is the builder's own judgment, plus occasional informal input from professors or peers — not a formal expert panel. Anything requiring a real expert panel (e.g. RQ6-style AI-vs-expert agreement studies) is future work, not part of the MVP.
- Prefer one generic, flexible table over many narrow entity-specific tables until extraction quality is proven. Splitting into dedicated tables (research_problems, research_methods, datasets, limitations, research_gaps, applications, etc.) is a Phase 3+ refactor, not a Phase 1 starting point.
- Start with exactly one data source (arXiv). Add a second source only once ingestion, normalization, and dedup work end-to-end on the first.
- Confidence values must be things a solo builder can actually justify — categorical (High/Medium/Low) tied to explicit rules, not decimal floats implying a calibration process that doesn't exist yet.
- The roadmap below is written in phases; treat "Phase 1" as the only phase with committed weekly milestones. Phases 2–5 stay directional until Phase 1 ships and is evaluated.

**Free/Open-Source First.** This is a project-wide constraint, not a suggestion — it applies to every phase and every architectural decision below:

- Prefer free and open-source software, libraries, frameworks, databases, models, and tools whenever a suitable option exists.
- Do not introduce paid APIs or cloud services by default.
- Anthropic/OpenAI or other paid LLM APIs must be treated as optional integrations, never core dependencies.
- The MVP must remain fully functional without any paid APIs.
- For LLM functionality, implement a provider abstraction and a free/local fallback wherever practical (see §29).
- During development and testing, prefer, in this order: local/open-source models, mock/stub providers, free API tiers where explicitly available, local PostgreSQL + pgvector, local processing, and free/open-source infrastructure generally.
- If a paid service would provide a significant advantage, propose it separately, state its cost/dependency explicitly, and do not make it mandatory without the builder's approval.
- Never assume an API is free — verify its current pricing/licensing before recommending it.
- Avoid vendor lock-in: keep every external provider behind a replaceable interface.
- Target state: the builder can develop and demonstrate the MVP with zero mandatory software/API costs, excluding optional hardware, electricity, or internet costs.

---

# 1. Project Overview

ResearchBridge addresses the gap between:

> **Scientific research → practical implementation → real-world application → product/technology → impact**

Scientific papers frequently contain valuable algorithms, models, methodologies, discoveries, datasets, optimization techniques, prototypes, and experimental results. However, research often stops at publication.

The result may never become:

- a deployable system,
- an industrial solution,
- a product,
- a service,
- a startup,
- a university spin-off,
- a technology-transfer project,
- or another real-world application.

The university wants a system that can help researchers and decision-makers determine:

1. Should we work on this research?
2. Is the problem actually novel?
3. Has this problem already been solved?
4. What related research already exists?
5. What has already been attempted?
6. What limitations remain?
7. What is the research gap?
8. What is scientifically interesting about the work?
9. What real-world problems could the research potentially address?
10. What applications could be built?
11. What product or technology opportunities could emerge?
12. Is the research technically feasible to develop further?
13. What resources would be required?
14. What risks exist?
15. What evidence supports the recommendation?
16. Which research projects should the university prioritize?

## How a user actually interacts with ResearchBridge

The platform's primary interaction is not "browse the corpus" — it is:

> A user provides a **ResearchInput**: either a free-text research idea/prompt,
> or an uploaded paper/document. ResearchBridge analyzes that input against the
> literature corpus and returns a **ResearchAssessment** — a single
> evidence-grounded report answering the questions above *for that specific
> input*.

Everything else in this document — the corpus, ingestion pipeline, retrieval
baselines, extraction pipeline, and the research-gap engine — exists to make
that one interaction possible and trustworthy. They are supporting
infrastructure, not independent products. Corpus-wide, automatic gap discovery
across every paper (§32's batch mode) is a valuable secondary capability that
pre-populates candidate observations for reuse inside assessments, but it is
not the user-facing product loop and must not be prioritized as if it were.

ResearchBridge is therefore a:

> **Research Intelligence and Research-to-Impact Decision Support Platform**

It is NOT merely a scientific-paper scraper, PDF summarizer, generic RAG chatbot, paper search engine, or LLM wrapper.

---

# 2. Core Concept

The central pipeline is:

```text
Scientific Literature
        ↓
Reliable Ingestion
        ↓
Structured Research Corpus
        ↓
Knowledge Extraction
        ↓
Semantic Retrieval
        ↓
Research Comparison
        ↓
Novelty Analysis
        ↓
Research Gap Detection
        ↓
Technical Applicability Assessment
        ↓
Application Discovery
        ↓
Opportunity Assessment
        ↓
Human Validation
        ↓
Research Prioritization
```

The project should ultimately bridge:

> **Research → Knowledge → Gap → Opportunity → Application → Product → Impact**

---

# 2A. Primary Product Workflow: ResearchInput → ResearchAssessment

This is the central end-user loop the rest of the architecture exists to support.

(Deliberately lettered rather than renumbering every subsequent heading for one insertion — renumber the whole document only if that becomes genuinely necessary.)

```text
User
  ↓
ResearchInput (idea/text prompt OR uploaded paper)
  ↓
Input Analysis (normalize into the same shape retrieval/extraction already use)
  ↓
Literature Search (Retrieval Engine, §26-27 — over the existing corpus)
  ↓
Retrieve Related Papers
  ↓
Compare Input Against Retrieved Papers (Extraction Engine + Research Similarity, §28-30)
  ↓
Novelty Assessment (what's already solved vs. what the input adds)
  ↓
Research Gap Identification (§32 — scoped to this input's neighborhood, not the whole corpus)
  ↓
Application Discovery (§33)
  ↓
Opportunity Assessment (§18-23, three-layer model)
  ↓
Technical Feasibility / Risk / Missing-Evidence Assessment
  ↓
ResearchAssessment Report (§49)
  ↓
Human Review (§35)
```

## Uploaded-Paper Path (input_type = document)

An idea input's `raw_text` is already usable directly as the basis for Input
Analysis. An uploaded document is not — a raw PDF/text dump isn't itself a
good retrieval query, so it needs its own extraction pass first, using the
same Extraction Engine (§28-29) already built for corpus papers:

```text
Uploaded Paper
  ↓
Text / section extraction
  ↓
Input Analysis
  ↓
Retrieval Query / Research Representation
  ↓
Related Corpus Papers
  ↓
Comparison
  ↓
ResearchAssessment
```

This runs **regardless of whether the upload matches an existing corpus
paper**. `matched_paper_id` (below) is optional and purely an optimization:
when the upload is identified as an already-ingested paper (e.g. by
title/DOI/arXiv id), the pipeline may reuse that paper's existing extracted
claims/evidence/embedding instead of re-running extraction on the upload.
When unmatched — the default case for a genuinely new document — the system
extracts problem/method/dataset/contribution directly from the uploaded
text to build its own research representation and retrieval query. A
corpus match is never required for an assessment to proceed.

## ResearchInput (first-class concept)

One user-submitted item to be assessed:

```text
research_inputs
----------------
id
input_type          -- idea | document
raw_text            -- the prompt itself, or the uploaded document's extracted text
title               NULL   -- inferred, or user-supplied
matched_paper_id     NULL   -- FK -> papers; optional, set only when matched (see Uploaded-Paper Path above) - never required
source_filename      NULL   -- original upload filename, if input_type = document
submitted_by         NULL   -- reserved for later multi-user support
created_at
```

## ResearchAssessment (first-class concept)

One analysis session over one ResearchInput — the object the UI is actually
built around, the same way one `candidate_gaps` row already represents one
detected pattern:

```text
research_assessments
---------------------
id
research_input_id            FK -> research_inputs
status                       -- pending | running | completed | failed
retrieved_paper_ids          -- ordered list/JSONB: the related-papers set used
comparison_summary           TEXT NULL   -- existing solutions, grounded in retrieved_paper_ids
novelty_level                 -- categorical: high | medium | low | not_assessed
novelty_reasoning            TEXT NULL
research_gap_text            TEXT NULL
research_gap_source           -- 'input_specific' | 'reused_candidate_gap'
candidate_gap_id              NULL FK -> candidate_gaps  -- set when reused
potential_applications        JSONB NULL
potential_opportunities       JSONB NULL
technical_feasibility_level   -- categorical: high | medium | low | not_assessed
risks_and_limitations         TEXT NULL
external_validation_needed    TEXT NULL
recommendation                 -- HIGH PRIORITY | MEDIUM PRIORITY | LOW PRIORITY | INSUFFICIENT EVIDENCE | REQUIRES HUMAN REVIEW
confidence                    -- categorical: high | medium | low
human_reviewed                 BOOLEAN default false
created_at
completed_at                  NULL
```

`research_gap_source` matters: an assessment should reuse an already-detected,
already-reviewed `candidate_gaps` row when the input's neighborhood overlaps
one (cheap, benefits from prior human review), but must be able to run
`detect_candidate_gaps` scoped to just this input's own neighborhood when
nothing existing applies — same machinery as §32, invoked per-assessment
rather than only via the corpus-wide batch job.

`research_assessments` is the single primary assessment object for the
MVP — not a summary layer sitting on top of a separate `opportunity_assessments`
persistence table. §41 describes a normalized three-layer breakdown
(scientific / technical / external), which is a plausible future refactor
once the MVP's flat categorical fields (`novelty_level`,
`technical_feasibility_level`, etc.) prove insufficient — it is not a
parallel model to build now, and nothing in the MVP should write to both.

## Evidence linkage — keep the assessment lightweight, not the source of truth

A `research_assessments` row must stay a lightweight MVP summary — the fields
above are what render in the UI/report, but they are never the ground truth.
The ground truth is real `evidence` rows already grounded in retrieved papers
(§15), linked explicitly:

```text
research_assessment_evidence
------------------------------
id
research_assessment_id        FK -> research_assessments
evidence_id                   FK -> evidence
role                           -- comparison | novelty | research_gap | application | feasibility | risk
created_at
```

Every populated text/categorical field in `research_assessments`
(`comparison_summary`, `novelty_reasoning`, `research_gap_text`,
`potential_applications`, `risks_and_limitations`, etc.) must be traceable to
at least one row here. This mirrors `candidate_gap_evidence`'s existing
pattern (§32) exactly — the same "never invent, always ground in a real
quoted passage" rule that already governs claims extraction (§15) and
candidate gaps (§32) applies to assessments too. If a ResearchAssessment
field cannot point to real evidence, it must be left `NULL`/`NOT_ASSESSED`
rather than filled with unsupported prose. `research_gap_text` additionally
inherits grounding from `candidate_gap_evidence` whenever
`research_gap_source = 'reused_candidate_gap'`.

---

# 3. Major Strategic Change: Do NOT Start With Heavy PDF Processing

## Risk

Scientific PDFs can be difficult to parse because of multi-column layouts, mathematical notation, equations, tables, figures, references, unusual formatting, scans, and embedded images.

Building a sophisticated PDF/OCR/GROBID pipeline immediately could consume a large part of the engineering effort before the core research-intelligence functionality is validated.

## New decision

For the MVP, prioritize **native structured APIs and JSON/XML endpoints**.

Examples:

- arXiv API
- Semantic Scholar API
- PubMed Central Open Access subset
- other legitimate structured scientific APIs/repositories

The primary Phase 1 flow should be:

```text
Scientific API
      ↓
JSON/XML
      ↓
Normalization
      ↓
Validation
      ↓
Deduplication
      ↓
PostgreSQL
```

rather than:

```text
Web page
      ↓
PDF
      ↓
OCR
      ↓
GROBID
      ↓
Text extraction
```

PDF processing remains a future or secondary ingestion path.

---

# 4. Source Strategy

For a solo build, the MVP should begin with **one** source, not a set.

## arXiv (Phase 1 — the only source)

Single API, generous rate limits, clean structured metadata, and it fully covers the initial CS/AI/ML scope. This alone is enough to prove the ingestion → normalization → dedup → corpus pipeline works.

Useful for:

- computer science,
- machine learning,
- AI,
- systems,
- abstracts,
- authors,
- categories,
- dates,
- identifiers.

## Semantic Scholar (Phase 1.5 / Phase 2 — add only after arXiv ingestion is stable)

Its main value is citation-relationship richness, which matters once the citation graph and retrieval baselines are already working on arXiv data alone. Do not add it in parallel with the first connector — the second connector should be built against an already-working `NormalizedPaper` pipeline, so it validates the abstraction rather than being built alongside it.

Useful for:

- metadata,
- abstracts,
- citation relationships,
- references,
- authors,
- paper relationships.

## PubMed Central Open Access

Keep as a later-expansion source, although it is less central to the initial CS/AI corpus.

## Future sources

Potential future sources include:

- CORE
- Crossref
- Springer Nature APIs
- IEEE
- ACM
- institutional repositories
- other legitimate scientific providers

Always respect:

- API terms,
- licensing,
- robots.txt,
- rate limits,
- publisher restrictions,
- copyright,
- access control,
- redistribution restrictions.

Do not assume that online availability means unrestricted bulk ingestion rights.

Where full text is unavailable, use metadata, abstracts, legitimate open-access copies, repository versions, or permitted API responses.

---

# 5. Source Connector Architecture

Each source should have an independent connector:

```text
ArxivConnector
SemanticScholarConnector
PubMedConnector
FutureSourceConnector
```

All connectors should convert source-specific responses into one canonical internal representation:

```text
NormalizedPaper
```

Downstream components must operate on the normalized representation rather than directly on individual APIs.

This makes the system extensible and avoids vendor/source-specific logic throughout the application.

---

# 6. Initial Corpus Scope

Do NOT attempt to ingest all scientific disciplines in the MVP.

Restrict the initial corpus to:

> **Computer Science / AI, especially Machine Learning and Systems**

A possible taxonomy:

```text
Computer Science
├── Machine Learning
│   ├── Deep Learning
│   ├── Natural Language Processing
│   ├── Computer Vision
│   ├── Reinforcement Learning
│   └── Generative AI
│
└── Systems
    ├── Distributed Systems
    ├── Cloud Computing
    ├── Databases
    └── Operating Systems
```

The taxonomy should remain extensible.

---

# 7. Phase 1 — Corpus and Data Foundation

## Objective

Build a reliable, reproducible, structured Computer Science / AI research corpus.

Phase 1 is NOT about advanced research-gap reasoning.

It is primarily about:

> **Data quality, provenance, normalization, deduplication, and benchmark preparation.**

---

# 8. Phase 1 Ingestion Pipeline

Initial pipeline:

```text
Scientific APIs
       ↓
Source connectors
       ↓
Schema normalization
       ↓
Metadata validation
       ↓
Deduplication
       ↓
PostgreSQL
       ↓
Citation relationships
       ↓
Basic embeddings
```

The ingestion system should store:

1. normalized fields,
2. source-specific raw metadata.

This preserves information that differs by provider.

## Ingestion Reliability (non-negotiable, even for a solo MVP)

A solo builder cannot afford silent data loss — there is no team to notice a quietly broken pipeline. The ingestion layer must, from week 1:

- **fail loudly, not silently**: a malformed API response, unexpected field, or schema drift must raise/log an explicit error, never get silently dropped or coerced into empty values,
- **handle pagination and rate limits explicitly**: retry with backoff on 429/5xx, and persist enough state to resume a partial ingestion run rather than restarting from zero,
- **record ingestion run metadata** (timestamp, source, page/cursor, record counts, error counts) so a broken run is discoverable without re-reading logs line by line.

This is cheap to build in from the start and expensive to retrofit once thousands of records exist with unknown gaps. It is a reliability requirement, not a "Phase 2 nice-to-have."

---

# 9. Canonical Paper Schema

Initial `papers` table:

```text
papers
------
id                  UUID PRIMARY KEY
source              VARCHAR
source_id           VARCHAR
doi                 VARCHAR NULL
title               TEXT
abstract            TEXT
publication_date    DATE
venue               TEXT
document_type       VARCHAR
language            VARCHAR
url                 TEXT
open_access         BOOLEAN
raw_metadata        JSONB
ingestion_metadata  JSONB
created_at          TIMESTAMP
updated_at          TIMESTAMP
```

Recommended uniqueness:

```text
(source, source_id)
```

Normalize DOI where available.

Example source-specific data preserved in `raw_metadata`:

```json
{
  "arxiv": {
    "primary_category": "cs.LG",
    "categories": ["cs.LG", "cs.AI"]
  }
}
```

---

# 10. Authors

Use normalized relational tables.

```text
authors
-------
id
name
orcid
metadata
```

Many-to-many:

```text
paper_authors
-------------
paper_id
author_id
author_order
```

---

# 11. Categories / Domains

Use:

```text
paper_categories
----------------
paper_id
category
confidence
source
```

Examples:

- cs.LG
- cs.AI
- cs.CL
- cs.CV
- cs.DB
- cs.DC

Later map source categories to the project's own domain taxonomy.

---

# 12. Citation Relationships Without Neo4j

Do NOT introduce Neo4j or another graph database during the initial MVP.

Use PostgreSQL:

```text
paper_citations
---------------
citing_paper_id
cited_paper_id
source
confidence
```

PostgreSQL recursive CTEs can support multi-hop traversal.

Example:

```sql
WITH RECURSIVE citation_graph AS (
    SELECT
        citing_paper_id,
        cited_paper_id,
        1 AS depth
    FROM paper_citations

    UNION ALL

    SELECT
        cg.citing_paper_id,
        pc.cited_paper_id,
        cg.depth + 1
    FROM citation_graph cg
    JOIN paper_citations pc
      ON cg.cited_paper_id = pc.citing_paper_id
    WHERE cg.depth < 3
)
SELECT *
FROM citation_graph;
```

Only move to a dedicated graph database if there is an empirically demonstrated:

- query-performance bottleneck,
- traversal complexity problem,
- data-modeling limitation,
- or scaling requirement

that PostgreSQL cannot handle efficiently.

Do NOT adopt Neo4j simply because the system contains graph-like relationships.

---

# 13. PostgreSQL + pgvector

Use:

> **PostgreSQL + pgvector**

as the initial persistence architecture.

PostgreSQL stores:

- papers,
- authors,
- categories,
- citations,
- extracted knowledge,
- evidence,
- annotations,
- assessments,
- provenance,
- JSONB metadata.

pgvector stores:

- paper embeddings,
- chunk/abstract embeddings,
- query embeddings,
- potentially entity embeddings.

This avoids maintaining multiple databases unnecessarily.

---

# 14. Knowledge Representation

## Phase 1 (solo build): one generic table, not ten

Do NOT create a dedicated relational table per concept type until extraction quality is proven on a benchmark. For a solo builder, ten narrow entity tables built before any extraction has been validated is speculative schema design — you don't yet know which categories need real relational structure (e.g. many-to-many dataset reuse across papers) versus which are just labeled text spans.

Start with one generic table:

```text
extracted_claims
-----------------
id
paper_id
claim_type        -- problem | method | dataset | metric | result | limitation | research_gap | application | contribution
text
evidence_id
confidence         -- categorical: high | medium | low
created_at
```

This still supports every Phase 1–2 use case (extraction evaluation, evidence linking, gap/application surfacing) without committing to a relational shape you haven't earned yet.

**Note: "one generic table" reduces schema complexity, not system complexity.** The real complexity in Phase 1 is the evidence/claims reasoning layer itself — `evidence` → `extracted_claims` → (later) `analysis_claims` → `claim_evidence`, each with its own provenance fields (§17). Collapsing the schema does not make that layered reasoning model simple; it stays the hardest part of the MVP. Don't let a simple schema create false confidence that the system is simple — budget engineering time for the reasoning layer accordingly, not just the tables.

## Phase 3+: split into dedicated tables once justified

Once the benchmark shows which claim types actually need first-class relational structure (e.g. `datasets` reused across many papers, `research_gaps` linked to multiple supporting papers), split them out:

```text
research_problems
research_methods
datasets
metrics
experiments
results
limitations
research_gaps
technologies
applications
```

With junction tables of the same shape as `paper_problems`, `paper_methods`, `paper_limitations`, `paper_gaps`, `paper_applications` (each: paper_id, entity_id, evidence_id, confidence) — this remains the target end-state, just not the Phase 1 starting point.

---

# 15. Critical New Architecture: Evidence as a First-Class Entity

ResearchBridge must avoid the:

> **Grounding Illusion**

An LLM can generate a plausible scientific-sounding claim without the underlying paper actually supporting it.

Create:

```text
evidence
--------
id
paper_id
evidence_type
section
text
source_locator
extraction_method
model_version
confidence
created_at
```

Possible `evidence_type`:

```text
problem
method
contribution
result
limitation
research_gap
application
dataset
claim
```

Example:

```text
paper_id:
P123

evidence_type:
limitation

section:
Discussion

text:
"The proposed system was evaluated only on static datasets."

source_locator:
page 8 / paragraph 3

extraction_method:
LLM

model_version:
model-x.y

confidence:
high   -- rule: explicit statement located in a named section (Discussion/Limitations), not paraphrased or inferred
```

**Confidence is categorical (high/medium/low), not a decimal.** A number like `0.94` implies a calibration process (comparison against labeled ground truth) that doesn't exist until the Phase 1 benchmark is built and evaluated. Until then, assign confidence by explicit rule, e.g.:

- **high** — explicit sentence in a named section, extracted verbatim or near-verbatim
- **medium** — paraphrased from explicit content, or extracted from an abstract only
- **low** — inferred by the LLM without a direct textual anchor

Every downstream conclusion should be able to point back to actual source evidence.

---

# 16. Analysis Claims

Create a separate representation for system-generated reasoning.

```text
analysis_claims
---------------
id
claim_type
claim_text
confidence
status
created_at
```

Possible types:

```text
FACT
INFERENCE
HYPOTHESIS
OPPORTUNITY
SPECULATION
```

Example:

```text
claim_type:
INFERENCE

claim_text:
"Existing research appears to focus primarily on offline evaluation."

confidence:
medium   -- inferred across multiple papers, no single paper states this directly

status:
CANDIDATE
```

Connect claims to evidence:

```text
claim_evidence
--------------
claim_id
evidence_id
relationship
```

Relationships:

```text
supports
contradicts
contextualizes
```

The resulting structure is:

```text
Source Evidence
      ↓
System Inference
      ↓
Opportunity / Recommendation
```

---

# 17. Provenance Across the Entire System

Every extracted or generated object should carry provenance where applicable:

```text
source
source_id
evidence_id
extraction_method
model_version
confidence
human_validated
created_at
```

Example:

```text
Research Gap
------------
Description:
"Most existing approaches evaluate static datasets."

Confidence:
medium

Supporting Evidence:
E103
E207
E241

Supported By:
3 papers

Human Validated:
No

Status:
Candidate
```

After expert review:

```text
Human Validated:
Yes

Status:
Accepted
```

This provenance system is a major part of the project's scientific credibility.

---

# 18. Opportunity Scoring — Major Revision

Do NOT allow arbitrary outputs such as:

```text
Market Potential = 87/100
Economic Impact = 91/100
```

based only on a paper and an LLM.

Divide assessment into layers.

---

# 19. Layer 1 — Scientific Opportunity

Can be substantially grounded in the scientific corpus.

Potential dimensions:

- novelty relative to analyzed corpus,
- research-gap strength,
- methodological contribution,
- evidence strength,
- experimental evidence,
- reproducibility indicators,
- technical maturity indicators.

---

# 20. Layer 2 — Technical Applicability

Potential dimensions:

- deployment feasibility,
- computational requirements,
- data availability,
- scalability,
- integration complexity,
- reproducibility,
- validation environment,
- implementation maturity.

Support these with:

- reported experiments,
- technical specifications,
- datasets,
- hardware requirements,
- methodology,
- documented limitations.

---

# 21. Layer 3 — External Impact

These should NOT be confidently inferred solely from scientific papers:

- market potential,
- economic impact,
- customer demand,
- competitive landscape,
- willingness to pay,
- industry adoption,
- commercialization potential.

If no external evidence exists, output:

```text
Market Potential:
NOT ASSESSED

Reason:
Insufficient external market evidence.

Required validation:
Market research / industry expert review
```

Do not fabricate numbers.

---

# 22. Confidence Must Fall When Evidence Is Weak

For example:

```text
Market Potential:
Moderate / Unverified

Confidence:
Low

Evidence:
No external market dataset available.

Action:
Requires external validation.
```

Every score should have:

```text
Score
Definition
Evidence
Calculation Method
Confidence
Explanation
Validation Status
```

---

# 23. Recommended Opportunity Model

At the MVP stage, avoid one universal weighted score.

Provide three separate assessments:

```text
Scientific Opportunity
----------------------
Novelty
Gap Strength
Evidence Strength
Technical Contribution


Technical Applicability
-----------------------
Feasibility
Data Availability
Compute Requirements
Maturity
Reproducibility


External Impact
---------------
Market Potential
Economic Value
Industry Demand
Competition
Regulation
```

The third category should allow:

```text
Not Evaluated
```

where appropriate.

Do not hide missing evidence behind numerical scores.

---

# 24. Phase 1 Benchmark Dataset

Before building advanced reasoning loops, create a manually annotated benchmark dataset.

Target:

> **30–50 papers**, annotated solo (this is a deliberately small, hand-labeled set a single person can realistically complete — do not scale this up before Phase 1 ships)

Do not sample completely randomly.

Use stratified sampling across the initial corpus.

Example:

```text
40 papers

10 Machine Learning
8 NLP
8 Computer Vision
7 Systems
7 General AI / Other
```

Include diversity in:

- publication year,
- citation counts,
- research maturity,
- methodological complexity,
- explicit limitations,
- explicit future work,
- paper structure.

**Effort warning:** the full annotation schema in §25 (problem, RQ, method, dataset, contribution, results, limitations, gap, applications, plus supporting evidence passages, per paper) is a real research-methods effort, not a checklist to speed-run. Annotating 30–50 papers to this depth, solo, realistically takes longer than two weeks if done carefully — do not compress it to hit a roadmap date. If time pressure hits, prefer annotating fewer papers to full depth over more papers shallowly; a smaller, trustworthy benchmark is more useful than a larger, sloppy one.

---

# 25. Annotation Schema

For every benchmark paper, manually annotate:

## Metadata
- title
- domain
- year

## Problem
What problem does the paper address?

## Research Question
What research question is being investigated?

## Method
What approach/method is proposed?

## Dataset
What data is used?

## Main Contribution
What is actually new?

## Results
What are the major findings?

## Limitations
What limitations are explicitly stated?

## Research Gap
What gap does the paper address?
What gap remains?

## Applications
What applications are explicitly supported?

## Key Evidence
Which passages support these annotations?

The benchmark is the initial ground truth for system evaluation.

---

# 26. Phase 2 — Research Intelligence Baselines

Phase 2 should NOT begin with complex multi-agent reasoning loops.

Objective:

> Establish measurable performance for scientific-paper retrieval and structured knowledge extraction.

Pipeline:

```text
Corpus
   ↓
Benchmark
   ↓
Baseline Retrieval
   ↓
Baseline Extraction
   ↓
Evaluation
   ↓
Improvement
   ↓
Advanced Reasoning Later
```

---

# 27. Retrieval Evaluation

Implement and compare:

## Baseline 1
TF-IDF + cosine similarity

## Baseline 2
BM25

## Baseline 3
Sentence embeddings

## Baseline 4
Hybrid lexical + semantic retrieval

Potential future additions:

- rerankers,
- domain-specific embeddings,
- graph-enhanced retrieval.

Measure:

- Precision@K
- Recall@K
- nDCG
- MRR where appropriate

Use manually judged relevance labels.

This supports a genuine research question:

> Does semantic or hybrid retrieval outperform simple lexical retrieval for scientific-paper discovery?

---

# 28. Extraction Evaluation

Start with:

- abstract,
- structured metadata,
- and later structured full text.

Expected fields:

```text
problem
research_question
method
contribution
dataset
results
limitations
research_gap
applications
```

Evaluate against manual annotations.

Metrics:

- Precision
- Recall
- F1

Evaluate fields independently.

Example:

```text
Problem Extraction
Precision:
Recall:
F1:

Method Extraction
Precision:
Recall:
F1:
```

Do not report only one overall number.

**Also check whether confidence buckets are actually meaningful.** The high/medium/low rules in §15 are heuristics (e.g. "verbatim + named section = high"), assigned before any accuracy measurement exists. Once the benchmark evaluation runs, check whether "high confidence" extractions actually have higher precision than "medium" or "low" ones. If they don't correlate, the confidence rule is providing false reassurance and should be revised — do not let a plausible-sounding rule stand in for a validated one indefinitely.

---

# 29. Phase 2 Extraction Architecture

A simple baseline:

```text
Paper Abstract / Structured Content
              ↓
     Extraction Interface
              ↓
┌──────────────────────┐
│ Local / Free Model   │  ← default
│ Mock Extractor       │  ← testing
│ Paid API             │  ← optional
└──────────────────────┘
              ↓
      JSON structured output
              ↓
       Validation layer
              ↓
           Evidence
              ↓
         PostgreSQL
```

Per the Free/Open-Source First constraint (Builder Context), extraction sits behind a single provider-agnostic interface with three interchangeable implementations:

- **Local / Free Model** — the default. An open-source or local model (a small local LLM, or a non-LLM scientific NLP model) performs extraction with no paid API call involved.
- **Mock Extractor** — used in tests and for validating pipeline mechanics (schema, evidence grounding, persistence) without depending on any model at all.
- **Paid API** — strictly optional. Anthropic/OpenAI/other paid LLM APIs may be wired in behind the same interface, but must never be required for the MVP to build, run, or be demonstrated. Any paid option must be proposed separately with its cost stated explicitly, and adopted only with the builder's approval.

Potential approaches within any of the three implementations:

- schema-constrained LLM extraction (local or paid),
- scientific NLP models,
- rule-based extraction for deterministic fields,
- hybrid methods.

Compare methods rather than assuming an LLM — local or paid — is automatically best.

---

# 30. Research Similarity

Similarity should not rely only on keywords.

Consider:

- title,
- abstract,
- problem,
- methodology,
- dataset,
- contribution,
- applications,
- citations,
- semantic embeddings.

Distinguish:

```text
Lexically Similar
```

from:

```text
Scientifically Similar
```

Two papers can use different terminology while addressing essentially the same problem.

---

# 31. Citation Analysis

Use citation relationships as one signal for:

- foundational papers,
- influential work,
- research lineages,
- clusters,
- emerging research,
- isolated work.

Do NOT equate:

```text
citation count = research quality
```

Citation count is only contextual evidence.

---

# 32. Phase 3 — Research Gap Engine

Only implement advanced gap detection after Phases 1 and 2 are measurable.

**Note on primary use:** the pipeline below runs two ways:
1. **Per-assessment (primary, product-facing):** scoped to one ResearchInput's
   neighborhood, invoked as part of a single ResearchAssessment (§2A). This is
   the common case the product depends on.
2. **Corpus-wide batch (supporting/background capability):** run independently
   across every embedded paper to pre-populate `candidate_gaps` with
   reviewable candidates, so future assessments can reuse an already-vetted
   pattern instead of recomputing it. Valuable infrastructure — not itself the
   user-facing workflow, and not to be prioritized over per-assessment
   integration.

Pipeline:

```text
Related Papers
      ↓
Paper Comparison
      ↓
Limitations
      ↓
Future Work
      ↓
Experimental Gaps
      ↓
Temporal Patterns
      ↓
Cross-Paper Synthesis
      ↓
Candidate Research Gaps
      ↓
Evidence Validation
      ↓
Human Review
```

Distinguish:

## Explicit gaps

Directly stated by authors:

- future work,
- limitations,
- unresolved problems,
- missing experiments.

## Implicit cross-paper gaps

Derived from recurring patterns across papers.

Example:

```text
Paper A → offline evaluation
Paper B → offline evaluation
Paper C → offline evaluation

Observation:
Most evaluated systems are offline.

Potential research gap:
Real-time deployment under production constraints.
```

The latter must be labeled as an inference, not presented as an author-stated fact.

---

# 33. Phase 4 — Opportunity Engine (produces the ResearchAssessment report body)

After gap detection is validated, this stage runs as the final steps of a
ResearchAssessment (§2A) — turning that assessment's `research_gap_text`,
`retrieved_paper_ids`, and `comparison_summary` into the report fields shown
in §49.

```text
Research Gap
      ↓
Technical Capability
      ↓
Potential Application
      ↓
Technical Feasibility
      ↓
Evidence-Grounded Opportunity Assessment
      ↓
Human Validation
```

Identify:

- direct applications,
- adjacent applications,
- speculative opportunities.

Example:

```text
Direct:
Fraud detection API

Adjacent:
Real-time payment risk platform

Speculative:
National cross-bank fraud intelligence network
```

Do not blur these categories.

---

# 34. Facts vs Inferences vs Ideas

The UI and data model must distinguish:

## Evidence / Fact
Directly supported by literature.

## Analysis
Reasoning based on retrieved evidence.

## Inference
Conclusion derived from multiple evidence items.

## Hypothesis
Potential research opportunity.

## Opportunity
Plausible practical use of a capability.

## Product Idea
Generated concept requiring validation.

This distinction is essential for scientific credibility.

---

# 35. Human-in-the-Loop

The system must not be presented as an oracle.

**For a solo build, "human review" in Phase 1–2 means the builder's own disciplined review against the benchmark annotations, not a panel of researchers.** Occasional spot-checks from a professor or peer are valuable but should not be assumed as an available, ongoing resource. Design the review UI/workflow so a single reviewer (you) can use it efficiently — batch review screens, keyboard-driven approve/reject, not a workflow built for a review team.

Once the platform has real users (professors, other students), the same workflow extends to them:

- approve extracted knowledge,
- reject incorrect extraction,
- approve/reject research gaps,
- modify scores,
- add missing evidence,
- validate applications,
- label recommendations,
- annotate relationships.

Workflow:

```text
AI Analysis
     ↓
Human Review
     ↓
Correction
     ↓
Validated Knowledge
     ↓
Evaluation / Future Improvement
```

Human validation data can later become training or evaluation data.

---

# 36. Development Environment Constraints

Development machine:

```text
CPU:
12th Gen Intel Core i7-12650H

RAM:
16 GB

GPU:
NVIDIA RTX 2050 4 GB VRAM

Storage:
477 GB total
approximately 58 GB free
```

The MVP must remain lightweight.

Recommended local components:

```text
Python
FastAPI
PostgreSQL
pgvector
Next.js
Docker
PyMuPDF
GROBID only when needed
Pandas
NumPy
scikit-learn
Sentence Transformers / embedding model
```

Potentially local:

```text
Small NLP models
Small quantized LLMs
Limited OCR experiments
```

Cloud/paid infrastructure is optional, not a default progression. Per the Free/Open-Source First constraint (Builder Context), the MVP must run fully on local/free infrastructure. If any of the following ever becomes genuinely necessary, treat it as a separate, explicitly-approved proposal with cost stated upfront — not a default next step:

```text
Large LLM inference (paid API)
Large-scale embeddings (paid API)
Fine-tuning (cloud compute)
Large corpus processing (cloud compute)
Production deployment (cloud hosting)
```

Do not design the MVP around requiring a powerful GPU, and do not design it around requiring paid cloud infrastructure either.

---

# 37. Storage Strategy

The development machine has limited free storage.

Do NOT download a huge scientific corpus locally.

Start with:

- controlled corpus,
- benchmark data,
- cached metadata,
- selected permitted full-text content.

Production can use:

- object storage,
- university servers,
- cloud infrastructure,
- larger dedicated storage.

---

# 38. Minimize Infrastructure

Initial infrastructure:

```text
PostgreSQL
+
pgvector
```

Potentially:

```text
Redis
```

for background jobs if later justified.

Do NOT initially deploy:

- PostgreSQL
- Neo4j
- Qdrant
- Elasticsearch
- a triplestore
- multiple message queues

unless each one is justified by measurable requirements.

---

# 39. Revised Architecture

```text
                     Scientific APIs
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
       arXiv       Semantic Scholar      PMC OA
          │               │                │
          └───────────────┼────────────────┘
                          ▼
                 Ingestion → Normalization → Dedup → Provenance
                          ▼
                 PostgreSQL + pgvector
                 (Papers, Authors, Categories, Citations,
                  Evidence, Claims, Embeddings, Benchmark,
                  candidate_gaps)
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
       Retrieval Engine         Extraction Engine
        (shared by both consumers below)
              │                        │
              └───────────┬────────────┘
                          │
        ┌─────────────────┴──────────────────────┐
        ▼                                          ▼
Corpus-wide batch jobs                    PRODUCT WORKFLOW:
(evaluation, background                   ResearchInput (idea/paper)
 candidate_gaps population)                       │
        │                                          ▼
        ▼                                 Literature Search + Retrieve
   Human Review (bulk)                             │
                                                     ▼
                                          Compare vs. Existing Research
                                                     │
                                                     ▼
                                          Novelty / Gap / Feasibility
                                                     │
                                                     ▼
                                          Opportunity Reasoning
                                                     │
                                                     ▼
                                          ResearchAssessment Report
                                                     │
                                                     ▼
                                          Human Review (per-assessment)
                                                     │
                                                     ▼
                                              Decision UI
```

Retrieval and Extraction are genuinely shared infrastructure: the left branch
(corpus-wide batch jobs — evaluation runs, background `candidate_gaps`
population) and the right branch (the per-input product workflow) both call
into the same Retrieval Engine and Extraction Engine. Neither branch owns
them.

`ResearchInput (idea/paper)` in the right branch is itself two paths: a
free-text idea goes straight into Literature Search, while an uploaded
document first runs through its own Text/section extraction (Extraction
Engine) to build a research representation before Literature Search can run
— see the Uploaded-Paper Path diagram in §2A. Both paths converge at
"Retrieve Related Papers." A corpus match (`matched_paper_id`) is optional
and never required for either path to proceed.

---

# 40. Database Schema Summary

**Phase 1 schema (solo build — this is the actual starting point):**

```text
papers
authors
paper_authors
paper_categories

paper_citations

evidence
extracted_claims        -- generic, replaces the per-concept tables below until justified (see §14)

embeddings

benchmark_papers
benchmark_annotations
```

**Product-facing schema** (needed once retrieval + extraction exist — not a
Phase 1 starting table, but not deferred to "Phase 3+ once justified" either,
since the product's core loop needs it as soon as there's a corpus to compare
against — see §2A):

```text
research_inputs
research_assessments
research_assessment_evidence
```

**Phase 3+ schema (deferred until extraction quality is proven):**

```text
research_problems / paper_problems
research_methods / paper_methods
datasets / paper_datasets
limitations / paper_limitations
research_gaps / paper_gaps
applications / paper_applications

analysis_claims
claim_evidence
```

**Deferred normalization, not a parallel model (§41):**

```text
opportunity_assessments
assessment_evidence
```

`research_assessments` is the one assessment table the MVP writes to. The
two tables above describe what a future normalized three-layer breakdown
could look like if the flat categorical fields on `research_assessments`
ever prove insufficient — build them only if and when that need is
demonstrated, not alongside `research_assessments` from the start.

Use JSONB for source-specific metadata and flexible extracted metadata, but keep core entities relational.

---

# 41. Opportunity Assessment Schema (future normalization, not an MVP table)

**Do not build this alongside `research_assessments` (§2A).** For the MVP,
`research_assessments` is the single primary assessment object — its flat
`novelty_level`, `technical_feasibility_level`, and related categorical
fields already cover the three-layer distinction (§18-23) at MVP fidelity.
This section sketches what a dedicated normalized breakdown could look like
*if* those flat fields later prove insufficient (e.g. once a research
cluster needs to aggregate across many assessments) — a future refactor to
consider, not a parallel persistence layer to implement now.

Potential table:

```text
opportunity_assessments
-----------------------
id
paper_id
research_cluster_id NULL

scientific_novelty_level NULL      -- categorical: high | medium | low
research_gap_level NULL
evidence_strength_level NULL

technical_feasibility_level NULL
data_availability_level NULL
maturity_level NULL

market_potential_level NULL        -- almost always NULL / NOT_ASSESSED pre-Phase-5
economic_impact_level NULL

market_validation_status
economic_validation_status

overall_recommendation

confidence

reasoning

created_at
updated_at
```

Important:

Allow scores to be `NULL`.

`NULL` is preferable to fabricated certainty.

---

# 42. Recommendation Status

Possible outputs:

```text
HIGH PRIORITY
MEDIUM PRIORITY
LOW PRIORITY
INSUFFICIENT EVIDENCE
REQUIRES HUMAN REVIEW
```

The system must be able to conclude:

> "Insufficient evidence"

A trustworthy system does not always force a recommendation.

---

# 43. Academic Research Questions

RQ1–RQ5 below are exercised through the ResearchAssessment workflow (§2A),
not as abstract corpus-level benchmarks — e.g. RQ2 is tested by how well
retrieval finds related work for a real user-submitted ResearchInput.

Potential research questions:

### RQ1
How effectively can structured scientific APIs and LLM-based extraction convert scientific literature into a machine-readable research knowledge representation?

### RQ2
Does semantic or hybrid retrieval outperform lexical retrieval for finding scientifically relevant papers?

### RQ3
Can cross-paper synthesis identify research gaps that are not explicitly stated in individual publications?

### RQ4
Does combining citation information, embeddings, structured knowledge, and LLM reasoning improve research-gap discovery?

### RQ5
How reliably can a system assess technical applicability using evidence extracted from scientific literature?

### RQ6 (Future work — requires an expert panel not available to a solo builder)
How closely do AI-generated opportunity assessments agree with expert judgments?

RQ1–RQ5 are answerable solo, against the self-annotated benchmark. RQ6 needs a real panel of domain experts rating assessments independently — treat it as a Phase 4+/publication-stage question, not something the MVP evaluation plan should claim to answer.

**Circularity risk in solo evaluation.** For RQ1–RQ5, the same person designs the extraction/gap-detection logic AND hand-labels the ground truth it is scored against (§25). This is a real methodological weakness, not a formality — the builder's mental model of "what counts as a limitation" or "what counts as a gap" shapes both the system's rules/prompts and the labels used to judge it, which can inflate measured performance without inflating real performance. Mitigate it rather than ignore it:

- annotate the benchmark **before** looking at any model/system output for those papers, not after,
- where feasible, hold out a small batch and annotate it "blind" (without re-reading prior extraction results) as a rough check against self-consistency drift,
- state this limitation explicitly in any write-up of RQ1–RQ5 results — it does not invalidate the numbers, but it caps how strongly they can be interpreted without an independent reviewer (which is what RQ6/Phase 4+ is for).

Refine these into experimentally testable hypotheses.

---

# 44. Evaluation Plan

Evaluation must be built into the architecture.

## Retrieval

Metrics:

- Precision@K
- Recall@K
- nDCG
- MRR where appropriate

## Extraction

Metrics:

- Precision
- Recall
- F1

## Gap Detection

Human evaluation dimensions:

- correctness,
- relevance,
- novelty,
- evidence support,
- usefulness.

## Opportunity Assessment

Self-rating (solo builder, using the same rubric consistently — not a substitute for expert validation, just the Phase 1–2 stand-in for it):

```text
0 = irrelevant
1 = weak
2 = plausible
3 = highly relevant
```

Compare:

```text
AI assessment
vs
builder's own judgment (Phase 1–2)
vs
expert judgment (Phase 4+, once available — see RQ6)
```

## Explainability

Evaluate:

- evidence traceability,
- citation correctness,
- source attribution,
- reasoning transparency.

---

# 45. Revised Roadmap

**Product framing:** every phase below is infrastructure in service of one
end-user loop — ResearchInput → ResearchAssessment (§2A). Treat "does a
user's idea or paper produce a real ResearchAssessment report?" as the
standing acceptance test across phases, not just corpus-level metrics.

## Phase 1 — Corpus & Data Foundation (Solo, ~6–12 weeks — the only phase with committed weekly milestones)

| Weeks | Deliverable |
|---|---|
| 1–2 | arXiv connector only, `NormalizedPaper` representation, raw ingestion into Postgres, with ingestion reliability (failure logging, backoff, resumable runs — see §8) built in from the start, not retrofitted |
| 3 | Normalization + dedup (source, source_id) + DOI handling |
| 4 | Core schema live: `papers`, `authors`, `paper_authors`, `paper_categories`, `paper_citations` |
| 5 | `evidence` table + generic `extracted_claims` table (see §14) wired to at least abstract-level extraction |
| 6 | Basic embeddings (sentence-transformers) + pgvector similarity search working end-to-end |
| 7–9 | Hand-annotate the 30–50 paper benchmark (§24–25) — this is the step most likely to run long; if it does, let it eat the buffer week first rather than compressing annotation quality |
| 10–12 | Buffer / whatever slipped — do not start Phase 2 extraction baselines until the benchmark is done |

Deliver:

- arXiv connector (only),
- normalization,
- deduplication,
- core PostgreSQL schema + generic `extracted_claims` table,
- citation relationships,
- provenance,
- initial embeddings,
- benchmark dataset (self-annotated).

Success condition:

> A clean, reproducible, searchable CS/AI corpus exists — built and validated by one person, on one source, before adding either a second source or a second knowledge-extraction category.

---

## Phase 2 — Retrieval & Extraction Baselines

Deliver:

- TF-IDF baseline,
- BM25 baseline,
- embedding retrieval,
- hybrid retrieval,
- structured extraction,
- evidence storage,
- benchmark evaluation.

Success condition:

> We have measurable retrieval and extraction performance.

---

## Phase 3 — Research Gap Intelligence

Deliver:

- paper comparison,
- limitation aggregation,
- future-work analysis,
- temporal analysis,
- cross-paper synthesis,
- candidate research gaps,
- evidence validation.

Note: the corpus-wide candidate-gap batch job (`rb-gaps-detect --all`) is a
supporting capability that pre-populates reviewable patterns; it is not the
primary way research gaps get identified — that happens per-assessment,
scoped to one ResearchInput's neighborhood (§2A).

Success condition:

> The system can identify defensible candidate research gaps with supporting evidence.

---

## Phase 4 — Research-to-Impact

Deliver:

- application mapping,
- technical feasibility,
- opportunity assessment,
- risk analysis,
- confidence levels,
- human validation.

This phase is where the ResearchAssessment report (§2A, §49) becomes fully
populated for a real user-submitted input — the primary product-completion
milestone, not merely "opportunity scoring exists in the abstract."

Success condition:

> The system can help researchers move from validated research gaps toward practical opportunities.
>
> A user can submit an idea or paper and receive a complete, evidence-grounded ResearchAssessment report (§49).

---

## Phase 5 — Expansion

Potential additions:

- external market data,
- patents,
- companies,
- products,
- industry reports,
- regulations,
- funding opportunities,
- additional scientific domains.

Do NOT start Phase 5 until earlier phases have measurable evidence of value.

---

# 46. Future PDF Processing

Introduce PDF processing only when justified.

Possible later pipeline:

```text
PDF
 ↓
GROBID / parser
 ↓
Structured sections
 ↓
Tables / figures / references
 ↓
Scientific knowledge extraction
```

Reasons to introduce it:

- abstracts are insufficient,
- methodology details are required,
- full-text evidence is needed,
- tables contain critical experimental evidence.

Do not add OCR/GROBID merely because scientific PDFs exist.

---

# 47. Future Knowledge Graph

A dedicated graph database is optional.

Only introduce Neo4j or another graph system if measurements show:

- PostgreSQL traversal is too slow,
- graph queries are too complex,
- graph algorithms become central,
- or the dataset reaches a scale where a graph system provides clear benefits.

Until then:

> PostgreSQL relational tables + recursive CTEs + JSONB + pgvector.

---

# 48. Future External Intelligence

The system may later integrate:

```text
Scientific Research
       ↓
Technical Capability
       ↓
Industry Problems
       ↓
Companies
       ↓
Products
       ↓
Markets
       ↓
Regulations
       ↓
Commercialization
```

Potential external sources:

- patents,
- company information,
- market reports,
- industrial datasets,
- regulations,
- public product catalogs,
- job postings,
- technology databases.

This is a separate expansion layer.

Do not use scientific papers alone to claim market demand.

---

# 49. End-to-End Example (ResearchAssessment Report)

Suppose a user submits a ResearchInput:

> **Input:** "Should we work on real-time fraud detection using graph transformers?" (idea)

The platform creates a `research_assessments` row (§2A) and runs the workflow:

1. Search the CS/AI corpus (Retrieval Engine).
2. Retrieve related papers.
3. Extract problems, methods, datasets, results, and limitations from each.
4. Compare the input against retrieved studies.
5. Identify what is already solved (Existing Solutions).
6. Identify recurring limitations across the neighborhood.
7. Detect a candidate research gap scoped to this input (reusing an existing
   `candidate_gaps` row if the neighborhood already has one, else running
   detection fresh).
8. Attach supporting evidence (`research_assessment_evidence`, §2A).
9. Assess scientific novelty.
10. Assess technical feasibility.
11. Identify plausible applications and product/technology opportunities.
12. Mark market/commercialization assessment `NOT ASSESSED` unless external
    data is available.
13. Generate a recommendation with confidence.

The resulting ResearchAssessment report:

```text
Input:
"Should we work on real-time fraud detection using graph transformers?"

Related Research:
Paper A - Graph Transformers for Fraud Detection (offline benchmark)
Paper B - Real-Time Graph Neural Networks for Payment Risk
Paper C - Scalable GNN Inference on Streaming Data

Existing Solutions:
Batch/offline graph-transformer fraud models with strong benchmark accuracy;
no evaluated system meets sub-100ms real-time constraints.

Novelty Assessment:
Moderate to High - most existing work targets offline evaluation.

Research Gap:
Existing studies predominantly focus on offline or static evaluation.
Limited evidence was found for real-time graph-based deployment under
strict latency constraints. (inference across Papers A/B/C - see Evidence)

Evidence:
Paper A, §5.2 (offline-only evaluation setup)
Paper B, §4 (reports 250ms+ inference latency)
Paper C, limitations section (no production deployment reported)

Potential Applications:
- Banking fraud detection
- Payment processing risk scoring
- E-commerce transaction screening
- Insurance claim anomaly detection

Potential Product/Technology Opportunities:
Direct: Real-time fraud-scoring API for a payment processor
Adjacent: Streaming risk-monitoring platform for multiple transaction types
Speculative: Cross-bank real-time fraud intelligence network

Technical Feasibility:
Moderate - requires low-latency graph inference, which is measured but
not yet demonstrated at production scale in the retrieved papers.

Risks / Limitations:
- data privacy
- inference latency under load
- concept drift
- compute requirements

External Validation Needed:
Market demand, cost sensitivity, and regulatory constraints - no external
market evidence is currently integrated (see §21).

Recommendation:
HIGH PRIORITY / REQUIRES VALIDATION

Confidence:
Medium
```

## Variant: Uploaded Document Input

If the user instead uploads a paper draft (`input_type = document`) rather
than typing an idea, steps 1-2 change; steps 3-13 are identical:

1. **Text/section extraction** runs on the uploaded document (Extraction
   Engine, §28-29) to produce a research representation — problem, method,
   dataset, contribution — the same fields already extracted from corpus
   papers.
2. That representation, not the raw uploaded text, becomes the retrieval
   query for Literature Search.

If the upload happens to match an already-ingested corpus paper (checked by
title/DOI/arXiv id), `matched_paper_id` is set and the pipeline may reuse
that paper's existing extraction instead of re-running it — a pure
optimization. If it doesn't match, extraction runs directly on the upload
and the assessment proceeds exactly the same way. A corpus match is never
required to produce a ResearchAssessment.

---

# 50. What to Avoid

Avoid:

- building a multi-agent architecture prematurely,
- introducing Neo4j before proving need,
- scraping large PDF collections before the ingestion model is validated,
- scoring market potential from paper text alone,
- using one LLM prompt as the entire research-gap engine,
- claiming absolute novelty,
- assuming citation count means quality,
- using arbitrary weights without validation,
- building for every scientific domain from day one,
- processing millions of papers before evaluating a 30–50 paper benchmark,
- hiding uncertainty behind confident language,
- splitting knowledge into ten narrow entity tables before extraction quality is proven on one generic table (solo build — every extra table is schema you must migrate alone later),
- adding a second data source before the first source's pipeline is fully validated,
- assigning decimal confidence scores (e.g. 0.94) without a calibration process behind them — use categorical high/medium/low instead,
- assuming a domain-expert review panel is available; design Phase 1–2 review workflows for one reviewer (yourself),
- letting ingestion failures pass silently — a solo builder has no team to notice a quietly broken pipeline, so failures must be loud and logged from week 1,
- annotating the benchmark after looking at system output, or presenting RQ1–RQ5 results without acknowledging that the same person designed the system and labeled its ground truth,
- treating a categorical confidence rule as validated just because it's plausible — check it against benchmark precision once results exist,
- introducing a paid API or cloud service as a default/mandatory dependency instead of an optional, explicitly-approved one (see Builder Context: Free/Open-Source First),
- assuming an API or model is free without verifying current pricing/licensing first,
- wiring an LLM integration directly against one paid provider instead of behind a replaceable provider interface.

---

# 51. Core Strategic Principle

The architecture should follow:

```text
Reliable Data
      ↓
Measurable Retrieval
      ↓
Reliable Extraction
      ↓
Evidence Representation
      ↓
Cross-Paper Analysis
      ↓
Research Gap Detection
      ↓
Opportunity Reasoning
      ↓
Human Validation
```

Not:

```text
Huge Corpus
      ↓
LLM
      ↓
Magic Research Recommendation
```

---

# 52. Final Objective

ResearchBridge should help a university move from:

> "We have thousands of scientific papers and many possible research directions. Which ones are worth pursuing?"

to:

> "Here are the strongest opportunities, what research they build on, what is already solved, what remains unresolved, what evidence supports the gap, what practical applications are plausible, what technical barriers exist, and what requires external validation."

The platform should produce:

> **Evidence-based research intelligence, not AI-generated certainty.**

The fundamental architecture is:

> **Reliable Corpus → Evidence → Retrieval → Knowledge → Gap → Opportunity → Human Validation**

This is being built by one person. Every phase beyond Phase 1 is directional, not committed — re-plan Phase 2's weekly milestones only once Phase 1 has shipped and the benchmark evaluation is in hand. Vision documents don't need to match execution timelines; roadmaps do.

Use this revised blueprint as the **current source of truth** for future technical decisions about ResearchBridge.

When proposing changes, always explain:

1. why the change is needed,
2. what problem it solves,
3. what alternative approaches exist,
4. how it affects the database schema,
5. how it affects evaluation,
6. how it affects scalability,
7. how it affects the MVP timeline,
8. and whether the change is justified given the current hardware and project scope.
