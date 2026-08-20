"""Stratified sampling for the Phase 1 benchmark dataset (Sec 24).

Selects papers per domain bucket, spread evenly across publication year
within each bucket (a cheap proxy for the blueprint's "diversity in
publication year ... research maturity" guidance - full diversity judgment
is still the annotator's call, this just avoids handing them 40 papers
all from the same month).
"""

from __future__ import annotations

import random
from collections import defaultdict
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.benchmark.domains import DEFAULT_TARGETS, classify_domain
from researchbridge.db.models import Paper


def stratified_sample(
    session: Session,
    targets: dict[str, int] | None = None,
    seed: int = 42,
) -> dict[str, list[Paper]]:
    """Return {domain: [papers]}, each list capped at its target count.

    A domain with fewer available papers than its target simply returns
    everything it has - callers should report the shortfall rather than
    treat it as an error (the corpus may not be large/diverse enough yet).
    """
    targets = targets if targets is not None else DEFAULT_TARGETS

    by_domain: dict[str, list[Paper]] = defaultdict(list)
    for paper in session.execute(select(Paper)).scalars():
        by_domain[classify_domain(paper)].append(paper)

    rng = random.Random(seed)
    result: dict[str, list[Paper]] = {}
    for domain, target in targets.items():
        result[domain] = _pick_spread_across_years(by_domain.get(domain, []), target, rng)
    return result


def _pick_spread_across_years(pool: list[Paper], target: int, rng: random.Random) -> list[Paper]:
    if len(pool) <= target or target <= 0:
        return list(pool)

    ordered = sorted(pool, key=lambda p: p.publication_date or date.min)
    if target == 1:
        indices = {len(ordered) // 2}
    else:
        indices = {round(i * (len(ordered) - 1) / (target - 1)) for i in range(target)}

    chosen = [ordered[i] for i in sorted(indices)]
    if len(chosen) < target:
        # rounding collapsed some evenly-spaced slots onto the same index -
        # top up randomly from whatever wasn't already picked
        remaining = [p for i, p in enumerate(ordered) if i not in indices]
        rng.shuffle(remaining)
        chosen.extend(remaining[: target - len(chosen)])
    return chosen
