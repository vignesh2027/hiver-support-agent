"""Invariants for the taxonomy.

Most of these exist because something actually went wrong. The coverage test
is the clearest case: the LLM consolidation asserted "no clusters were
dropped" while leaving two clusters (4.1% of traffic) unassigned. A
self-reported completeness claim is not evidence, so it is now a test.
"""

from __future__ import annotations

import json

import pytest

from hsa.classify.baselines import cluster_to_intent
from hsa.taxonomy.schema import ARTIFACTS, VALID_DISPOSITIONS, load


@pytest.fixture(scope="module")
def tax():
    return load()


def test_loads_and_validates(tax):
    assert len(tax.intents) >= 8
    assert "other" in tax.names


def test_prior_shares_sum_to_one(tax):
    total = sum(i.prior_share for i in tax.intents)
    assert abs(total - 1.0) < 0.02, f"prior shares sum to {total:.4f}, not 1.0"


def test_intent_names_unique_and_snake_case(tax):
    assert len(set(tax.names)) == len(tax.names)
    for n in tax.names:
        assert n == n.lower(), n
        assert " " not in n, n


def test_every_intent_has_usable_annotation_guidance(tax):
    for i in tax.intents:
        assert i.definition.strip()
        assert i.disposition_rationale.strip(), f"{i.name} has no rationale for its disposition"
        if i.name != "other":
            assert i.includes, f"{i.name} has no inclusion examples"
            assert i.excludes, f"{i.name} has no exclusion rules"
            assert i.hard_case.strip(), f"{i.name} has no boundary case"


def test_dispositions_are_valid(tax):
    for i in tax.intents:
        assert i.default_disposition in VALID_DISPOSITIONS


def test_every_cluster_is_assigned_exactly_once(tax):
    """The regression test for curation_log C-1."""
    named = json.loads((ARTIFACTS / "clusters_named.json").read_text())
    mapping = cluster_to_intent(tax)  # raises if a cluster is claimed twice
    missing = sorted(set(named) - set(mapping))
    assert not missing, (
        f"{len(missing)} clusters are in no intent's source_clusters: {missing}. "
        "Every leaf cluster must be assigned."
    )
    unknown = sorted(set(mapping) - set(named))
    assert not unknown, f"taxonomy references clusters that do not exist: {unknown}"


def test_catch_all_is_not_a_dumping_ground(tax):
    """A large 'other' hides the system's true coverage (curation_log C-2)."""
    other = tax.get("other")
    assert other.prior_share < 0.08, (
        f"'other' holds {other.prior_share:.1%} of traffic. Above ~8% the taxonomy is "
        "hiding real intents behind an escalate-by-default bucket."
    )


def test_live_data_intents_are_never_auto(tax):
    """Auto-sending a reply about something we cannot observe is the core risk."""
    for i in tax.intents:
        if i.needs_live_data:
            assert i.default_disposition != "auto", (
                f"{i.name} needs live data but defaults to auto"
            )


def test_the_two_confusable_intents_stay_separate(tax):
    """Status, explanation and compensation must not be merged (C-4)."""
    for n in ("live_service_status", "compensation_and_refunds", "service_complaint"):
        assert tax.is_valid(n), f"{n} must exist as its own intent"


def test_guideline_block_renders_for_prompts(tax):
    full = tax.guideline_block()
    compact = tax.guideline_block(compact=True)
    assert len(compact) < len(full)
    for n in tax.names:
        assert n in full and n in compact
