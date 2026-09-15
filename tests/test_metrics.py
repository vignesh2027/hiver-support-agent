"""Tests for the statistics the report's claims rest on.

These matter more than they look. Every headline number in the report is
produced by this module, so a silent bug here would not crash anything -- it
would just make the conclusions wrong in a way no reviewer could detect from
the outside.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hsa.evalx.metrics import (
    area_under_risk_coverage,
    bootstrap_ci,
    classification_report,
    cohens_kappa,
    kappa_with_ci,
    risk_coverage_curve,
    selective_metrics,
    top_confusions,
    wilson_ci,
)


# ------------------------------------------------------------------ basics

def test_perfect_classification():
    y = ["a", "b", "c", "a"]
    r = classification_report(y, y)
    assert r.accuracy == 1.0
    assert r.macro_f1 == 1.0


def test_macro_f1_punishes_majority_class_collapse():
    """The point of reporting macro-F1 alongside accuracy."""
    y_true = ["a"] * 90 + ["b"] * 5 + ["c"] * 5
    y_pred = ["a"] * 100
    r = classification_report(y_true, y_pred)
    assert r.accuracy == pytest.approx(0.90)
    assert r.macro_f1 < 0.35, "macro-F1 must expose the collapse that accuracy hides"


def test_confusion_matrix_rows_sum_to_support():
    y_true = ["a", "a", "b", "c", "c", "c"]
    y_pred = ["a", "b", "b", "c", "a", "b"]
    r = classification_report(y_true, y_pred)
    for cls, row in r.confusion.items():
        assert sum(row.values()) == r.support.get(cls, 0)


def test_top_confusions_finds_the_dominant_error():
    y_true = ["a"] * 10 + ["b"] * 10
    y_pred = ["b"] * 10 + ["b"] * 10
    conf = top_confusions(classification_report(y_true, y_pred), k=1)
    assert conf[0][:2] == ("a", "b")
    assert conf[0][2] == 10


def test_report_is_json_serialisable():
    import json

    r = classification_report(["a", "b"], ["a", "a"])
    json.dumps(r.to_dict())  # must not raise on numpy scalars


# --------------------------------------------------------------- intervals

def test_bootstrap_ci_brackets_the_point_estimate():
    v = [1.0] * 80 + [0.0] * 20
    point, lo, hi = bootstrap_ci(v)
    assert point == pytest.approx(0.8)
    assert lo < point < hi
    assert 0.0 <= lo and hi <= 1.0


def test_bootstrap_ci_narrows_with_more_data():
    small = bootstrap_ci([1.0] * 8 + [0.0] * 2)
    large = bootstrap_ci([1.0] * 800 + [0.0] * 200)
    assert (large[2] - large[1]) < (small[2] - small[1])


def test_bootstrap_ci_handles_empty():
    assert all(math.isnan(x) for x in bootstrap_ci([]))


def test_wilson_never_returns_impossible_bounds():
    """The reason we use Wilson: normal approx goes negative near zero."""
    for k, n in [(0, 50), (0, 5), (50, 50), (1, 200)]:
        p, lo, hi = wilson_ci(k, n)
        assert 0.0 <= lo <= p <= hi <= 1.0, (k, n, lo, p, hi)


def test_wilson_zero_events_has_nonzero_upper_bound():
    _, lo, hi = wilson_ci(0, 40)
    assert lo == 0.0
    assert hi > 0.0, "zero observed failures does not mean zero risk"


# --------------------------------------------------------------- agreement

def test_kappa_perfect_and_chance():
    assert cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    # Systematically opposite labels: worse than chance.
    assert cohens_kappa(["a", "a", "b", "b"], ["b", "b", "a", "a"]) < 0


def test_kappa_is_lower_than_raw_agreement_on_skewed_labels():
    """The reason kappa is reported instead of raw agreement."""
    a = ["x"] * 90 + ["y"] * 10
    b = ["x"] * 88 + ["y"] * 2 + ["x"] * 10
    res = kappa_with_ci(a, b, n_boot=500)
    assert res["raw_agreement"] > 0.85
    assert res["kappa"] < res["raw_agreement"]


def test_kappa_ci_contains_point_estimate():
    a = ["x", "y"] * 40
    b = ["x", "y"] * 35 + ["y", "x"] * 5
    res = kappa_with_ci(a, b, n_boot=500)
    lo, hi = res["kappa_ci"]
    assert lo <= res["kappa"] <= hi


def test_kappa_interpretation_bands():
    res = kappa_with_ci(["a", "b"] * 30, ["a", "b"] * 30, n_boot=200)
    assert res["interpretation"] == "almost perfect"


# ----------------------------------------------------------- risk/coverage

def test_risk_coverage_is_monotone_in_coverage_for_a_perfect_ranker():
    scores = [0.99, 0.95, 0.9, 0.4, 0.3, 0.1]
    correct = [True, True, True, False, False, False]
    curve = risk_coverage_curve(scores, correct)
    errs = [e for _, e, _ in curve]
    assert errs == sorted(errs), "a perfect confidence ranking must not decrease error"
    assert curve[-1][0] == 1.0


def test_aurc_rewards_a_useful_confidence_signal():
    correct = [True] * 5 + [False] * 5
    good = risk_coverage_curve([0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1], correct)
    useless = risk_coverage_curve([0.5] * 10, correct)
    assert area_under_risk_coverage(good) < area_under_risk_coverage(useless)


# ---------------------------------------------------------------- selective

def test_selective_metrics_separates_coverage_from_quality():
    actions = ["auto"] * 4 + ["escalate"] * 6
    correct = [True, True, True, False] + [False] * 6
    m = selective_metrics(actions, correct)
    assert m["auto_coverage"] == pytest.approx(0.4)
    assert m["quality_on_auto"] == pytest.approx(0.75)
    # The whole point: overall quality is much worse than quality-on-auto.
    assert m["quality_overall_if_all_sent"] < m["quality_on_auto"]


def test_catastrophic_escapes_are_counted_only_on_auto():
    actions = ["auto", "escalate"]
    m = selective_metrics(actions, [True, True], catastrophic=[False, True])
    assert m["catastrophic_escapes"] == 0
    assert m["catastrophic_rate_overall"] == pytest.approx(0.5)


def test_selective_metrics_json_safe():
    import json

    m = selective_metrics(["auto", "escalate"], [True, False], [False, True])
    json.dumps(m)
