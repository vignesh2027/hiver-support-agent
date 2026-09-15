"""Metrics, confidence intervals and agreement statistics.

Small module, but it carries most of the weight of the claim "the proof is
worth more than the system". Three things here are non-default on purpose:

* **Every headline number ships with a bootstrap CI.** With n=120 in the
  unbiased stratum, the 95% CI on a proportion near 0.9 is roughly +/- 5
  points. Quoting "91.7%" without that interval invites a comparison between
  two systems that the data cannot support.

* **Cohen's kappa, not raw agreement.** Two annotators who both label
  everything `live_service_status` agree 41% of the time by accident on this
  corpus. Kappa removes the chance agreement; on a skewed label distribution
  the difference is large and always flattering in the wrong direction.

* **Risk-coverage, not a single accuracy.** A selective system's quality is a
  curve, not a point. `risk_coverage_curve` produces the (coverage, error)
  pairs and `area_under_risk_coverage` summarises it, so two routers can be
  compared without cherry-picking an operating point.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np


# ---------------------------------------------------------------- intervals

def bootstrap_ci(
    values: list[float] | np.ndarray,
    *,
    stat=np.mean,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 20260913,
) -> tuple[float, float, float]:
    """Percentile bootstrap. Returns (point estimate, lo, hi)."""
    v = np.asarray(list(values), dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = stat(v[idx], axis=1)
    return float(stat(v)), float(np.percentile(boots, 100 * alpha / 2)), float(
        np.percentile(boots, 100 * (1 - alpha / 2))
    )


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson interval for a proportion.

    Preferred over the normal approximation for the rates that matter most
    here -- catastrophic-error rate is near zero, where the normal interval
    famously produces negative lower bounds.
    """
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * float(np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / d
    return float(p), float(max(0.0, centre - half)), float(min(1.0, centre + half))


# ------------------------------------------------------------ classification

@dataclass
class ClassificationReport:
    n: int
    accuracy: float
    accuracy_ci: tuple[float, float]
    macro_f1: float
    macro_f1_ci: tuple[float, float]
    weighted_f1: float
    per_class: dict
    confusion: dict
    support: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def classification_report(
    y_true: list[str], y_pred: list[str], labels: list[str] | None = None, seed: int = 20260913
) -> ClassificationReport:
    assert len(y_true) == len(y_pred), "length mismatch"
    labels = labels or sorted(set(y_true) | set(y_pred))
    n = len(y_true)

    correct = np.array([t == p for t, p in zip(y_true, y_pred)], dtype=float)
    acc, acc_lo, acc_hi = bootstrap_ci(correct, seed=seed)

    per_class = {}
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec, rec, f1 = _f1(tp, fp, fn)
        per_class[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in y_true if t == c),
        }

    # Macro-F1 is averaged over classes that actually occur in the reference;
    # including zero-support classes would silently drag it toward zero.
    present = [c for c in labels if per_class[c]["support"] > 0]
    macro = float(np.mean([per_class[c]["f1"] for c in present])) if present else 0.0

    # Bootstrap macro-F1 by resampling examples and recomputing, which is the
    # only honest way to get an interval on a non-linear statistic.
    rng = np.random.default_rng(seed)
    boots = []
    yt, yp = np.array(y_true), np.array(y_pred)
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        bt, bp = yt[idx], yp[idx]
        fs = []
        for c in present:
            tp = int(np.sum((bt == c) & (bp == c)))
            fp = int(np.sum((bt != c) & (bp == c)))
            fn = int(np.sum((bt == c) & (bp != c)))
            fs.append(_f1(tp, fp, fn)[2])
        boots.append(np.mean(fs) if fs else 0.0)
    m_lo, m_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    weighted = float(
        sum(per_class[c]["f1"] * per_class[c]["support"] for c in present) / max(n, 1)
    )

    conf: dict[str, dict[str, int]] = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1

    return ClassificationReport(
        n=n,
        accuracy=round(acc, 4),
        accuracy_ci=(round(acc_lo, 4), round(acc_hi, 4)),
        macro_f1=round(macro, 4),
        macro_f1_ci=(round(m_lo, 4), round(m_hi, 4)),
        weighted_f1=round(weighted, 4),
        per_class=per_class,
        confusion={k: v for k, v in conf.items()},
        support=dict(Counter(y_true)),
    )


def top_confusions(report: ClassificationReport, k: int = 5) -> list[tuple[str, str, int]]:
    out = []
    for t, row in report.confusion.items():
        for p, cnt in row.items():
            if t != p and cnt:
                out.append((t, p, cnt))
    return sorted(out, key=lambda x: -x[2])[:k]


# ---------------------------------------------------------------- agreement

def cohens_kappa(a: list, b: list) -> float:
    """Chance-corrected agreement between two raters on the same items."""
    assert len(a) == len(b) and a, "need equal, non-empty sequences"
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if abs(1 - pe) < 1e-12:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def kappa_with_ci(a: list, b: list, n_boot: int = 4000, seed: int = 20260913) -> dict:
    """Kappa plus a bootstrap interval and the raw agreement it corrects."""
    rng = np.random.default_rng(seed)
    n = len(a)
    arr_a, arr_b = np.array(a, dtype=object), np.array(b, dtype=object)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boots.append(cohens_kappa(list(arr_a[idx]), list(arr_b[idx])))
        except AssertionError:
            continue
    raw = sum(1 for x, y in zip(a, b) if x == y) / n
    return {
        "kappa": round(cohens_kappa(a, b), 4),
        "kappa_ci": (round(float(np.percentile(boots, 2.5)), 4),
                     round(float(np.percentile(boots, 97.5)), 4)),
        "raw_agreement": round(raw, 4),
        "n": n,
        "interpretation": _kappa_label(cohens_kappa(a, b)),
    }


def _kappa_label(k: float) -> str:
    # Landis & Koch (1977) bands. Quoted because a bare kappa means little to
    # a reader who does not work with them daily.
    if k < 0.0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


# ----------------------------------------------------------- risk / coverage

def risk_coverage_curve(
    scores: list[float], correct: list[bool], *, seed: int = 20260913, n_tiebreaks: int = 20
) -> list[tuple[float, float, float]]:
    """(coverage, error rate, threshold) as the abstention threshold sweeps.

    Sorted by confidence descending: answering only the most confident items
    first. A useful selective system's error rate should fall as coverage
    falls; a flat curve means the confidence signal carries no information.

    **Ties are broken randomly and averaged.** A plain ``argsort`` resolves
    equal scores by input order, which is not a property of the model -- it is
    a property of how the rows happen to sit in a file. That silently rewards
    an uninformative confidence signal: a model that emits 0.9 for everything
    would trace a perfect curve whenever the correct items happened to be
    listed first. This is not hypothetical; LLM self-reported confidences are
    heavily tied. Averaging over `n_tiebreaks` random permutations gives the
    expected curve instead of an artefact of row order.
    """
    assert len(scores) == len(correct), "length mismatch"
    s_all = np.asarray(scores, dtype=float)
    c_all = np.asarray(correct, dtype=bool)
    n = len(s_all)
    if n == 0:
        return []

    rng = np.random.default_rng(seed)
    err_acc = np.zeros(n)
    thr_acc = np.zeros(n)
    for _ in range(n_tiebreaks):
        perm = rng.permutation(n)
        # Stable sort over a shuffled array = uniformly random tie-breaking.
        order = perm[np.argsort(-s_all[perm], kind="stable")]
        c = c_all[order]
        s = s_all[order]
        err_acc += 1 - np.cumsum(c) / np.arange(1, n + 1)
        thr_acc += s
    err = err_acc / n_tiebreaks
    thr = thr_acc / n_tiebreaks
    return [(i / n, float(err[i - 1]), float(thr[i - 1])) for i in range(1, n + 1)]


def area_under_risk_coverage(curve: list[tuple[float, float, float]]) -> float:
    """Lower is better. Trapezoidal integral of error over coverage."""
    if not curve:
        return float("nan")
    cov = np.array([p[0] for p in curve])
    err = np.array([p[1] for p in curve])
    return float(np.trapezoid(err, cov))


def selective_metrics(
    actions: list[str], correct: list[bool], catastrophic: list[bool] | None = None
) -> dict:
    """Headline numbers for a system that is allowed to abstain.

    Deliberately reports quality *conditional on auto-handling* alongside the
    coverage it was achieved at. Either number alone is misleading: 100%
    quality at 2% coverage is a system that does nothing, and 60% coverage at
    70% quality is a system nobody can trust.
    """
    n = len(actions)
    auto = [i for i, a in enumerate(actions) if a == "auto"]
    assist = [i for i, a in enumerate(actions) if a == "assist"]
    esc = [i for i, a in enumerate(actions) if a == "escalate"]

    def rate(idx, arr):
        return float(np.mean([arr[i] for i in idx])) if idx else float("nan")

    k_auto = sum(1 for i in auto if correct[i])
    p, lo, hi = wilson_ci(k_auto, len(auto)) if auto else (float("nan"),) * 3

    res = {
        "n": n,
        "auto_coverage": round(len(auto) / n, 4),
        "assist_rate": round(len(assist) / n, 4),
        "escalate_rate": round(len(esc) / n, 4),
        "quality_on_auto": round(p, 4),
        "quality_on_auto_ci": (round(lo, 4), round(hi, 4)),
        "quality_on_assist": round(rate(assist, correct), 4),
        "quality_overall_if_all_sent": round(float(np.mean(correct)), 4),
    }
    if catastrophic is not None:
        k_cat = sum(1 for i in auto if catastrophic[i])
        cp, clo, chi = wilson_ci(k_cat, len(auto)) if auto else (float("nan"),) * 3
        res["catastrophic_rate_on_auto"] = round(cp, 4)
        res["catastrophic_rate_on_auto_ci"] = (round(clo, 4), round(chi, 4))
        res["catastrophic_rate_overall"] = round(float(np.mean(catastrophic)), 4)
        res["catastrophic_escapes"] = int(k_cat)
    return res
