"""Intent-classification baselines.

The assignment asks for a trivial baseline and a simple one. Both exist here,
and both are stronger than they look, which is the point: a headline number is
only meaningful against a baseline that someone might actually have shipped.

* **MajorityBaseline** (trivial). Always predicts the largest intent. On this
  corpus that is `live_service_status` at ~41% of traffic, so it scores a
  deceptively healthy accuracy while having macro-F1 near 0.06. Reporting both
  is how the report shows why accuracy is the wrong headline metric here.

* **WeakSupervisedLogReg** (simple). TF-IDF over word and character n-grams
  into a logistic regression. The interesting part is where its labels come
  from: the corpus has none, so we take the cluster each training message fell
  into during taxonomy induction and map it through the curated
  cluster -> intent table. That is weak supervision -- the labels are wrong for
  any message that sat in an impure cluster -- but it costs nothing, needs no
  LLM, and runs in milliseconds. If the LLM classifier cannot beat it by a
  clear margin, the LLM is not worth its latency and cost.

Both expose the same `predict` / `predict_proba` interface as the LLM
classifier so the harness can treat them interchangeably.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..config import ARTIFACTS, INTERIM, SEED
from ..taxonomy.schema import Taxonomy, load as load_taxonomy

CLUSTERS_RAW = ARTIFACTS / "clusters_raw.json"


def cluster_to_intent(tax: Taxonomy | None = None) -> dict[str, str]:
    """Invert the taxonomy's source_clusters into a cluster -> intent lookup."""
    tax = tax or load_taxonomy()
    m: dict[str, str] = {}
    for intent in tax.intents:
        for c in intent.source_clusters:
            if c in m:
                raise ValueError(
                    f"cluster {c} claimed by both {m[c]} and {intent.name}; "
                    "source_clusters must partition the clusters"
                )
            m[c] = intent.name
    return m


def weak_labelled_train() -> pd.DataFrame:
    """Training messages with a weak intent label from their cluster.

    Returns the 6,000 messages that went through clustering, not the full
    training split: only those have a cluster assignment.
    """
    raw = json.loads(CLUSTERS_RAW.read_text())
    c2i = cluster_to_intent()
    missing = sorted(set(raw["assignment"]) - set(c2i))
    if missing:
        raise ValueError(
            f"clusters with no intent in taxonomy_final.json: {missing}. "
            "Every leaf cluster must be assigned (see curation_log C-1)."
        )
    return pd.DataFrame(
        {
            "customer_msg": raw["messages"],
            "cluster": raw["assignment"],
            "weak_intent": [c2i[c] for c in raw["assignment"]],
        }
    )


@dataclass
class Prediction:
    intent: str
    confidence: float
    scores: dict[str, float]


class MajorityBaseline:
    """Trivial baseline: always the most common intent."""

    name = "majority"

    def __init__(self) -> None:
        self.intent: str | None = None
        self.classes_: list[str] = []

    def fit(self, texts, labels) -> "MajorityBaseline":
        counts = Counter(labels)
        self.intent = counts.most_common(1)[0][0]
        self.classes_ = sorted(set(labels))
        return self

    def predict(self, texts: list[str]) -> list[Prediction]:
        assert self.intent is not None, "call fit() first"
        return [
            Prediction(self.intent, 1.0, {c: float(c == self.intent) for c in self.classes_})
            for _ in texts
        ]


class WeakSupervisedLogReg:
    """Simple baseline: TF-IDF -> logistic regression on cluster-derived labels."""

    name = "tfidf_logreg"

    def __init__(self, seed: int = SEED) -> None:
        self.seed = seed
        self._w: TfidfVectorizer | None = None
        self._c: TfidfVectorizer | None = None
        self._clf: LogisticRegression | None = None
        self.classes_: list[str] = []

    def _features(self, texts: list[str], fit: bool = False):
        if fit:
            self._w = TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode"
            )
            self._c = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True
            )
            return hstack([self._w.fit_transform(texts), self._c.fit_transform(texts)]).tocsr()
        assert self._w is not None and self._c is not None
        return hstack([self._w.transform(texts), self._c.transform(texts)]).tocsr()

    def fit(self, texts: list[str], labels: list[str]) -> "WeakSupervisedLogReg":
        X = self._features(texts, fit=True)
        # balanced: the weak labels are heavily skewed toward live_service_status,
        # and an unbalanced fit collapses to the majority class, which would make
        # this baseline indistinguishable from the trivial one.
        self._clf = LogisticRegression(
            max_iter=2000, C=2.0, class_weight="balanced", random_state=self.seed
        )
        self._clf.fit(X, labels)
        self.classes_ = list(self._clf.classes_)
        return self

    def predict(self, texts: list[str]) -> list[Prediction]:
        assert self._clf is not None, "call fit() first"
        P = self._clf.predict_proba(self._features(texts))
        out = []
        for row in P:
            j = int(np.argmax(row))
            out.append(
                Prediction(
                    intent=str(self.classes_[j]),
                    confidence=float(row[j]),
                    scores={str(c): float(p) for c, p in zip(self.classes_, row)},
                )
            )
        return out


def train_baselines() -> tuple[MajorityBaseline, WeakSupervisedLogReg]:
    df = weak_labelled_train()
    texts, labels = df["customer_msg"].tolist(), df["weak_intent"].tolist()
    return (
        MajorityBaseline().fit(texts, labels),
        WeakSupervisedLogReg().fit(texts, labels),
    )


def main() -> None:
    df = weak_labelled_train()
    print(f"weakly-labelled training messages: {len(df):,}")
    dist = df["weak_intent"].value_counts(normalize=True)
    for k, v in dist.items():
        print(f"  {k:<30} {v:>6.1%}")

    maj, lr = train_baselines()
    print(f"\nmajority baseline predicts: {maj.intent}")

    demo = [
        "left my north face jacket on the 8.01 from Bristol, how do I get it back?",
        "is the 18:21 from Cardiff running or is it cancelled again",
        "how do I claim delay repay for a 45 minute delay",
        "absolutely rammed this morning, no seats at all",
    ]
    print("\nlogreg baseline on held-out phrasings:")
    for t, p in zip(demo, lr.predict(demo)):
        print(f"  {p.intent:<28} {p.confidence:.2f}  {t[:62]}")

    (ARTIFACTS / "metrics" / "weak_label_distribution.json").write_text(
        json.dumps({k: round(float(v), 4) for k, v in dist.items()}, indent=2)
    )


if __name__ == "__main__":
    main()
