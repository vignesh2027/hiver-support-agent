"""LLM intent classifier, optionally grounded in retrieved neighbours.

Two knobs worth understanding, both of which are ablated in the report:

**Batching.** Messages are classified several per request. This is a budget
decision -- the free tier allows 1,000 requests a day and the full evaluation
needs classification, generation and judging -- but it is not free of risk:
items in a batch can influence each other. `--batch 1` reproduces the
unbatched behaviour, and the harness runs a 40-item A/B to measure whether
batching actually moves the labels. Reporting a batched number without that
check would be quietly unsound.

**kNN few-shot.** With `--knn k`, each message is shown alongside its k most
similar *training* messages and the intent implied by the cluster each fell
into. These neighbour labels are weak -- they come from clustering, not from
humans -- so they are presented to the model as "similar past messages and
their provisional labels", not as ground truth. Whether this helps is an
empirical question the ablation answers rather than an assumption.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from ..config import CLS_MODEL
from ..llm import chat_json
from ..retrieve.index import PrecedentIndex
from ..taxonomy.schema import load as load_taxonomy
from .baselines import Prediction, cluster_to_intent, weak_labelled_train

_SYS = (
    "You classify customer messages sent to Great Western Railway's Twitter support desk "
    "into a fixed intent taxonomy. You label what the customer WANTS, not the topic they "
    "mention. You apply the written boundaries literally. You never invent a category, and "
    "when a message genuinely straddles two intents you pick the higher-stakes one and say "
    "you were unsure."
)

_USER = """Intent taxonomy:
{guideline}

Rules:
1. Label what the customer wants, not the topic mentioned.
2. Several asks in one message: label the highest-stakes one
   (money > stranded passenger > factual question > venting).
3. Sarcasm does not change intent. Label the grievance underneath it.
4. Use only the message shown; you cannot see later turns.
5. `confidence` must reflect real uncertainty. A message you could defend two
   labels for should score below 0.6.
{knn_block}
Classify each message. Return JSON:
{{"labels": [{{"id": "...", "intent": "...", "confidence": 0.0-1.0}}]}}

Messages:
{messages}"""


@dataclass
class _Neighbour:
    text: str
    weak_intent: str
    score: float


class LLMClassifier:
    name = "llm"

    def __init__(
        self,
        model: str = CLS_MODEL,
        batch_size: int = 6,
        knn: int = 0,
        index: PrecedentIndex | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.knn = knn
        self.tax = load_taxonomy()
        self._nn_index: PrecedentIndex | None = None
        self._nn_labels: dict[int, str] = {}
        if knn:
            self._build_knn(index)

    def _build_knn(self, index: PrecedentIndex | None) -> None:
        """Index the weakly-labelled training messages for neighbour lookup.

        Deliberately a *separate* index from the precedent index: that one is
        filtered to substantive replies, which would bias which neighbours are
        reachable for classification.
        """
        wl = weak_labelled_train()
        df = pd.DataFrame(
            {
                "episode_id": range(len(wl)),
                "customer_msg": wl["customer_msg"],
                "brand_reply": wl["weak_intent"],
                "created_at": "",
                "customer_thanked": False,
                "reply_substantive": True,
            }
        )
        self._nn_index = PrecedentIndex(backend="tfidf").fit(df, substantive_only=False)
        self._nn_labels = dict(zip(range(len(wl)), wl["weak_intent"]))

    def _neighbours(self, msg: str) -> list[_Neighbour]:
        if not self.knn or self._nn_index is None:
            return []
        return [
            _Neighbour(text=p.customer_msg, weak_intent=p.brand_reply, score=p.score)
            for p in self._nn_index.search(msg, k=self.knn)
        ]

    def predict(self, texts: list[str], ids: list[str] | None = None) -> list[Prediction]:
        ids = ids or [f"m{i}" for i in range(len(texts))]
        guideline = self.tax.guideline_block(compact=self.batch_size > 3)
        results: dict[str, Prediction] = {}

        for b in range(0, len(texts), self.batch_size):
            chunk_t = texts[b : b + self.batch_size]
            chunk_i = ids[b : b + self.batch_size]

            lines, knn_lines = [], []
            for eid, t in zip(chunk_i, chunk_t):
                lines.append(f"[{eid}] {t}")
                for nb in self._neighbours(t):
                    knn_lines.append(
                        f"  for [{eid}] similar past message (sim {nb.score:.2f}, "
                        f"provisional label '{nb.weak_intent}'): {nb.text[:130]}"
                    )
            knn_block = ""
            if knn_lines:
                knn_block = (
                    "\nSimilar past messages with PROVISIONAL labels from clustering. "
                    "These labels are weak and sometimes wrong -- treat them as a hint, "
                    "not as ground truth:\n" + "\n".join(knn_lines) + "\n"
                )

            res = chat_json(
                [
                    {"role": "system", "content": _SYS},
                    {
                        "role": "user",
                        "content": _USER.format(
                            guideline=guideline,
                            knn_block=knn_block,
                            messages="\n".join(lines),
                        ),
                    },
                ],
                model=self.model,
                tag=f"classify:knn{self.knn}:b{self.batch_size}",
                max_tokens=180 * len(chunk_t) + 300,
            )
            got = {str(x.get("id")): x for x in (res.get("labels") or [])}
            for eid in chunk_i:
                rec = got.get(eid)
                if rec is None or not self.tax.is_valid(str(rec.get("intent"))):
                    # An unusable answer is recorded as low-confidence `other`
                    # so the router escalates it, rather than being dropped and
                    # quietly shrinking the evaluation set.
                    results[eid] = Prediction("other", 0.0, {})
                    continue
                results[eid] = Prediction(
                    intent=str(rec["intent"]),
                    confidence=float(rec.get("confidence") or 0.0),
                    scores={},
                )
        return [results[i] for i in ids]


def main() -> None:
    import argparse

    from ..config import GOLDEN

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--knn", type=int, default=0)
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    pool = pd.read_json(GOLDEN / "pool.jsonl", lines=True).head(args.limit)
    clf = LLMClassifier(batch_size=args.batch, knn=args.knn)
    preds = clf.predict(pool["customer_msg"].tolist(), pool["example_id"].tolist())
    for r, p in zip(pool.itertuples(), preds):
        print(f"{p.intent:<28} {p.confidence:.2f}  {r.customer_msg[:66]}")


if __name__ == "__main__":
    main()
