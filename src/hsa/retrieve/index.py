"""Retrieve historical (customer message -> brand reply) precedents.

This is the "grounded in how that brand has historically resolved similar
issues" half of the task. Two decisions inside are worth stating plainly
because they change the numbers a lot:

**We index only substantive replies.** Roughly 16% of GWR's first replies are
hand-offs ("DM us your booking reference") or bare apologies. Indexing them
means the retriever happily returns "sorry to hear that" as the precedent for
everything, and the generator learns to apologise instead of answering. We
drop them from the index and keep them in the corpus for analysis. This costs
recall on intents where GWR genuinely only ever deflects -- which is itself
something the escalation policy should know, so we surface it rather than
papering over it.

**The index is built from the training split only.** Retrieval over the whole
corpus would let a test message retrieve a precedent written minutes later
about the same incident, which is not a precedent, it is the answer. On this
corpus that leak is severe: hundreds of near-duplicate tweets follow a single
disruption within the hour. See DECISIONS.md D-04.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..config import ARTIFACTS, INTERIM


@dataclass
class Precedent:
    episode_id: int
    customer_msg: str
    brand_reply: str
    score: float
    created_at: str
    customer_thanked: bool

    def as_prompt_block(self, i: int) -> str:
        return (
            f"[precedent {i}] (similarity {self.score:.2f})\n"
            f"  customer: {self.customer_msg}\n"
            f"  GWR replied: {self.brand_reply}"
        )


class PrecedentIndex:
    """Similarity search over historical resolved exchanges.

    Backends
    --------
    ``tfidf``  word 1-2 grams + char 3-5 grams, cosine. No model download, no
               torch, deterministic. This is the default so that `make
               reproduce` works on a clean machine in minutes.
    ``embed``  sentence-transformers all-MiniLM-L6-v2. Better on paraphrase,
               costs a 90 MB download. Reported as an ablation, not as the
               headline configuration, so that the headline does not depend on
               an optional dependency.
    """

    def __init__(self, backend: str = "tfidf") -> None:
        self.backend = backend
        self.df: pd.DataFrame | None = None
        self._vec: TfidfVectorizer | None = None
        self._cvec: TfidfVectorizer | None = None
        self._M: np.ndarray | None = None
        self._model = None

    # -- build -------------------------------------------------------------

    def fit(self, train: pd.DataFrame, *, substantive_only: bool = True) -> "PrecedentIndex":
        df = train.copy()
        if substantive_only:
            before = len(df)
            df = df[df["reply_substantive"]].copy()
            self.dropped_nonsubstantive = before - len(df)
        else:
            self.dropped_nonsubstantive = 0
        df = df.reset_index(drop=True)
        self.df = df
        texts = df["customer_msg"].tolist()

        if self.backend == "embed":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._M = normalize(
                np.asarray(self._model.encode(texts, batch_size=64, show_progress_bar=False))
            )
        else:
            # Char n-grams matter here: customers write "cancelled", "canceled",
            # "cancled" and "cx". A word-only index misses all but the first.
            self._vec = TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode"
            )
            self._cvec = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True
            )
            W = self._vec.fit_transform(texts)
            C = self._cvec.fit_transform(texts)
            from scipy.sparse import hstack

            self._M = normalize(hstack([W, C]).tocsr())
        return self

    # -- query -------------------------------------------------------------

    def _embed_query(self, q: str):
        if self.backend == "embed":
            return normalize(np.asarray(self._model.encode([q])))
        from scipy.sparse import hstack

        return normalize(hstack([self._vec.transform([q]), self._cvec.transform([q])]).tocsr())

    def search(self, query: str, k: int = 4, min_score: float = 0.0) -> list[Precedent]:
        assert self.df is not None and self._M is not None, "call fit() first"
        qv = self._embed_query(query)
        sims = (self._M @ qv.T)
        sims = np.asarray(sims.todense()).ravel() if hasattr(sims, "todense") else sims.ravel()

        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        top = top[np.argsort(-sims[top])]
        out = []
        for i in top:
            if sims[i] < min_score:
                continue
            r = self.df.iloc[int(i)]
            out.append(
                Precedent(
                    episode_id=int(r["episode_id"]),
                    customer_msg=str(r["customer_msg"]),
                    brand_reply=str(r["brand_reply"]),
                    score=float(sims[i]),
                    created_at=str(r["created_at"]),
                    customer_thanked=bool(r["customer_thanked"]),
                )
            )
        return out

    # -- persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "backend": self.backend,
                    "df": self.df,
                    "vec": self._vec,
                    "cvec": self._cvec,
                    "M": self._M if self.backend != "embed" else None,
                    "E": self._M if self.backend == "embed" else None,
                    "dropped": getattr(self, "dropped_nonsubstantive", 0),
                },
                fh,
            )

    @classmethod
    def load(cls, path: Path) -> "PrecedentIndex":
        with path.open("rb") as fh:
            st = pickle.load(fh)
        ix = cls(backend=st["backend"])
        ix.df = st["df"]
        ix._vec = st["vec"]
        ix._cvec = st["cvec"]
        ix._M = st["M"] if st["backend"] != "embed" else st["E"]
        ix.dropped_nonsubstantive = st.get("dropped", 0)
        if st["backend"] == "embed":
            from sentence_transformers import SentenceTransformer

            ix._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return ix


def build(brand: str = "GWRHelp", backend: str = "tfidf") -> PrecedentIndex:
    train = pd.read_parquet(INTERIM / f"episodes_{brand}_train.parquet")
    ix = PrecedentIndex(backend=backend).fit(train)
    ix.save(ARTIFACTS / f"index_{brand}_{backend}.pkl")
    return ix


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="GWRHelp")
    ap.add_argument("--backend", default="tfidf", choices=["tfidf", "embed"])
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    ix = build(args.brand, args.backend)
    assert ix.df is not None
    print(
        f"indexed {len(ix.df):,} substantive precedents "
        f"(dropped {ix.dropped_nonsubstantive:,} deflections/acks) backend={args.backend}"
    )
    (ARTIFACTS / "metrics" / f"index_{args.backend}.json").write_text(
        json.dumps(
            {
                "backend": args.backend,
                "indexed": int(len(ix.df)),
                "dropped_nonsubstantive": int(ix.dropped_nonsubstantive),
            },
            indent=2,
        )
    )

    if args.demo:
        for q in [
            "why is the 18:03 from Paddington to Oxford cancelled?",
            "how do I claim delay repay for a 45 minute delay",
            "no seats at all, absolutely rammed again this morning",
        ]:
            print(f"\nQ: {q}")
            for i, p in enumerate(ix.search(q, k=3), 1):
                print(f"  {p.score:.3f}  C: {p.customer_msg[:90]}")
                print(f"         B: {p.brand_reply[:90]}")


if __name__ == "__main__":
    main()
