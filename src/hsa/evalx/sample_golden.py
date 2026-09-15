"""Build the golden evaluation pool: 220 messages, three strata, documented.

Why three strata instead of one random sample
---------------------------------------------
A single random sample of 220 from this corpus contains roughly 91 live-status
messages and *four* missed-connection messages. Per-class F1 on four examples
is noise, and the rare intents are exactly the ones where mistakes are
expensive. But if you only sample hard and rare cases, your headline number no
longer describes the traffic the system will actually see.

So the pool is split and the strata are never mixed into one number:

  A  natural   (n=120)  uniform random from the held-out test split.
                        The ONLY stratum used for population estimates.
  B  enriched  (n=60)   over-samples intents the weak baseline thinks are rare,
                        so per-class metrics have enough support to mean
                        anything. Biased by construction.
  C  adversarial (n=40) targeted selection of the failure shapes that matter
                        operationally: multi-intent, sarcasm, money amounts,
                        safety language, image-only, existing-case follow-ups.
                        A stress test, not a sample of anything.

Every row records its stratum and the reason it was selected, so the harness
can refuse to compute a population metric over B or C. See DECISIONS.md D-08.

Known bias, stated up front: stratum B uses a model to decide what is rare,
so it can only enrich intents that model can already find. Rare intents the
baseline is blind to stay under-represented. Stratum C is hand-built from
regexes and inherits whatever those regexes miss.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import GOLDEN, INTERIM, SEED
from ..classify.baselines import train_baselines

POOL = GOLDEN / "pool.jsonl"

# --- stratum C probes -------------------------------------------------------
# Each probe names an operational failure shape. They are deliberately
# high-precision and low-recall: a probe that fires on everything selects
# nothing interesting.

MONEY = re.compile(r"(£\s?\d+|\d+\s?(pounds|quid)|refund of|compensat\w+ of)", re.I)
SAFETY = re.compile(
    r"\b(unsafe|assault\w*|attack\w*|abuse\w*|racist|threat\w*|injur\w*|accident|"
    r"emergency|police|999|collaps\w*|faint\w*|ill on|paramedic|ambulance|"
    r"disabled|wheelchair|accessib\w*|guide dog|pram)\b",
    re.I,
)
SARCASM = re.compile(
    r"(thanks? (a lot|for (the|nothing)|so much)[^.!?]{0,40}(late|delay|cancel|miss)|"
    r"\b(well done|great job|brilliant|fantastic|outstanding|wonderful|marvellous)\b"
    r"[^.!?]{0,60}\b(late|delay|cancel|again|miss|packed|crowd)\b|"
    r"\banother (outstanding|excellent|great|fine) \w+\b|"
    r"#?(joke|shambles|shameful|farce)\b)",
    re.I,
)
FOLLOWUP = re.compile(
    r"(sent (you )?a? ?dm|as per my dm|still waiting|any update|chased|"
    r"weeks? ago|no (reply|response)|heard nothing|reference number|ref no|case number)",
    re.I,
)
IMAGE_ONLY = re.compile(r"^\W*(<url>\s*)+\W*$|^.{0,45}<url>\s*$")
MULTI_Q = re.compile(r"\?[^?]*\?")
LEGAL = re.compile(r"\b(ombudsman|solicitor|legal|sue|small claims|rail ombudsman|complaint to)\b", re.I)

PROBES = {
    "money_amount": MONEY,
    "safety_or_accessibility": SAFETY,
    "sarcasm": SARCASM,
    "existing_case_followup": FOLLOWUP,
    "image_only": IMAGE_ONLY,
    "multi_question": MULTI_Q,
    "legal_escalation": LEGAL,
}


@dataclass
class PoolRow:
    example_id: str
    episode_id: int
    stratum: str
    selection_reason: str
    customer_msg: str
    created_at: str
    brand_reply_reference: str
    reply_substantive: bool
    n_followups: int
    customer_thanked: bool
    baseline_intent: str
    baseline_confidence: float
    probes: list[str] = field(default_factory=list)


def _probe_hits(msg: str) -> list[str]:
    return [name for name, rx in PROBES.items() if rx.search(msg)]


def build_pool(
    brand: str = "GWRHelp",
    n_natural: int = 120,
    n_enriched: int = 60,
    n_adversarial: int = 40,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    test = pd.read_parquet(INTERIM / f"episodes_{brand}_test.parquet").reset_index(drop=True)

    _, lr = train_baselines()
    preds = lr.predict(test["customer_msg"].tolist())
    test["baseline_intent"] = [p.intent for p in preds]
    test["baseline_confidence"] = [p.confidence for p in preds]
    test["probes"] = test["customer_msg"].map(_probe_hits)

    rows: list[PoolRow] = []
    used: set[int] = set()

    def add(idx: int, stratum: str, reason: str) -> bool:
        if idx in used:
            return False
        used.add(idx)
        r = test.iloc[idx]
        rows.append(
            PoolRow(
                example_id=f"{stratum[0].upper()}{len(rows):03d}",
                episode_id=int(r["episode_id"]),
                stratum=stratum,
                selection_reason=reason,
                customer_msg=str(r["customer_msg"]),
                created_at=str(r["created_at"]),
                brand_reply_reference=str(r["brand_reply"]),
                reply_substantive=bool(r["reply_substantive"]),
                n_followups=int(r["n_followups"]),
                customer_thanked=bool(r["customer_thanked"]),
                baseline_intent=str(r["baseline_intent"]),
                baseline_confidence=float(r["baseline_confidence"]),
                probes=list(r["probes"]),
            )
        )
        return True

    # -- A: natural ---------------------------------------------------------
    order = rng.permutation(len(test))
    for i in order:
        if sum(r.stratum == "natural" for r in rows) >= n_natural:
            break
        add(int(i), "natural", "uniform random draw from held-out test split")

    # -- B: enriched --------------------------------------------------------
    # Target the intents the baseline predicts least often, so that each has a
    # chance of reaching usable support. Take them round-robin rather than
    # filling the rarest first, which would just swap one skew for another.
    freq = test["baseline_intent"].value_counts()
    rare_first = list(freq.sort_values().index)
    per_intent: dict[str, int] = {k: 0 for k in rare_first}
    cap = max(3, n_enriched // max(len(rare_first), 1) + 2)
    added = 0
    for _ in range(cap * len(rare_first)):
        if added >= n_enriched:
            break
        for intent in rare_first:
            if added >= n_enriched or per_intent[intent] >= cap:
                continue
            cand = [
                int(i)
                for i in test.index[test["baseline_intent"] == intent]
                if int(i) not in used
            ]
            if not cand:
                continue
            pick = int(rng.choice(cand))
            if add(pick, "enriched", f"enrich rare predicted intent '{intent}'"):
                per_intent[intent] += 1
                added += 1

    # -- C: adversarial -----------------------------------------------------
    # Round-robin across probes so no single probe dominates the stress set.
    probe_names = list(PROBES)
    per_probe = max(2, n_adversarial // len(probe_names))
    added = 0
    for rounds in range(per_probe + 3):
        if added >= n_adversarial:
            break
        for name in probe_names:
            if added >= n_adversarial:
                break
            cand = [
                int(i)
                for i in test.index
                if int(i) not in used and name in test.at[i, "probes"]
            ]
            if not cand:
                continue
            pick = int(rng.choice(cand))
            if add(pick, "adversarial", f"probe:{name}"):
                added += 1

    # Low-confidence items are where a router most needs to be right, so top up
    # any shortfall with the baseline's most uncertain predictions.
    if added < n_adversarial:
        unc = test.assign(_i=test.index).sort_values("baseline_confidence")
        for _, r in unc.iterrows():
            if added >= n_adversarial:
                break
            if add(int(r["_i"]), "adversarial", "probe:low_baseline_confidence"):
                added += 1

    df = pd.DataFrame([vars(r) for r in rows])
    df["example_id"] = [
        f"{s[0].upper()}{i:03d}" for i, s in enumerate(df["stratum"].tolist())
    ]
    return df


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="GWRHelp")
    ap.add_argument("--natural", type=int, default=120)
    ap.add_argument("--enriched", type=int, default=60)
    ap.add_argument("--adversarial", type=int, default=40)
    args = ap.parse_args()

    df = build_pool(args.brand, args.natural, args.enriched, args.adversarial)
    GOLDEN.mkdir(parents=True, exist_ok=True)
    with POOL.open("w") as fh:
        for _, r in df.iterrows():
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    print(f"golden pool: {len(df)} examples -> {POOL}")
    print(df["stratum"].value_counts().to_string())
    print("\npredicted-intent spread (baseline, NOT labels):")
    print(df["baseline_intent"].value_counts().to_string())
    print("\nadversarial probe coverage:")
    probes = pd.Series(
        [p for ps in df[df.stratum == "adversarial"]["probes"] for p in ps]
    ).value_counts()
    print(probes.to_string() if len(probes) else "  (none)")
    print(f"\nduplicate episodes: {df['episode_id'].duplicated().sum()}")


if __name__ == "__main__":
    main()
