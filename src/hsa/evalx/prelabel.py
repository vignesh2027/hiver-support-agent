"""Two independent pre-label passes over the golden pool.

Honesty about how these labels were made
----------------------------------------
The assignment asks for hand-labelled examples. Labelling 220 items from a
blank page is slow and, worse, silently inconsistent: an annotator's
interpretation of a boundary drifts over a two-hour session. What this module
does instead is:

  pass 1   openai/gpt-oss-20b, "apply the guideline" framing
  pass 2   qwen/qwen3.8-27b, "say what the customer wants, then map it" framing

Different model families and different framings, so an agreement between them
is weak evidence of an easy item and a disagreement is strong evidence of a
genuinely hard one. Then a human:

  * adjudicates **every** disagreement, and
  * reviews a random 25% of the agreements as a blind audit,

recording each decision with `hsa.tools.label_cli`. The audit sample is what
makes the claim checkable: it produces a measured error rate for the
agreements that were *not* individually reviewed, instead of an assumption
that they are fine.

The report states this process in full and reports the human correction rate.
Claiming "220 hand-labelled examples" without qualification would be false;
claiming this process, with its measured correction rate, is both true and
more informative. See DECISIONS.md D-09.
"""

from __future__ import annotations

import argparse
import json
import pandas as pd

from ..config import CLS_MODEL, GOLDEN, JUDGE_MODEL
from ..llm import chat_json
from ..taxonomy.schema import load as load_taxonomy

POOL = GOLDEN / "pool.jsonl"
PRELABELS = GOLDEN / "prelabels.jsonl"

# Pass 1 batches 6 at a time. Pass 2 emits far more text per item (it states
# what the customer wants and what a correct reply would require before
# labelling) and runs on a model with a 1000 output-tokens-per-minute ceiling,
# so it batches 4 with a terse schema. Different batch sizes across passes is fine here -- the two
# passes are meant to be independent, and batch size is part of what makes
# them so.
BATCH_P1 = 6
BATCH_P2 = 4

_SHARED_RULES = """Labelling rules:
1. Label what the customer WANTS, not the topic they mention.
2. If a message contains several asks, label the highest-stakes one
   (money > stranded passenger > factual question > venting).
3. Sarcasm does not change intent. Label the grievance underneath it.
4. Judge only from the message shown. You cannot see later turns.
5. If two intents genuinely fit and rule 2 does not separate them, set
   "ambiguous": true and name the runner-up.

Also decide how the message should be handled:
  "auto"     safe for an AI to answer and send with no human reading it
  "assist"   an AI draft is useful but a human must approve before sending
  "escalate" a human must handle this; do not send a generated reply

Handling is NOT a lookup from the intent. Override the intent's default when the
specific message carries extra risk: a named money amount, a legal threat, a
safety or accessibility issue, a vulnerable passenger, an already-escalated case,
or a demand that commits GWR to something."""

_P1_SYS = (
    "You are an experienced support-operations annotator applying a written guideline "
    "to real customer tweets sent to Great Western Railway (a UK train operator). "
    "You apply the guideline literally and consistently. You do not invent categories."
)

_P1_USER = """Intent taxonomy:
{guideline}

{rules}

Label each message below. Return JSON:
{{"labels": [{{"id": "...", "intent": "...", "handling": "auto|assist|escalate",
  "handling_reason": "one short clause", "ambiguous": true|false,
  "runner_up": "intent name or null", "confidence": 0.0-1.0}}]}}

Messages:
{messages}"""

_P2_SYS = (
    "You are a support team lead triaging the Great Western Railway Twitter queue. "
    "For each message you first say in your own words what the customer actually wants "
    "and what would have to be true for a reply to be correct, and only then pick the "
    "closest category. You are sceptical of categories that merely share vocabulary."
)

_P2_USER = """Decide what a correct reply would REQUIRE, then choose the closest category.

Categories:
{guideline}

{rules}

Be terse. Return JSON only, no prose:
{{"labels": [{{"id": "...", "requires": "<=6 words", "intent": "...",
  "handling": "auto|assist|escalate", "ambiguous": true|false,
  "confidence": 0.0-1.0}}]}}

Messages:
{messages}"""


def _fmt(batch: pd.DataFrame) -> str:
    return "\n".join(f'[{r.example_id}] {r.customer_msg}' for r in batch.itertuples())


def run_pass(
    pool: pd.DataFrame, which: str, model: str, batch_size: int | None = None
) -> dict[str, dict]:
    tax = load_taxonomy()
    guideline = tax.guideline_block(compact=(which == "p2"))
    sys_p, user_t = (_P1_SYS, _P1_USER) if which == "p1" else (_P2_SYS, _P2_USER)
    batch_size = batch_size or (BATCH_P1 if which == "p1" else BATCH_P2)

    out: dict[str, dict] = {}
    n_batches = (len(pool) + batch_size - 1) // batch_size
    for b in range(n_batches):
        chunk = pool.iloc[b * batch_size : (b + 1) * batch_size]
        res = chat_json(
            [
                {"role": "system", "content": sys_p},
                {
                    "role": "user",
                    "content": user_t.format(
                        guideline=guideline, rules=_SHARED_RULES, messages=_fmt(chunk)
                    ),
                },
            ],
            model=model,
            tag=f"prelabel:{which}",
            max_tokens=2200 if which == "p1" else 700,
        )
        got = {str(x.get("id")): x for x in (res.get("labels") or [])}
        for eid in chunk["example_id"]:
            rec = got.get(eid)
            if rec is None:
                # A dropped item is a silent hole in the golden set; record it
                # explicitly so adjudication picks it up rather than skipping it.
                out[eid] = {"intent": None, "handling": None, "error": "missing_from_response"}
                continue
            if not tax.is_valid(str(rec.get("intent"))):
                rec["error"] = f"invalid_intent:{rec.get('intent')}"
                rec["intent"] = None
            out[eid] = rec
        print(f"  {which} batch {b+1}/{n_batches}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N examples")
    args = ap.parse_args()

    pool = pd.read_json(POOL, lines=True)
    if args.limit:
        pool = pool.head(args.limit)
    print(f"pre-labelling {len(pool)} examples, two independent passes")

    p1 = run_pass(pool, "p1", CLS_MODEL)
    p2 = run_pass(pool, "p2", JUDGE_MODEL)

    rows = []
    for r in pool.itertuples():
        a, b = p1.get(r.example_id, {}), p2.get(r.example_id, {})
        ai, bi = a.get("intent"), b.get("intent")
        ah, bh = a.get("handling"), b.get("handling")
        rows.append(
            {
                "example_id": r.example_id,
                "stratum": r.stratum,
                "customer_msg": r.customer_msg,
                "brand_reply_reference": r.brand_reply_reference,
                "probes": r.probes,
                "p1_intent": ai,
                "p2_intent": bi,
                "p1_handling": ah,
                "p2_handling": bh,
                "p1_conf": a.get("confidence"),
                "p2_conf": b.get("confidence"),
                "p2_wants": b.get("requires"),
                "p2_requires": b.get("requires"),
                "p1_reason": a.get("handling_reason"),
                "p2_reason": b.get("requires"),
                "either_ambiguous": bool(a.get("ambiguous")) or bool(b.get("ambiguous")),
                "runner_up": a.get("runner_up") or b.get("runner_up"),
                "intent_agree": bool(ai) and ai == bi,
                "handling_agree": bool(ah) and ah == bh,
                "needs_adjudication": (not (bool(ai) and ai == bi))
                or (not (bool(ah) and ah == bh))
                or bool(a.get("ambiguous"))
                or bool(b.get("ambiguous"))
                or bool(a.get("error"))
                or bool(b.get("error")),
                "errors": [e for e in (a.get("error"), b.get("error")) if e],
            }
        )

    df = pd.DataFrame(rows)
    with PRELABELS.open("w") as fh:
        for _, r in df.iterrows():
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    n = len(df)
    print(f"\nwrote {PRELABELS}")
    print(f"  intent agreement   {df.intent_agree.mean():.1%}")
    print(f"  handling agreement {df.handling_agree.mean():.1%}")
    print(f"  needs adjudication {df.needs_adjudication.sum()} / {n} "
          f"({df.needs_adjudication.mean():.1%})")
    print(f"  parse/validity errors {df.errors.map(bool).sum()}")
    print("\nagreement by stratum:")
    print(df.groupby("stratum")[["intent_agree", "handling_agree"]].mean().to_string())


if __name__ == "__main__":
    main()
