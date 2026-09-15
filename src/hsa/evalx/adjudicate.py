"""Third-pass adjudication of pre-label disagreements.

Where the two independent pre-label passes disagreed, something has to break
the tie. The honest options are a human or a third model, and they are not
equivalent, so this module does the second and labels it as such: rows it
writes carry `review_kind="llm_adjudicated"`, never `adjudicate` (which is
reserved for items a person actually looked at).

Why this exists at all: an unadjudicated disagreement carries no gold label
and is excluded from scoring entirely. With 138 of 220 items in disagreement,
excluding them would leave a golden set of 82 easy, agreed items -- which
would make every metric look better than the truth, because the hard cases
would be gone. A machine-adjudicated label is worse than a human one and much
better than silently dropping the hardest 63% of the sample.

The adjudicator differs from both earlier passes in three ways that matter:

* a larger model (gpt-oss-120b rather than 20b),
* it *sees both prior labels and both reasons* and has to choose between them
  or reject both, rather than labelling cold,
* it is told the specific failure mode the pre-label passes exhibit: they are
  systematically too eager to automate (pass 1 marked 144 of 220 messages
  `auto`, pass 2 marked 78), so it is asked to apply the disposition rules
  literally rather than averaging the two.

Any human review recorded later overrides these rows, because
`hsa.tools.label_cli` skips ids already present and the harness prefers the
last write. The report breaks the golden set down by `review_kind` so the
human-verified fraction is always visible rather than implied.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from ..config import GOLDEN, TAXONOMY_MODEL
from ..llm import chat_json
from ..taxonomy.schema import load as load_taxonomy

PRELABELS = GOLDEN / "prelabels.jsonl"
GOLD = GOLDEN / "golden.jsonl"

BATCH = 4

_SYS = (
    "You are the senior reviewer on a support-operations annotation team. Two annotators "
    "have independently labelled each message and disagreed. You see both of their answers "
    "and the written guideline. You decide. You are allowed to reject both answers if both "
    "are wrong.\n\n"
    "You know one specific thing about these two annotators: both are far too willing to "
    "mark messages as safe to automate. Apply the disposition rules literally. Do not split "
    "the difference between them."
)

_USER = """Intent taxonomy and boundaries:
{guideline}

Rules for choosing the intent:
1. Label what the customer WANTS, not the topic they mention.
2. Several asks in one message: label the highest-stakes one
   (money > stranded passenger > factual question > venting).
3. Sarcasm does not change intent. Label the grievance underneath it.
4. Use only the message shown.

Rules for choosing the handling. These are not suggestions:
- "escalate" if the intent is live_service_status, missed_connection or other.
  A correct answer to these needs real-time running data or an existing case
  history, and the system has neither.
- "escalate" if the message names a specific money amount, threatens legal or
  ombudsman action, raises safety, injury, accessibility or a vulnerable
  passenger, or chases a case already open with the team.
- "auto" ONLY for lost_property with a clear request, or genuine unambiguous
  praise. Sarcastic praise is not praise.
- "assist" for everything else: an AI draft is useful but a human must approve
  it before it is sent.

For each message below you are given annotator A's answer and annotator B's answer.

{items}

Return JSON:
{{"decisions": [{{"id": "...", "intent": "...", "handling": "auto|assist|escalate",
  "chose": "A|B|neither", "why": "one short clause"}}]}}"""


def _fmt(chunk: pd.DataFrame) -> str:
    out = []
    for r in chunk.itertuples():
        out.append(
            f"[{r.example_id}] {r.customer_msg}\n"
            f"   annotator A: {r.p1_intent} / {r.p1_handling}\n"
            f"   annotator B: {r.p2_intent} / {r.p2_handling}"
        )
    return "\n\n".join(out)


def run(limit: int = 0, model: str = TAXONOMY_MODEL) -> pd.DataFrame:
    tax = load_taxonomy()
    pre = pd.read_json(PRELABELS, lines=True)

    done: set[str] = set()
    if GOLD.exists():
        for line in GOLD.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["example_id"])

    todo = pre[pre.needs_adjudication & ~pre.example_id.isin(done)]
    if limit:
        todo = todo.head(limit)
    if not len(todo):
        print("nothing to adjudicate")
        return pd.DataFrame()

    print(f"adjudicating {len(todo)} disagreements with {model}")
    guideline = tax.guideline_block(compact=True)
    written = 0

    for b in range(0, len(todo), BATCH):
        chunk = todo.iloc[b : b + BATCH]
        try:
            res = chat_json(
                [
                    {"role": "system", "content": _SYS},
                    {"role": "user", "content": _USER.format(
                        guideline=guideline, items=_fmt(chunk))},
                ],
                model=model,
                tag="adjudicate",
                max_tokens=1200,
            )
        except Exception as e:  # noqa: BLE001 - keep partial progress
            print(f"  batch {b//BATCH+1} failed: {type(e).__name__}: {e}")
            break

        got = {str(d.get("id")): d for d in (res.get("decisions") or [])}
        with GOLD.open("a") as fh:
            for r in chunk.itertuples():
                d = got.get(r.example_id)
                if not d or not tax.is_valid(str(d.get("intent"))):
                    continue
                handling = str(d.get("handling"))
                if handling not in ("auto", "assist", "escalate"):
                    handling = "escalate"
                fh.write(json.dumps({
                    "example_id": r.example_id,
                    "stratum": r.stratum,
                    "customer_msg": r.customer_msg,
                    "brand_reply_reference": r.brand_reply_reference,
                    "probes": list(r.probes),
                    "intent": str(d["intent"]),
                    "handling": handling,
                    "review_kind": "llm_adjudicated",
                    "p1_intent": r.p1_intent,
                    "p2_intent": r.p2_intent,
                    "p1_handling": r.p1_handling,
                    "p2_handling": r.p2_handling,
                    "human_changed_prelabel": False,
                    "adjudicator_chose": d.get("chose"),
                    "note": str(d.get("why") or "")[:200],
                    "seconds_spent": 0.0,
                }, ensure_ascii=False) + "\n")
                written += 1
        print(f"  batch {b//BATCH+1}/{(len(todo)+BATCH-1)//BATCH} ({written} written)", flush=True)

    g = pd.read_json(GOLD, lines=True)
    print(f"\ngolden set now {len(g)} rows")
    print(g["review_kind"].value_counts().to_string())
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
