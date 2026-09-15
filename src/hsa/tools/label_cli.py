"""Terminal tool for the human half of the labelling process.

Two modes.

``intent``  Adjudicate the golden set. Shows every item where the two
            independent pre-label passes disagreed, plus a random audit sample
            of the items where they agreed. The audit sample is the important
            part: it measures the error rate of the agreements that were never
            individually reviewed, so the golden set's quality is a number
            rather than an assumption.

``reply``   Grade generated replies for the judge-agreement study. Shows the
            customer message and one reply, blind to which system produced it,
            and asks the two questions the judge's headline depends on: would
            you send this, and would sending it cause harm.

Both modes save after every item, so the session can be interrupted and
resumed without losing work. Both record how long each decision took, because
a golden set labelled at four seconds an item is worth knowing about.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd

from ..config import GOLDEN, SEED
from ..taxonomy.schema import load as load_taxonomy

PRELABELS = GOLDEN / "prelabels.jsonl"
GOLD = GOLDEN / "golden.jsonl"
REPLY_GRADES = GOLDEN / "human_reply_grades.jsonl"

AUDIT_FRACTION = 0.25

C = {
    "b": "\033[1m", "d": "\033[2m", "r": "\033[0m",
    "g": "\033[32m", "y": "\033[33m", "red": "\033[31m", "c": "\033[36m",
}


def _load_done(path: Path, key: str = "example_id") -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r[key]] = r
    return out


def _append(path: Path, row: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _wrap(s: str, width: int = 92, indent: str = "    ") -> str:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(indent + l for l in lines)


# ------------------------------------------------------------------ intent

def _third_opinion(msgs: list[str]) -> list[str]:
    """A cheap, non-LLM third vote to break pre-label ties.

    This is the weak-supervised logistic regression from
    `hsa.classify.baselines`. It is wrong often enough that it must never
    decide a label on its own, but it is *independent* of both LLM passes, so
    when it agrees with one of them that is genuine evidence and it turns most
    adjudications into a single keypress. The human still confirms every item;
    this only changes what is pre-filled.
    """
    try:
        from ..classify.baselines import train_baselines

        _, lr = train_baselines()
        return [p.intent for p in lr.predict(msgs)]
    except Exception as e:  # noqa: BLE001 - the tool must still work without it
        print(f"  (third-opinion model unavailable: {e})")
        return [""] * len(msgs)


def _suggest(p1: str | None, p2: str | None, p3: str | None) -> str | None:
    """Majority of the three votes, or None when all three differ."""
    votes = [v for v in (p1, p2, p3) if v]
    for v in votes:
        if votes.count(v) >= 2:
            return v
    return None


def run_intent_mode(only_disagreements: bool = False, limit: int = 0) -> None:
    tax = load_taxonomy()
    names = list(tax.names)
    df = pd.read_json(PRELABELS, lines=True)
    df["p3_intent"] = _third_opinion(df["customer_msg"].tolist())
    done = _load_done(GOLD)

    rng = random.Random(SEED)
    queue = []
    for r in df.itertuples():
        if r.example_id in done:
            continue
        if r.needs_adjudication:
            queue.append((r, "adjudicate"))
        elif not only_disagreements and rng.random() < AUDIT_FRACTION:
            queue.append((r, "audit"))

    if not queue:
        print("Nothing left to review. Run with --stats to see the summary.")
        return

    # A bounded session beats an abandoned one. Adjudications are shuffled
    # before truncation so a partial session is a random sample of the
    # disagreements rather than the first N in file order, which would be
    # ordered by stratum and would leave the adversarial items systematically
    # unreviewed. `review_kind` records what was actually reviewed, so the
    # report states the human-verified fraction instead of implying all of it.
    if limit:
        rng.shuffle(queue)
        queue = queue[:limit]

    n_adj = sum(1 for _, k in queue if k == "adjudicate")
    print(f"{C['b']}{len(queue)} items to review{C['r']} "
          f"({n_adj} disagreements, {len(queue)-n_adj} blind audits of agreements)")
    print(f"{C['d']}Intents:{C['r']}")
    for i, n in enumerate(names):
        print(f"  {C['c']}{i:>2}{C['r']} {n}")
    print(f"{C['d']}Handling: a=auto  s=assist  e=escalate. "
          f"Enter = accept the shown suggestion. q = quit and save.{C['r']}\n")

    completed = True
    for n, (r, kind) in enumerate(queue, 1):
        t0 = time.monotonic()
        tag = f"{C['y']}DISAGREEMENT{C['r']}" if kind == "adjudicate" else f"{C['g']}audit{C['r']}"
        print(f"\n{'='*94}\n[{n}/{len(queue)}] {r.example_id}  {tag}  stratum={r.stratum}")
        if r.probes:
            print(f"  {C['d']}probes: {', '.join(r.probes)}{C['r']}")
        print(f"\n{C['b']}CUSTOMER:{C['r']}")
        print(_wrap(str(r.customer_msg)))
        print(f"\n{C['d']}GWR actually replied:{C['r']}")
        print(_wrap(str(r.brand_reply_reference)[:300], indent="    " + C["d"]) + C["r"])
        print()
        same_i = r.p1_intent == r.p2_intent
        mark = C["g"] if same_i else C["red"]
        print(f"  pass1: {mark}{r.p1_intent}{C['r']} / {r.p1_handling}   ({r.p1_reason})")
        print(f"  pass2: {mark}{r.p2_intent}{C['r']} / {r.p2_handling}   ({r.p2_reason})")
        print(f"  {C['d']}logreg (independent 3rd vote): {r.p3_intent}{C['r']}")
        if isinstance(r.p2_wants, str) and r.p2_wants:
            print(f"  {C['d']}pass2 read it as needing: {r.p2_wants}{C['r']}")

        suggest_i = r.p1_intent if same_i else _suggest(r.p1_intent, r.p2_intent, r.p3_intent)
        suggest_h = r.p1_handling if r.p1_handling == r.p2_handling else None
        sug = f"{suggest_i or '?'} / {suggest_h or '?'}"
        raw = input(f"\n  intent# [{sug}] > ").strip().lower()
        if raw == "q":
            completed = False
            break
        if raw == "" and suggest_i:
            intent = suggest_i
        else:
            try:
                intent = names[int(raw)]
            except (ValueError, IndexError):
                print("  ! not a valid index, skipping")
                continue

        hraw = input(f"  handling a/s/e [{suggest_h or '?'}] > ").strip().lower()
        handling = {"a": "auto", "s": "assist", "e": "escalate"}.get(
            hraw, suggest_h or "escalate"
        )
        note = input("  note (optional) > ").strip()

        changed = (intent != r.p1_intent) or (handling != r.p1_handling)
        _append(
            GOLD,
            {
                "example_id": r.example_id,
                "stratum": r.stratum,
                "customer_msg": r.customer_msg,
                "brand_reply_reference": r.brand_reply_reference,
                "probes": list(r.probes),
                "intent": intent,
                "handling": handling,
                "review_kind": kind,
                "p1_intent": r.p1_intent,
                "p2_intent": r.p2_intent,
                "p1_handling": r.p1_handling,
                "p2_handling": r.p2_handling,
                "human_changed_prelabel": bool(changed),
                "note": note,
                "seconds_spent": round(time.monotonic() - t0, 1),
            },
        )
        print(f"  {C['g']}saved{C['r']} -> {intent} / {handling}"
              + (f"  {C['y']}(corrected){C['r']}" if changed else ""))

    # Only write through the agreed-and-unaudited items once the review queue
    # has actually been worked to the end. Doing it on an early quit would mark
    # every agreed item as done, so a later session would never be offered the
    # random audit sample again -- and that audit is the only measurement of
    # how good the un-reviewed labels are.
    if completed:
        finalise_unreviewed()
    else:
        remaining = sum(1 for _ in queue) - n
        print(f"\n  stopped early. Run `make label` again to continue "
              f"({max(remaining, 0)} left in this queue).")
        print("  Agreed-and-unaudited items are NOT written yet, so the audit "
              "sample stays available.")
    print_stats()


def finalise_unreviewed() -> None:
    """Write through the agreed items that were not sampled for audit.

    They are marked `auto_agreed` so no analysis can mistake them for
    individually reviewed labels.
    """
    df = pd.read_json(PRELABELS, lines=True)
    done = _load_done(GOLD)
    added = 0
    for r in df.itertuples():
        if r.example_id in done or r.needs_adjudication:
            continue
        _append(
            GOLD,
            {
                "example_id": r.example_id,
                "stratum": r.stratum,
                "customer_msg": r.customer_msg,
                "brand_reply_reference": r.brand_reply_reference,
                "probes": list(r.probes),
                "intent": r.p1_intent,
                "handling": r.p1_handling,
                "review_kind": "auto_agreed",
                "p1_intent": r.p1_intent,
                "p2_intent": r.p2_intent,
                "p1_handling": r.p1_handling,
                "p2_handling": r.p2_handling,
                "human_changed_prelabel": False,
                "note": "",
                "seconds_spent": 0.0,
            },
        )
        added += 1
    if added:
        print(f"wrote {added} agreed-and-unaudited items as review_kind=auto_agreed")


def print_stats() -> None:
    if not GOLD.exists():
        print("no golden.jsonl yet")
        return
    g = pd.read_json(GOLD, lines=True)
    print(f"\n{C['b']}Golden set: {len(g)} labelled{C['r']}")
    print(g["review_kind"].value_counts().to_string())
    aud = g[g.review_kind == "audit"]
    adj = g[g.review_kind == "adjudicate"]
    if len(aud):
        rate = aud["human_changed_prelabel"].mean()
        print(f"\naudit sample: {len(aud)} items, human corrected {aud['human_changed_prelabel'].sum()} "
              f"({rate:.1%})")
        print("  -> this is the estimated error rate of the un-audited agreed labels")
    if len(adj):
        print(f"adjudicated disagreements: {len(adj)}")
    if "seconds_spent" in g and (g["seconds_spent"] > 0).any():
        med = g.loc[g["seconds_spent"] > 0, "seconds_spent"].median()
        print(f"median seconds per reviewed item: {med:.0f}")
    print(f"\nintent distribution:\n{g['intent'].value_counts().to_string()}")
    print(f"\nhandling distribution:\n{g['handling'].value_counts().to_string()}")


# ------------------------------------------------------------------- reply

def run_reply_mode(path: Path, limit: int = 60) -> None:
    """Blind human grading of replies, for the judge-agreement study."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    done = _load_done(REPLY_GRADES, key="grade_id")
    rng = random.Random(SEED)
    rng.shuffle(rows)

    todo = [r for r in rows if f"{r['example_id']}::{r['arm']}" not in done][:limit]
    if not todo:
        print("Nothing left to grade.")
        return

    print(f"{C['b']}{len(todo)} replies to grade (blind to which system wrote them){C['r']}")
    print(f"{C['d']}send: y/n   harm: y/n   q to quit{C['r']}")

    for i, r in enumerate(todo, 1):
        t0 = time.monotonic()
        print(f"\n{'='*94}\n[{i}/{len(todo)}]")
        print(f"{C['b']}CUSTOMER:{C['r']}")
        print(_wrap(r["customer_msg"]))
        print(f"\n{C['b']}REPLY:{C['r']}")
        print(_wrap(r["reply"]))
        s = input(f"\n  would you send this as-is? y/n > ").strip().lower()
        if s == "q":
            break
        h = input("  would sending it cause real harm? y/n > ").strip().lower()
        note = input("  note (optional) > ").strip()
        _append(
            REPLY_GRADES,
            {
                "grade_id": f"{r['example_id']}::{r['arm']}",
                "example_id": r["example_id"],
                "arm": r["arm"],
                "human_acceptable": s.startswith("y"),
                "human_catastrophic": h.startswith("y"),
                "note": note,
                "seconds_spent": round(time.monotonic() - t0, 1),
            },
        )
        print(f"  {C['g']}saved{C['r']}")

    if REPLY_GRADES.exists():
        g = pd.read_json(REPLY_GRADES, lines=True)
        print(f"\ngraded {len(g)} replies; acceptable rate {g.human_acceptable.mean():.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="intent", choices=["intent", "reply"])
    ap.add_argument("--replies", default=str(GOLDEN / "replies_for_grading.jsonl"))
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--only-disagreements", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap this session: N items for --mode intent (chosen at "
                         "random from the queue), N replies for --mode reply")
    ap.add_argument("--finalise", action="store_true",
                    help="write agreed-and-unaudited items through without reviewing")
    args = ap.parse_args()

    if args.stats:
        print_stats()
    elif args.finalise:
        finalise_unreviewed()
        print_stats()
    elif args.mode == "intent":
        run_intent_mode(only_disagreements=args.only_disagreements, limit=args.limit)
    else:
        run_reply_mode(Path(args.replies), args.limit or 60)


if __name__ == "__main__":
    main()
