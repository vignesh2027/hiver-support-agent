"""Mine the run logs for failure modes, with real examples.

Written so the report's failure analysis is found rather than chosen. Each
detector states a hypothesis, counts how often it fires, and pulls verbatim
examples. Cherry-picking five interesting-looking failures by hand would
produce a nicer story and a less trustworthy one: you cannot tell whether a
hand-picked failure happens twice or two hundred times.

Every mode reports its rate over the relevant denominator, so a vivid failure
that occurs once is visibly rare and a boring one that occurs constantly is
visibly common.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd

from ..config import ARTIFACTS, METRICS

RUNS = ARTIFACTS / "runs"
OUT = METRICS / "failure_modes.json"


@dataclass
class FailureMode:
    key: str
    title: str
    hypothesis: str
    n: int
    denominator: int
    examples: list[dict] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.n / self.denominator if self.denominator else 0.0

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "hypothesis": self.hypothesis,
            "n": self.n,
            "denominator": self.denominator,
            "rate": round(self.rate, 4),
            "examples": self.examples,
        }


def _ex(r, extra: dict | None = None) -> dict:
    d = {
        "example_id": r.example_id,
        "stratum": r.stratum,
        "customer_msg": str(r.customer_msg)[:260],
        "reply": str(getattr(r, "reply", ""))[:260],
        "pred_intent": getattr(r, "pred_intent", None),
        "action": getattr(r, "action", None),
        "action_rule": getattr(r, "action_rule", None),
    }
    if extra:
        d.update(extra)
    return d


def find_modes(system_arm: str = "grounded_llm", k_examples: int = 3) -> list[FailureMode]:
    gen = pd.read_json(RUNS / "generation.jsonl", lines=True)
    sysg = gen[gen.arm == system_arm].copy()
    modes: list[FailureMode] = []

    jpath = RUNS / "judgements.jsonl"
    jud = pd.read_json(jpath, lines=True) if jpath.exists() else pd.DataFrame()
    have_judge = len(jud) > 0
    if have_judge:
        j = jud[jud.arm == system_arm][
            ["example_id", "factual_safety", "addresses_need", "actionability",
             "tone_fit", "catastrophic", "acceptable_to_send", "one_line_verdict",
             "factual_safety_evidence"]
        ]
        sysg = sysg.merge(j, on="example_id", how="left")

    # ---------------------------------------------------------------- F1
    # The failure the whole design is built to prevent, so its rate is the
    # single most important number in the report.
    hit = sysg[sysg.regex_live_claim.fillna(False).astype(bool)]
    modes.append(FailureMode(
        key="invented_concrete_fact",
        title="Draft states a time, platform or amount the customer never mentioned",
        hypothesis=(
            "Precedents are full of concrete operational facts about other journeys. "
            "A model asked to imitate them will carry a number across even when told "
            "not to. This is the failure that makes an auto-reply dangerous rather "
            "than merely unhelpful."
        ),
        n=len(hit), denominator=len(sysg),
        examples=[_ex(r, {"novel_claims": getattr(r, "auto_concrete_claims", None)})
                  for r in hit.head(k_examples).itertuples()],
    ))

    # ---------------------------------------------------------------- F2
    mis = sysg[sysg.self_report_disagrees.fillna(False).astype(bool)]
    modes.append(FailureMode(
        key="model_misreports_itself",
        title="Model declares it asserted no time-sensitive fact, but it did",
        hypothesis=(
            "The generator is asked to self-declare whether its draft contains an "
            "unverifiable claim. A model that hallucinates a fact is not a reliable "
            "witness to having hallucinated it, so any safety design that trusts the "
            "self-report alone inherits the same blind spot."
        ),
        n=len(mis), denominator=len(sysg),
        examples=[_ex(r) for r in mis.head(k_examples).itertuples()],
    ))

    # ---------------------------------------------------------------- F3
    # Retrieval can only help where a precedent exists.
    weak = sysg[sysg.precedent_score < 0.22]
    modes.append(FailureMode(
        key="no_usable_precedent",
        title="Nothing similar enough in the archive to ground a reply",
        hypothesis=(
            "Retrieval-grounded generation silently degrades to unguided generation "
            "when the index has no close match. Without an explicit grounding floor "
            "the system keeps answering confidently on exactly the messages it knows "
            "least about."
        ),
        n=len(weak), denominator=len(sysg),
        examples=[_ex(r, {"precedent_score": float(r.precedent_score)})
                  for r in weak.head(k_examples).itertuples()],
    ))

    # ---------------------------------------------------------------- F4
    # Deflection: the agent reproduces the brand's worst habit.
    defl = sysg[sysg.auto_is_deflection.fillna(False).astype(bool)]
    modes.append(FailureMode(
        key="learned_to_deflect",
        title="Reply is a hand-off rather than an answer",
        hypothesis=(
            "Even with deflections filtered out of the index, the register of the "
            "corpus pulls the generator toward 'send us a DM'. When it does this it "
            "scores well on tone and badly on usefulness, which is precisely the "
            "combination a tone-weighted rubric would miss."
        ),
        n=len(defl), denominator=len(sysg),
        examples=[_ex(r) for r in defl.head(k_examples).itertuples()],
    ))

    # ---------------------------------------------------------------- F5
    over = sysg[sysg.auto_over_tweet_limit.fillna(False).astype(bool)]
    modes.append(FailureMode(
        key="over_length",
        title="Reply exceeds the 280-character limit of the channel",
        hypothesis=(
            "A constraint stated in the prompt is not a constraint. Length is the "
            "cheapest possible thing to verify in code, and any reply over the limit "
            "is unusable regardless of how good it is."
        ),
        n=len(over), denominator=len(sysg),
        examples=[_ex(r, {"chars": int(r.auto_length_chars)})
                  for r in over.head(k_examples).itertuples()],
    ))

    # ---------------------------------------------------------------- F6
    empt = sysg[sysg.reply.fillna("").astype(str).str.strip() == ""]
    if len(empt):
        modes.append(FailureMode(
            key="empty_reply",
            title="Generator returned nothing at all",
            hypothesis=(
                "Reasoning-family models can spend their entire output budget on "
                "hidden reasoning and return empty content. Scored naively this looks "
                "like a merely bad reply rather than a broken call."
            ),
            n=len(empt), denominator=len(sysg),
            examples=[_ex(r) for r in empt.head(k_examples).itertuples()],
        ))

    # ------------------------------------------------- judge-dependent modes
    if have_judge and "catastrophic" in sysg:
        cat = sysg[sysg.catastrophic.fillna(False).astype(bool)]
        modes.append(FailureMode(
            key="judged_catastrophic",
            title="Judge flagged the reply as actively harmful to send",
            hypothesis=(
                "The rate that matters operationally is not average quality but how "
                "often something genuinely damaging gets through, and whether the "
                "router caught it."
            ),
            n=len(cat), denominator=len(sysg),
            examples=[_ex(r, {"why": getattr(r, "one_line_verdict", None),
                              "evidence": getattr(r, "factual_safety_evidence", None)})
                      for r in cat.head(k_examples).itertuples()],
        ))

        # The dangerous quadrant: judge says do not send, router said send.
        if "action" in sysg:
            escaped = sysg[
                (~sysg.acceptable_to_send.fillna(True).astype(bool))
                & (sysg.action == "auto")
            ]
            modes.append(FailureMode(
                key="router_escape",
                title="Router auto-sent a reply the judge would not have sent",
                hypothesis=(
                    "Every guard has a gap. These are the cases where the whole "
                    "pipeline failed together, and they set the floor on how much "
                    "autonomy this system can be given."
                ),
                n=len(escaped), denominator=int((sysg.action == "auto").sum()),
                examples=[_ex(r, {"why": getattr(r, "one_line_verdict", None)})
                          for r in escaped.head(k_examples).itertuples()],
            ))

    return modes


def main() -> None:
    modes = find_modes()
    payload = {"modes": [m.to_dict() for m in modes]}
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print(f"{'mode':<30} {'n':>5} {'/ denom':>9} {'rate':>7}")
    print("-" * 56)
    for m in sorted(modes, key=lambda x: -x.rate):
        print(f"{m.key:<30} {m.n:>5} {m.denominator:>9} {m.rate:>7.1%}")
    print(f"\nwrote {OUT}")

    top = sorted(modes, key=lambda x: -x.rate)[:3]
    for m in top:
        if not m.examples:
            continue
        print(f"\n### {m.title}  ({m.n}/{m.denominator}, {m.rate:.1%})")
        for e in m.examples[:2]:
            print(f"  C: {e['customer_msg'][:110]}")
            print(f"  A: {e['reply'][:110]}")
            print(f"     -> {e.get('action')} via {e.get('action_rule')}")


if __name__ == "__main__":
    main()
