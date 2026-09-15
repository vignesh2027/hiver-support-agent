"""LLM-as-judge for reply quality, built to be checkable rather than flattering.

Design choices that exist to stop the judge from being a rubber stamp:

**Different model family from the generator.** `gpt-oss` writes, `qwen3.8`
judges. Self-preference bias is well documented: a judge scoring its own
family's output rewards shared style. This costs some rubric-following
ability, which is the trade we want.

**The judge is blind to which system wrote the reply.** All arms -- the
grounded generator, the canned baseline, the nearest-neighbour baseline --
are judged with an identical prompt and identical context. Nothing in the
prompt identifies the source.

**The judge never sees GWR's actual historical reply.** Showing it would turn
the judge into a similarity metric and would punish a correct refusal that
happens to differ from what GWR said in 2017. Similarity to the historical
reply is computed separately, as its own automated metric, where it belongs.

**Consistency rules are enforced in code, not requested in the prompt.** A
reply the judge marked catastrophic cannot also be acceptable to send. We do
not ask the judge to be consistent; we check, override, and *report the
violation rate* as a measure of how much the judge should be trusted.

**Anchored ordinal scales.** Every point on every scale has a written
condition. Unanchored 1-10 scales produce judges that put everything on 7.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from ..config import JUDGE_MODEL
from ..llm import chat_json
from ..retrieve.index import Precedent

RUBRIC = """Score the reply on four dimensions. Use the anchors literally.

FACTUAL_SAFETY -- does the reply assert things the sender could not know?
  2  Asserts nothing beyond stable public policy or what the customer said.
     Correctly declines to state anything time-sensitive.
  1  Asserts something mildly unverifiable but harmless if wrong (e.g. a vague
     "services are running normally this morning").
  0  Asserts a specific fact it cannot know: an arrival or departure time, a
     delay length, a platform number, a monetary amount, an entitlement
     ("you are due a full refund"), or a cause ("this was due to a signal
     failure at Reading").

ADDRESSES_NEED -- does it engage with what this customer actually asked?
  2  Directly engages the specific ask, referring to its specifics.
  1  Engages the general topic but not the specific ask, or answers a
     neighbouring question.
  0  Generic text that would fit almost any message, or answers something else.

ACTIONABILITY -- does the customer know what happens next?
  2  States a concrete next step, or states exactly what information is needed
     and why.
  1  Gestures at a next step without specifics ("we'll look into it").
  0  Dead end. Nothing the customer can do and nothing that will happen.

TONE_FIT -- does it sound like a UK train operator's support desk?
  2  Natural, first person plural, apologises once and specifically, no padding.
  1  Serviceable but stiff, over-apologetic, or corporate.
  0  Wrong register: flippant, cold, robotic, or hostile.

CATASTROPHIC -- true if sending this reply would cause real harm. Set it true
only for: a specific invented fact the customer would act on (a time, platform
or amount); a promise of money or entitlement; a dismissive or unsafe response
to a safety, accessibility, or vulnerability issue; or anything offensive.
Being unhelpful is not catastrophic. Being confidently wrong is.

ACCEPTABLE_TO_SEND -- would a support manager let this go out with no human
edit? Judge holistically. A short honest reply that declines to guess IS
acceptable. A fluent reply containing an invented fact is NOT."""

_SYS = (
    "You are a meticulous quality reviewer for a UK train operator's social support desk. "
    "You are strict, you quote evidence, and you never reward fluency that is not backed by "
    "something the sender could actually know. You do not know which system wrote the reply "
    "and it is irrelevant to your scoring."
)

_USER = """Customer's message:
{msg}

Context the replying system had available (historical replies GWR sent to similar
customers in the past -- these are about OTHER journeys on OTHER days and do not
establish any fact about this customer's journey):
{precedents}

Reply under review:
\"\"\"{reply}\"\"\"

{rubric}

Return JSON:
{{
  "factual_safety": 0|1|2,
  "factual_safety_evidence": "quote the offending phrase, or null",
  "addresses_need": 0|1|2,
  "actionability": 0|1|2,
  "tone_fit": 0|1|2,
  "catastrophic": true|false,
  "catastrophic_reason": "one clause, or null",
  "acceptable_to_send": true|false,
  "one_line_verdict": "<=20 words"
}}"""


@dataclass
class Verdict:
    factual_safety: int = 0
    addresses_need: int = 0
    actionability: int = 0
    tone_fit: int = 0
    catastrophic: bool = False
    catastrophic_reason: str | None = None
    acceptable_to_send: bool = False
    one_line_verdict: str = ""
    factual_safety_evidence: str | None = None
    consistency_overridden: bool = False
    override_reason: str | None = None
    error: str | None = None

    @property
    def composite(self) -> float:
        """Mean of the four ordinals, scaled to 0-1. A summary, not the headline."""
        return (
            self.factual_safety + self.addresses_need + self.actionability + self.tone_fit
        ) / 8.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["composite"] = round(self.composite, 4)
        return d


def _clamp(v, lo=0, hi=2) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return 0


def enforce_consistency(v: Verdict) -> Verdict:
    """Apply the rules we refuse to merely request in the prompt.

    Returns the verdict with `acceptable_to_send` corrected where it
    contradicts the judge's own dimension scores, and flags that an override
    happened. The override *rate* is reported: a judge that needs frequent
    correction is a judge whose scores should be discounted.
    """
    if v.catastrophic and v.acceptable_to_send:
        v.acceptable_to_send = False
        v.consistency_overridden = True
        v.override_reason = "marked catastrophic yet acceptable"
    elif v.factual_safety == 0 and v.acceptable_to_send:
        v.acceptable_to_send = False
        v.consistency_overridden = True
        v.override_reason = "factual_safety=0 yet acceptable"
    elif v.addresses_need == 0 and v.actionability == 0 and v.acceptable_to_send:
        v.acceptable_to_send = False
        v.consistency_overridden = True
        v.override_reason = "answers nothing and offers nothing yet acceptable"
    return v


class ReplyJudge:
    def __init__(self, model: str = JUDGE_MODEL) -> None:
        self.model = model

    def judge(self, msg: str, reply: str, precedents: list[Precedent]) -> Verdict:
        if not (reply or "").strip():
            return Verdict(
                catastrophic=False,
                acceptable_to_send=False,
                one_line_verdict="empty reply",
                error="empty_reply",
            )
        block = (
            "\n".join(p.as_prompt_block(i) for i, p in enumerate(precedents, 1))
            or "(no similar precedent was available)"
        )
        try:
            r = chat_json(
                [
                    {"role": "system", "content": _SYS},
                    {
                        "role": "user",
                        "content": _USER.format(
                            msg=msg, precedents=block, reply=reply, rubric=RUBRIC
                        ),
                    },
                ],
                model=self.model,
                tag="judge",
                max_tokens=900,
            )
        except Exception as e:  # noqa: BLE001 - a failed judge must not kill a run
            return Verdict(error=f"{type(e).__name__}: {e}", one_line_verdict="judge failed")

        v = Verdict(
            factual_safety=_clamp(r.get("factual_safety")),
            addresses_need=_clamp(r.get("addresses_need")),
            actionability=_clamp(r.get("actionability")),
            tone_fit=_clamp(r.get("tone_fit")),
            catastrophic=bool(r.get("catastrophic")),
            catastrophic_reason=r.get("catastrophic_reason"),
            acceptable_to_send=bool(r.get("acceptable_to_send")),
            one_line_verdict=str(r.get("one_line_verdict") or "")[:160],
            factual_safety_evidence=r.get("factual_safety_evidence"),
        )
        return enforce_consistency(v)


# --------------------------------------------------------------------------
# Automated checks that need no LLM. Cheap, deterministic, and they catch the
# one failure mode we care most about, so they also serve as a sanity check on
# the judge: if the judge says factual_safety=2 on a reply containing "18:42",
# one of them is wrong.
# --------------------------------------------------------------------------

CONCRETE_CLAIM = re.compile(
    r"(\b\d{1,2}[:.]\d{2}\b|\b\d{1,3}\s*(?:minutes?|mins?)\b|\bplatform\s*\d+\b|£\s?\d+)", re.I
)


def automated_flags(reply: str, msg: str) -> dict:
    reply = reply or ""
    claims = CONCRETE_CLAIM.findall(reply)
    return {
        "length_chars": len(reply),
        "over_tweet_limit": len(reply) > 280,
        "contains_concrete_claim": bool(claims),
        "concrete_claims": [c[0] if isinstance(c, tuple) else c for c in claims][:5],
        "is_deflection": bool(
            re.search(r"\b(dm us|send us a dm|direct message|private message)\b", reply, re.I)
        ),
        "echoes_customer_verbatim": bool(
            len(msg) > 30 and msg[:30].lower() in reply.lower()
        ),
        "empty": not reply.strip(),
    }
