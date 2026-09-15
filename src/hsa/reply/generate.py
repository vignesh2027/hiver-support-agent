"""Draft a reply grounded in how GWR historically resolved similar issues.

The generator's hardest requirement is not fluency, it is **refusal**. 53% of
this brand's traffic needs live running data the agent does not have. A model
given four precedents that each contain a concrete arrival time will happily
produce a fifth concrete arrival time, and it will be fluent, on-brand, and
invented. So the system prompt's main job is to make "I cannot know that"
a first-class output, and the response schema forces the model to declare
whether its own draft asserts anything time-sensitive.

That self-declaration is not trusted on its own -- a model that hallucinates a
train time may also misreport that it did. It is one input to the router,
which cross-checks it against a regex for concrete claims. See
`hsa.route.policy`.

Baselines, for the report's comparison table:

* `CannedBaseline` (trivial) -- one fixed apology for every message. This is
  what a bad deployment actually looks like, and on a rubric that rewards tone
  it scores much better than it deserves, which is a useful warning about the
  rubric.
* `NearestNeighbourBaseline` (simple) -- return, verbatim, the reply GWR sent
  to the most similar historical message. No LLM at all. This is a genuinely
  strong baseline for a brand with formulaic replies and it is the bar the
  generated replies have to clear.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import GEN_MODEL
from ..llm import chat_json
from ..retrieve.index import Precedent, PrecedentIndex
from ..taxonomy.schema import load as load_taxonomy

# Concrete, checkable assertions. Used to cross-check the generator's own
# claim that it did not assert a live fact.
LIVE_CLAIM = re.compile(
    r"("
    r"\b\d{1,2}[:.]\d{2}\b|"                      # a clock time
    r"\b\d{1,3}\s*(minutes?|mins?|hours?)\b|"     # a duration
    r"\bplatform\s*\d+\b|"                        # a platform number
    r"\b(is|will be|should be|due)\s+(running|arriving|departing|delayed|cancelled|on time)\b|"
    r"\b(due (in|at)|arriv\w+ at|depart\w+ at)\b|"
    r"£\s?\d+"                                     # a money amount
    r")",
    re.I,
)

_SYS = """You draft replies for Great Western Railway's Twitter support desk.

You are given the customer's message, its intent, and real replies GWR sent to
similar customers in the past. Write the next reply.

Non-negotiable rules:

1. You have NO access to live running information, the customer's booking, or
   any account. You cannot know whether a specific train is delayed right now,
   when it will arrive, what platform it leaves from, or how much a customer is
   owed.

2. The precedents show you GWR's TONE and PROCESS. They do not transfer facts.
   If a precedent says "the 18:03 is running 11 minutes late", that was true
   about a different train on a different day. Never carry a specific time,
   platform, duration or amount out of a precedent into your reply.

3. When the customer needs a fact you cannot have, say what you can do instead:
   acknowledge the specific problem, and name the exact next step or the exact
   information you need from them. A short honest reply beats a fluent invented
   one.

4. Match the historical register: first person plural, apologise once and
   specifically, no corporate padding, under 280 characters. Do not open with
   "We're sorry to hear that" unless something bad actually happened.

5. Never promise compensation, refunds, or an amount. Never state a policy
   detail that does not appear in at least one precedent.

Return JSON only."""

_USER = """Customer message:
{msg}

Classified intent: {intent}
Intent definition: {definition}
This intent typically requires live data: {needs_live}

Historical precedents from GWR (most similar first):
{precedents}

Return JSON:
{{
  "reply": "the reply text, under 280 characters",
  "grounded_in": [precedent numbers you actually used, e.g. [1,3]],
  "asserts_time_sensitive_fact": true|false,
  "asserts_time_sensitive_fact_note": "what fact, or null",
  "missing_information": "what you would need to answer fully, or null",
  "confidence": 0.0-1.0
}}"""


def _norm_claim(s: str) -> str:
    """Normalise a claim so 19:22, 19.22 and 1922 compare equal."""
    return re.sub(r"[^0-9a-z]", "", s.lower())


# A draft asking the customer for something they already wrote. Added after a
# reply got past every safety guard and was still unsendable:
#
#   customer: "I left my hat on one your trains, it was 16:29 Swansea to Cardiff"
#   draft:    "Which service was this on? Please let us know the train number."
#
# The judge scored it factual_safety 2 and not catastrophic, because nothing in
# it is false. It is simply not listening. Every guard in the router is about
# truthfulness and none of them noticed. See REPORT.md F0.
_ASKS_FOR_SERVICE = re.compile(
    r"("
    r"\b(which|what)\s+(service|train|time)\b|"
    r"\blet us know\b[^.?!]{0,40}\b(service|train|time|number)\b|"
    r"\b(could|can) you\s+(tell|let|confirm|provide|share)\b|"
    r"\bplease\s+(provide|share|confirm|let us know|send)\b"
    r")",
    re.I,
)
# Something that identifies a specific service. A clock time or a four-digit
# departure ("16:29", "06.24", "1927"), or a named route between two places.
#
# The first version accepted any "<word> to <word>", which matched ordinary
# English ("only leave when we got to the station") and fired the guard on
# messages that named no service at all. Requiring capitalised words either
# side, or a CRS-style code pair like WSM-PAD, keeps the genuine catches
# ("the 8.19 from Paddington to Newbury", where the draft then asks for the
# departure time) and drops the noise.
_GIVES_SERVICE = re.compile(
    r"("
    r"\b\d{1,2}[:.]\d{2}\b|"                   # 16:29, 06.24
    r"\b[01]\d{3}\b|\b2[0-3]\d{2}\b|"          # 1927, 0624
    r"\b[A-Z][a-z]{2,}\s+to\s+[A-Z][a-z]{2,}\b|"  # Swansea to Cardiff
    r"\b[A-Z]{3}\s*[-–]\s*[A-Z]{3}\b"          # WSM-PAD
    r")"
)


def asks_for_already_given(reply: str, customer_msg: str) -> bool:
    """True when the draft requests service details the customer already gave.

    Deliberately narrow. It only fires when the reply asks for a service or
    time *and* the message already identifies one, because the expensive
    mistake is asking a customer to repeat themselves, not asking a genuinely
    under-specified question.
    """
    if not reply or not customer_msg:
        return False
    return bool(_ASKS_FOR_SERVICE.search(reply) and _GIVES_SERVICE.search(customer_msg))


@dataclass
class Draft:
    reply: str
    customer_msg: str = ""
    grounded_in: list[int] = field(default_factory=list)
    asserts_live_fact: bool = False
    asserts_live_fact_note: str | None = None
    missing_information: str | None = None
    confidence: float = 0.0
    precedents: list[Precedent] = field(default_factory=list)
    source: str = "llm"

    @property
    def max_precedent_score(self) -> float:
        return max((p.score for p in self.precedents), default=0.0)

    @property
    def novel_live_claims(self) -> list[str]:
        """Concrete claims in the reply that the customer did not already state.

        The naive version of this check -- "does the reply contain a clock
        time?" -- fires on safe replies that simply echo the customer back:

            customer: "any news on the delayed 19:22 to Plymouth?"
            reply:    "We don't have live running details; please check the
                       app for the 19:22 status."

        Repeating the customer's own service reference asserts nothing. What
        is dangerous is a time, platform, duration or amount that appears in
        the reply but *not* in the message it answers, because that can only
        have come from the model or from a precedent about a different
        journey. Comparing against the customer's text turns a noisy tripwire
        into a specific one. Found by reading drafts, not by a test.
        """
        haystack = _norm_claim(self.customer_msg)
        out = []
        for m in LIVE_CLAIM.findall(self.reply):
            claim = m[0] if isinstance(m, tuple) else m
            if not claim:
                continue
            if _norm_claim(claim) and _norm_claim(claim) not in haystack:
                out.append(claim)
        return out

    @property
    def regex_flags_live_claim(self) -> bool:
        """Independent check on the model's self-report."""
        return bool(self.novel_live_claims)

    @property
    def repeats_a_question_already_answered(self) -> bool:
        """The draft asks for service details the customer already supplied."""
        return asks_for_already_given(self.reply, self.customer_msg)

    @property
    def self_report_disagrees(self) -> bool:
        """The model said it asserted no live fact, but the text contains one."""
        return self.regex_flags_live_claim and not self.asserts_live_fact


class GroundedGenerator:
    name = "grounded_llm"

    def __init__(self, index: PrecedentIndex, model: str = GEN_MODEL, top_k: int = 4) -> None:
        self.index = index
        self.model = model
        self.top_k = top_k
        self.tax = load_taxonomy()

    def draft(self, msg: str, intent: str) -> Draft:
        precedents = self.index.search(msg, k=self.top_k)
        it = self.tax.get(intent) if self.tax.is_valid(intent) else None
        block = (
            "\n".join(p.as_prompt_block(i) for i, p in enumerate(precedents, 1))
            or "(no similar precedent found)"
        )
        res = chat_json(
            [
                {"role": "system", "content": _SYS},
                {
                    "role": "user",
                    "content": _USER.format(
                        msg=msg,
                        intent=intent,
                        definition=it.definition if it else "unknown",
                        needs_live=it.needs_live_data if it else "unknown",
                        precedents=block,
                    ),
                },
            ],
            model=self.model,
            tag="generate",
            max_tokens=900,
            temperature=0.2,
        )
        return Draft(
            reply=str(res.get("reply", "")).strip(),
            customer_msg=msg,
            grounded_in=[int(x) for x in (res.get("grounded_in") or []) if str(x).isdigit()],
            asserts_live_fact=bool(res.get("asserts_time_sensitive_fact")),
            asserts_live_fact_note=res.get("asserts_time_sensitive_fact_note"),
            missing_information=res.get("missing_information"),
            confidence=float(res.get("confidence") or 0.0),
            precedents=precedents,
        )


class CannedBaseline:
    """Trivial baseline: the same apology for every message."""

    name = "canned"
    TEXT = (
        "Sorry to hear about this. Please send us a DM with your journey details "
        "and we'll look into it for you."
    )

    def __init__(self, index: PrecedentIndex | None = None) -> None:
        self.index = index

    def draft(self, msg: str, intent: str) -> Draft:
        return Draft(reply=self.TEXT, customer_msg=msg, confidence=1.0, source="canned")


class NearestNeighbourBaseline:
    """Simple baseline: replay GWR's reply to the most similar past message."""

    name = "nearest_neighbour"

    def __init__(self, index: PrecedentIndex) -> None:
        self.index = index

    def draft(self, msg: str, intent: str) -> Draft:
        hits = self.index.search(msg, k=1)
        if not hits:
            return Draft(reply="", customer_msg=msg, confidence=0.0, source="nn_empty")
        p = hits[0]
        return Draft(
            reply=p.brand_reply,
            customer_msg=msg,
            grounded_in=[1],
            confidence=p.score,
            precedents=hits,
            source="nearest_neighbour",
        )
