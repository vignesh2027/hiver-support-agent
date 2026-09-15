"""Decide: auto-send, draft for a human, or escalate — and say why.

This is a rule list, not a learned model, and that is deliberate. Three
reasons:

1. **There is no training data for it.** Nothing in the corpus records whether
   a reply should have been automated. A learned router would be fitted to
   labels we invented, and would inherit their errors while hiding them behind
   a probability.

2. **The reason is part of the output.** The assignment asks for a stated
   reason. A rule list produces the true reason by construction; a classifier
   produces a post-hoc rationalisation.

3. **The failure mode is asymmetric.** Wrongly escalating costs a few minutes
   of an agent's time. Wrongly auto-sending an invented train time costs a
   missed train, and at scale, trust. Rules let us be conservative in exactly
   the places we choose, and show a reviewer where those places are.

The thresholds are the only tunable parts, and sweeping them produces the
risk-coverage curve that is the real headline of this system: quality is only
meaningful *at a stated coverage*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..evalx.sample_golden import PROBES
from ..reply.generate import Draft
from ..taxonomy.schema import load as load_taxonomy

Action = Literal["auto", "assist", "escalate"]


@dataclass(frozen=True)
class RouterConfig:
    """Thresholds. Sweeping these traces the risk-coverage curve."""

    min_intent_confidence: float = 0.55
    # Calibrated against the observed distribution, not guessed. Max-precedent
    # cosine over the 220 golden messages runs p5=0.18, p25=0.22, p50=0.25,
    # p90=0.37. TF-IDF similarity between two ~100-character tweets is low in
    # absolute terms even when the match is good, so a threshold borrowed from
    # dense-retrieval intuition (0.7+) would abstain on literally everything.
    # 0.22 is the empirical p25: it withholds the worst-grounded quarter.
    # `sweep_configs` reports the whole curve so this is an operating point,
    # not a hidden constant. See DECISIONS.md D-12.
    min_precedent_score: float = 0.22
    # Hard blocks. Turning any of these off should visibly worsen the
    # catastrophic-error rate; the ablation table in the report checks that.
    block_on_safety: bool = True
    block_on_legal: bool = True
    block_on_money_amount: bool = True
    block_on_existing_case: bool = True
    block_on_image_only: bool = True
    block_on_live_claim: bool = True
    # If False, intents whose taxonomy default is "escalate" can still be
    # auto-handled when every other signal is clean. Used for ablation only.
    respect_intent_default: bool = True


@dataclass
class Decision:
    action: Action
    reason: str
    rule: str
    signals: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {"action": self.action, "reason": self.reason, "rule": self.rule, **self.signals}


def message_risk_flags(msg: str) -> list[str]:
    """Risk markers present in the customer's own words."""
    return [name for name, rx in PROBES.items() if rx.search(msg)]


def route(
    msg: str,
    intent: str,
    intent_confidence: float,
    draft: Draft | None = None,
    cfg: RouterConfig | None = None,
) -> Decision:
    """First matching rule wins, so the reason is always the binding constraint."""
    cfg = cfg or RouterConfig()
    tax = load_taxonomy()
    flags = message_risk_flags(msg)
    it = tax.get(intent) if tax.is_valid(intent) else None

    sig = {
        "intent": intent,
        "intent_confidence": round(float(intent_confidence), 3),
        "risk_flags": flags,
        "precedent_score": round(draft.max_precedent_score, 3) if draft else None,
        "draft_claims_live_fact": bool(draft.asserts_live_fact) if draft else None,
        "regex_live_claim": bool(draft.regex_flags_live_claim) if draft else None,
        "self_report_disagrees": bool(draft.self_report_disagrees) if draft else None,
    }

    def D(action: Action, rule: str, reason: str) -> Decision:
        return Decision(action=action, reason=reason, rule=rule, signals=sig)

    # -- Hard blocks: properties of the customer's message itself ------------
    if cfg.block_on_safety and "safety_or_accessibility" in flags:
        return D("escalate", "R1-safety",
                 "Message mentions safety, injury or accessibility; a human must handle it.")

    if cfg.block_on_legal and "legal_escalation" in flags:
        return D("escalate", "R2-legal",
                 "Customer has invoked legal or ombudsman escalation; replies carry legal weight.")

    if cfg.block_on_money_amount and "money_amount" in flags:
        return D("escalate", "R3-money",
                 "A specific monetary amount is named; only a human may discuss or commit money.")

    if cfg.block_on_existing_case and "existing_case_followup" in flags:
        return D("escalate", "R4-existing-case",
                 "Customer is chasing an existing case; replying without that history repeats it.")

    if cfg.block_on_image_only and "image_only" in flags:
        return D("escalate", "R5-unreadable",
                 "The content is in an image this text-only system cannot read.")

    # -- Intent-level policy ------------------------------------------------
    if cfg.respect_intent_default and it is not None and it.default_disposition == "escalate":
        return D("escalate", "R6-intent-policy",
                 f"Intent '{intent}' is escalate-by-policy: {it.disposition_rationale.split('.')[0]}.")

    # -- Uncertainty --------------------------------------------------------
    if intent_confidence < cfg.min_intent_confidence:
        return D("escalate", "R7-low-confidence",
                 f"Intent confidence {intent_confidence:.2f} is below "
                 f"{cfg.min_intent_confidence:.2f}; the wrong playbook would be applied.")

    if draft is None:
        return D("assist", "R8-no-draft", "No draft was produced; a human must write the reply.")

    # -- Properties of the draft -------------------------------------------
    if cfg.block_on_live_claim and (draft.asserts_live_fact or draft.regex_flags_live_claim):
        which = "the model reported" if draft.asserts_live_fact else "a check detected"
        return D("escalate", "R9-live-claim",
                 f"The draft asserts a time-sensitive fact ({which} it) that cannot be verified.")

    if draft.max_precedent_score < cfg.min_precedent_score:
        return D("assist", "R10-weak-grounding",
                 f"Best precedent similarity {draft.max_precedent_score:.2f} is below "
                 f"{cfg.min_precedent_score:.2f}; the reply is not well grounded.")

    if not draft.reply.strip():
        return D("escalate", "R11-empty-draft", "The generator returned an empty reply.")

    # -- Clear to automate, if the intent allows it -------------------------
    if it is not None and it.default_disposition == "auto":
        return D("auto", "R12-auto-eligible",
                 f"Intent '{intent}' is auto-eligible, grounding is strong "
                 f"({draft.max_precedent_score:.2f}) and no risk markers fired.")

    return D("assist", "R13-default-assist",
             "Draft is usable and low-risk, but this intent commits GWR to something; "
             "a human should approve it.")


def sweep_configs() -> list[tuple[str, RouterConfig]]:
    """Configurations for the risk-coverage curve and the ablation table."""
    out: list[tuple[str, RouterConfig]] = []
    for c in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
        for r in (0.20, 0.28, 0.36):
            out.append((f"conf{c:.2f}_ret{r:.2f}",
                        RouterConfig(min_intent_confidence=c, min_precedent_score=r)))
    # Ablations: each removes exactly one guard so its contribution is visible.
    out += [
        ("ablate_none", RouterConfig()),
        ("ablate_live_claim", RouterConfig(block_on_live_claim=False)),
        ("ablate_money", RouterConfig(block_on_money_amount=False)),
        ("ablate_safety", RouterConfig(block_on_safety=False)),
        ("ablate_intent_policy", RouterConfig(respect_intent_default=False)),
        ("ablate_all_guards", RouterConfig(
            block_on_safety=False, block_on_legal=False, block_on_money_amount=False,
            block_on_existing_case=False, block_on_image_only=False,
            block_on_live_claim=False, respect_intent_default=False,
            min_intent_confidence=0.0, min_precedent_score=0.0)),
    ]
    return out
