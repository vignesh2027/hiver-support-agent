"""Behavioural tests for the escalation policy.

The router is the component whose mistakes are expensive, so it is tested by
behaviour ("a message naming a refund amount is never auto-sent") rather than
by asserting which rule fired. That way the tests survive a reordering of the
rules but still fail if a guard is genuinely removed.
"""

from __future__ import annotations

import pytest

from hsa.reply.generate import Draft
from hsa.retrieve.index import Precedent
from hsa.route.policy import RouterConfig, message_risk_flags, route


def mk_draft(reply="Sorry about that, here is what to do next.", score=0.75, live=False):
    p = Precedent(
        episode_id=1,
        customer_msg="prior message",
        brand_reply="prior reply",
        score=score,
        created_at="2017-11-01",
        customer_thanked=True,
    )
    return Draft(reply=reply, confidence=0.9, precedents=[p], asserts_live_fact=live)


# ------------------------------------------------------------- hard blocks

@pytest.mark.parametrize(
    "msg",
    [
        "I was assaulted on the 18:03 and no staff helped",
        "my wife collapsed on the train, we need help",
        "the wheelchair ramp was not available at Reading",
        "there was an accident and someone is injured",
    ],
)
def test_safety_and_accessibility_always_escalate(msg):
    d = route(msg, "lost_property", 0.99, mk_draft())
    assert d.action == "escalate"
    assert "safety" in d.reason.lower() or "human" in d.reason.lower()


@pytest.mark.parametrize(
    "msg",
    [
        "I want my £45 back for last night",
        "you owe me a refund of 32 pounds",
        "compensation of £120 please",
    ],
)
def test_named_money_amounts_never_auto_send(msg):
    d = route(msg, "lost_property", 0.99, mk_draft())
    assert d.action == "escalate", f"money message was routed {d.action}"


def test_legal_threats_escalate():
    d = route("I am taking this to the rail ombudsman", "service_complaint", 0.95, mk_draft())
    assert d.action == "escalate"


def test_existing_case_followup_escalates():
    d = route("I sent a DM three weeks ago and heard nothing", "other", 0.95, mk_draft())
    assert d.action == "escalate"


def test_image_only_message_escalates():
    d = route("<url>", "other", 0.95, mk_draft())
    assert d.action == "escalate"


# --------------------------------------------------------- intent policy

def test_live_status_is_escalate_by_policy():
    d = route("is the 18:21 to Cardiff running?", "live_service_status", 0.99, mk_draft())
    assert d.action == "escalate"


def test_missed_connection_escalates():
    d = route("I have missed my connection, what do I do", "missed_connection", 0.98, mk_draft())
    assert d.action == "escalate"


# ------------------------------------------------------------ uncertainty

def test_low_confidence_escalates():
    d = route("left my bag on the train", "lost_property", 0.20, mk_draft())
    assert d.action == "escalate"
    assert "confidence" in d.reason.lower()


def test_weak_grounding_downgrades_to_assist_not_auto():
    d = route("left my bag on the train", "lost_property", 0.95, mk_draft(score=0.05))
    assert d.action == "assist"


def test_empty_draft_escalates():
    d = route("left my bag on the train", "lost_property", 0.95, mk_draft(reply="   "))
    assert d.action == "escalate"


# ------------------------------------------------------- draft properties

def test_draft_asserting_a_time_escalates_even_if_model_denies_it():
    """The regex cross-check must catch a model that misreports itself."""
    d = mk_draft(reply="That service is due into Paddington at 18:42.", live=False)
    assert d.regex_flags_live_claim is True
    assert d.self_report_disagrees is True
    out = route("when does it arrive?", "lost_property", 0.99, d)
    assert out.action == "escalate"


def test_draft_asserting_a_platform_escalates():
    d = mk_draft(reply="It departs from platform 4.", live=False)
    out = route("where does it go from", "lost_property", 0.99, d)
    assert out.action == "escalate"


def test_clean_lost_property_reply_is_auto():
    d = route(
        "I left my coat on the 1606 from Paddington, how do I get it back?",
        "lost_property",
        0.94,
        mk_draft(reply="Sorry about your coat. Report it on our lost property form and "
                       "quote the service and coach so the team can trace it."),
    )
    assert d.action == "auto", f"expected auto, got {d.action}: {d.reason}"


def test_clean_praise_is_not_auto_when_intent_default_is_auto_but_grounding_weak():
    d = route("thanks, great service today", "feedback_positive", 0.95, mk_draft(score=0.01))
    assert d.action == "assist"


# ------------------------------------------------------------- ablations

def test_removing_the_live_claim_guard_lets_an_invented_time_through():
    """If this ever passes with the guard on, the guard is not doing anything."""
    d = mk_draft(reply="It should arrive at 19:05.", live=True)
    guarded = route("when will it get in?", "lost_property", 0.99, d)
    unguarded = route(
        "when will it get in?", "lost_property", 0.99, d,
        RouterConfig(block_on_live_claim=False),
    )
    assert guarded.action == "escalate"
    assert unguarded.action != "escalate"


def test_every_decision_carries_a_reason_and_a_rule_id():
    for msg, intent in [
        ("is the 9:15 late", "live_service_status"),
        ("left my hat", "lost_property"),
        ("£20 refund please", "compensation_and_refunds"),
        ("thanks!", "feedback_positive"),
    ]:
        d = route(msg, intent, 0.9, mk_draft())
        assert d.reason.strip() and d.rule.strip()
        assert d.action in ("auto", "assist", "escalate")


def test_unknown_intent_does_not_crash():
    d = route("something", "not_a_real_intent", 0.9, mk_draft())
    assert d.action in ("auto", "assist", "escalate")


# ------------------------------------------------------------- risk flags

def test_risk_flags_are_precise_enough_to_be_useful():
    """A probe that fires on ordinary messages selects nothing interesting."""
    benign = [
        "is the train to Bath running on time",
        "left my umbrella on the 0812",
        "what platform please",
        "thanks for the help today",
    ]
    for m in benign:
        assert message_risk_flags(m) == [], f"{m!r} spuriously flagged {message_risk_flags(m)}"
