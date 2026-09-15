"""Tests for draft construction and the live-claim cross-check.

The cross-check is the system's main safety net: it is what stops a fluent
invented arrival time from being auto-sent. Its first version flagged *any*
clock time, which fired on perfectly safe replies that echoed the customer's
own service reference back at them. These tests pin the distinction.
"""

from __future__ import annotations

import pytest

from hsa.reply.generate import CannedBaseline, Draft, LIVE_CLAIM, _norm_claim
from hsa.retrieve.index import Precedent


def mk(reply: str, msg: str = "", **kw) -> Draft:
    return Draft(reply=reply, customer_msg=msg, **kw)


# ------------------------------------------------------- echo vs invention

@pytest.mark.parametrize(
    "reply,msg",
    [
        ("Please check the app for the 19:22 status.", "any news on the delayed 19:22 to Plymouth?"),
        ("We can look into the 07.53 for you.", "why the delay on the 07.53 TWY-PAD?"),
        ("Sorry about the £45.", "I want my £45 back"),
        ("A 40 minute delay is eligible.", "I had a 40 minute delay yesterday"),
    ],
)
def test_echoing_the_customer_is_not_a_novel_claim(reply, msg):
    assert mk(reply, msg).novel_live_claims == [], "echoing the customer asserts nothing"
    assert mk(reply, msg).regex_flags_live_claim is False


@pytest.mark.parametrize(
    "reply",
    [
        "It is due into Paddington at 18:42.",
        "That service departs from platform 4.",
        "It is running 11 minutes late.",
        "You are owed £120.",
    ],
)
def test_facts_the_customer_never_mentioned_are_flagged(reply):
    d = mk(reply, "where is my train?")
    assert d.regex_flags_live_claim is True
    assert d.novel_live_claims


def test_time_formats_normalise_together():
    """19:22, 19.22 and 1922 must count as the same reference."""
    assert _norm_claim("19:22") == _norm_claim("19.22") == "1922"
    assert mk("check the 19.22", "the 19:22 is late").novel_live_claims == []


def test_empty_customer_message_still_flags_invented_facts():
    assert mk("Arrives at 18:42.", "").regex_flags_live_claim is True


# --------------------------------------------------------- self-reporting

def test_self_report_disagreement_detects_a_model_misreporting_itself():
    d = mk("It arrives at 18:42.", "where is it?", asserts_live_fact=False)
    assert d.self_report_disagrees is True


def test_no_disagreement_when_model_admits_the_claim():
    d = mk("It arrives at 18:42.", "where is it?", asserts_live_fact=True)
    assert d.self_report_disagrees is False


def test_no_disagreement_on_a_clean_refusal():
    d = mk(
        "We don't have live running details. Please check the GWR app, or send "
        "your booking reference and we'll look into it.",
        "is my train late?",
        asserts_live_fact=False,
    )
    assert d.regex_flags_live_claim is False
    assert d.self_report_disagrees is False


# -------------------------------------------------------------- mechanics

def test_max_precedent_score_of_empty_is_zero():
    assert mk("x").max_precedent_score == 0.0


def test_max_precedent_score_picks_the_best():
    ps = [
        Precedent(1, "a", "b", 0.3, "", False),
        Precedent(2, "c", "d", 0.7, "", False),
    ]
    assert mk("x", precedents=ps).max_precedent_score == pytest.approx(0.7)


def test_canned_baseline_is_constant_and_carries_the_message():
    b = CannedBaseline()
    d1 = b.draft("first message", "lost_property")
    d2 = b.draft("totally different", "feedback_positive")
    assert d1.reply == d2.reply
    assert d2.customer_msg == "totally different"


def test_canned_baseline_is_a_deflection():
    """It should score badly on the rubric for exactly this reason."""
    from hsa.evalx.judge import automated_flags

    d = CannedBaseline().draft("anything", "other")
    assert automated_flags(d.reply, "anything")["is_deflection"] is True


def test_live_claim_pattern_ignores_ordinary_prose():
    for s in ["We are sorry about this.", "Please contact the team.", "Thanks for letting us know."]:
        assert not LIVE_CLAIM.search(s), s
