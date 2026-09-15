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


# ------------------------------------------- unresponsiveness (REPORT.md F0)

@pytest.mark.parametrize(
    "reply,msg",
    [
        # The verbatim case that got past every guard and was still unsendable.
        ("Which service was this on? Please let us know the train number.",
         "hi I left my hat on one your trains, it was 16:29 service from Swansea to Cardiff"),
        ("Could you tell us which train you were on?",
         "the 1927 from Pangbourne to Oxford was cancelled"),
    ],
)
def test_asking_for_details_already_given_is_flagged(reply, msg):
    from hsa.reply.generate import asks_for_already_given

    assert asks_for_already_given(reply, msg) is True
    assert mk(reply, msg).repeats_a_question_already_answered is True


@pytest.mark.parametrize(
    "reply,msg",
    [
        # Genuinely under-specified: asking is the right thing to do.
        ("Which service was this on?", "I left my bag on a train"),
        # Not a request at all.
        ("Sorry about your hat. Report it on our lost property form.",
         "left my hat on the 16:29 from Swansea"),
    ],
)
def test_legitimate_questions_are_not_flagged(reply, msg):
    from hsa.reply.generate import asks_for_already_given

    assert asks_for_already_given(reply, msg) is False


def test_unresponsive_draft_is_not_auto_sent():
    """The regression test for F0: every truthfulness guard passes, and it
    still must not be auto-sent."""
    from hsa.route.policy import RouterConfig, route

    d = mk(
        "Hi there. Which service was this on? Please let us know the train number.",
        "hi I left my hat on one your trains, it was 16:29 service from Swansea to Cardiff",
    )
    d.precedents = [Precedent(1, "a", "b", 0.8, "", True)]
    assert d.regex_flags_live_claim is False       # asserts nothing false
    out = route(d.customer_msg, "lost_property", 0.98, d)
    assert out.action != "auto", f"unresponsive reply was routed {out.action}"
    # And turning the guard off reproduces the original failure.
    off = route(d.customer_msg, "lost_property", 0.98, d,
                RouterConfig(block_on_unresponsive=False))
    assert off.action == "auto"
