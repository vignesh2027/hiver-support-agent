"""Tests for corpus preparation.

The deflection detector gets the most attention here because it decided which
brand this whole project is built on. Its first version scored AmazonHelp at
1.3% deflection; the corrected version scores it at 16.8%. A regression that
quietly reverted it would invalidate the brand choice without failing anything
else, so the phrasings that caught the bug are pinned as test cases.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hsa.data.prepare import (
    _ACK_ONLY,
    _DEFLECT,
    _THANKS,
    chronological_split,
    clean_text,
    latin_ratio,
)


# ----------------------------------------------------------- normalisation

def test_clean_text_strips_mentions_and_urls():
    out = clean_text("@GWRHelp my train is late https://t.co/abc123")
    assert "@GWRHelp" not in out
    assert "t.co" not in out
    assert "<url>" in out
    assert "my train is late" in out


def test_clean_text_unescapes_html_entities():
    assert "&" in clean_text("fish &amp; chips")
    assert "&amp;" not in clean_text("fish &amp; chips")


def test_clean_text_handles_non_strings():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\n\nc") == "a b c"


# -------------------------------------------------------------- deflection

@pytest.mark.parametrize(
    "reply",
    [
        # The phrasings that the FIRST detector missed. These are the
        # regression cases for the brand-selection bug (DECISIONS.md D-02).
        "Kindly drop in your details through the link provided earlier",
        "I am sorry to hear this. Can I ask for you to reach out to us here please",
        "That's odd! We'd like to look into it. Kindly report this to our support team",
        "Please share your order number via the link below",
        "Our team will contact you shortly",
        "Please fill out this form and we will investigate",
        # The obvious ones the first detector did catch.
        "Please DM us your booking reference",
        "Send us a direct message with your details",
        "Please call us on 0345 7000 125",
    ],
)
def test_deflection_detector_catches_handoffs(reply):
    assert _DEFLECT.search(reply), f"missed deflection: {reply!r}"


@pytest.mark.parametrize(
    "reply",
    [
        "That service is currently running 11 minutes late but is expected to reach Filton at 11:18.",
        "You don't need to collect the tickets. Just detail on the form that they haven't been collected.",
        "Compensation is only due for delays of at least an hour on our High Speed Services.",
        "More carriages than usual are undergoing maintenance repairs.",
        "We will run 24 additional trains to and from Bath Spa this weekend.",
    ],
)
def test_deflection_detector_does_not_fire_on_real_answers(reply):
    """A detector that fires on substantive replies would invert the scorecard."""
    assert not _DEFLECT.search(reply), f"false positive: {reply!r}"


@pytest.mark.parametrize(
    "reply",
    [
        "Sorry to hear this.",
        "Hi, so sorry about that!",
        "Apologies for this.",
        "Oh no!",
    ],
)
def test_ack_only_detector(reply):
    assert _ACK_ONLY.search(reply)


def test_ack_only_does_not_fire_on_apology_plus_substance():
    """An apology that carries information is not an empty acknowledgement."""
    assert not _ACK_ONLY.search(
        "Sorry for the delay. The 18:03 was cancelled due to a fault and the next "
        "service leaves in 20 minutes."
    )


def test_thanks_detector():
    assert _THANKS.search("thanks, that worked!")
    assert _THANKS.search("all good now")
    assert not _THANKS.search("this is still broken")


# ---------------------------------------------------------------- language

def test_latin_ratio_separates_scripts():
    assert latin_ratio("hello world") == 1.0
    assert latin_ratio("今更Amazonビデオやべえことに気づく") < 0.5
    assert latin_ratio("Ar y ffordd i Llundain") == 1.0  # Welsh is Latin script
    assert latin_ratio("") == 0.0
    assert latin_ratio("12345 !!!") == 0.0


# ------------------------------------------------------------------ splits

def _fake_episodes(n=100):
    return pd.DataFrame(
        {
            "episode_id": range(n),
            "created_at": pd.date_range("2017-10-01", periods=n, freq="h", tz="UTC"),
            "customer_msg": [f"msg {i}" for i in range(n)],
        }
    )


def test_chronological_split_has_no_time_overlap():
    """The property that prevents retrieval leakage (DECISIONS.md D-04)."""
    s = chronological_split(_fake_episodes())
    assert s["train"]["created_at"].max() < s["dev"]["created_at"].min()
    assert s["dev"]["created_at"].max() < s["test"]["created_at"].min()


def test_chronological_split_partitions_every_episode_exactly_once():
    ep = _fake_episodes()
    s = chronological_split(ep)
    ids = pd.concat([v["episode_id"] for v in s.values()])
    assert len(ids) == len(ep)
    assert set(ids) == set(ep["episode_id"])
    assert not ids.duplicated().any()


def test_chronological_split_respects_requested_proportions():
    s = chronological_split(_fake_episodes(1000), train_frac=0.7, dev_frac=0.1)
    assert len(s["train"]) == 700
    assert len(s["dev"]) == 100
    assert len(s["test"]) == 200


def test_chronological_split_is_order_independent():
    """Shuffled input must produce the same split, since it sorts by time."""
    ep = _fake_episodes(50)
    a = chronological_split(ep)
    b = chronological_split(ep.sample(frac=1, random_state=1))
    assert list(a["test"]["episode_id"]) == list(b["test"]["episode_id"])
