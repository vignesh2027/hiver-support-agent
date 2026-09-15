"""Tests for the LLM client's non-network behaviour.

None of these make an API call. They cover the parts that would break the
reproducibility claim or silently corrupt a long run: cache keying, JSON
extraction from imperfect model output, the rate limiters, and the per-model
output ceiling that killed two labelling runs before it existed.
"""

from __future__ import annotations

import json
import time

import pytest

from hsa.llm import (
    DEFAULT_MAX_OUTPUT,
    MODEL_MAX_OUTPUT,
    NO_JSON_MODE,
    _TokenBucket,
    _cache_key,
    _cache_path,
    _estimate_tokens,
    _extract_json,
)


# ------------------------------------------------------------------- cache

def test_cache_key_is_stable_and_order_independent():
    a = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.0}
    b = {"temperature": 0.0, "messages": [{"role": "user", "content": "hi"}], "model": "m"}
    assert _cache_key(a) == _cache_key(b)


def test_cache_key_changes_with_any_parameter():
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.0}
    for field, value in [
        ("model", "other"),
        ("temperature", 0.7),
        ("messages", [{"role": "user", "content": "bye"}]),
    ]:
        variant = dict(base)
        variant[field] = value
        assert _cache_key(variant) != _cache_key(base), field


def test_cache_key_survives_unicode():
    k = _cache_key({"messages": [{"role": "user", "content": "£45 refund, 今更"}]})
    assert len(k) == 32


def test_cache_paths_are_sharded():
    p = _cache_path("ab" + "0" * 30)
    assert p.parent.name == "ab"
    assert p.name.endswith(".json")


# -------------------------------------------------------------- json parse

def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_from_fenced_block():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_surrounding_prose():
    """Models add commentary even when told not to."""
    assert _extract_json('Sure! Here you go:\n{"a": 1}\nHope that helps.') == {"a": 1}


def test_extract_json_handles_nested_braces():
    text = 'blah {"a": {"b": [1, 2]}, "c": "}"} trailing'
    assert _extract_json(text) == {"a": {"b": [1, 2]}, "c": "}"}


def test_extract_json_handles_braces_inside_strings():
    """The brace matcher must not be fooled by braces in string literals."""
    text = 'x {"quote": "a } b", "n": 1} y'
    assert _extract_json(text) == {"quote": "a } b", "n": 1}


def test_extract_json_handles_escaped_quotes():
    text = r'{"q": "she said \"hi\"", "n": 2}'
    assert _extract_json(text) == {"q": 'she said "hi"', "n": 2}


def test_extract_json_finds_arrays():
    assert _extract_json('here: [1, 2, 3]') == [1, 2, 3]


def test_extract_json_raises_on_no_json():
    with pytest.raises(ValueError):
        _extract_json("I'm afraid I can't do that.")
    with pytest.raises(ValueError):
        _extract_json("")


# ------------------------------------------------------------ token bucket

def test_bucket_allows_within_capacity_without_waiting():
    b = _TokenBucket(capacity=1000, refill_per_sec=1000)
    assert b.take(500) == 0.0
    assert b.take(500) == 0.0


def test_bucket_blocks_when_exhausted():
    b = _TokenBucket(capacity=100, refill_per_sec=1000)  # refills fast
    b.take(100)
    t0 = time.monotonic()
    waited = b.take(50)
    assert waited > 0
    assert time.monotonic() - t0 > 0


def test_bucket_take_never_deadlocks_on_oversized_request():
    """A request larger than capacity must be clamped, not wait forever."""
    b = _TokenBucket(capacity=100, refill_per_sec=10000)
    b.take(999_999)  # returns rather than hanging


def test_bucket_settle_refunds_overestimates():
    b = _TokenBucket(capacity=1000, refill_per_sec=0.0001)
    b.take(800)
    b.settle(estimated=800, actual=100)
    # 700 tokens were handed back, so a 600-token request must not block.
    assert b.take(600) == 0.0


def test_bucket_settle_never_exceeds_capacity():
    b = _TokenBucket(capacity=1000, refill_per_sec=1)
    b.take(10)
    b.settle(estimated=10, actual=0)
    assert b.tokens <= b.capacity


# ------------------------------------------------------ per-model ceilings

def test_qwen_has_a_low_output_ceiling():
    """Regression guard for the limit that killed two labelling runs."""
    assert MODEL_MAX_OUTPUT["qwen/qwen3.8-27b"] <= 1000
    assert DEFAULT_MAX_OUTPUT > 1000


def test_qwen_is_excluded_from_server_side_json_mode():
    assert any("qwen/qwen3" in m for m in NO_JSON_MODE)


def test_token_estimate_is_generous():
    """Underestimating would let the client drift past the real limit."""
    msgs = [{"role": "user", "content": "x" * 360}]
    est = _estimate_tokens(msgs, max_tokens=100)
    assert est >= 100 + 360 / 4


# -------------------------------------------------------------- ledger I/O

def test_ledger_summary_shape():
    from hsa.llm import ledger_summary

    s = ledger_summary()
    assert {"calls", "total_tokens", "by_tag"} <= set(s)
    json.dumps(s)  # must be serialisable for the report
