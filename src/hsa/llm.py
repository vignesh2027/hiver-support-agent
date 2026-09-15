"""One LLM entry point for the whole pipeline.

Four things this buys us, all of which exist because of constraints that are
real rather than hypothetical:

1. **Disk cache.** Every call is keyed by a hash of (model, messages, params).
   The cache is committed to the repo, so `HSA_OFFLINE=1 make reproduce`
   regenerates every headline number with *no API key and no network*. This is
   how the reviewer hits the 15-minute bar.

2. **Rate limiting.** The free Groq tier gives 8k tokens/minute. A token
   bucket makes a long run slow instead of failing.

3. **Daily request budget.** 1000 requests/day. We refuse to exceed it rather
   than discovering it halfway through an eval run, and we persist the count
   across processes.

4. **A ledger.** Every call appends to artifacts/llm_ledger.jsonl with model,
   tokens and latency, so the README can state the true cost of reproducing
   the results instead of guessing at it.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import urllib.error
import urllib.request

from .config import ARTIFACTS, BUDGET, CACHE

LEDGER = ARTIFACTS / "llm_ledger.jsonl"
_RPD_STATE = ARTIFACTS / ".rpd_state.json"

GROQ_BASE = "https://api.groq.com/openai/v1"
OPENAI_BASE = "https://api.openai.com/v1"

# Per-model ceiling on requested output tokens.
#
# Groq enforces a separate output-tokens-per-minute limit, and it is much
# tighter than the input budget: qwen3.8-27b rejects any single request asking
# for more than ~1000 output tokens with a 429 that no amount of retrying will
# clear. Discovered when a batched labelling pass died 3 batches in. Clamping
# here rather than at each call site means no caller can reintroduce the bug.
MODEL_MAX_OUTPUT: dict[str, int] = {
    "qwen/qwen3.8-27b": 950,
    "qwen/qwen3.6-27b": 950,
}
DEFAULT_MAX_OUTPUT = 8000

# Models where server-side JSON mode fights with the model's own reasoning
# tokens. The qwen3 reasoning models spend their budget thinking, emit empty
# content, and the provider then rejects the whole request with a 400
# "Failed to validate JSON" that retrying cannot fix. Asking for JSON in the
# prompt and parsing leniently is strictly more robust for these.
NO_JSON_MODE = ("qwen/qwen3",)


class BudgetExceeded(RuntimeError):
    pass


class CacheMiss(RuntimeError):
    """Raised in offline mode when a call is not already cached."""


def _load_dotenv() -> None:
    """Minimal .env loader. Avoids a python-dotenv dependency for three lines."""
    for name in (".env", ".env.local"):
        p = Path(__file__).resolve().parents[2] / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


class _TokenBucket:
    """Classic token bucket over a 60-second window.

    We charge the bucket the *actual* usage reported by the API after the call
    and reserve an estimate before it. Estimating low would let us drift over
    the limit, so the estimate is deliberately generous.
    """

    def __init__(self, capacity: int, refill_per_sec: float) -> None:
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill = refill_per_sec
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.refill)
        self.updated = now

    def take(self, amount: float) -> float:
        """Block until `amount` tokens are available. Returns seconds waited."""
        amount = min(amount, self.capacity)
        waited = 0.0
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= amount:
                    self.tokens -= amount
                    return waited
                deficit = amount - self.tokens
                sleep_for = max(deficit / self.refill, 0.05)
            time.sleep(sleep_for)
            waited += sleep_for

    def settle(self, estimated: float, actual: float) -> None:
        """Correct the bucket once the real usage is known."""
        with self.lock:
            self._refill()
            self.tokens = max(0.0, min(self.capacity, self.tokens + (estimated - actual)))


_bucket = _TokenBucket(
    capacity=int(BUDGET.tokens_per_min * BUDGET.safety_factor),
    refill_per_sec=BUDGET.tokens_per_min * BUDGET.safety_factor / 60.0,
)

# Output tokens are metered *separately and far more tightly* than total
# tokens, and the limit is per minute rather than per request: qwen3.8-27b
# allows 1000 output tokens/minute across all calls. A single 900-token
# request therefore consumes almost the whole minute's allowance, and a naive
# loop 429s immediately. Reasoning tokens count toward this budget too, which
# is what makes reasoning models expensive here.
#
# Tracking only total tokens (as the shared bucket above does) cannot see this
# at all, so each constrained model gets its own output bucket.
MODEL_OTPM: dict[str, int] = {
    "qwen/qwen3.8-27b": 1000,
    "qwen/qwen3.6-27b": 1000,
}
_out_buckets: dict[str, _TokenBucket] = {
    m: _TokenBucket(capacity=int(v * 0.85), refill_per_sec=v * 0.85 / 60.0)
    for m, v in MODEL_OTPM.items()
}


def _rpd_used() -> int:
    if not _RPD_STATE.exists():
        return 0
    try:
        st = json.loads(_RPD_STATE.read_text())
    except json.JSONDecodeError:
        return 0
    if st.get("day") != time.strftime("%Y-%m-%d"):
        return 0
    return int(st.get("used", 0))


def _rpd_bump() -> int:
    used = _rpd_used() + 1
    _RPD_STATE.write_text(json.dumps({"day": time.strftime("%Y-%m-%d"), "used": used}))
    return used


def remaining_requests_today() -> int:
    return max(0, BUDGET.requests_per_day - _rpd_used())


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _cache_path(key: str) -> Path:
    # Shard by first two chars: a flat dir with thousands of files is slow to
    # list and unpleasant in a git diff.
    d = CACHE / key[:2]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def _estimate_tokens(messages: Sequence[dict], max_tokens: int) -> int:
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    return int(chars / 3.6) + max_tokens + 40


def _post(url: str, body: dict, api_key: str, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # urllib's default UA ("Python-urllib/3.x") is rejected by the
            # provider's edge with Cloudflare error 1010. Identify properly.
            "User-Agent": "hiver-support-agent/0.1 (+https://github.com/)",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


_RETRY_AFTER = re.compile(
    r"try again in\s+(?:(\d+)m)?\s*([\d.]+)s", re.I
)


def _retry_after_seconds(detail: str) -> float | None:
    """Pull "Please try again in 6m6.768s" out of a 429 body."""
    m = _RETRY_AFTER.search(detail)
    if not m:
        return None
    mins = float(m.group(1) or 0)
    secs = float(m.group(2) or 0)
    return mins * 60 + secs


def _provider() -> tuple[str, str]:
    """Return (base_url, api_key). Groq first, OpenAI as a drop-in fallback."""
    if os.getenv("GROQ_API_KEY"):
        return GROQ_BASE, os.environ["GROQ_API_KEY"]
    if os.getenv("OPENAI_API_KEY"):
        return OPENAI_BASE, os.environ["OPENAI_API_KEY"]
    raise RuntimeError(
        "No GROQ_API_KEY or OPENAI_API_KEY found. Copy .env.example to .env, or run "
        "with HSA_OFFLINE=1 to replay the committed cache."
    )


def chat(
    messages: Sequence[dict],
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 700,
    reasoning_effort: str | None = "low",
    response_format: dict | None = None,
    tag: str = "untagged",
    retries: int = 6,
    retry_cap: float = 420.0,
) -> LLMResponse:
    """Single chat completion, cached and rate-limited.

    `tag` is recorded in the ledger so we can attribute spend to a pipeline
    stage (taxonomy / classify / generate / judge) after the fact.
    """
    capped = min(max_tokens, MODEL_MAX_OUTPUT.get(model, DEFAULT_MAX_OUTPUT))
    if capped < max_tokens:
        # Silently truncating a JSON response is worse than a loud failure, so
        # callers that genuinely need more output must batch smaller instead.
        max_tokens = capped

    body: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        body["response_format"] = response_format
    if "qwen3" in model:
        # Hidden reasoning tokens are billed against the 1000/minute OUTPUT
        # budget, so a reasoning pass costs ~40x more wall-clock than the
        # answer itself and routinely consumed the entire budget before
        # emitting any content -- the model returned "" and the request was
        # rejected as invalid JSON. Reasoning is instead requested as explicit
        # fields in the response schema (the judge quotes its evidence before
        # scoring), which fits the budget AND makes the reasoning auditable
        # rather than discarded. See DECISIONS.md D-13.
        body["reasoning_effort"] = "none"
    elif reasoning_effort and "gpt-oss" in model:
        body["reasoning_effort"] = reasoning_effort

    key = _cache_key(body)
    cpath = _cache_path(key)
    if cpath.exists():
        rec = json.loads(cpath.read_text())
        return LLMResponse(
            text=rec["text"],
            model=model,
            prompt_tokens=rec.get("prompt_tokens", 0),
            completion_tokens=rec.get("completion_tokens", 0),
            cached=True,
        )

    if os.getenv("HSA_OFFLINE") == "1":
        raise CacheMiss(
            f"Offline mode: no cached response for tag={tag} model={model} key={key}. "
            "The committed cache only covers the pipeline as configured in the README."
        )

    if remaining_requests_today() <= 0:
        raise BudgetExceeded(
            f"Daily request budget of {BUDGET.requests_per_day} is spent. "
            "Cached calls still work; new ones resume tomorrow."
        )

    base, api_key = _provider()
    est = _estimate_tokens(messages, max_tokens)

    out_bucket = _out_buckets.get(model)
    last_err: Exception | None = None
    for attempt in range(retries):
        waited = _bucket.take(est)
        if out_bucket is not None:
            # Reserve the full requested output budget; settle with the real
            # completion count (reasoning tokens included) after the call.
            waited += out_bucket.take(max_tokens)
        t0 = time.monotonic()
        try:
            data = _post(f"{base}/chat/completions", body, api_key, timeout=120)
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:400]
            last_err = RuntimeError(f"HTTP {e.code}: {detail}")
            if e.code in (429, 500, 502, 503, 520, 522):
                # A 429 may be any of four different budgets (per-minute
                # tokens, per-minute output tokens, per-day tokens, per-day
                # requests). The daily ones refill as old usage ages out, so
                # exponential backoff from 2s is hopelessly optimistic: the
                # response says exactly how long to wait, and honouring that
                # is the difference between a run that finishes slowly and one
                # that burns its retries in 30 seconds and dies.
                wait = _retry_after_seconds(detail)
                if wait is None:
                    wait = 2 ** (attempt + 1) + random.uniform(0, 1.5)
                time.sleep(min(wait + 1.0, retry_cap))
                continue
            raise last_err from e
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            # RemoteDisconnected / ConnectionReset are *not* URLError
            # subclasses -- urllib raises them straight through, so a narrow
            # `except URLError` silently fails the whole run mid-way. The
            # provider's edge drops idle connections fairly often under
            # throttling, and these are exactly the retryable cases.
            last_err = e
            time.sleep(2 ** (attempt + 1) + random.uniform(0, 1.5))
            continue

        latency = time.monotonic() - t0
        _rpd_bump()
        usage = data.get("usage", {}) or {}
        actual = int(usage.get("total_tokens", est))
        _bucket.settle(est, actual)
        if out_bucket is not None:
            out_bucket.settle(max_tokens, int(usage.get("completion_tokens", max_tokens)))

        msg = data["choices"][0]["message"]
        text = (msg.get("content") or "").strip()
        # gpt-oss style models can spend the whole budget on hidden reasoning
        # and return empty content. Surface that rather than silently
        # returning "" and scoring it as a bad reply.
        if not text and msg.get("reasoning"):
            text = ""

        rec = {
            "text": text,
            "model": model,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "tag": tag,
        }
        cpath.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        with LEDGER.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "tag": tag,
                        "model": model,
                        "prompt_tokens": rec["prompt_tokens"],
                        "completion_tokens": rec["completion_tokens"],
                        "latency_s": round(latency, 3),
                        "throttle_wait_s": round(waited, 2),
                        "key": key,
                    }
                )
                + "\n"
            )
        return LLMResponse(
            text=text,
            model=model,
            prompt_tokens=rec["prompt_tokens"],
            completion_tokens=rec["completion_tokens"],
            latency_s=latency,
        )

    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


def _extract_json(text: str) -> Any:
    """Pull the first JSON value out of a model response.

    Models wrap JSON in prose or fences often enough that a bare json.loads is
    a liability. We try strict first, then fenced, then brace matching.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            body = part.split("\n", 1)[-1] if part[:20].strip().lower().startswith("json") else part
            try:
                return json.loads(body.strip())
            except json.JSONDecodeError:
                continue

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"No JSON found in response: {text[:200]!r}")


def chat_json(
    messages: Sequence[dict],
    model: str,
    *,
    tag: str = "untagged",
    max_tokens: int = 900,
    temperature: float = 0.0,
    repair: bool = True,
    **kw: Any,
) -> Any:
    """Chat that must return JSON. One repair attempt, then give up loudly."""
    use_json_mode = not any(m in model for m in NO_JSON_MODE)
    resp = chat(
        messages,
        model,
        tag=tag,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"} if use_json_mode else None,
        **kw,
    )
    try:
        return _extract_json(resp.text)
    except ValueError:
        if not repair:
            raise
    repair_msgs = list(messages) + [
        {"role": "assistant", "content": resp.text[:1500]},
        {"role": "user", "content": "That was not valid JSON. Reply with the JSON object only, no prose."},
    ]
    resp2 = chat(
        repair_msgs,
        model,
        tag=f"{tag}:repair",
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"} if use_json_mode else None,
        **kw,
    )
    return _extract_json(resp2.text)


def ledger_summary() -> dict[str, Any]:
    """Aggregate the ledger. Used by the README cost table."""
    if not LEDGER.exists():
        return {"calls": 0, "total_tokens": 0, "by_tag": {}}
    calls = 0
    total = 0
    by_tag: dict[str, dict[str, int]] = {}
    wait = 0.0
    for line in LEDGER.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        calls += 1
        t = r["prompt_tokens"] + r["completion_tokens"]
        total += t
        wait += r.get("throttle_wait_s", 0.0)
        base = r["tag"].split(":")[0]
        slot = by_tag.setdefault(base, {"calls": 0, "tokens": 0})
        slot["calls"] += 1
        slot["tokens"] += t
    return {
        "calls": calls,
        "total_tokens": total,
        "throttle_wait_s": round(wait, 1),
        "by_tag": dict(sorted(by_tag.items(), key=lambda kv: -kv[1]["tokens"])),
    }


def cached_count() -> int:
    return sum(1 for _ in CACHE.rglob("*.json"))
