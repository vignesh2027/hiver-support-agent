"""Central paths and run configuration.

Everything downstream imports paths from here so that the repo is relocatable
and the reviewer never has to edit a hard-coded path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
GOLDEN = DATA / "golden"

ARTIFACTS = ROOT / "artifacts"
CACHE = ARTIFACTS / "cache"
FIGURES = ARTIFACTS / "figures"
METRICS = ARTIFACTS / "metrics"

REPORTS = ROOT / "reports"

TWCS_CSV = RAW / "twcs.csv"

for _p in (DATA, RAW, INTERIM, GOLDEN, ARTIFACTS, CACHE, FIGURES, METRICS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Models
#
# Deliberate choice: the *generator* and the *judge* come from different model
# families. An LLM judge scoring text produced by its own family inflates
# scores (self-preference bias). Keeping them apart is not free -- the judge is
# weaker at following the rubric -- but it makes the headline quality number
# mean something. See DECISIONS.md D-11.
# --------------------------------------------------------------------------

GEN_MODEL = os.getenv("HSA_GEN_MODEL", "openai/gpt-oss-20b")
CLS_MODEL = os.getenv("HSA_CLS_MODEL", "openai/gpt-oss-20b")
JUDGE_MODEL = os.getenv("HSA_JUDGE_MODEL", "qwen/qwen3.8-27b")
TAXONOMY_MODEL = os.getenv("HSA_TAXONOMY_MODEL", "openai/gpt-oss-120b")

SEED = 20260913


@dataclass(frozen=True)
class Budget:
    """Free-tier Groq limits, measured from response headers on 2026-09-13.

    tokens_per_min comes from `x-ratelimit-limit-tokens` (reset ~1s) and
    requests_per_day from `x-ratelimit-limit-requests` (reset ~86.4s/request).
    We enforce both client-side so a long run degrades into waiting rather
    than into a wall of 429s.
    """

    tokens_per_min: int = int(os.getenv("HSA_TPM", "8000"))
    requests_per_day: int = int(os.getenv("HSA_RPD", "1000"))
    # Leave headroom: a 429 storm costs more wall-clock than throttling does.
    safety_factor: float = 0.85


BUDGET = Budget()


@dataclass
class RunConfig:
    brand: str = os.getenv("HSA_BRAND", "")
    sample_threads: int = int(os.getenv("HSA_SAMPLE_THREADS", "20000"))
    retriever: str = os.getenv("HSA_RETRIEVER", "tfidf")  # tfidf | embed
    top_k: int = 4
    seed: int = SEED
    tags: dict = field(default_factory=dict)
