# Results

Generated from `artifacts/metrics/results.json`. Do not edit by hand —
run `make report`.

## Golden set

220 labelled examples across three strata.

| stratum | n | role |
| --- | --- | --- |
| `natural` | 120 | uniform random from held-out test split — **the only stratum used for population estimates** |
| `enriched` | 60 | over-samples rare predicted intents so per-class F1 has support — biased by construction |
| `adversarial` | 40 | targeted stress probes (money, safety, sarcasm, legal, image-only) — a stress test, not a sample |

Measured intent distribution (all strata pooled — **not** a population estimate):

| intent | share of golden set |
| --- | --- |
| `service_complaint` | 27.3% |
| `capacity_and_overcrowding` | 14.5% |
| `live_service_status` | 11.8% |
| `compensation_and_refunds` | 10.0% |
| `onboard_facilities` | 6.4% |
| `missed_connection` | 5.5% |
| `ticketing_and_booking` | 5.5% |
| `other` | 5.0% |
| `service_information` | 4.5% |
| `feedback_positive` | 4.1% |
| `seat_reservation` | 3.6% |
| `lost_property` | 1.8% |

Gold handling decisions:

| handling | share |
| --- | --- |
| `assist` | 49.5% |
| `auto` | 25.9% |
| `escalate` | 24.6% |

## Intent classification

Measured on the **natural stratum only** (the unbiased sample), so these
numbers describe traffic the system would actually see. 95% bootstrap CIs.

| system | n | accuracy | macro-F1 | weighted-F1 |
| --- | --- | --- | --- | --- |
| majority class (trivial baseline) | 120 | 15.0% [9.2–21.7] | 0.022 [0.014–0.030] | 0.039 |
| TF-IDF + logreg on weak labels (simple baseline) | 120 | 38.3% [30.0–46.7] | 0.343 [0.256–0.405] | 0.375 |
| LLM classifier (system) | 120 | 75.0% [67.5–82.5] | 0.755 [0.605–0.826] | 0.754 |

> The majority baseline scores 15.0% accuracy and
> 0.022 macro-F1. It predicts the intent that dominates the
> *cluster-derived* prior, which turns out not to dominate the adjudicated
> labels at all, so it does badly on both. That mismatch is itself a result:
> see the note on `prior_share` in REPORT.md. Macro-F1 is reported alongside
> accuracy throughout because on a skewed label set accuracy alone can hide
> a model that has collapsed onto one class.

Most frequent confusions (LLM classifier, natural stratum):

| true intent | predicted as | count |
| --- | --- | --- |
| `live_service_status` | `service_complaint` | 3 |
| `service_complaint` | `live_service_status` | 3 |
| `service_complaint` | `onboard_facilities` | 2 |
| `service_complaint` | `service_information` | 2 |
| `compensation_and_refunds` | `service_complaint` | 2 |
| `capacity_and_overcrowding` | `service_complaint` | 2 |

### Per-class performance (all strata)

Pooled across strata so rare intents have enough support to measure.
Because the enriched and adversarial strata over-represent hard cases,
treat these as a **lower bound**, not a population estimate.

| intent | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| `service_complaint` | 0.78 | 0.77 | 0.77 | 60 |
| `capacity_and_overcrowding` | 0.93 | 0.81 | 0.87 | 32 |
| `live_service_status` | 0.73 | 0.85 | 0.79 | 26 |
| `compensation_and_refunds` | 0.94 | 0.77 | 0.85 | 22 |
| `onboard_facilities` | 0.71 | 0.71 | 0.71 | 14 |
| `ticketing_and_booking` | 0.90 | 0.75 | 0.82 | 12 |
| `missed_connection` | 1.00 | 0.50 | 0.67 | 12 |
| `other` | 0.77 | 0.91 | 0.83 | 11 |
| `service_information` | 0.56 | 0.90 | 0.69 | 10 |
| `feedback_positive` | 0.89 | 0.89 | 0.89 | 9 |
| `seat_reservation` | 0.62 | 1.00 | 0.76 | 8 |
| `lost_property` | 1.00 | 1.00 | 1.00 | 4 |

## Reply quality (LLM judge, natural stratum)

Judge is `qwen3.8-27b`; the generator is `gpt-oss-20b`. Different model
families, and the judge is blind to which system wrote each reply.
Scales are 0–2 per dimension.

| system | n | acceptable to send | catastrophic | factual safety | addresses need | actionability | tone |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed apology (trivial baseline) | 61 | 70.5% [58.1–80.5] | 0.0% [0.0–5.9] | 2.00 | 0.82 | 1.00 | 1.05 |
| replay nearest historical reply (simple baseline) | 61 | 24.6% [15.5–36.7] | 36.1% [25.2–48.6] | 0.97 | 0.56 | 0.47 | 1.00 |
| grounded generator (system) | 61 | 85.2% [74.3–92.0] | 1.6% [0.3–8.7] | 1.84 | 1.16 | 1.54 | 1.52 |

Judge self-consistency: 0/183 verdicts (0.0%) contradicted the judge's own dimension scores and were corrected in code. A high rate here is a reason to discount the judge, not a reason to celebrate the correction.

## The headline: quality at a stated coverage

A system allowed to abstain has no single quality number. These two must
always be quoted together.

- **Auto-handled coverage: 3.3%** of messages
- **Acceptable-reply rate on auto-handled: 100.0%** [34.2–100.0]
- Catastrophic replies among auto-sent: 0.0% [0.0–65.8] (0 escapes)
- Drafted for a human: 60.7%; escalated: 36.1%
- If every reply were sent unfiltered, acceptable rate would be 85.2% — the gap is what the router buys.

### By stratum

| stratum | auto coverage | quality on auto | escalate rate | catastrophic on auto |
| --- | --- | --- | --- | --- |
| `natural` | 3.3% | 100.0% | 36.1% | 0.0% |

### Router ablations

Each row removes exactly one guard, scored on identical drafts.

Read this table with two caveats. First, it measures each guard's effect
**on the auto-handled subset only**, so a guard that works by escalating a
message before it ever reaches the auto branch (safety, money, legal,
intent policy) correctly shows no effect here even though it is doing its
job -- its effect is on the escalate rate, not on auto quality. Second,
the auto subset is small, so these differences are directional rather
than significant.

| config | auto coverage | quality on auto | catastrophic escapes |
| --- | --- | --- | --- |
| `ablate_all_guards` | 8.2% | 80.0% | 0 |
| `ablate_intent_policy` | 3.3% | 100.0% | 0 |
| `ablate_live_claim` | 3.3% | 100.0% | 0 |
| `ablate_money` | 3.3% | 100.0% | 0 |
| `ablate_none` | 3.3% | 100.0% | 0 |
| `ablate_safety` | 3.3% | 100.0% | 0 |
| `ablate_unresponsive` | 6.6% | 75.0% | 0 |

Threshold sweep (the risk–coverage trade-off):

| config | auto coverage | quality on auto |
| --- | --- | --- |
| `conf0.35_ret0.20` | 4.9% | 100.0% |
| `conf0.35_ret0.28` | 1.6% | 100.0% |
| `conf0.35_ret0.36` | 1.6% | 100.0% |
| `conf0.45_ret0.20` | 4.9% | 100.0% |
| `conf0.45_ret0.28` | 1.6% | 100.0% |
| `conf0.45_ret0.36` | 1.6% | 100.0% |
| `conf0.55_ret0.20` | 4.9% | 100.0% |
| `conf0.55_ret0.28` | 1.6% | 100.0% |
| `conf0.55_ret0.36` | 1.6% | 100.0% |
| `conf0.65_ret0.20` | 4.9% | 100.0% |
| `conf0.65_ret0.28` | 1.6% | 100.0% |
| `conf0.65_ret0.36` | 1.6% | 100.0% |
| `conf0.75_ret0.20` | 4.9% | 100.0% |
| `conf0.75_ret0.28` | 1.6% | 100.0% |
| `conf0.75_ret0.36` | 1.6% | 100.0% |
| `conf0.85_ret0.20` | 4.9% | 100.0% |
| `conf0.85_ret0.28` | 1.6% | 100.0% |
| `conf0.85_ret0.36` | 1.6% | 100.0% |

## Which rule decided each message

| rule | count |
| --- | --- |
| `R13-default-assist` | 78 |
| `R6-intent-policy` | 36 |
| `R9b-unresponsive` | 31 |
| `R10-weak-grounding` | 29 |
| `R3-money` | 11 |
| `R5-unreadable` | 8 |
| `R1-safety` | 7 |
| `R12-auto-eligible` | 6 |
| `R4-existing-case` | 5 |
| `R9-live-claim` | 2 |
| `R2-legal` | 2 |

The generator's self-report disagreed with the independent regex check on **1.4%** of drafts — i.e. it asserted a concrete time, platform or amount while reporting that it had not. This is why the router cross-checks rather than trusting the model's own declaration.

## What this cost to produce

614 LLM calls, 795,756 tokens, 94 minutes spent waiting on the free-tier rate limit.

| stage | calls | tokens |
| --- | --- | --- |
| `judge` | 182 | 224,241 |
| `prelabel` | 103 | 217,180 |
| `generate` | 215 | 213,873 |
| `adjudicate` | 35 | 53,987 |
| `classify` | 38 | 49,021 |
| `taxonomy` | 39 | 37,269 |
| `smoke2` | 1 | 118 |
| `smoke` | 1 | 67 |

