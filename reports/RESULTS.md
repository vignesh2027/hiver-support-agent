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
| majority class (trivial baseline) | 120 | 15.0% [9.2–21.7] | 0.022 [1.4–3.0] | 0.039 |
| TF-IDF + logreg on weak labels (simple baseline) | 120 | 38.3% [30.0–46.7] | 0.343 [25.6–40.5] | 0.375 |
| LLM classifier (system) | 120 | 75.0% [67.5–82.5] | 0.755 [60.5–82.6] | 0.754 |

> The gap between accuracy and macro-F1 for the majority baseline is the
> reason accuracy is not the headline metric: predicting the largest intent
> for everything scores respectably on accuracy while being useless.

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
| fixed apology (trivial baseline) | 60 | 70.0% [57.5–80.1] | 0.0% [0.0–6.0] | 2.00 | 0.82 | 1.00 | 1.05 |
| replay nearest historical reply (simple baseline) | 60 | 25.0% [15.8–37.2] | 35.0% [24.2–47.6] | 0.98 | 0.57 | 0.47 | 1.00 |
| grounded generator (system) | 60 | 86.7% [75.8–93.1] | 1.7% [0.3–8.9] | 1.85 | 1.17 | 1.55 | 1.53 |

Judge self-consistency: 0/180 verdicts (0.0%) contradicted the judge's own dimension scores and were corrected in code. A high rate here is a reason to discount the judge, not a reason to celebrate the correction.

## The headline: quality at a stated coverage

A system allowed to abstain has no single quality number. These two must
always be quoted together.

- **Auto-handled coverage: 3.3%** of messages
- **Acceptable-reply rate on auto-handled: 100.0%** [34.2–100.0]
- Catastrophic replies among auto-sent: 0.0% [0.0–65.8] (0 escapes)
- Drafted for a human: 60.0%; escalated: 36.7%
- If every reply were sent unfiltered, acceptable rate would be 86.7% — the gap is what the router buys.

### By stratum

| stratum | auto coverage | quality on auto | escalate rate | catastrophic on auto |
| --- | --- | --- | --- | --- |
| `natural` | 3.3% | 100.0% | 36.7% | 0.0% |

### Router ablations

Each row removes exactly one guard. If removing a guard does not worsen
anything, the guard is not earning its place.

| config | auto coverage | quality on auto | catastrophic escapes |
| --- | --- | --- | --- |
| `ablate_all_guards` | 8.3% | 80.0% | 0 |
| `ablate_intent_policy` | 6.7% | 75.0% | 0 |
| `ablate_live_claim` | 6.7% | 75.0% | 0 |
| `ablate_money` | 6.7% | 75.0% | 0 |
| `ablate_none` | 6.7% | 75.0% | 0 |
| `ablate_safety` | 6.7% | 75.0% | 0 |
| `ablate_unresponsive` | 6.7% | 75.0% | 0 |

Threshold sweep (the risk–coverage trade-off):

| config | auto coverage | quality on auto |
| --- | --- | --- |
| `conf0.35_ret0.20` | 8.3% | 80.0% |
| `conf0.35_ret0.28` | 5.0% | 66.7% |
| `conf0.35_ret0.36` | 1.7% | 100.0% |
| `conf0.45_ret0.20` | 8.3% | 80.0% |
| `conf0.45_ret0.28` | 5.0% | 66.7% |
| `conf0.45_ret0.36` | 1.7% | 100.0% |
| `conf0.55_ret0.20` | 8.3% | 80.0% |
| `conf0.55_ret0.28` | 5.0% | 66.7% |
| `conf0.55_ret0.36` | 1.7% | 100.0% |
| `conf0.65_ret0.20` | 8.3% | 80.0% |
| `conf0.65_ret0.28` | 5.0% | 66.7% |
| `conf0.65_ret0.36` | 1.7% | 100.0% |
| `conf0.75_ret0.20` | 8.3% | 80.0% |
| `conf0.75_ret0.28` | 5.0% | 66.7% |
| `conf0.75_ret0.36` | 1.7% | 100.0% |
| `conf0.85_ret0.20` | 8.3% | 80.0% |
| `conf0.85_ret0.28` | 5.0% | 66.7% |
| `conf0.85_ret0.36` | 1.7% | 100.0% |

## Which rule decided each message

| rule | count |
| --- | --- |
| `R13-default-assist` | 78 |
| `R6-intent-policy` | 35 |
| `R9b-unresponsive` | 31 |
| `R10-weak-grounding` | 29 |
| `R3-money` | 9 |
| `R5-unreadable` | 7 |
| `R1-safety` | 6 |
| `R12-auto-eligible` | 6 |
| `R4-existing-case` | 5 |
| `R9-live-claim` | 2 |
| `R2-legal` | 2 |

The generator's self-report disagreed with the independent regex check on **1.4%** of drafts — i.e. it asserted a concrete time, platform or amount while reporting that it had not. This is why the router cross-checks rather than trusting the model's own declaration.

## What this cost to produce

613 LLM calls, 794,475 tokens, 94 minutes spent waiting on the free-tier rate limit.

| stage | calls | tokens |
| --- | --- | --- |
| `judge` | 181 | 222,960 |
| `prelabel` | 103 | 217,180 |
| `generate` | 215 | 213,873 |
| `adjudicate` | 35 | 53,987 |
| `classify` | 38 | 49,021 |
| `taxonomy` | 39 | 37,269 |
| `smoke2` | 1 | 118 |
| `smoke` | 1 | 67 |

