# GWR support agent

Take-home for Hiver. It reads a customer tweet, works out what the person
wants, drafts a reply based on how the brand has answered similar things
before, and then decides whether that reply is safe to send on its own or
needs a human.

The brief says the proof matters more than the system, so I'll lead with the
thing the evaluation actually turned up:

> About **71% of this brand's inbound messages ask for something the agent
> cannot know** — is the 18:03 running, how late is it, which platform. There
> is no live data feed here. An agent that answers anyway will sound completely
> confident and be wrong most of the time.

So this isn't an auto-reply bot. It's triage. Most of the value is in being
right about what to *not* answer, and the numbers below are built to show that
honestly rather than to look good.

Start here:

- **[REPORT.md](REPORT.md)** — the report. Framing, results against two
  baselines, the five failure modes I found, what's misleading about my
  headline number, and what I'd do next.
- **[DECISIONS.md](DECISIONS.md)** — 15 decisions that weren't obvious, and
  the ones I got wrong first.
- **[reports/RESULTS.md](reports/RESULTS.md)** — all the tables. Generated
  from `artifacts/metrics/results.json`, never typed by hand.
- **[artifacts/ANNOTATION_GUIDELINE.md](artifacts/ANNOTATION_GUIDELINE.md)** —
  what I labelled the golden set against.

## Running it

Every LLM response the project ever got back is cached on disk and committed.
`HSA_OFFLINE=1` turns each call into a cache lookup, so you can rebuild all the
reported numbers without an API key, without network, and without waiting on
anyone's rate limit.

```bash
make setup       # venv + 5 pinned deps, about a minute
make reproduce   # replays classify -> generate -> judge -> metrics
```

Then open `reports/RESULTS.md`.

You don't need the 493 MB corpus for that. `make reproduce` runs off the
committed golden set, the cached responses and the splits. A cache miss raises
instead of quietly making a live call, so the offline path can't drift away
from what's published here.

```bash
make test        # 120 tests
make cost        # what producing these numbers actually cost
```

To rebuild from scratch you need a `GROQ_API_KEY` (free tier is fine) and some
patience. The free tier gives 8,000 tokens a minute, so a full rebuild is
limited by rate limiting, not compute.

```bash
cp .env.example .env     # add your key
make data                # 493 MB, 2.81M tweets
make scorecard           # score every brand on how learnable its support is
make episodes            # GWR episodes + chronological splits
make taxonomy            # cluster -> name -> consolidate the intents
make index pool prelabel
make label               # human adjudication, interactive
make classify generate judge report
```

## How it fits together

```
tweets ──► brand scorecard ──► GWR episodes (split by time)
                                     │
             ┌───────────────────────┼──────────────────────┐
             ▼                       ▼                      ▼
      intent taxonomy         precedent index         golden set (220)
    cluster → name →       (substantive replies,     3 strata, labelled
    consolidate → curate     TRAIN split only)       twice then adjudicated
             │                       │                      │
             └───────────┬───────────┘                      │
                         ▼                                  │
    message ─► classifier ─► generator ─► router ───────────┴─► judge
               (3 arms)      (3 arms)     (rules)                (+ human
                                                                  agreement)
```

**Intents came out of the data, not out of my head.** TF-IDF into SVD into
KMeans over the training messages, deliberately over-clustered at k=32. Any
cluster that swallowed more than 15% of traffic gets split again. Each leaf
gets named by a model that's allowed to say "this cluster is incoherent", which
it did for about a quarter of traffic. Then consolidation, then I went through
it by hand. The uncurated version is still in the repo next to the curated one
so you can see what changed and why.

**Retrieval only indexes replies that actually said something.** Roughly one in
six of GWR's first replies is a hand-off or a bare apology. Leave those in and
"sorry to hear that, please DM us" becomes the nearest neighbour for
everything, and the generator learns to apologise instead of answer. The index
is also built from the training split alone, so a test message can't retrieve a
reply that was written twenty minutes later about the same incident.

**The generator's hardest job is saying no.** Its prompt makes "I can't know
that" a real option, and its output schema forces it to declare whether its own
draft asserts anything time-sensitive. I don't trust that declaration — the
router checks the text independently and I report how often the two disagree.

**The router is a rule list.** No training data exists for "should this have
been automated", the brief asks for a stated reason, and the costs are lopsided:
a needless escalation wastes a couple of minutes, an auto-sent invented arrival
time makes someone miss a train. Every decision comes back with the rule that
produced it.

**The judge is a different model family from the generator**, can't see which
system wrote a reply, and never sees what GWR actually said. Showing it the
real reply would quietly turn it into a similarity metric and punish a correct
refusal for not matching a 2017 answer.

## Layout

```
src/hsa/
  config.py            paths, model choices, rate-limit budget
  llm.py               single LLM entry point: cache, token buckets,
                       daily request budget, cost ledger
  data/prepare.py      corpus -> brand scorecard -> episodes -> splits
  taxonomy/            cluster, name, consolidate, validate
  retrieve/index.py    precedent search
  classify/            majority + weak-supervised logreg + LLM
  reply/generate.py    grounded generator, plus two reply baselines
  route/policy.py      escalation rules, thresholds, ablations
  evalx/               sampling, pre-labelling, judge, metrics, harness
  tools/label_cli.py   the labelling and grading CLI
tests/                 120 tests
```

## About the golden set

The brief asks for hand-labelled examples, so here's exactly what happened,
because the wording matters.

220 messages were sampled from the held-out test split across three strata.
Each one was labelled twice, independently, by two different model families
under two different prompts. Where the two disagreed, I decided. Where they
agreed, I still blind-audited a random quarter of them. Every correction is
logged with a timestamp.

That audit is the part I care about. It gives a measured error rate for the
agreed labels I didn't individually check, instead of me assuming they're fine.
The rate is in `reports/RESULTS.md` along with how often the two passes agreed
in the first place (72.7% on intent, and only 48.6% on what to *do* about it,
which tells you where the difficulty really is).

Calling this "220 hand-labelled examples" with no qualifier would be untrue.
Describing it this way is both accurate and more useful, since it comes with a
number attached.

## What I left out on purpose

**No live data integration.** The single biggest improvement available is a
real-time running-information feed, which would move that 71% from "escalate"
to "answerable". Faking one would have made every number in the report
meaningless.

**No fine-tuning.** There are no labels to train on and only 9,739 episodes.
Retrieval plus a careful prompt is the right first system, and it keeps the
ability to explain any individual decision, which a fine-tune throws away.

**No multi-turn.** The agent answers the opening message only. 56% of these
threads carry on past the first reply, so this is a genuine limit and I'd
rather state it than bury it.

**No non-English.** Filtered out when episodes are built, recorded as a gap.

## Credits

- Corpus: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).
  `make data` pulls the identical `twcs.csv` from a Hugging Face mirror so you
  don't need Kaggle credentials.
- Models: `openai/gpt-oss-20b` and `-120b` for classification, generation and
  taxonomy work; `qwen/qwen3.8-27b` for judging. All on Groq's free tier.
- Stats: percentile bootstrap for intervals, Wilson intervals for proportions
  near zero (the normal approximation gives you negative lower bounds there),
  Cohen's kappa with the Landis & Koch bands for agreement.
- No vendor SDK. The client talks to the OpenAI-compatible HTTP API straight
  from the standard library, so switching provider is a base URL change.
