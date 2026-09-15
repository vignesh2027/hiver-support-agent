# Building a support agent for GWR, and working out whether to trust it

Brand: **GWRHelp** (Great Western Railway), 19,237 first-reply exchanges,
9,739 usable episodes after cleaning. Everything below runs from
[the repo](README.md); tables are generated into
[reports/RESULTS.md](reports/RESULTS.md) rather than typed by hand.

---

## 1. Framing: what "good" means here, and what I didn't build

I started by asking what a correct reply would actually require, message by
message. That turned out to be the whole project.

When the intent taxonomy was induced from the training data, I had the naming
step also mark whether answering correctly needs real-time information. It
flagged **70.9% of traffic**. After I consolidated and hand-curated the
taxonomy, **52.9%** of traffic sits in intents where a correct answer needs
live running data or the customer's booking — is the 18:03 cancelled, how late
is it, which platform, is there an unreserved coach on today's formation.

This agent has none of that. It has a text archive from 2017.

So "good" cannot mean "answers most messages". For this brand it means:

1. **Never state something it cannot know.** A wrong arrival time is worse
   than no reply, because the customer acts on it and misses a train. This is
   the only failure I treat as unrecoverable.
2. **Route correctly.** Get the message to a human fast when a human is
   needed, and say why.
3. **Automate the narrow band that is genuinely safe** — and be honest that
   the band is narrow. On my taxonomy only **6.4% of traffic**
   (`lost_property`, `feedback_positive`) is auto-eligible before any
   per-message risk check.

That reframes the product from deflection to triage. A support manager reading
this should conclude "this saves my team time on a specific slice and protects
me everywhere else", not "this replaces my team".

**What I deliberately didn't build.**

- **A live data integration.** This is the single biggest available win and it
  would move half the traffic from "escalate" to "answerable". I didn't stub
  one, because a fake feed would make every number here meaningless.
- **Fine-tuning.** No labels exist and there are only 9,739 episodes.
  Retrieval plus a careful prompt is the right first system, and it keeps the
  ability to explain any individual decision.
- **Multi-turn.** The agent answers the opening message only. 56% of these
  threads continue past the first reply, so this is a real limit.
- **A learned router.** Nothing in the corpus records whether a reply *should*
  have been automated. A learned router would fit labels I invented and hide
  their errors behind a probability. Rules give the true reason instead.
- **Non-English.** Filtered at episode build; recorded as a coverage gap.

### Choosing the brand was itself a result

The obvious pick is AmazonHelp: 168,814 pairs, 8.8× more data than anything
else. I scored every brand with ≥2,000 pairs on how often its first reply is
neither a hand-off nor a bare apology.

My first deflection detector looked for "DM", "direct message", "call us",
"email us". It scored AmazonHelp at **1.3% deflection** and ranked it fourth
overall. Then I read the actual replies:

> *"Kindly drop in your details through the link provided earlier"*
> *"I am sorry to hear this. Can I ask for you to reach out to us here please: <url>"*
> *"That's odd! We'd like to look into it. Kindly report this to our support team"*

Same behaviour, none of my words. I rewrote the detector to target the *act*
of handing off rather than one phrasing, and AmazonHelp went to **16.8%**. A
13× move on the metric that decides the entire project, and nothing about the
first number looked wrong.

GWRHelp won on 82.5% substantive replies, 2.3% deflection, no non-English
traffic, and 56% multi-turn. Its replies contain facts you can check
(*"You don't need to collect the tickets — just detail on the form that they
haven't been collected"*), which is what makes grounded generation possible at
all.

---

## 2. How it works

Twelve intents, induced rather than invented: TF-IDF → SVD → KMeans over 6,000
training messages, deliberately over-clustered at k=32, with any cluster above
15% of traffic recursively split. That split mattered — the flat clustering
produced one 27% "everything else" blob, and inside it was `lost_property`
("left my North Face jacket on the 8:01 from Bristol"), which is the single
most automatable intent in the corpus: fixed public procedure, no live data,
no money. A naive taxonomy would have buried it in an escalate-by-default
bucket.

Clustering also first organised itself by **station name** rather than problem
type, because station names are the highest-TF-IDF tokens here. Normalising
stations and times to placeholders (for induction only) fixed it.

Retrieval indexes only substantive historical replies, from the training split
only. Both matter: index the deflections and "sorry, please DM us" becomes the
nearest neighbour for everything; index the whole corpus and a test message
retrieves a reply written twenty minutes later about the same incident.

The generator is told it has no live data and that precedents transfer *tone
and process, not facts*. It must declare whether its own draft asserts anything
time-sensitive. I don't trust that declaration — the router re-checks the text
independently.

The router is an ordered rule list. Every decision returns the rule that bound
it, so `R9-live-claim` and `R3-money` are auditable in the run log.

---

## 3. Results

Full tables: [reports/RESULTS.md](reports/RESULTS.md). Population estimates come
from the `natural` stratum only (n=120, uniform random from the held-out test
split). The `enriched` and `adversarial` strata are biased by construction and
are never pooled into a headline; the harness refuses to do it.

**Baselines.** Trivial and simple, for both tasks:

| task | trivial | simple |
| --- | --- | --- |
| intent | always predict the largest intent (`live_service_status`) | TF-IDF + logistic regression trained on cluster-derived weak labels |
| reply | one fixed apology for every message | replay, verbatim, GWR's reply to the nearest historical message |

The simple baselines are deliberately strong. The logreg needs no labels and
no LLM and runs in milliseconds; the nearest-neighbour replier is genuinely
competitive for a brand with formulaic replies. If the LLM can't clear them by
a visible margin it isn't worth its latency.

**Router behaviour** over 180 messages (grounded arm):

| action | share |
| --- | --- |
| assist (drafted for a human) | 72.2% |
| escalate | 23.3% |
| auto-send | 4.4% |

and the rule that decided each: `R13-default-assist` 93, `R10-weak-grounding`
37, `R6-intent-policy` 32, `R12-auto-eligible` 8, `R3-money` 4,
`R5-unreadable` 3, `R9-live-claim` 2, `R1-safety` 1.

A 4.4% auto rate is a small number and I'm reporting it as the headline anyway,
because it's the true one. The alternative — auto-sending confident guesses
about train times — is the thing this whole design exists to prevent.

---

## 4. Failure analysis

Mined from the run logs by `hsa.evalx.failures`, so these are found rather than
chosen, and each comes with a rate over its real denominator. Hand-picking five
interesting failures would read better and tell you less: you couldn't tell
whether one happens twice or two hundred times.

**F1 — No usable precedent. 43/180 (23.9%).**
Nearly a quarter of messages have no historical match above the grounding
floor. *"A) I can see why you have halved the compensation B) I don't know why
I bother setting the alarm C) …"* — a multi-part rant with no close neighbour.
Retrieval-grounded generation degrades silently into unguided generation
exactly where it knows least, which is why there's an explicit grounding
threshold at all; these route to `assist` via `R10-weak-grounding` instead of
being answered confidently.

**F2 — Invented concrete facts. 3/180 (1.7%).**
*"the train is running in reverse formation today"* — plausible, on-brand, and
unknowable. All three were caught by `R9-live-claim` and escalated. The rate is
low because the prompt works; it is not zero because prompts are not
guarantees.

**F3 — The model misreports itself. 3/180 (1.7%).**
The same three drafts. Each declared `asserts_time_sensitive_fact: false` while
containing exactly such a fact. A model that hallucinates a fact is not a
reliable witness to having hallucinated it, which is the entire argument for
checking the text independently rather than trusting self-reported confidence.

**F4 — Over-length replies. 1/180 (0.6%).**
Twitter's limit is stated in the prompt and the prompt is not a constraint.
Cheap to check in code, and a reply over the limit is unusable no matter how
good it is.

**F5 — Disagreement about what to *do*, not what it *is*.**
This one is about the labels rather than the model, and it's the most
interesting thing I found. Two independent pre-label passes agreed on **intent
72.7%** of the time but on **handling only 48.6%**. Pass 1 wanted to auto-send
144 of 220 messages; pass 2 wanted 78. Same taxonomy, same rules, same
messages. If two capable models disagree this sharply about what may be
automated, the automation boundary is not something to delegate to a model —
which is precisely why a human adjudicates every disagreement here, and why the
router is rules rather than a classifier.

---

## 5. What is misleading about my headline number?

Six things, roughly in order of how much they'd bother me if I were reading
this.

**1. The auto-handled rate and the quality on it are meaningless apart.**
"4.4% auto-handled" and "quality on auto-handled replies" only mean something
together. 100% quality at 2% coverage is a system that does nothing; 60%
coverage at 70% quality is one nobody can trust. I report them as a pair and
sweep the thresholds so the operating point is visible rather than chosen for
me. Any single-number version of this result is misleading by construction.

**2. The judge and the thing it judges are not independent enough.**
I used different model families (gpt-oss generates, qwen judges), kept the
judge blind to the arm, and never showed it GWR's real reply. That reduces
self-preference bias. It does not remove shared blind spots: both models were
trained on overlapping internet text and may find the same wrong answer
plausible. The human-agreement study is there to bound this, and it's a sample
of 60, not a proof.

**3. My confidence signal barely works.**
Mean self-reported classifier confidence is **0.914**, and only 1 of 220
messages scored below my 0.55 threshold. That rule is nearly inert. The router
works because of structural signals — risk markers, intent policy, grounding
score, the live-claim check — not because the model knows when it's unsure. Any
risk–coverage curve driven by that confidence should be read sceptically.
(Related: my first risk–coverage implementation broke ties by row order, which
would have let a useless confidence signal trace a perfect curve. A test caught
it; ties are now broken randomly and averaged.)

**4. The test window is ten days of one winter.**
99.7% of this corpus is October–December 2017. The chronological split gives a
test set spanning 2017-11-23 to 12-03. No summer engineering works, no
Christmas timetable, no strike. Daily volume is stable across the window so
it's not one freak incident, but nothing here demonstrates seasonal
robustness.

**5. The golden labels are not purely hand-made, and I'd rather say so.**
220 messages, labelled independently twice by two model families, then: every
disagreement adjudicated by me, plus a blind audit of a random 25% of the
agreements. That audit exists to produce a *measured* error rate for the
agreed labels I didn't individually check. Calling this "220 hand-labelled
examples" with no qualifier would be untrue.

**6. Per-class F1 is computed on a deliberately biased sample.**
Rare intents only have enough support to measure because the `enriched`
stratum over-samples them, and that stratum uses a *model* to decide what's
rare — so it can only enrich intents that model can already find. Rare intents
the baseline is blind to stay under-represented. Treat per-class numbers as a
lower bound, not a population estimate.

And one that isn't about statistics: **the brand was chosen partly because its
data suits the task.** GWR resolves in-thread more than most. On a brand like
AmazonHelp, which mostly hands off, the same pipeline would produce a much
worse system and I'd have found that out later.

---

## 6. With one more week

In the order I'd actually do them:

1. **Wire a live running-information feed** (National Rail Darwin or similar).
   This is not incremental — it moves ~53% of traffic from "must escalate" to
   "answerable" and changes what the product is. Everything else on this list
   is worth less than this one item.
2. **Fix the abstention signal.** Self-reported confidence is useless here.
   I'd calibrate on the golden set, or replace it with agreement across
   sampled generations, and re-derive the risk–coverage curve from something
   that actually carries information.
3. **Grow the golden set to ~600 and get a second annotator.** n=120 for
   population estimates gives roughly ±8-point intervals — too wide to compare
   two systems honestly. A second human would also let me report inter-human
   agreement, which is the missing denominator for the judge study: I currently
   compare the judge to one person, with no measure of how much two people
   would disagree.
4. **Multi-turn.** 56% of threads continue. The obvious next slice is
   "customer replied to our reply", where intent is much more constrained.
5. **Ship the retriever ablation properly.** Dense embeddings are wired up
   behind a flag but not in the headline, because I didn't want the reproduce
   path to depend on a 90 MB download. I'd measure it and, if it wins, make it
   the default and accept the dependency.
6. **Close the loop on real outcomes.** The corpus has a weak signal — did the
   customer thank us afterwards — that I used only for sampling. In production
   that becomes real feedback, and it's the only way to learn what "good" means
   without me deciding it.

---

## Appendix: what this cost

Everything ran on Groq's free tier, which enforces four separate budgets
(8,000 tokens/minute, 1,000 output tokens/minute on the judge model, 200,000
tokens/day, ~1,000 requests/day). All four are enforced client-side, and
`make cost` prints the real ledger: calls, tokens, and time spent waiting on
rate limits. Every response is cached and committed, so `make reproduce`
rebuilds every number with no API key and no network.
