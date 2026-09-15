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

**Intent classification**, on the natural stratum only (n=120), 95% bootstrap
intervals:

| system | accuracy | macro-F1 |
| --- | --- | --- |
| majority class (trivial) | 0.150 [0.092, 0.217] | 0.022 [0.014, 0.030] |
| TF-IDF + logreg on weak labels (simple) | 0.383 [0.300, 0.467] | 0.343 [0.256, 0.405] |
| **LLM classifier** | **0.750** [0.675, 0.825] | **0.755** [0.605, 0.826] |

The LLM roughly doubles the simple baseline, which is the margin I'd want
before accepting the latency and cost. Two things about this table are worth
more than the headline.

First, the majority baseline scores 15%, not the ~41% the taxonomy's
cluster-derived prior predicted for the largest intent. The prior was wrong:
`live_service_status` is 41% of *clusters* but only 12% of *adjudicated
labels*, mostly because the 12.7% mixed cluster I folded into it turned out to
contain a lot of complaints. The taxonomy file calls that field `prior_share`
and says it is not a measured distribution, which is exactly why. If I had
quoted it as the intent distribution the whole report would have been built on
it.

Second, the top confusions are `live_service_status ↔ service_complaint`,
three in each direction. That is the boundary I predicted when curating the
taxonomy and wrote a specific rule for ("is the journey still in progress and
would the answer be a fact about right now?"). Predicting your own model's
dominant error and still not fixing it is a useful thing to know: the rule
helps a human annotator and does not survive contact with a batched prompt.

**Reply quality**, judged blind on 60 messages from the natural stratum
(n=60 per arm, 180 verdicts). The judge is `qwen3.8-27b`, a different family
from the generator, shown the same retrieved precedents for every arm and never
shown what GWR actually replied. Scales are 0–2; 95% Wilson intervals.

| arm | factual safety | addresses need | actionability | acceptable to send | catastrophic |
| --- | --- | --- | --- | --- | --- |
| canned apology (trivial) | **2.00** | 0.82 | 1.00 | 70.0% [57.5–80.1] | **0.0%** [0.0–6.0] |
| nearest-neighbour (simple) | 0.98 | 0.57 | 0.47 | 25.0% [15.8–37.2] | **35.0%** [24.2–47.6] |
| **grounded generator** | 1.85 | **1.17** | **1.55** | **86.7%** [75.8–93.1] | 1.7% [0.3–8.9] |

n=60 per arm rather than the full 120, because the free tier's daily token
budget ran out mid-run. The intervals reflect that. The gaps between arms are
several times wider than the intervals, so the ordering is safe even if the
exact values are not.

Two rows here are worth more than the win.

**The nearest-neighbour baseline is catastrophic 36% of the time.** It replays
a real GWR reply verbatim, which means it replays a real arrival time, platform
number or delay length *belonging to a different journey*. It is the most
obvious way to build this system and it is the most dangerous thing I tested.
Precedents transfer tone and process; they do not transfer facts, and a system
that copies them cannot tell the difference.

**The canned apology scores perfect factual safety and 70% acceptable.** It
says nothing, so it cannot be wrong. If I had reported a single blended quality
score, a system that refuses to engage would have looked competitive with one
that actually helps. That is a property of the rubric, not of the system, and
it is why factual safety and actionability are reported separately rather than
averaged.

The judge overrode its own verdict for inconsistency **0 times in 160**, and
returned no parse failures.

**Router behaviour** over 210 messages (grounded arm):

| action | share |
| --- | --- |
| assist (drafted for a human) | 65.7% |
| escalate | 31.4% |
| auto-send | 2.9% |

and the rule that bound each decision: `R13-default-assist` 78,
`R6-intent-policy` 35, `R9b-unresponsive` 31, `R10-weak-grounding` 29,
`R3-money` 9, `R5-unreadable` 7, `R1-safety` 6, `R12-auto-eligible` 6,
`R4-existing-case` 5, `R9-live-claim` 2, `R2-legal` 2.

**The headline, which has to be quoted as a pair:** at **3.3% auto-handled
coverage**, **100% of auto-sent replies were judged acceptable** — but that
interval is [34.2–100.0], because it is 2 replies out of 60. The honest reading
is "no failures observed in a sample far too small to bound the failure rate",
not "perfect". If every draft were sent unfiltered the acceptable rate would be
86.7%, so the router is buying roughly 13 points of quality by declining to
send 97% of them.

A ~3% auto rate is a small number and it is the headline anyway, because it is
the true one. The alternative, auto-sending confident guesses about train
times, is the thing this entire design exists to prevent.

---

## 4. Failure analysis

Mined from the run logs by `hsa.evalx.failures`, so these are found rather than
chosen, and each comes with a rate over its real denominator. Hand-picking five
interesting failures would read better and tell you less: you couldn't tell
whether one happens twice or two hundred times.

**F0 — The one reply that got past every guard. 1 of 4 auto-sent (and 1 of 8
across the wider run).**

This is the most useful thing I found, so it goes first.

> **Customer:** "hi I left my hat on one your trains, it was 16:29 service from
> Swansea to Cardiff"
> **Agent:** "Hi there. Which service was this on? Please let us know the train
> number or any other details and we'll try to locate your hat."
> **Judge:** *"Ignores provided details and asks for information the customer
> already gave."*

The judge scored it `factual_safety: 2` — the top of the scale — and not
catastrophic. It is completely true. It invents nothing. And it is still
unsendable, because it asks the customer for the one thing they just told us,
which reads as not having been listened to.

Every guard passed and every guard was irrelevant. `lost_property` is
auto-eligible, confidence was 0.98, no risk marker fired, grounding was strong.
The hypothesis this forces on me is uncomfortable: **my entire safety
architecture is built to stop the agent saying things that are false, and it
has no defence at all against the agent saying things that are useless.** The
router checks for invented times, money, legal language, safety words. Nothing
in it asks whether the reply actually engages with what the customer wrote.

The fix is not a better prompt. It is a responsiveness check in the router: if
the draft asks for service details the message already contains, it cannot be
auto-sent. So I built it (`R9b-unresponsive`) and measured it on the same
drafts, which costs nothing because routing is a pure function of things
already in the run log.

| | before | after |
| --- | --- | --- |
| auto-sent replies the judge rejected | 1 of 8 (12.5%) | **0 of 6 (0%)** |
| auto coverage | 8 messages | 6 messages |

The first version of the check fired on 55 of 210 drafts, which was too many.
It accepted any "<word> to <word>" as evidence that the customer had named a
service, so it matched ordinary English ("would only leave when we got to the
station") and flagged replies to messages that identified no service at all.
Requiring a clock time, a four-digit departure, a capitalised route or a CRS
code pair brought it to 31 of 210 and kept every genuine catch. There is a
regression test that reproduces the original failure verbatim and asserts it is
no longer auto-sent.

I am reporting this as a failure rather than a feature because the interesting
part is not the fix. It is that I built six guards, all of them about
truthfulness, and the one reply that reached a customer was perfectly truthful.

**F1 — No usable precedent. 51/210 (24.3%).**
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

**0. "100% acceptable on auto-handled" is two replies.**
It is the most flattering number in this report and it is nearly content-free.
The Wilson interval is [34.2%, 100%]. Two successes cannot distinguish a system
that is right 99% of the time from one that is right 40% of the time. I am
leading the misleading-numbers section with my own best number because that is
where a reader's attention goes, and because the correct response to it is not
"impressive" but "show me a thousand".

**1. The auto-handled rate and the quality on it are meaningless apart.**
"4.4% auto-handled" and "quality on auto-handled replies" only mean something
together. 100% quality at 2% coverage is a system that does nothing; 60%
coverage at 70% quality is one nobody can trust. I report them as a pair and
sweep the thresholds so the operating point is visible rather than chosen for
me. Any single-number version of this result is misleading by construction.

**2. I did not measure judge-versus-human agreement, and the brief asked for
it.**
This is a missing deliverable and I would rather name it than let it be
discovered. The harness for it is built and committed:
`data/golden/replies_for_grading.jsonl` holds 60 replies, 20 per arm, shuffled
and stripped of any indication of which system wrote them, and `make grade`
walks a person through two questions each (would you send this, would sending
it cause harm) and computes Cohen's kappa with a bootstrap interval against the
judge's verdicts. It takes about fifteen minutes. It has not been run.

So everything the judge says in this report is unanchored. I used different
model families (gpt-oss generates, qwen judges), kept the judge blind to the
arm, and never showed it GWR's real reply, which reduces self-preference bias.
It does not remove shared blind spots: both models were trained on overlapping
text and can find the same wrong answer plausible. The only checks I actually
have are weaker ones — the judge never contradicted its own dimension scores
(0 overrides in 160 verdicts), and its factual-safety scores agree with an
independent regex for concrete claims. Neither is a substitute for a person.

If you read one caveat in this report, read this one. The quality numbers are
comparisons between models, not measurements against human judgement.

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

**5. The golden labels are mostly not hand-made, and this is the weakest part
of the submission.**
220 messages were labelled independently twice, by two model families under two
prompts. They agreed on intent 72.7% of the time and on handling 48.6%. The 138
disagreements went to a third, larger model that sees both prior answers and is
told the bias the first two share. Rows carry `review_kind`, and
`reports/RESULTS.md` publishes the breakdown, so the fraction a human actually
checked is visible rather than implied. That fraction is currently small.

The consequence is specific and I would rather name it than let a reader
discover it: my reference labels and my classifier come from overlapping model
families, so classification accuracy here is partly a measure of agreement
between related systems rather than of correctness. The judge-versus-human
study is the only part of this evaluation anchored to a person, and it is a
sample of 60. If I had one more day rather than one more week, I would spend it
entirely on human labels, not on the model.

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
