# Decision log

Fifteen decisions that were not obvious, with the reasoning and, where there
was one, the thing that went wrong first. Ordered roughly by when they were
made rather than by importance.

---

### D-01 — Picked GWRHelp over AmazonHelp, on a measurement rather than volume

**Decision.** Build the agent for `GWRHelp` (Great Western Railway, 19,237
first-reply pairs) rather than `AmazonHelp` (168,814 pairs, 8.8× the data).

**Why.** The task is to draft replies *grounded in how the brand historically
resolved issues*. That is only possible if the brand actually resolves things
in-thread. I scored every brand with ≥2,000 pairs on the share of first
replies that are neither a hand-off to another channel nor a bare apology.
AmazonHelp resolves in-thread far less than its volume suggests — its replies
are dominated by "kindly drop in your details through the link provided" — and
5.6% of its traffic is not in English. GWRHelp: 82.5% substantive replies,
2.3% deflection, 0% non-English, 56% multi-turn.

**Cost.** A niche UK-rail domain a reviewer may not know, and 8× less data.
Accepted: data quality dominates data quantity for retrieval grounding, and
the scorecard makes the choice auditable (`data/interim/brand_scorecard.csv`).

---

### D-02 — Rewrote the deflection detector after it gave a 13× wrong answer

**Decision.** Detect deflection by the *act* of handing off, not by the word
"DM".

**Why.** My first detector looked for DM / direct message / call us / email us.
It scored AmazonHelp at **1.3% deflection** and ranked it 4th-best brand
overall. Reading actual Amazon replies showed the same behaviour expressed
without any of those words: *"kindly drop in your details through the link
provided"*, *"reach out to us here: <url>"*, *"kindly report this to our
support team"*. The rewritten pattern targets the act — requesting details,
pointing at a link, promising a callback — and scored AmazonHelp at **16.8%**.

**Why it matters beyond one brand.** A lexical detector flatters whichever
brand has the largest phrasebook. The first number would have sent the whole
project to a brand whose data cannot support the task. This is the clearest
example of the report's theme: the metric moved 13× because of a regex, and
nothing about the first number looked suspicious.

---

### D-03 — Indexed only substantive replies as precedents

**Decision.** Exclude deflections and bare apologies from the retrieval index.

**Why.** "Sorry to hear that, please DM us" is similar to *every* customer
message. Leaving those in the index makes them the nearest neighbour for
everything, and the generator learns to apologise rather than answer.

**Cost.** Recall drops for intents where GWR genuinely only ever deflects. That
is real information, not a bug: those intents should escalate, and the router
sees the weak retrieval score and does exactly that.

---

### D-04 — Split chronologically, never randomly

**Decision.** Train / dev / test are contiguous time windows (train ends
2017-11-17, test is 2017-11-23 → 12-03).

**Why.** A random split leaks badly on a support corpus. One disruption
produces hundreds of near-identical tweets within an hour; a random split puts
near-copies of test items into the retrieval index, and retrieval quality
looks far better than it will be in production, where every incident is new.
The chronological split is also what deployment actually looks like: the model
only ever has the past.

**Cost.** The test window is ten days of a single winter, so seasonal effects
are untested. Recorded as a limitation rather than fixed — the corpus does not
contain enough history to do better (99.7% of it is Oct–Dec 2017).

---

### D-05 — Normalised station names and times before clustering, but nowhere else

**Decision.** Replace station names, clock times and bare numbers with
placeholders for taxonomy induction only.

**Why.** The first clustering run produced clusters organised by *geography*:
a Paddington cluster, a Bristol Parkway cluster, a Temple Meads cluster, a
Didcot cluster. Station names are the highest-TF-IDF tokens in the corpus.
"Delayed at Bristol" and "delayed at Reading" are the same intent and must
land together. After normalisation the clusters reorganised around problem
type — delay queries, refunds, seat reservations, ticket machines.

**Why "nowhere else".** A station name is genuinely useful evidence when
matching a message to a precedent. It is only harmful when deciding what the
*categories* are. So the classifier and retriever see original text.

---

### D-06 — Over-clustered deliberately, then consolidated, then curated by hand

**Decision.** k=32 for an expected 8–12 intents, then LLM naming, then LLM
consolidation, then human curation — with the un-curated output kept in the
repo as evidence.

**Why.** Merging two clusters that mean the same thing is easy for a human.
Noticing that one cluster is secretly two intents is hard. Over-clustering
converts the hard error into the easy one. The naming step is explicitly
allowed to answer "this cluster is incoherent", because a namer that cannot
refuse will happily name noise — 26.8% of traffic landed in clusters it
flagged as incoherent, which is information I would otherwise not have had.

---

### D-07 — Recursively split any cluster holding more than 15% of traffic

**Decision.** Re-cluster oversized clusters and let the children replace them.

**Why.** KMeans on short noisy text reliably produces one enormous
"everything else" cluster; here it took **27%** of the corpus. Naming that is
meaningless, and leaving it unexamined means a quarter of real customer
traffic is never looked at. After splitting, the largest leaf is 12.7% and
there are 38 leaves.

**What it found.** The split surfaced `lost_property` — a fixed public
procedure, no live data, no money, highly consistent historical replies. It is
the single most auto-handleable intent in the corpus and the flat clustering
had buried it inside a mixed bucket that would have defaulted to escalate.

---

### D-08 — Built the golden set in three strata and refused to pool them

**Decision.** 220 examples: 120 uniform-random (`natural`), 60 rare-intent
enriched (`enriched`), 40 adversarial probes (`adversarial`). Population
estimates are computed on `natural` only; the harness will not compute one
over the pooled set.

**Why.** A single random sample of 220 contains ~91 live-status messages and
about four missed-connection messages. Per-class F1 on four examples is noise,
and the rare intents are exactly where mistakes are expensive. But a set
enriched with hard cases no longer describes real traffic. Both are needed and
mixing them produces a number that is neither.

**Known biases, stated in the repo.** `enriched` uses a model to decide what
is rare, so it can only enrich intents that model can already find.
`adversarial` is regex-built and inherits whatever those regexes miss.

---

### D-09 — Two independent pre-label passes, then human adjudication with an audit sample

**Decision.** Label with two different model families under two different
prompt framings. A human adjudicates **every** disagreement plus a random 25%
audit of the agreements, and every correction is logged.

**Why.** Labelling 220 items from a blank page is slow and drifts: an
annotator's reading of a boundary at item 200 is not their reading at item 10.
Two independent passes make disagreement a signal — it concentrates human
attention on genuinely hard items. The 25% audit is what makes the result
*checkable*: it yields a measured error rate for the agreed labels that were
never individually reviewed, instead of an assumption that they are fine.

**On the wording.** Claiming "220 hand-labelled examples" without qualification
would be false. The report states the process and reports the human correction
rate, which is both true and more informative.

---

### D-10 — The router is a rule list, not a learned model

**Decision.** Escalation decisions come from ordered rules, first match wins.

**Why.** Three reasons. (1) There is no training data for it — nothing in the
corpus records whether a reply *should* have been automated, so a learned
router would be fitted to labels I invented and would hide their errors behind
a probability. (2) The assignment requires a stated reason; a rule list
produces the true binding reason by construction, whereas a classifier
produces a post-hoc rationalisation. (3) The failure modes are wildly
asymmetric — a wrong escalation costs minutes, a wrongly auto-sent invented
train time costs a missed train — and rules let me be conservative in exactly
the places I choose, visibly.

---

### D-11 — Generator and judge come from different model families

**Decision.** `gpt-oss-20b` generates; `qwen3.8-27b` judges. The judge is blind
to which system produced a reply and never sees GWR's actual historical reply.

**Why.** A judge scoring its own family's output rewards shared style
(self-preference bias). Withholding the historical reply matters just as much:
showing it would turn the judge into a similarity metric and would punish a
correct refusal that happens to differ from what GWR said in 2017. Similarity
to the historical reply is computed separately as its own automated metric.

**Cost.** The cross-family judge follows the rubric less precisely. Accepted,
and partly mitigated by D-12.

---

### D-12 — Judge consistency is enforced in code, and the override rate is published

**Decision.** A verdict marked catastrophic cannot also be "acceptable to
send"; `factual_safety=0` cannot be acceptable. These are corrected
programmatically and the correction rate is reported.

**Why.** Asking a model to be self-consistent in a prompt does not make it so.
Checking is cheap. Publishing the override rate turns a hidden weakness into a
stated one: a judge needing frequent correction is a judge whose scores should
be discounted, and the reader should be able to see that.

---

### D-13 — Calibrated the retrieval threshold to the observed distribution instead of guessing

**Decision.** `min_precedent_score = 0.22`, the empirical 25th percentile.

**Why.** I first set 0.28 by intuition. Measuring the actual distribution of
top-1 similarity over the 220 golden messages gave p5=0.18, p25=0.22, p50=0.25,
p90=0.37 — so 0.28 would have abstained on **68%** of traffic. TF-IDF cosine
between two ~100-character tweets is low in absolute terms even for a good
match; a threshold carried over from dense-retrieval intuition (0.7+) would
reject everything. The full sweep is reported so this is an operating point,
not a hidden constant.

---

### D-14 — Disabled hidden reasoning and put the reasoning in the output schema

**Decision.** `reasoning_effort="none"` for the qwen judge; its reasoning is
requested as explicit fields (quote the offending phrase, then score).

**Why.** The provider meters **output tokens per minute separately and far
more tightly** than total tokens — 1,000/minute for this model — and hidden
reasoning tokens count against it. With reasoning on, the judge consumed the
entire minute's budget before emitting any content, returned an empty string,
and the request was rejected as invalid JSON. With reasoning off it answers in
14 tokens.

**The upside.** Reasoning in the schema is auditable. Hidden reasoning tokens
are discarded; a quoted evidence string can be read, checked, and put in the
report.

---

### D-15 — The headline metric is quality *at a stated coverage*, never a single number

**Decision.** Report auto-handled coverage and acceptable-reply rate on the
auto-handled subset together, always, plus the risk–coverage curve. Never a
bare "accuracy".

**Why.** A system allowed to abstain has no single quality number. 100%
quality at 2% coverage is a system that does nothing; 60% coverage at 70%
quality is a system nobody can trust. Quoting either alone is the most
available way to mislead a reader about this kind of product, and the
assignment asks specifically what is misleading about the headline number —
so the headline is constructed to resist it.

**Consequence for this brand.** 53% of GWR's traffic needs live running data
the agent cannot have. The honest automation ceiling is small, and the product
is triage rather than deflection. A single accuracy figure would have hidden
that completely.
