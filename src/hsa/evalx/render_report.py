"""Render artifacts/metrics/results.json into reports/RESULTS.md.

Kept separate from the harness so every table in the report is regenerated
from saved numbers rather than retyped. Nothing in the report should be a
figure a human copied by hand; that is how stale numbers survive edits.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import METRICS, REPORTS

RESULTS = METRICS / "results.json"
OUT = REPORTS / "RESULTS.md"


def _pct(x, nd=1) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{100*float(x):.{nd}f}%"


def _ci(pair) -> str:
    if not pair or any(v is None for v in pair):
        return ""
    lo, hi = pair
    if lo != lo or hi != hi:
        return ""
    return f" [{100*float(lo):.1f}–{100*float(hi):.1f}]"


def render(r: dict) -> str:
    L: list[str] = []
    add = L.append

    add("# Results")
    add("")
    add("Generated from `artifacts/metrics/results.json`. Do not edit by hand —")
    add("run `make report`.")
    add("")

    # -- golden set ---------------------------------------------------------
    g = r.get("golden_set", {})
    add("## Golden set")
    add("")
    add(f"{g.get('n', 0)} labelled examples across three strata.")
    add("")
    add("| stratum | n | role |")
    add("| --- | --- | --- |")
    roles = {
        "natural": "uniform random from held-out test split — **the only stratum used for population estimates**",
        "enriched": "over-samples rare predicted intents so per-class F1 has support — biased by construction",
        "adversarial": "targeted stress probes (money, safety, sarcasm, legal, image-only) — a stress test, not a sample",
    }
    for s, n in sorted(g.get("by_stratum", {}).items(), key=lambda kv: -kv[1]):
        add(f"| `{s}` | {n} | {roles.get(s,'')} |")
    add("")

    if g.get("intent_distribution"):
        add("Measured intent distribution (all strata pooled — **not** a population estimate):")
        add("")
        add("| intent | share of golden set |")
        add("| --- | --- |")
        for k, v in sorted(g["intent_distribution"].items(), key=lambda kv: -kv[1]):
            add(f"| `{k}` | {_pct(v)} |")
        add("")

    if g.get("handling_distribution"):
        add("Gold handling decisions:")
        add("")
        add("| handling | share |")
        add("| --- | --- |")
        for k, v in sorted(g["handling_distribution"].items(), key=lambda kv: -kv[1]):
            add(f"| `{k}` | {_pct(v)} |")
        add("")

    # -- classification -----------------------------------------------------
    c = r.get("classification", {})
    nat = c.get("overall_natural_only", {})
    if nat:
        add("## Intent classification")
        add("")
        add("Measured on the **natural stratum only** (the unbiased sample), so these")
        add("numbers describe traffic the system would actually see. 95% bootstrap CIs.")
        add("")
        add("| system | n | accuracy | macro-F1 | weighted-F1 |")
        add("| --- | --- | --- | --- | --- |")
        pretty = {
            "majority": "majority class (trivial baseline)",
            "tfidf_logreg": "TF-IDF + logreg on weak labels (simple baseline)",
            "llm": "LLM classifier (system)",
        }
        for arm in ("majority", "tfidf_logreg", "llm"):
            a = nat.get(arm)
            if not a:
                continue
            add(
                f"| {pretty.get(arm, arm)} | {a['n']} | {_pct(a['accuracy'])}{_ci(a.get('accuracy_ci'))} "
                f"| {a['macro_f1']:.3f}{_ci(a.get('macro_f1_ci')).replace('%','')} | {a['weighted_f1']:.3f} |"
            )
        add("")
        add("> The gap between accuracy and macro-F1 for the majority baseline is the")
        add("> reason accuracy is not the headline metric: predicting the largest intent")
        add("> for everything scores respectably on accuracy while being useless.")
        add("")

        conf = nat.get("llm", {}).get("top_confusions") or []
        if conf:
            add("Most frequent confusions (LLM classifier, natural stratum):")
            add("")
            add("| true intent | predicted as | count |")
            add("| --- | --- | --- |")
            for t, p, n in conf:
                add(f"| `{t}` | `{p}` | {n} |")
            add("")

    pc = c.get("per_class_llm_all_strata")
    if pc:
        add("### Per-class performance (all strata)")
        add("")
        add("Pooled across strata so rare intents have enough support to measure.")
        add("Because the enriched and adversarial strata over-represent hard cases,")
        add("treat these as a **lower bound**, not a population estimate.")
        add("")
        add("| intent | precision | recall | F1 | support |")
        add("| --- | --- | --- | --- | --- |")
        for k, v in sorted(pc.items(), key=lambda kv: -kv[1]["support"]):
            add(f"| `{k}` | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} | {v['support']} |")
        add("")

    # -- reply quality ------------------------------------------------------
    q = r.get("reply_quality_natural_stratum", {})
    if q:
        add("## Reply quality (LLM judge, natural stratum)")
        add("")
        add("Judge is `qwen3.8-27b`; the generator is `gpt-oss-20b`. Different model")
        add("families, and the judge is blind to which system wrote each reply.")
        add("Scales are 0–2 per dimension.")
        add("")
        add("| system | n | acceptable to send | catastrophic | factual safety | addresses need | actionability | tone |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- |")
        pretty = {
            "canned": "fixed apology (trivial baseline)",
            "nearest_neighbour": "replay nearest historical reply (simple baseline)",
            "grounded_llm": "grounded generator (system)",
        }
        for arm in ("canned", "nearest_neighbour", "grounded_llm"):
            a = q.get(arm)
            if not a:
                continue
            add(
                f"| {pretty.get(arm, arm)} | {a['n']} | {_pct(a['acceptable_rate'])}{_ci(a.get('acceptable_ci'))} "
                f"| {_pct(a['catastrophic_rate'])}{_ci(a.get('catastrophic_ci'))} "
                f"| {a['mean_factual_safety']:.2f} | {a['mean_addresses_need']:.2f} "
                f"| {a['mean_actionability']:.2f} | {a['mean_tone_fit']:.2f} |"
            )
        add("")
        overrides = sum(a.get("judge_consistency_overrides", 0) for a in q.values())
        total = sum(a.get("n", 0) for a in q.values())
        if total:
            add(f"Judge self-consistency: {overrides}/{total} verdicts "
                f"({_pct(overrides/total)}) contradicted the judge's own dimension scores "
                "and were corrected in code. A high rate here is a reason to discount the "
                "judge, not a reason to celebrate the correction.")
            add("")

    # -- selective ----------------------------------------------------------
    s = r.get("selective", {})
    op = s.get("operating_point")
    if op:
        add("## The headline: quality at a stated coverage")
        add("")
        add("A system allowed to abstain has no single quality number. These two must")
        add("always be quoted together.")
        add("")
        add(f"- **Auto-handled coverage: {_pct(op['auto_coverage'])}** of messages")
        add(f"- **Acceptable-reply rate on auto-handled: {_pct(op['quality_on_auto'])}**"
            f"{_ci(op.get('quality_on_auto_ci'))}")
        if "catastrophic_rate_on_auto" in op:
            add(f"- Catastrophic replies among auto-sent: {_pct(op['catastrophic_rate_on_auto'])}"
                f"{_ci(op.get('catastrophic_rate_on_auto_ci'))} "
                f"({op.get('catastrophic_escapes', 0)} escapes)")
        add(f"- Drafted for a human: {_pct(op['assist_rate'])}; escalated: {_pct(op['escalate_rate'])}")
        add(f"- If every reply were sent unfiltered, acceptable rate would be "
            f"{_pct(op['quality_overall_if_all_sent'])} — the gap is what the router buys.")
        add("")

    bs = s.get("by_stratum")
    if bs:
        add("### By stratum")
        add("")
        add("| stratum | auto coverage | quality on auto | escalate rate | catastrophic on auto |")
        add("| --- | --- | --- | --- | --- |")
        for k, v in bs.items():
            add(f"| `{k}` | {_pct(v['auto_coverage'])} | {_pct(v['quality_on_auto'])} "
                f"| {_pct(v['escalate_rate'])} | {_pct(v.get('catastrophic_rate_on_auto'))} |")
        add("")

    sweep = s.get("router_sweep")
    if sweep:
        add("### Router ablations")
        add("")
        add("Each row removes exactly one guard. If removing a guard does not worsen")
        add("anything, the guard is not earning its place.")
        add("")
        add("| config | auto coverage | quality on auto | catastrophic escapes |")
        add("| --- | --- | --- | --- |")
        for k in sorted(sweep):
            if not k.startswith("ablate"):
                continue
            v = sweep[k]
            add(f"| `{k}` | {_pct(v['auto_coverage'])} | {_pct(v['quality_on_auto'])} "
                f"| {v.get('catastrophic_escapes', 0)} |")
        add("")
        add("Threshold sweep (the risk–coverage trade-off):")
        add("")
        add("| config | auto coverage | quality on auto |")
        add("| --- | --- | --- |")
        for k in sorted(sweep):
            if k.startswith("ablate"):
                continue
            v = sweep[k]
            add(f"| `{k}` | {_pct(v['auto_coverage'])} | {_pct(v['quality_on_auto'])} |")
        add("")

    # -- router behaviour ---------------------------------------------------
    rt = r.get("router", {})
    if rt.get("rule_distribution"):
        add("## Which rule decided each message")
        add("")
        add("| rule | count |")
        add("| --- | --- |")
        for k, v in sorted(rt["rule_distribution"].items(), key=lambda kv: -kv[1]):
            add(f"| `{k}` | {v} |")
        add("")
        srd = rt.get("self_report_disagreement_rate")
        if srd is not None:
            add(f"The generator's self-report disagreed with the independent regex check on "
                f"**{_pct(srd)}** of drafts — i.e. it asserted a concrete time, platform or "
                "amount while reporting that it had not. This is why the router cross-checks "
                "rather than trusting the model's own declaration.")
            add("")

    # -- cost ---------------------------------------------------------------
    cost = r.get("cost", {})
    if cost.get("calls"):
        add("## What this cost to produce")
        add("")
        add(f"{cost['calls']} LLM calls, {cost['total_tokens']:,} tokens, "
            f"{cost.get('throttle_wait_s', 0)/60:.0f} minutes spent waiting on the "
            "free-tier rate limit.")
        add("")
        add("| stage | calls | tokens |")
        add("| --- | --- | --- |")
        for k, v in cost.get("by_tag", {}).items():
            add(f"| `{k}` | {v['calls']} | {v['tokens']:,} |")
        add("")

    return "\n".join(L) + "\n"


def main() -> None:
    if not RESULTS.exists():
        raise SystemExit(f"{RESULTS} not found — run `make report` first.")
    md = render(json.loads(RESULTS.read_text()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(md)
    print(f"wrote {OUT} ({len(md.splitlines())} lines)")


if __name__ == "__main__":
    main()
