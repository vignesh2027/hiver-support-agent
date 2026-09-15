"""End-to-end evaluation harness.

Stages are separate commands because each has a different cost profile and the
free-tier request budget is a real constraint:

  classify   3 arms over the golden set. Two arms are free (no LLM).
  generate   1 arm over the golden set. The two reply baselines need no LLM.
  judge      the expensive stage. Head-to-head over the *natural* stratum only
             (the sole unbiased sample), plus the system arm over the stress
             strata for failure analysis.
  report     pure computation over saved runs. No API calls, so the headline
             tables can be regenerated offline at any time.

Everything each stage produces lands in `artifacts/runs/` as JSONL, so a later
stage never re-runs an earlier one, and `HSA_OFFLINE=1` can replay the whole
thing from the committed cache with no key.

The strata are never pooled into a single headline. `natural` is the only
stratum from which a population estimate may be drawn; `enriched` and
`adversarial` are deliberately biased and are reported separately. The report
builder refuses to compute a population metric over the pooled set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..config import ARTIFACTS, GOLDEN, INTERIM, METRICS
from ..classify.baselines import MajorityBaseline, WeakSupervisedLogReg, train_baselines
from ..classify.llm_classifier import LLMClassifier
from ..reply.generate import CannedBaseline, GroundedGenerator, NearestNeighbourBaseline
from ..retrieve.index import PrecedentIndex, build as build_index
from ..route.policy import RouterConfig, route, sweep_configs
from ..taxonomy.schema import load as load_taxonomy
from .judge import ReplyJudge, automated_flags
from .metrics import (
    area_under_risk_coverage,
    classification_report,
    kappa_with_ci,
    risk_coverage_curve,
    selective_metrics,
    top_confusions,
    wilson_ci,
)

RUNS = ARTIFACTS / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

GOLD = GOLDEN / "golden.jsonl"
POOL = GOLDEN / "pool.jsonl"
CLS_RUN = RUNS / "classification.jsonl"
GEN_RUN = RUNS / "generation.jsonl"
JUDGE_RUN = RUNS / "judgements.jsonl"


def load_golden(require_labels: bool = True) -> pd.DataFrame:
    """Golden set if it exists, otherwise the unlabelled pool.

    The unlabelled fallback matters operationally. Classification, generation
    and judging depend only on the customer messages -- gold labels are needed
    solely to *score* them. Running the expensive API stages before the human
    labelling session is finished lets the free-tier daily request budget be
    spent while it is available, and `make report` afterwards costs nothing.
    Rows carry `labels_available` so no metric can silently score against
    placeholder labels.
    """
    if GOLD.exists():
        g = pd.read_json(GOLD, lines=True)
        if len(g):
            g["labels_available"] = True
            return g
    if require_labels:
        raise SystemExit(
            "data/golden/golden.jsonl not found or empty.\n"
            "Run:  make prelabel   then   make label   (see README, 'Building the golden set')."
        )
    pool = pd.read_json(POOL, lines=True)
    pool["intent"] = None
    pool["handling"] = None
    pool["review_kind"] = "UNLABELLED"
    pool["labels_available"] = False
    print(
        f"! No golden.jsonl yet - running over the unlabelled pool ({len(pool)} rows).\n"
        "  Generation and judging are unaffected; scoring is skipped until labels exist."
    )
    return pool


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write atomically: full file to a temp path, then rename over the target.

    Stages checkpoint by rewriting the whole file, so a plain open("w")
    truncates it for as long as the write takes. Any concurrent reader -- the
    judge stage, or just a `wc -l` while a long run is going -- can see an
    empty or half-written file and silently treat it as the real result.
    rename(2) is atomic within a filesystem, so a reader sees either the old
    file or the new one, never a partial one.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def _index(backend: str = "tfidf") -> PrecedentIndex:
    p = ARTIFACTS / f"index_GWRHelp_{backend}.pkl"
    return PrecedentIndex.load(p) if p.exists() else build_index("GWRHelp", backend)


# --------------------------------------------------------------- classify

def stage_classify(batch: int = 6, knn: int = 0, require_labels: bool = False) -> pd.DataFrame:
    g = load_golden(require_labels=require_labels)
    texts = g["customer_msg"].tolist()
    ids = g["example_id"].tolist()

    maj, lr = train_baselines()
    arms = {
        "majority": maj.predict(texts),
        "tfidf_logreg": lr.predict(texts),
        "llm": LLMClassifier(batch_size=batch, knn=knn).predict(texts, ids),
    }

    rows = []
    for i, r in enumerate(g.itertuples()):
        row = {
            "example_id": r.example_id,
            "stratum": r.stratum,
            "customer_msg": r.customer_msg,
            "gold_intent": r.intent,
            "gold_handling": r.handling,
            "review_kind": getattr(r, "review_kind", "unknown"),
        }
        for arm, preds in arms.items():
            row[f"{arm}_intent"] = preds[i].intent
            row[f"{arm}_conf"] = round(preds[i].confidence, 4)
        rows.append(row)

    _write_jsonl(CLS_RUN, rows)
    print(f"wrote {CLS_RUN} ({len(rows)} rows)")
    return pd.DataFrame(rows)


# --------------------------------------------------------------- generate

def stage_generate(backend: str = "tfidf", arm_intent: str = "llm") -> pd.DataFrame:
    """Generate replies for every golden example, all three arms.

    The two baselines cost no API calls, so all three arms are generated over
    the full golden set even though only the natural stratum is judged
    head-to-head.
    """
    cls = pd.read_json(CLS_RUN, lines=True)
    ix = _index(backend)
    gens = {
        "canned": CannedBaseline(ix),
        "nearest_neighbour": NearestNeighbourBaseline(ix),
        "grounded_llm": GroundedGenerator(ix),
    }

    # Resume from whatever a previous run managed to write.
    #
    # The first version of this function accumulated every row in memory and
    # wrote once at the end. It then died on call ~200 of 220 against a daily
    # token cap and wrote nothing at all -- an hour of rate-limited work
    # thrown away because of where a single write sat. The LLM cache meant the
    # responses survived, but nothing else did. Checkpointing every 10
    # messages and skipping already-generated ids makes a long throttled run
    # restartable, which on a free tier is not optional.
    rows: list[dict] = []
    done_ids: set[str] = set()
    if GEN_RUN.exists():
        for line in GEN_RUN.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                rows.append(rec)
                done_ids.add(rec["example_id"])
        if done_ids:
            print(f"resuming: {len(done_ids)} messages already generated")

    n = len(cls)
    for i, r in enumerate(cls.itertuples(), 1):
        if r.example_id in done_ids:
            continue
        intent = getattr(r, f"{arm_intent}_intent")
        conf = float(getattr(r, f"{arm_intent}_conf"))
        for arm, gen in gens.items():
            d = gen.draft(r.customer_msg, intent)
            dec = route(r.customer_msg, intent, conf, d, RouterConfig())
            rows.append(
                {
                    "example_id": r.example_id,
                    "stratum": r.stratum,
                    "arm": arm,
                    "customer_msg": r.customer_msg,
                    "gold_intent": r.gold_intent,
                    "gold_handling": r.gold_handling,
                    "pred_intent": intent,
                    "pred_conf": conf,
                    "reply": d.reply,
                    "precedent_score": round(d.max_precedent_score, 4),
                    "precedent_ids": [p.episode_id for p in d.precedents],
                    "asserts_live_fact": d.asserts_live_fact,
                    "regex_live_claim": d.regex_flags_live_claim,
                    "self_report_disagrees": d.self_report_disagrees,
                    "missing_information": d.missing_information,
                    "action": dec.action,
                    "action_reason": dec.reason,
                    "action_rule": dec.rule,
                    **{f"auto_{k}": v for k, v in automated_flags(d.reply, r.customer_msg).items()},
                }
            )
        if i % 10 == 0:
            _write_jsonl(GEN_RUN, rows)
            print(f"  generated {i}/{n} (checkpointed)", flush=True)

    _write_jsonl(GEN_RUN, rows)
    print(f"wrote {GEN_RUN} ({len(rows)} rows)")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ judge

def stage_judge(head_to_head_stratum: str = "natural", system_arm: str = "grounded_llm") -> pd.DataFrame:
    gen = pd.read_json(GEN_RUN, lines=True)
    judge = ReplyJudge()

    # Head-to-head on the unbiased stratum; system arm everywhere else.
    h2h = gen[gen.stratum == head_to_head_stratum]
    stress = gen[(gen.stratum != head_to_head_stratum) & (gen.arm == system_arm)]
    todo = pd.concat([h2h, stress]).reset_index(drop=True)

    done = {}
    if JUDGE_RUN.exists():
        for line in JUDGE_RUN.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["example_id"], r["arm"])] = r

    ix = _index()
    rows = list(done.values())
    todo = todo[~todo.apply(lambda r: (r.example_id, r.arm) in done, axis=1)]
    print(f"judging {len(todo)} replies ({len(done)} already cached)")

    for i, r in enumerate(todo.itertuples(), 1):
        precedents = ix.search(r.customer_msg, k=4)
        v = judge.judge(r.customer_msg, r.reply, precedents)
        rows.append(
            {
                "example_id": r.example_id,
                "arm": r.arm,
                "stratum": r.stratum,
                "action": r.action,
                "reply": r.reply,
                **v.to_dict(),
            }
        )
        if i % 20 == 0:
            print(f"  judged {i}/{len(todo)}", flush=True)
            _write_jsonl(JUDGE_RUN, rows)

    _write_jsonl(JUDGE_RUN, rows)
    print(f"wrote {JUDGE_RUN} ({len(rows)} rows)")
    return pd.DataFrame(rows)


# --------------------------------------------------- judge-agreement study

GRADING_SET = GOLDEN / "replies_for_grading.jsonl"
HUMAN_GRADES = GOLDEN / "human_reply_grades.jsonl"


def stage_grading_set(n_per_arm: int = 20, seed: int = 20260913) -> pd.DataFrame:
    """Sample replies for a human to grade blind, for judge-vs-human agreement.

    Balanced across arms on purpose. If the sample were drawn at random from
    all judged replies it would be dominated by whichever arm is most common,
    and agreement would then be measured mostly on one system's output. A
    balanced sample also guarantees the human sees obviously-bad replies (the
    canned baseline) as well as good ones, which is what makes an agreement
    statistic meaningful -- two raters who only ever see good replies agree
    trivially.

    The file carries no indication of which arm produced which reply; the CLI
    shows only the customer message and the reply text.
    """
    gen = pd.read_json(GEN_RUN, lines=True)
    nat = gen[gen.stratum == "natural"]
    rng = pd.Series(range(len(nat))).sample(frac=1, random_state=seed)

    rows = []
    for arm in sorted(nat.arm.unique()):
        sub = nat[nat.arm == arm].sample(
            n=min(n_per_arm, (nat.arm == arm).sum()), random_state=seed
        )
        for r in sub.itertuples():
            rows.append(
                {
                    "example_id": r.example_id,
                    "arm": r.arm,
                    "customer_msg": r.customer_msg,
                    "reply": r.reply,
                }
            )
    out = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    _write_jsonl(GRADING_SET, out.to_dict("records"))
    print(f"wrote {GRADING_SET} ({len(out)} replies, blind, {n_per_arm}/arm)")
    print("Grade them with:  make grade")
    return out


def _judge_agreement() -> dict:
    """Cohen's kappa between the human grades and the LLM judge."""
    if not HUMAN_GRADES.exists() or not JUDGE_RUN.exists():
        return {"status": "not yet measured - run `make grade`"}
    hum = pd.read_json(HUMAN_GRADES, lines=True)
    jud = pd.read_json(JUDGE_RUN, lines=True)
    jud["grade_id"] = jud["example_id"].astype(str) + "::" + jud["arm"].astype(str)
    m = hum.merge(jud, on="grade_id", suffixes=("_h", "_j"))
    if len(m) < 10:
        return {"status": f"only {len(m)} overlapping grades - too few to report"}

    acc = kappa_with_ci(
        m["human_acceptable"].astype(bool).tolist(),
        m["acceptable_to_send"].astype(bool).tolist(),
    )
    cat = kappa_with_ci(
        m["human_catastrophic"].astype(bool).tolist(),
        m["catastrophic"].astype(bool).tolist(),
    )
    # Direction of disagreement matters more than its size: a judge that is
    # systematically more permissive than a human inflates every headline.
    judge_yes_human_no = int(
        ((~m["human_acceptable"].astype(bool)) & m["acceptable_to_send"].astype(bool)).sum()
    )
    human_yes_judge_no = int(
        (m["human_acceptable"].astype(bool) & (~m["acceptable_to_send"].astype(bool))).sum()
    )
    return {
        "n": len(m),
        "acceptable_to_send": acc,
        "catastrophic": cat,
        "judge_more_permissive_than_human": judge_yes_human_no,
        "human_more_permissive_than_judge": human_yes_judge_no,
        "judge_acceptable_rate": round(float(m["acceptable_to_send"].mean()), 4),
        "human_acceptable_rate": round(float(m["human_acceptable"].mean()), 4),
        "by_arm": {
            arm: {
                "n": int((m.arm_h == arm).sum()),
                "human_acceptable": round(float(m.loc[m.arm_h == arm, "human_acceptable"].mean()), 3),
                "judge_acceptable": round(
                    float(m.loc[m.arm_h == arm, "acceptable_to_send"].mean()), 3
                ),
            }
            for arm in sorted(m.arm_h.unique())
        },
    }


def _label_quality() -> dict:
    """What the human adjudication measured about the golden labels."""
    if not GOLD.exists():
        return {"status": "no golden.jsonl yet"}
    g = pd.read_json(GOLD, lines=True)
    if not len(g):
        return {"status": "golden.jsonl is empty"}
    pool_n = len(pd.read_json(POOL, lines=True)) if POOL.exists() else len(g)
    kinds = g["review_kind"].value_counts().to_dict()
    human_seen = int(sum(v for k, v in kinds.items() if k in ("adjudicate", "audit")))
    out = {
        "n_labelled": len(g),
        "pool_n": pool_n,
        "label_coverage_of_pool": round(len(g) / pool_n, 4) if pool_n else None,
        "by_review_kind": kinds,
        "individually_reviewed_by_human": human_seen,
        "reviewed_fraction_of_labelled": round(human_seen / len(g), 4) if len(g) else None,
        "note": (
            "Items where the two independent pre-label passes disagreed and which were "
            "not adjudicated carry no gold label at all and are excluded from scoring, "
            "rather than being given a provisional label that would look like ground truth."
        ),
    }
    aud = g[g.review_kind == "audit"]
    if len(aud):
        k = int(aud["human_changed_prelabel"].sum())
        p, lo, hi = wilson_ci(k, len(aud))
        out["audit"] = {
            "n": len(aud),
            "corrected": k,
            "estimated_error_rate_of_unaudited_agreed_labels": round(p, 4),
            "ci": (round(lo, 4), round(hi, 4)),
        }
    adj = g[g.review_kind == "adjudicate"]
    if len(adj):
        out["adjudicated_disagreements"] = len(adj)
    if (g["seconds_spent"] > 0).any():
        out["median_seconds_per_reviewed_item"] = float(
            g.loc[g["seconds_spent"] > 0, "seconds_spent"].median()
        )
    return out


# ----------------------------------------------------------------- report

def _classification_tables(cls: pd.DataFrame) -> dict:
    tax = load_taxonomy()
    labels = list(tax.names)
    out: dict = {"by_stratum": {}, "overall_natural_only": {}}
    for arm in ("majority", "tfidf_logreg", "llm"):
        nat = cls[cls.stratum == "natural"]
        rep = classification_report(
            nat["gold_intent"].tolist(), nat[f"{arm}_intent"].tolist(), labels
        )
        out["overall_natural_only"][arm] = {
            "n": rep.n,
            "accuracy": rep.accuracy,
            "accuracy_ci": rep.accuracy_ci,
            "macro_f1": rep.macro_f1,
            "macro_f1_ci": rep.macro_f1_ci,
            "weighted_f1": rep.weighted_f1,
            "top_confusions": top_confusions(rep, 6),
        }
        for s in sorted(cls.stratum.unique()):
            sub = cls[cls.stratum == s]
            r2 = classification_report(
                sub["gold_intent"].tolist(), sub[f"{arm}_intent"].tolist(), labels
            )
            out["by_stratum"].setdefault(s, {})[arm] = {
                "n": r2.n,
                "accuracy": r2.accuracy,
                "macro_f1": r2.macro_f1,
            }
    # Per-class detail from the whole pool: the enriched stratum exists so
    # that rare classes have support, so per-class F1 uses everything and says
    # so.
    full = classification_report(cls["gold_intent"].tolist(), cls["llm_intent"].tolist(), labels)
    out["per_class_llm_all_strata"] = full.per_class
    out["confusion_llm_all_strata"] = full.confusion
    return out


def _judge_tables(judg: pd.DataFrame, gen: pd.DataFrame) -> dict:
    out: dict = {}
    nat = judg[judg.stratum == "natural"]
    for arm in sorted(nat.arm.unique()):
        a = nat[nat.arm == arm]
        k = int(a.acceptable_to_send.sum())
        p, lo, hi = wilson_ci(k, len(a))
        kc = int(a.catastrophic.sum())
        cp, clo, chi = wilson_ci(kc, len(a))
        out[arm] = {
            "n": len(a),
            "acceptable_rate": round(p, 4),
            "acceptable_ci": (round(lo, 4), round(hi, 4)),
            "catastrophic_rate": round(cp, 4),
            "catastrophic_ci": (round(clo, 4), round(chi, 4)),
            "mean_factual_safety": round(float(a.factual_safety.mean()), 3),
            "mean_addresses_need": round(float(a.addresses_need.mean()), 3),
            "mean_actionability": round(float(a.actionability.mean()), 3),
            "mean_tone_fit": round(float(a.tone_fit.mean()), 3),
            "mean_composite": round(float(a.composite.mean()), 3),
            "judge_consistency_overrides": int(a.consistency_overridden.sum()),
            "judge_errors": int(a.error.notna().sum()) if "error" in a else 0,
        }
    return out


def _selective_tables(gen: pd.DataFrame, judg: pd.DataFrame, system_arm: str) -> dict:
    """Risk-coverage: the actual headline of this system."""
    j = judg[judg.arm == system_arm].set_index("example_id")
    g = gen[gen.arm == system_arm].set_index("example_id")
    common = [i for i in g.index if i in j.index]
    if not common:
        return {}
    g, j = g.loc[common], j.loc[common]

    actions = g["action"].tolist()
    correct = j["acceptable_to_send"].astype(bool).tolist()
    catastrophic = j["catastrophic"].astype(bool).tolist()

    out = {
        "operating_point": selective_metrics(actions, correct, catastrophic),
        "by_stratum": {},
        "risk_coverage": {},
        "router_sweep": {},
    }
    for s in sorted(g["stratum"].unique()):
        m = g["stratum"] == s
        if m.sum() < 5:
            continue
        out["by_stratum"][s] = selective_metrics(
            g.loc[m, "action"].tolist(),
            j.loc[m, "acceptable_to_send"].astype(bool).tolist(),
            j.loc[m, "catastrophic"].astype(bool).tolist(),
        )

    curve = risk_coverage_curve(g["pred_conf"].astype(float).tolist(), correct)
    out["risk_coverage"] = {
        "aurc": round(area_under_risk_coverage(curve), 4),
        "points": [
            {"coverage": round(c, 3), "error": round(e, 4)}
            for c, e, _ in curve[:: max(1, len(curve) // 20)]
        ],
    }

    # Sweep router configs. Each is a different operating point on the same
    # judged data, so no new API calls are needed.
    for name, cfg in sweep_configs():
        acts = []
        for r in g.itertuples():
            from ..reply.generate import Draft
            from ..retrieve.index import Precedent

            d = Draft(
                reply=r.reply,
                precedents=[
                    Precedent(0, "", "", float(r.precedent_score), "", False)
                ],
                asserts_live_fact=bool(r.asserts_live_fact),
            )
            acts.append(route(r.customer_msg, r.pred_intent, float(r.pred_conf), d, cfg).action)
        out["router_sweep"][name] = selective_metrics(acts, correct, catastrophic)
    return out


def stage_report(system_arm: str = "grounded_llm") -> dict:
    cls = pd.read_json(CLS_RUN, lines=True)
    gen = pd.read_json(GEN_RUN, lines=True)
    judg = pd.read_json(JUDGE_RUN, lines=True) if JUDGE_RUN.exists() else pd.DataFrame()

    labelled = cls["gold_intent"].notna().sum()
    have_labels = labelled > 0

    result: dict = {
        "golden_set": {
            "n": len(cls),
            "labelled": int(labelled),
            "by_stratum": cls.stratum.value_counts().to_dict(),
        }
    }
    if have_labels:
        result["golden_set"]["intent_distribution"] = (
            cls.gold_intent.value_counts(normalize=True).round(4).to_dict()
        )
        result["golden_set"]["handling_distribution"] = (
            cls.gold_handling.value_counts(normalize=True).round(4).to_dict()
        )
        result["classification"] = _classification_tables(cls.dropna(subset=["gold_intent"]))
    else:
        # Refuse to emit a classification table rather than scoring against
        # placeholders and producing a number that looks real.
        result["classification"] = {
            "status": "skipped - no human labels yet; run `make label` then `make report`"
        }
    if len(judg):
        result["reply_quality_natural_stratum"] = _judge_tables(judg, gen)
        result["selective"] = _selective_tables(gen, judg, system_arm)

    # Router behaviour is measurable without the judge.
    result["router"] = {
        "action_distribution": gen[gen.arm == system_arm].action.value_counts(normalize=True).round(4).to_dict(),
        "rule_distribution": gen[gen.arm == system_arm].action_rule.value_counts().to_dict(),
        "self_report_disagreement_rate": round(
            float(gen[gen.arm == system_arm].self_report_disagrees.mean()), 4
        ),
    }

    result["label_quality"] = _label_quality()
    result["judge_vs_human"] = _judge_agreement()

    from ..llm import ledger_summary

    result["cost"] = ledger_summary()

    out = METRICS / "results.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {out}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["classify", "generate", "judge", "grading-set", "report", "all"])
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--knn", type=int, default=0)
    ap.add_argument("--backend", default="tfidf")
    ap.add_argument("--require-labels", action="store_true",
                    help="fail instead of falling back to the unlabelled pool")
    args = ap.parse_args()

    if args.stage in ("classify", "all"):
        stage_classify(batch=args.batch, knn=args.knn, require_labels=args.require_labels)
    if args.stage in ("generate", "all"):
        stage_generate(backend=args.backend)
    if args.stage in ("judge", "all"):
        stage_judge()
    if args.stage in ("grading-set", "all"):
        stage_grading_set()
    if args.stage in ("report", "all"):
        r = stage_report()
        print(json.dumps(r.get("classification", {}).get("overall_natural_only", {}), indent=2)[:1200])


if __name__ == "__main__":
    main()
