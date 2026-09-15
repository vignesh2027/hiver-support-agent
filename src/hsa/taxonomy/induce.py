"""Induce an intent taxonomy *from the data* rather than inventing one.

Three stages, deliberately separated so each can be inspected:

1. **Cluster.** TF-IDF -> SVD -> KMeans over training-split customer messages.
   Over-cluster on purpose (k ~ 30 for an expected 8-12 intents): it is far
   easier for a human to merge two clusters that mean the same thing than to
   notice that one cluster is secretly two intents.

2. **Name.** One LLM call per cluster, shown the top terms and the exemplars
   nearest the centroid, asked for a label, a definition, and -- importantly --
   permission to call the cluster incoherent. A naming step that cannot say
   "this is junk" will happily name noise.

3. **Consolidate.** One call that merges the cluster proposals into a small
   taxonomy with inclusion/exclusion rules. The output doubles as the
   annotation guideline for the golden set, which is the point: if the
   definition is too vague to label against, it is too vague to classify
   against.

The result is written to artifacts/taxonomy.json and artifacts/taxonomy.md.
Stage 3 output is then hand-edited (see DECISIONS.md D-06); the edited file is
what the rest of the pipeline consumes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from ..config import ARTIFACTS, INTERIM, SEED, TAXONOMY_MODEL
from ..llm import chat_json

TAXONOMY_JSON = ARTIFACTS / "taxonomy.json"
TAXONOMY_MD = ARTIFACTS / "taxonomy.md"
CLUSTERS_JSON = ARTIFACTS / "clusters_raw.json"

# Railway-specific stop terms. The brand handle and generic politeness words
# dominate TF-IDF otherwise.
EXTRA_STOP = {
    "gwr", "gwrhelp", "train", "trains", "service", "services", "please", "thanks",
    "thank", "hi", "hello", "just", "got", "get", "im", "ive", "dont", "cant",
    "url", "today", "pls", "amp", "station", "time", "times",
    # The normalisation placeholders themselves. They survive max_df because
    # they sit just under the threshold, and they were crowding out the real
    # discriminating terms in every cluster's top-term list.
    "num", "nums",
}

# GWR's network, plus the three-letter CRS codes customers actually type.
#
# This list exists because of a failure, not a premonition. The first
# clustering run produced clusters organised by *place* -- one for Paddington,
# one for Bristol Parkway, one for Temple Meads, one for Didcot -- because
# station names are the highest-TF-IDF tokens in the corpus. "Delayed at
# Bristol" and "delayed at Reading" are the same intent and must land in the
# same cluster. Normalising stations and departure times to placeholders moves
# the clustering from geography to problem type. See DECISIONS.md D-05.
_STATIONS = {
    "paddington", "pad", "reading", "rdg", "bristol", "parkway", "bpw", "temple",
    "meads", "btm", "bath", "spa", "swindon", "swi", "didcot", "did", "oxford",
    "oxf", "cardiff", "cdf", "swansea", "swa", "exeter", "exd", "plymouth", "ply",
    "penzance", "pnz", "taunton", "tau", "newbury", "nbr", "maidenhead", "mai",
    "slough", "slo", "twyford", "twy", "cheltenham", "chm", "gloucester", "gcr",
    "worcester", "wos", "hereford", "her", "weston", "truro", "tru", "newquay",
    "paignton", "torquay", "totnes", "liskeard", "bodmin", "redruth", "camborne",
    "erth", "westbury", "salisbury", "basingstoke", "guildford", "gatwick",
    "bournemouth", "chippenham", "melksham", "trowbridge", "frome", "castle",
    "cary", "keynsham", "filton", "abbey", "wood", "yate", "nailsea", "clevedon",
    "portishead", "severn", "tunnel", "pilning", "patchway", "hayes", "southall",
    "ealing", "broadway", "hanwell", "iver", "langley", "burnham", "taplow",
    "wargrave", "henley", "marlow", "bourne", "windsor", "eton", "datchet",
    "ascot", "bracknell", "wokingham", "winnersh", "earley", "theale", "thatcham",
    "hungerford", "bedwyn", "pewsey", "warminster", "london", "cornwall", "devon",
    "penryn", "falmouth", "stroud", "kemble", "moreton", "marsh", "evesham",
    "pershore", "malvern", "ledbury", "yatton", "worle", "highbridge", "bridgwater",
    "tiverton", "honiton", "axminster", "dawlish", "teignmouth", "newton", "abbot",
}

# Times: "17:06", "1758", "7.17pm", "08.30", "the 4:41".
_TIME_RE = re.compile(
    r"\b(\d{1,2}[:.]\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm)|\b[01]?\d{3}\b|\b2[0-3]\d{2}\b)\b",
    re.I,
)
_NUM_RE = re.compile(r"\b\d+\b")
_STATION_RE = re.compile(r"\b(" + "|".join(sorted(_STATIONS, key=len, reverse=True)) + r")\b", re.I)


def normalize_for_clustering(s: str) -> str:
    """Collapse place and time specifics so clusters form around problem type.

    Applied only for taxonomy induction. The classifier and the retriever see
    the original text -- a station name is genuinely useful evidence when
    matching a message to a precedent, it is only harmful when deciding what
    the *categories* are.
    """
    s = _TIME_RE.sub(" <time> ", s)
    s = _STATION_RE.sub(" <station> ", s)
    s = _NUM_RE.sub(" <num> ", s)
    return re.sub(r"\s+", " ", s).strip()


def _vectorizer() -> TfidfVectorizer:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return TfidfVectorizer(
        lowercase=True,
        stop_words=list(ENGLISH_STOP_WORDS | EXTRA_STOP),
        ngram_range=(1, 2),
        min_df=8,
        max_df=0.35,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def cluster_messages(msgs: list[str], k: int, seed: int = SEED) -> dict:
    """Cluster on normalised text; report exemplars in the original wording.

    The LLM namer needs to see real messages -- "<station> delayed <time>" is
    unreadable -- so normalisation affects the geometry only.
    """
    norm = [normalize_for_clustering(m) for m in msgs]
    vec = _vectorizer()
    X = vec.fit_transform(norm)
    svd = make_pipeline(TruncatedSVD(n_components=120, random_state=seed), Normalizer(copy=False))
    Xr = svd.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=seed, n_init=8)
    labels = km.fit_predict(Xr)

    terms = np.array(vec.get_feature_names_out())
    # Project centroids back to term space to read off what each cluster is about.
    centroids_term = svd.named_steps["truncatedsvd"].inverse_transform(km.cluster_centers_)

    out = {}
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        top_terms = terms[np.argsort(centroids_term[c])[::-1][:12]].tolist()
        # Exemplars nearest the centroid are the most representative, but a
        # purely central sample hides the cluster's spread, so we also take a
        # few from the edge.
        d = np.linalg.norm(Xr[idx] - km.cluster_centers_[c], axis=1)
        order = np.argsort(d)
        near = idx[order[:10]]
        far = idx[order[len(order) // 2 : len(order) // 2 + 4]]
        out[str(c)] = {
            "size": int(len(idx)),
            "share": round(len(idx) / len(msgs), 4),
            "top_terms": top_terms,
            "exemplars": [msgs[i][:240] for i in near],
            "spread_exemplars": [msgs[i][:240] for i in far],
            "member_idx": idx.tolist(),
        }
    return {"k": k, "n": len(msgs), "clusters": out, "labels": labels.tolist()}


def cluster_with_splits(
    msgs: list[str], k: int, seed: int = SEED, split_threshold: float = 0.15, sub_k: int = 7
) -> dict:
    """Cluster, then re-cluster any cluster that swallowed too much traffic.

    KMeans on short, noisy text reliably produces one enormous "everything
    else" cluster -- here it took 27% of the corpus. Naming that cluster is
    meaningless and leaving it unexamined means a quarter of real customer
    traffic never gets looked at. So any cluster above `split_threshold` is
    recursively split and its children replace it. See DECISIONS.md D-07.
    """
    base = cluster_messages(msgs, k=k, seed=seed)
    final: dict[str, dict] = {}
    n = len(msgs)
    # Per-message leaf-cluster id. Downstream this becomes weak supervision:
    # cluster -> intent (via the curated taxonomy) gives a training label for
    # the logistic-regression baseline without any hand-labelling, which is
    # what lets that baseline exist at all on an unlabelled corpus.
    assignment: list[str | None] = [None] * n

    for cid, c in base["clusters"].items():
        if c["share"] <= split_threshold or c["size"] < sub_k * 20:
            final[cid] = c
            for i in c["member_idx"]:
                assignment[i] = cid
            continue

        member_idx = c["member_idx"]
        sub_msgs = [msgs[i] for i in member_idx]
        print(
            f"  cluster {cid} holds {c['share']:.1%} of traffic -- splitting into {sub_k}",
            flush=True,
        )
        sub = cluster_messages(sub_msgs, k=sub_k, seed=seed + 1)
        for scid, sc in sub["clusters"].items():
            sc = dict(sc)
            sc["share"] = round(sc["size"] / n, 4)
            sc["member_idx"] = [member_idx[i] for i in sc["member_idx"]]
            sc["split_from"] = cid
            final[f"{cid}.{scid}"] = sc
            for i in sc["member_idx"]:
                assignment[i] = f"{cid}.{scid}"

    for c in final.values():
        c.pop("member_idx", None)
    assert all(a is not None for a in assignment), "every message must land in a leaf cluster"
    return {
        "k": k,
        "n": n,
        "clusters": final,
        "split_threshold": split_threshold,
        "assignment": assignment,
        "messages": msgs,
    }


_NAME_SYS = (
    "You are a support operations analyst reading real customer tweets sent to a UK train "
    "operator's support desk. You name what customers WANT, not what topic they mention. "
    "'Delay' is a topic; 'asking whether a specific service is delayed right now' and "
    "'demanding compensation for a past delay' are two different intents with different "
    "handling. Be willing to say a cluster is incoherent."
)

_NAME_USER = """Cluster {cid} ({share:.1%} of messages).

Top TF-IDF terms: {terms}

Representative messages (nearest centroid):
{near}

Messages from the cluster's edge:
{far}

Return JSON:
{{
  "label": "snake_case_intent_name",
  "definition": "one sentence: what the customer wants",
  "coherent": true|false,
  "coherence_note": "if false, what the cluster actually mixes together",
  "needs_live_data": true|false,
  "needs_live_data_note": "what real-time or account-specific data a correct answer requires, or null"
}}"""


def name_clusters(clusters: dict, model: str = TAXONOMY_MODEL) -> dict:
    named = {}
    for cid, c in clusters["clusters"].items():
        msgs = "\n".join(f"- {m}" for m in c["exemplars"])
        far = "\n".join(f"- {m}" for m in c["spread_exemplars"])
        res = chat_json(
            [
                {"role": "system", "content": _NAME_SYS},
                {
                    "role": "user",
                    "content": _NAME_USER.format(
                        cid=cid, share=c["share"], terms=", ".join(c["top_terms"]), near=msgs, far=far
                    ),
                },
            ],
            model=model,
            tag="taxonomy:name",
            max_tokens=600,
        )
        res["size"] = c["size"]
        res["share"] = c["share"]
        res["top_terms"] = c["top_terms"]
        res["sample"] = c["exemplars"][:3]
        named[cid] = res
        flag = "" if res.get("coherent", True) else "  [INCOHERENT]"
        live = "  [LIVE-DATA]" if res.get("needs_live_data") else ""
        print(f"  c{cid:>2} {c['share']:>6.1%}  {res.get('label','?'):<34}{flag}{live}", flush=True)
    return named


_CONSOLIDATE_SYS = (
    "You design intent taxonomies for customer support automation. A good taxonomy is small, "
    "mutually exclusive, collectively exhaustive over the observed data, and every category "
    "must be distinguishable by a human annotator reading one message with no extra context. "
    "You are ruthless about merging categories that would be handled identically."
)

_CONSOLIDATE_USER = """Below are {n} raw clusters induced from {total} real customer messages to a UK
train operator's Twitter support desk, with the share of traffic each represents.

{blocks}

Design a FINAL taxonomy of 8-11 intents (including exactly one catch-all "other").

Hard requirements:
- Merge clusters that a support agent would handle the same way.
- Separate intents whose correct HANDLING differs, even if the topic is the same.
  In particular keep apart: asking about a live/ongoing disruption vs. claiming
  compensation for a past one.
- Every intent needs inclusion rules AND exclusion rules a human can apply to a single
  message, plus a hard case that shows the boundary with its nearest neighbour intent.
- Mark `needs_live_data` true when a *correct* answer requires real-time running
  information or access to the customer's booking/account, which a text-only agent does
  not have.
- Mark `default_disposition` as "auto" (safe to answer from historical precedent),
  "assist" (draft for a human to send), or "escalate" (must reach a human).

Return JSON:
{{
  "intents": [
    {{
      "name": "snake_case",
      "definition": "...",
      "includes": ["...", "..."],
      "excludes": ["...", "..."],
      "hard_case": "a message that looks like this intent but is not, and why",
      "needs_live_data": true|false,
      "default_disposition": "auto|assist|escalate",
      "disposition_rationale": "...",
      "source_clusters": ["3","7"],
      "est_share": 0.00
    }}
  ],
  "merges_explained": "what you merged and why",
  "dropped": "any cluster you treated as noise, and why"
}}"""


def consolidate(named: dict, total: int, model: str = TAXONOMY_MODEL) -> dict:
    blocks = []
    for cid, c in sorted(named.items(), key=lambda kv: -kv[1].get("share", 0)):
        blocks.append(
            f"[cluster {cid}] share={c.get('share',0):.1%} label={c.get('label')} "
            f"coherent={c.get('coherent')} needs_live_data={c.get('needs_live_data')}\n"
            f"  definition: {c.get('definition')}\n"
            f"  terms: {', '.join(c.get('top_terms', [])[:8])}\n"
            f"  example: {(c.get('sample') or [''])[0][:160]}"
        )
    return chat_json(
        [
            {"role": "system", "content": _CONSOLIDATE_SYS},
            {
                "role": "user",
                "content": _CONSOLIDATE_USER.format(
                    n=len(named), total=total, blocks="\n\n".join(blocks)
                ),
            },
        ],
        model=model,
        tag="taxonomy:consolidate",
        max_tokens=4000,
    )


def write_markdown(tax: dict, path: Path = TAXONOMY_MD) -> None:
    """Render the taxonomy as the annotation guideline used for the golden set."""
    lines = [
        "# Intent taxonomy — GWRHelp",
        "",
        "Induced from training-split customer messages by clustering, named and",
        "consolidated by an LLM, then hand-edited. This file is also the annotation",
        "guideline: every golden-set label was assigned by applying these rules.",
        "",
        "| intent | share | live data needed | default disposition |",
        "| --- | --- | --- | --- |",
    ]
    for it in tax["intents"]:
        lines.append(
            f"| `{it['name']}` | {it.get('est_share', 0):.1%} | "
            f"{'yes' if it.get('needs_live_data') else 'no'} | {it.get('default_disposition','-')} |"
        )
    lines.append("")
    for it in tax["intents"]:
        lines += [
            f"## `{it['name']}`",
            "",
            it["definition"],
            "",
            "**Includes**",
            *[f"- {x}" for x in it.get("includes", [])],
            "",
            "**Excludes**",
            *[f"- {x}" for x in it.get("excludes", [])],
            "",
            f"**Boundary case.** {it.get('hard_case','—')}",
            "",
            f"**Disposition.** `{it.get('default_disposition','-')}` — {it.get('disposition_rationale','')}",
            "",
        ]
    if tax.get("merges_explained"):
        lines += ["## Consolidation notes", "", tax["merges_explained"], ""]
    if tax.get("dropped"):
        lines += ["## Dropped clusters", "", tax["dropped"], ""]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="GWRHelp")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--max-msgs", type=int, default=6000)
    ap.add_argument("--stage", default="all", choices=["cluster", "name", "consolidate", "all"])
    args = ap.parse_args()

    train = pd.read_parquet(INTERIM / f"episodes_{args.brand}_train.parquet")
    msgs = train["customer_msg"].tolist()[: args.max_msgs]

    if args.stage in ("cluster", "all"):
        print(f"clustering {len(msgs)} messages into k={args.k} ...", flush=True)
        clusters = cluster_with_splits(msgs, k=args.k)
        CLUSTERS_JSON.write_text(json.dumps(clusters, indent=1))
        print(f"  wrote {CLUSTERS_JSON}")
    else:
        clusters = json.loads(CLUSTERS_JSON.read_text())

    if args.stage == "cluster":
        return

    if args.stage in ("name", "all"):
        print("naming clusters ...", flush=True)
        named = name_clusters(clusters)
        (ARTIFACTS / "clusters_named.json").write_text(json.dumps(named, indent=1))
    else:
        named = json.loads((ARTIFACTS / "clusters_named.json").read_text())

    if args.stage == "name":
        return

    if args.stage in ("consolidate", "all"):
        print("consolidating ...", flush=True)
        tax = consolidate(named, total=len(msgs))
        TAXONOMY_JSON.write_text(json.dumps(tax, indent=2))
        write_markdown(tax)
        print(f"\nFinal taxonomy ({len(tax['intents'])} intents):")
        for it in tax["intents"]:
            print(
                f"  {it['name']:<32} {it.get('est_share',0):>6.1%}  "
                f"live={str(it.get('needs_live_data')):<5} {it.get('default_disposition')}"
            )
        print(f"\nwrote {TAXONOMY_JSON} and {TAXONOMY_MD}")


if __name__ == "__main__":
    main()
