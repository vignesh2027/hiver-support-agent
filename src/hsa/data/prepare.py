"""Turn the raw TWCS dump into per-brand support *episodes*.

The raw file is a flat table of 2.8M tweets linked by `in_response_to_tweet_id`.
What a support agent actually needs is an episode: the customer's opening
message, the brand's first reply, and the rest of the exchange as ground truth
for how this brand historically resolved that issue.

Two things here are load-bearing for the evaluation and are easy to get wrong:

* **Brand selection is measured, not assumed.** A brand whose first reply is
  almost always "DM us" teaches a reply model nothing except how to say "DM
  us". We score every candidate brand on how often it resolves in-thread and
  pick on that, not on volume. See `brand_scorecard`.

* **Episodes carry a timestamp.** Every split downstream is chronological.
  Randomly splitting a support corpus leaks: the same outage produces hundreds
  of near-identical tweets within an hour, so a random split puts near-copies
  of test items into train and flatters retrieval enormously. See DECISIONS.md
  D-04.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from ..config import INTERIM, TWCS_CSV

# Handles that front the brand's support desk. Everything not inbound is a
# brand tweet, so the author_id of an outbound tweet *is* the brand.
_MENTION = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")

# A first reply matching this is a *deflection*: it moves the conversation off
# Twitter without resolving anything. High deflection => poor training signal.
#
# The first version of this pattern only looked for "DM". That scored
# AmazonHelp at 1.3% deflection and ranked it 4th overall -- which is wrong.
# Reading actual Amazon replies showed the same behaviour expressed without
# the word "DM": "kindly drop in your details through the link provided",
# "reach out to us here: <url>", "kindly report this to our support team".
# Lexical deflection detectors flatter brands with a large phrasebook, so the
# pattern below targets the *act* of handing off rather than one phrasing.
# See DECISIONS.md D-02.
_DEFLECT = re.compile(
    r"("
    r"\bd\.?m\b|\bdm us\b|\bdm me\b|direct message|private message|"
    r"\bpm us\b|via dm|in a dm|to a dm|"
    # hand-off to another channel
    r"call us|give us a call|contact (our|the|us)|get in touch with (our|us)|"
    r"email us|drop us an email|"
    r"reach out to (us|our)|report (this|it) to (our|the)|"
    # "give us your details" -- the classic non-answer
    r"(drop|share|send|provide|dm) (in |us )?(your|the) (details|contact|info|"
    r"order (id|number)|email|phone)|"
    r"fill (out|in) (this|the) form|"
    # bare link hand-off
    r"(through|via|using|click|use) (the |this )?link|"
    r"link (provided|below|above|shared)|"
    r"our team will (contact|reach|get)"
    r")",
    re.I,
)

# Replies that are pure acknowledgement carry no resolution either, but they
# are a different failure mode from deflection and worth measuring separately.
_ACK_ONLY = re.compile(
    r"^(hi|hello|hey)?[\s,.!]*("
    r"(so |really |very )?sorry (to hear|about|for)|"
    r"thanks for (reaching out|getting in touch|letting us know|the feedback)|"
    r"we('| a)re (sorry|sad|sorry to hear)|"
    r"apologies|oh no|that'?s (not good|odd|strange)"
    r")[^.!?]*[.!?]?\s*$",
    re.I,
)

# Rough language gate. TWCS is multilingual (AmazonHelp carries a lot of
# Japanese); mixing scripts into one intent taxonomy is a silent quality
# killer, so we measure it per brand and filter it per episode.
_LATIN = re.compile(r"[A-Za-z]")


def latin_ratio(s: str) -> float:
    if not isinstance(s, str) or not s:
        return 0.0
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(_LATIN.match(c)) for c in letters) / len(letters)

# Customer expressing that the issue is handled. A weak but genuinely useful
# outcome signal; we never treat it as a label, only as a sampling stratum.
_THANKS = re.compile(
    r"\b(thank you|thanks|thankyou|thx|ty|cheers|appreciate it|"
    r"that worked|it works now|sorted|fixed now|all good|resolved)\b",
    re.I,
)


def clean_text(s: str, *, strip_mentions: bool = True) -> str:
    """Normalise a tweet for modelling.

    Mentions are stripped because every customer tweet starts with the brand
    handle and every brand reply starts with the (already anonymised) customer
    id; leaving them in gives a classifier a free, useless feature. URLs become
    a token because the *presence* of a link matters but t.co hashes do not.
    """
    if not isinstance(s, str):
        return ""
    s = _URL.sub(" <url> ", s)
    if strip_mentions:
        s = _MENTION.sub(" ", s)
    s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return _WS.sub(" ", s).strip()


def load_raw(columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "in_response_to_tweet_id",
    ]
    df = pd.read_csv(
        TWCS_CSV,
        usecols=cols,
        dtype={
            "tweet_id": "int64",
            "author_id": "string",
            "text": "string",
            "in_response_to_tweet_id": "float64",
        },
    )
    df["inbound"] = df["inbound"].astype(str).str.lower().eq("true")
    return df


def brand_scorecard(df: pd.DataFrame, min_pairs: int = 2000) -> pd.DataFrame:
    """Score brands on how learnable their support behaviour is.

    `usable_score` trades volume off against *substance*. A brand needs enough
    pairs to build an index from, but the thing we actually care about is
    whether its replies contain resolutions. We weight deflection hardest
    because it is the failure mode that makes the whole exercise vacuous: an
    agent trained on a brand that always says "send us your order ID" learns
    to always say "send us your order ID", and every quality metric will look
    fine while the product is worthless.
    """
    parent = df.set_index("tweet_id")[["inbound", "text", "author_id"]]

    out = df[~df["inbound"]].copy()
    out = out[out["in_response_to_tweet_id"].notna()]
    out["parent_id"] = out["in_response_to_tweet_id"].astype("int64")

    joined = out.join(
        parent.rename(
            columns={"inbound": "p_inbound", "text": "p_text", "author_id": "p_author"}
        ),
        on="parent_id",
        how="inner",
    )
    # Only pairs where the parent really is a customer message.
    pairs = joined[joined["p_inbound"].fillna(False)].copy()

    # Did the customer come back after the brand replied? Cheap multi-turn proxy.
    replied_to = set(df.loc[df["inbound"], "in_response_to_tweet_id"].dropna().astype("int64"))

    pairs["reply_clean"] = pairs["text"].map(lambda s: clean_text(s))
    pairs["is_deflect"] = pairs["reply_clean"].str.contains(_DEFLECT, na=False)
    pairs["is_ack_only"] = pairs["reply_clean"].str.contains(_ACK_ONLY, na=False)
    pairs["reply_chars"] = pairs["reply_clean"].str.len()
    pairs["got_followup"] = pairs["tweet_id"].isin(replied_to)
    pairs["non_latin"] = pairs["p_text"].map(lambda s: latin_ratio(s) < 0.6)
    # "Substantive" = the brand said something that is neither a hand-off nor
    # a bare apology, and said enough of it to carry information.
    pairs["substantive"] = (
        ~pairs["is_deflect"] & ~pairs["is_ack_only"] & (pairs["reply_chars"] >= 60)
    )

    # Thanks from the customer anywhere in that customer's later messages.
    thanks_ids = set(
        df.loc[df["inbound"] & df["text"].str.contains(_THANKS, na=False), "in_response_to_tweet_id"]
        .dropna()
        .astype("int64")
    )
    pairs["got_thanks"] = pairs["tweet_id"].isin(thanks_ids)

    g = pairs.groupby("author_id", observed=True)
    stats = pd.DataFrame(
        {
            "first_reply_pairs": g.size(),
            "deflect_rate": g["is_deflect"].mean(),
            "ack_only_rate": g["is_ack_only"].mean(),
            "substantive_rate": g["substantive"].mean(),
            "non_latin_rate": g["non_latin"].mean(),
            "median_reply_chars": g["reply_chars"].median(),
            "thanks_rate": g["got_thanks"].mean(),
            "multi_turn_rate": g["got_followup"].mean(),
            "distinct_customers": g["p_author"].nunique(),
        }
    )
    stats = stats[stats["first_reply_pairs"] >= min_pairs].copy()

    # Volume is normalised on a log scale and capped: past ~20k pairs more
    # data stops mattering for retrieval quality, and we would rather have a
    # cleaner corpus than a bigger one.
    vol = np.log10(stats["first_reply_pairs"]).clip(upper=np.log10(40000))
    vol = (vol - vol.min()) / max(vol.max() - vol.min(), 1e-9)

    stats["usable_score"] = (
        0.45 * stats["substantive_rate"]
        + 0.15 * vol
        + 0.15 * (1 - stats["non_latin_rate"])
        + 0.10 * stats["thanks_rate"].clip(upper=0.15) / 0.15
        + 0.15 * stats["multi_turn_rate"].clip(upper=0.6) / 0.6
    )
    stats = stats.sort_values("usable_score", ascending=False)
    stats.index.name = "brand"
    return stats.reset_index()


def build_episodes(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """Episodes for one brand: opening customer message + brand's first reply.

    We keep only threads the customer *started*. Brand-initiated threads are a
    different product (proactive outreach) and mixing them in would blur what
    the agent is being asked to do.
    """
    brand_ids = df["author_id"] == brand
    parent = df.set_index("tweet_id")[["inbound", "text", "author_id", "created_at"]]

    out = df[brand_ids & df["in_response_to_tweet_id"].notna()].copy()
    out["parent_id"] = out["in_response_to_tweet_id"].astype("int64")
    j = out.join(
        parent.rename(
            columns={
                "inbound": "p_inbound",
                "text": "p_text",
                "author_id": "p_author",
                "created_at": "p_created",
            }
        ),
        on="parent_id",
        how="inner",
    )
    j = j[j["p_inbound"].fillna(False)].copy()

    # Opening message = customer tweet with no parent of its own.
    parent_of_parent = df.set_index("tweet_id")["in_response_to_tweet_id"]
    j["p_has_parent"] = j["parent_id"].map(parent_of_parent).notna()
    j = j[~j["p_has_parent"]].copy()

    # Later customer turns in the same thread, for the escalation signal.
    followups = (
        df[df["inbound"] & df["in_response_to_tweet_id"].notna()]
        .assign(parent_id=lambda d: d["in_response_to_tweet_id"].astype("int64"))
        .groupby("parent_id")["text"]
        .apply(list)
    )
    j["customer_followups"] = j["tweet_id"].map(followups).apply(
        lambda v: v if isinstance(v, list) else []
    )

    ep = pd.DataFrame(
        {
            "episode_id": j["parent_id"].astype("int64"),
            "brand": brand,
            "customer_id": j["p_author"],
            "created_at": pd.to_datetime(
                j["p_created"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True
            ),
            "customer_msg_raw": j["p_text"],
            "customer_msg": j["p_text"].map(clean_text),
            "brand_reply_raw": j["text"],
            "brand_reply": j["text"].map(clean_text),
            "customer_followups": j["customer_followups"],
        }
    )
    ep = ep.dropna(subset=["created_at"])
    ep = ep[ep["customer_msg"].str.len() >= 12]
    # Drop non-English traffic rather than letting it form its own junk
    # intent cluster. Recorded as a known coverage gap in the report.
    ep = ep[ep["customer_msg"].map(latin_ratio) >= 0.6]
    ep = ep.drop_duplicates(subset=["episode_id"]).sort_values("created_at")

    ep["n_followups"] = ep["customer_followups"].map(len)
    ep["reply_is_deflection"] = ep["brand_reply"].str.contains(_DEFLECT, na=False)
    ep["reply_is_ack_only"] = ep["brand_reply"].str.contains(_ACK_ONLY, na=False)
    ep["reply_substantive"] = (
        ~ep["reply_is_deflection"]
        & ~ep["reply_is_ack_only"]
        & (ep["brand_reply"].str.len() >= 60)
    )
    ep["customer_thanked"] = ep["customer_followups"].map(
        lambda msgs: any(_THANKS.search(m or "") for m in msgs)
    )
    return ep.reset_index(drop=True)


def chronological_split(
    ep: pd.DataFrame, train_frac: float = 0.7, dev_frac: float = 0.1
) -> dict[str, pd.DataFrame]:
    """Split by time, never at random. See module docstring / D-04."""
    ep = ep.sort_values("created_at").reset_index(drop=True)
    n = len(ep)
    # Compute each boundary from its own fraction rather than from a running
    # sum: `int(n * (0.7 + 0.1))` is 799 for n=1000 because 0.7+0.1 is
    # 0.7999... in binary floating point, which silently costs the dev split a
    # row and makes the reported split sizes not match the configuration.
    i_tr = int(round(n * train_frac))
    i_dev = i_tr + int(round(n * dev_frac))
    return {
        "train": ep.iloc[:i_tr].copy(),
        "dev": ep.iloc[i_tr:i_dev].copy(),
        "test": ep.iloc[i_dev:].copy(),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Prepare TWCS episodes for one brand.")
    ap.add_argument("--brand", default=None, help="Brand handle; omit to only print the scorecard.")
    ap.add_argument("--scorecard", action="store_true")
    args = ap.parse_args()

    print("loading raw tweets ...", flush=True)
    df = load_raw()
    print(f"  {len(df):,} tweets, {df['author_id'].nunique():,} authors")

    if args.scorecard or not args.brand:
        sc = brand_scorecard(df)
        sc.to_csv(INTERIM / "brand_scorecard.csv", index=False)
        cols = [
            "brand",
            "first_reply_pairs",
            "deflect_rate",
            "median_reply_chars",
            "thanks_rate",
            "multi_turn_rate",
            "usable_score",
        ]
        with pd.option_context("display.width", 160, "display.max_columns", 20):
            print(sc[cols].head(25).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\nwrote {INTERIM / 'brand_scorecard.csv'}")

    if args.brand:
        ep = build_episodes(df, args.brand)
        out = INTERIM / f"episodes_{args.brand}.parquet"
        ep.to_parquet(out, index=False)
        splits = chronological_split(ep)
        meta = {
            "brand": args.brand,
            "episodes": len(ep),
            "date_min": str(ep["created_at"].min()),
            "date_max": str(ep["created_at"].max()),
            "deflection_rate": float(ep["reply_is_deflection"].mean()),
            "thanks_rate": float(ep["customer_thanked"].mean()),
            "splits": {k: len(v) for k, v in splits.items()},
            "split_boundaries": {
                k: [str(v["created_at"].min()), str(v["created_at"].max())]
                for k, v in splits.items()
                if len(v)
            },
        }
        for k, v in splits.items():
            v.to_parquet(INTERIM / f"episodes_{args.brand}_{k}.parquet", index=False)
        (INTERIM / f"episodes_{args.brand}_meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
