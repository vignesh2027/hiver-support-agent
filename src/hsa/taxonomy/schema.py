"""Load and validate the curated taxonomy.

The taxonomy is the contract between four things that must not drift apart:
the annotation guideline, the classifier's prompt, the router's policy table,
and the report's intent distribution. Keeping one validated loader means a
change to the taxonomy either propagates everywhere or fails a test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..config import ARTIFACTS

TAXONOMY_FINAL = ARTIFACTS / "taxonomy_final.json"
TAXONOMY_RAW = ARTIFACTS / "taxonomy.json"
CLUSTERS_NAMED = ARTIFACTS / "clusters_named.json"

VALID_DISPOSITIONS = ("auto", "assist", "escalate")


@dataclass(frozen=True)
class Intent:
    name: str
    definition: str
    includes: tuple[str, ...]
    excludes: tuple[str, ...]
    hard_case: str
    needs_live_data: bool
    default_disposition: str
    disposition_rationale: str
    source_clusters: tuple[str, ...]
    prior_share: float


@dataclass(frozen=True)
class Taxonomy:
    brand: str
    version: str
    intents: tuple[Intent, ...]
    curation_log: tuple[dict, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(i.name for i in self.intents)

    def get(self, name: str) -> Intent:
        for i in self.intents:
            if i.name == name:
                return i
        raise KeyError(f"unknown intent {name!r}; known: {self.names}")

    def is_valid(self, name: str) -> bool:
        return any(i.name == name for i in self.intents)

    @property
    def auto_eligible(self) -> tuple[str, ...]:
        return tuple(i.name for i in self.intents if i.default_disposition == "auto")

    @property
    def live_data_intents(self) -> tuple[str, ...]:
        return tuple(i.name for i in self.intents if i.needs_live_data)

    def guideline_block(self, *, compact: bool = False) -> str:
        """The taxonomy rendered for an LLM prompt.

        `compact` drops the boundary cases. Used when many messages share one
        prompt during batched classification, where the full guideline would
        dominate the context window.
        """
        parts = []
        for i in self.intents:
            block = [f"- {i.name}: {i.definition}"]
            if not compact:
                if i.includes:
                    block.append("    includes: " + " | ".join(i.includes[:4]))
                if i.excludes:
                    block.append("    excludes: " + " | ".join(i.excludes[:3]))
                if i.hard_case:
                    block.append(f"    boundary: {i.hard_case}")
            parts.append("\n".join(block))
        return "\n".join(parts)


def _validate(tax: Taxonomy) -> None:
    names = tax.names
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate intent names: {names}")
    if "other" not in names:
        raise ValueError("taxonomy must contain an 'other' catch-all")
    for i in tax.intents:
        if i.default_disposition not in VALID_DISPOSITIONS:
            raise ValueError(f"{i.name}: bad disposition {i.default_disposition!r}")
        if not i.definition.strip():
            raise ValueError(f"{i.name}: empty definition")
        if i.name != "other" and not i.hard_case.strip():
            raise ValueError(f"{i.name}: every intent needs a boundary case")
    total = sum(i.prior_share for i in tax.intents)
    if abs(total - 1.0) > 0.02:
        raise ValueError(f"prior_share must sum to 1.0, got {total:.4f}")


@lru_cache(maxsize=1)
def load(path: Path | None = None) -> Taxonomy:
    p = path or TAXONOMY_FINAL
    raw = json.loads(p.read_text())
    intents = tuple(
        Intent(
            name=i["name"],
            definition=i["definition"],
            includes=tuple(i.get("includes", [])),
            excludes=tuple(i.get("excludes", [])),
            hard_case=i.get("hard_case", ""),
            needs_live_data=bool(i.get("needs_live_data", False)),
            default_disposition=i.get("default_disposition", "escalate"),
            disposition_rationale=i.get("disposition_rationale", ""),
            source_clusters=tuple(str(c) for c in i.get("source_clusters", [])),
            prior_share=float(i.get("prior_share", 0.0)),
        )
        for i in raw["intents"]
    )
    tax = Taxonomy(
        brand=raw["brand"],
        version=raw["version"],
        intents=intents,
        curation_log=tuple(raw.get("curation_log", [])),
    )
    _validate(tax)
    return tax


def write_markdown(path: Path | None = None) -> Path:
    """Emit the annotation guideline that golden-set labelling was done against."""
    tax = load()
    out = path or (ARTIFACTS / "ANNOTATION_GUIDELINE.md")
    L = [
        f"# Annotation guideline — {tax.brand} intents (v{tax.version})",
        "",
        "This is the document every golden-set label was assigned against. It is",
        "generated from `artifacts/taxonomy_final.json`; edit the JSON, not this file.",
        "",
        "Labelling rules that apply to every intent:",
        "",
        "1. Label what the customer **wants**, not the topic they mention.",
        "2. If a message contains several asks, label the one with the highest stakes",
        "   (money > stranded > factual question > venting).",
        "3. Sarcasm does not change intent. Label the grievance underneath it.",
        "4. Do not use information from later turns in the thread. The agent sees only",
        "   the opening message, so the label must be assignable from it alone.",
        "5. If two intents both genuinely fit and rule 2 does not separate them, mark the",
        "   item ambiguous rather than guessing. Ambiguous items are reported, not dropped.",
        "",
        "| intent | prior share | needs live data | default disposition |",
        "| --- | --- | --- | --- |",
    ]
    for i in tax.intents:
        L.append(
            f"| `{i.name}` | {i.prior_share:.1%} | {'yes' if i.needs_live_data else 'no'} "
            f"| `{i.default_disposition}` |"
        )
    L.append("")
    for i in tax.intents:
        L += [
            f"## `{i.name}`",
            "",
            i.definition,
            "",
            "**Includes**",
            *[f"- {x}" for x in i.includes],
            "",
            "**Excludes**",
            *[f"- {x}" for x in i.excludes],
            "",
            f"**Boundary case.** {i.hard_case}",
            "",
            f"**Disposition `{i.default_disposition}`.** {i.disposition_rationale}",
            "",
        ]
    L += ["## Curation log", ""]
    for c in tax.curation_log:
        L += [
            f"### {c['id']} — {c['change']}",
            "",
            c["detail"],
            "",
            f"*Why it matters.* {c['why_it_matters']}",
            "",
        ]
    out.write_text("\n".join(L))
    return out


if __name__ == "__main__":
    t = load()
    print(f"{t.brand} v{t.version}: {len(t.intents)} intents")
    for i in t.intents:
        print(
            f"  {i.name:<28} {i.prior_share:>6.1%}  live={str(i.needs_live_data):<5} "
            f"{i.default_disposition}"
        )
    print(f"\nauto-eligible: {t.auto_eligible}")
    print(f"live-data intents carry {sum(i.prior_share for i in t.intents if i.needs_live_data):.1%} of traffic")
    print(f"\nwrote {write_markdown()}")
