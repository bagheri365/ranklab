"""Deterministic M2.3 README status updater.

This avoids fragile line-context patches by replacing the README Status section
between stable Markdown headings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable


STATUS_BLOCK = """## Result at a glance

> **Winner identity is stable, but comparative margins are not.**

RankLab asks whether the **same trained recommenders** lead to the same offline
model-selection decision under standard-policy versus randomized exposure.

### Primary macro NDCG@10

| Logging regime | Target | Popularity | BPR | LightGCN |
| --- | --- | ---: | ---: | ---: |
| standard | `is_click` | **0.717** | 0.677 | 0.677 |
| standard | `long_view` | **0.648** | 0.602 | 0.602 |
| randomized | `is_click` | **0.432** | 0.344 | 0.344 |
| randomized | `long_view` | **0.370** | 0.265 | 0.265 |

### Three takeaways

- **Popularity is the decisive winner in all four primary cells.**
- **BPR and LightGCN are not decisively separated** in any primary cell.
- **The winner stays the same, but its measured advantage is much larger under
  randomized exposure.** Target definition also changes the measured margin.

These are **offline evaluation contrasts, not causal effects**.

### Go deeper

- **Reproduce the study:** `research/reporting/REPRODUCIBILITY.md`
- **Read Results + Discussion:** `research/reporting/M2.4_RESULTS_DISCUSSION.md`
- **Inspect the reporting contract:** `research/reporting/M2.1_REPORTING_CONTRACT.md`
- **Release:** `v1.0.0`

M0 and M1 are complete and frozen; M2 is the frozen publication/reporting
layer. The pre-specified shared-tabs sensitivity preserves the qualitative
primary result. The `tab=1` sensitivity also preserves the pattern but remains
descriptive only because the public KuaiRand documentation does not map every
numeric tab ID to a semantic UI label.

"""


def rewrite_readme(text: str) -> str:
    pattern = re.compile(
        r"(?ms)^## (?:Status|Result at a glance)\s*$.*?(?=^## Primary study\s*$)"
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one README status/results→Primary study block, found {len(matches)}"
        )

    updated = pattern.sub(STATUS_BLOCK, text, count=1)

    updated = updated.replace(
        "The primary protocol is not considered frozen until "
        "`research/protocol_frozen_m0.yaml` is complete and its SHA-256 is "
        "computed externally. Every valid M1 manifest must reference that hash.",
        "The final M0 protocol is frozen. Historical milestone notes below are "
        "retained as an audit trail; current status is summarized above and in "
        "`research/reporting/REPRODUCIBILITY.md`.",
    )
    updated = updated.replace(
        "The overall M0 protocol remains UNFROZEN until evaluation, support, "
        "and uncertainty decisions are complete.",
        "The full M0 protocol is now frozen; this sentence is retained as "
        "historical milestone context.",
    )
    return updated


def update_path(path: str | Path) -> None:
    path = Path(path)
    original = path.read_text(encoding="utf-8")
    updated = rewrite_readme(original)
    path.write_text(updated, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update README to the fast-reader frozen results summary.")
    parser.add_argument("--readme", default="README.md")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    update_path(args.readme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
