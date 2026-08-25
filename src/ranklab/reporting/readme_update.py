"""Deterministic M2.3 README status updater.

This avoids fragile line-context patches by replacing the README Status section
between stable Markdown headings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable


STATUS_BLOCK = """## Status

M0 and M1 are complete and frozen. M2 is the publication/reporting layer.

> **Winner identity is stable, but comparative margins are not.**

The frozen primary analysis finds Popularity as the decisive winner in all four
logging-regime × behavioral-target cells. BPR and LightGCN are not decisively
separated in any primary cell. Popularity's measured advantage is substantially
larger under randomized exposure than under standard logging, and the target
definition also changes the measured margin.

These are offline evaluation contrasts under different logged exposure
processes. They are **not causal treatment-effect estimates**.

### Primary macro NDCG@10

| Logging regime | Target | Popularity | BPR | LightGCN |
| --- | --- | ---: | ---: | ---: |
| standard | `is_click` | **0.716524** | 0.676985 | 0.676862 |
| standard | `long_view` | **0.648404** | 0.601746 | 0.601590 |
| randomized | `is_click` | **0.432459** | 0.343522 | 0.343527 |
| randomized | `long_view` | **0.370164** | 0.265048 | 0.265314 |

Support robustness preserves the same qualitative pattern under both the
pre-specified shared-tabs sensitivity and the descriptive `tab=1` sensitivity.
`tab=1` is not assigned a semantic UI meaning because the public KuaiRand
documentation does not map every numeric tab ID to a semantic label.

See:

- `research/reporting/M2.1_REPORTING_CONTRACT.md`
- `research/reporting/M2.2_PUBLICATION_TABLES_FIGURES.md`
- `research/reporting/REPRODUCIBILITY.md`

"""


def rewrite_readme(text: str) -> str:
    pattern = re.compile(
        r"(?ms)^## Status\s*$.*?(?=^## Primary study\s*$)"
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one README Status→Primary study block, found {len(matches)}"
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
    parser = argparse.ArgumentParser(description="Update README to frozen M2 status.")
    parser.add_argument("--readme", default="README.md")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    update_path(args.readme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
