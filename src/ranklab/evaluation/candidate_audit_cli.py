"""CLI for the M0.16 candidate-structure audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from ranklab.evaluation.candidate_audit import build_candidate_structure_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit RankLab evaluation candidate structure.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_candidate_structure_audit(args.data_dir)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
