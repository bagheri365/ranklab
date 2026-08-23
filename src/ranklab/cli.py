from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from ranklab.artifacts.hashing import sha256_file
from ranklab.benchmark.protocol import load_protocol
from ranklab.data.provenance import (
    KUAIRAND_PURE_EXPECTED_FILES,
    inventory_as_dicts,
)
from ranklab.data.validation import audit_log_header

app = typer.Typer(no_args_is_help=True, help="RankLab experiment CLI")


@app.command()
def audit(
    benchmark: Path = typer.Option(Path("configs/benchmark.yaml"), exists=True),
) -> None:
    """Show the current M0 benchmark freeze status."""
    payload = yaml.safe_load(benchmark.read_text())
    typer.echo(f"benchmark={benchmark}")
    typer.echo(f"status={payload.get('status', 'UNKNOWN')}")
    typer.echo("M0 benchmark decisions remain unfrozen until the protocol gate is complete.")


@app.command("audit-data")
def audit_data(
    data_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    output: Path | None = typer.Option(None, help="Optional JSON output path."),
) -> None:
    """Audit KuaiRand-Pure inventory, hashes, and documented log headers."""
    inventory = inventory_as_dicts(data_dir)
    log_names = [name for name in KUAIRAND_PURE_EXPECTED_FILES if name.startswith("log_")]
    header_audits = []
    failed = False

    for name in log_names:
        result = audit_log_header(data_dir / name)
        failed = failed or not result.ok
        header_audits.append(
            {
                "name": name,
                "columns": list(result.columns),
                "missing_required": list(result.missing_required),
                "ok": result.ok,
            }
        )

    payload = {
        "dataset": "KuaiRand-Pure",
        "status": "M0_PROVENANCE_AUDIT_ONLY",
        "data_dir": str(data_dir),
        "files": inventory,
        "log_headers": header_audits,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    typer.echo(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if failed:
        raise typer.Exit(code=1)


@app.command("freeze-protocol")
def freeze_protocol(
    protocol: Path = typer.Option(Path("research/protocol_frozen_m0.yaml"), exists=True),
) -> None:
    """Hash a fully frozen M0 protocol without mutating the hashed payload."""
    payload = load_protocol(protocol)
    if payload.get("status") != "FROZEN":
        raise typer.BadParameter("protocol status must be FROZEN before hashing")
    typer.echo(sha256_file(protocol))


@app.command()
def evaluate() -> None:
    """Primary M1 evaluation is blocked until the M0 protocol is frozen."""
    typer.echo("M1 evaluation is not implemented; complete and freeze M0 first.")
    raise typer.Exit(code=2)
