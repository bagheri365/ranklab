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
from ranklab.data.regime_audit import build_regime_audit
from ranklab.data.support_audit import build_support_audit
from ranklab.data.target_audit import build_target_audit
from ranklab.data.training_audit import build_training_audit
from ranklab.data.training_contract_audit import build_training_contract_audit
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


@app.command("audit-regimes")
def audit_regimes(
    data_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    output: Path | None = typer.Option(None, help="Optional JSON output path."),
    chunksize: int = typer.Option(250_000, min=1, help="CSV rows per processing chunk."),
) -> None:
    """Describe logging regimes, primary target fields, and evaluation-log overlap."""
    payload = build_regime_audit(data_dir, chunksize=chunksize)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    typer.echo(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


@app.command("audit-support")
def audit_support(
    data_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    output: Path | None = typer.Option(None, help="Optional JSON output path."),
    chunksize: int = typer.Option(250_000, min=1, help="CSV rows per processing chunk."),
) -> None:
    """Describe common user/video/scenario support across evaluation regimes."""
    payload = build_support_audit(data_dir, chunksize=chunksize)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    typer.echo(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


@app.command("audit-targets")
def audit_targets(
    data_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    output: Path | None = typer.Option(None, help="Optional JSON output path."),
    chunksize: int = typer.Option(250_000, min=1, help="CSV rows per processing chunk."),
) -> None:
    """Audit target semantics, duration dependence, and label-rule consistency."""
    payload = build_target_audit(data_dir, chunksize=chunksize)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    typer.echo(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


@app.command("audit-training")
def audit_training(
    data_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    output: Path | None = typer.Option(None, help="Optional JSON output path."),
    validation_days: int = typer.Option(3, min=1, help="Trailing observed history days to audit as validation."),
) -> None:
    """Audit candidate training interactions and a leakage-safe temporal validation split."""
    payload = build_training_audit(data_dir, validation_days=validation_days)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    typer.echo(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


@app.command("audit-training-contract")
def audit_training_contract(
    data_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    output: Path | None = typer.Option(None, help="Optional JSON output path."),
) -> None:
    """Audit the proposed click-based training/validation contract before freezing it."""
    payload = build_training_contract_audit(data_dir)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    typer.echo(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


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
