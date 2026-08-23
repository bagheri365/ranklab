from __future__ import annotations

from pathlib import Path

import typer
import yaml

from ranklab.artifacts.hashing import sha256_file
from ranklab.benchmark.protocol import load_protocol

app = typer.Typer(no_args_is_help=True, help="RankLab experiment CLI")


@app.command()
def audit(
    benchmark: Path = typer.Option(Path("configs/benchmark.yaml"), exists=True),
) -> None:
    """Show the current M0 benchmark freeze status."""
    payload = yaml.safe_load(benchmark.read_text())
    typer.echo(f"benchmark={benchmark}")
    typer.echo(f"status={payload.get('status', 'UNKNOWN')}")
    typer.echo("M0 audit execution is not implemented yet.")


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
