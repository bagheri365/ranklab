from typer.testing import CliRunner

from ranklab.cli import app

runner = CliRunner()


def test_evaluate_is_blocked_before_m0() -> None:
    result = runner.invoke(app, ["evaluate"])
    assert result.exit_code == 2
    assert "complete and freeze M0" in result.stdout
