from typer.testing import CliRunner

from synkro.cli import app


def test_generate_help_includes_pretty_flag():
    runner = CliRunner()
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0, result.output
    assert "--pretty" in result.output
