from click.testing import CliRunner

from dsa_sentiment.scripts.predict import main


def test_predict():
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0
