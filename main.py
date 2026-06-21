import subprocess
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def run_script(name: str) -> None:
    """Run scripts/<name> from the project root so ./data and ./models resolve."""
    click.echo(f"▶ running {name}")
    subprocess.run([sys.executable, str(SCRIPTS / name)], cwd=ROOT, check=True)


@click.group()
def cli():
    """World Cup Prediction Pipeline"""


@cli.command("elo-data")
def elo_data():
    """scrape eloratings.net for live results"""
    run_script("elo.py")


@cli.command("stats-data")
def stats_data():
    """clean stats info from footystats.org -- pre-loaded CSVs"""
    run_script("stats.py")


@cli.command("build-feats")
def build_feats():
    """build feats from raw data"""
    run_script("features.py")


@cli.command("train")
def train():
    """train eval and deploy models"""
    run_script("model.py")


@cli.command("eval-models")
def test():
    """run test script on eval and deploy models"""
    run_script("test.py")


@cli.command("predict")
def predict():
    """predict upcoming WC2026 fixtures (fetches live from eloratings.net)"""
    run_script("fixtures.py")


@cli.command("predict-played")
def predict_played():
    """run deploy model on already-played WC2026 games in features.csv"""
    run_script("inference.py")


@cli.command("retro-predict")
def retro_predict():
    """day-by-day retroactive predictions for past WC2026 games (clean model, no leakage)"""
    run_script("retro_predict.py")


@cli.command("save-predictions")
def save_predictions():
    """snapshot today's upcoming predictions → data/prediction_snapshots/"""
    run_script("save_predictions.py")


@cli.command("dash")
@click.option("--port", default=8050, show_default=True, help="Port to listen on")
@click.option("--debug", is_flag=True, default=False, help="Enable Dash debug mode")
def dash_app(port, debug):
    """launch the Dash prediction UI"""
    import subprocess
    subprocess.run(
        [sys.executable, str(ROOT / "dash_app.py")],
        cwd=ROOT,
        env={**__import__("os").environ, "DASH_PORT": str(port), "DASH_DEBUG": "1" if debug else "0"},
    )


@cli.command("pipeline")
def pipeline():
    """run entire pipeline"""
    for s in ["elo.py", "stats.py", "features.py", "model.py", "test.py"]:
        run_script(s)


@cli.command("model_pipeline")
def model_pipeline():
    """run pipeline sans raw data cleaning/scraping"""
    for s in ["features.py", "model.py", "test.py"]:
        run_script(s)


if __name__ == "__main__":
    cli()
