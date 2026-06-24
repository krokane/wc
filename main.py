import os
import subprocess
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def run_script(name: str, *args) -> None:
    """Run scripts/<name> from the project root so ./data and ./models resolve."""
    click.echo(f"▶ running {name}")
    env = {**os.environ, "PYTHONPATH": str(SCRIPTS)}
    subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args], cwd=ROOT, env=env, check=True
    )


@click.group()
def cli():
    """World Cup Prediction Pipeline"""


@cli.command("elo")
def elo_data():
    """scrape eloratings.net for live results"""
    run_script("data_collection/elo.py")


@cli.command("stats")
def stats_data():
    """clean stats info from footystats.org -- pre-loaded CSVs"""
    run_script("data_collection/stats.py")


@cli.command("features")
def build_feats():
    """build feats from raw data"""
    run_script("data_collection/features.py")


@cli.command("train")
@click.option(
    "--eval",
    "run_eval",
    is_flag=True,
    default=False,
    help="also train and report the eval model",
)
def train(run_eval):
    """train the deploy model (add --eval to also run the eval model comparison)"""
    run_script("model.py", *(["--eval"] if run_eval else []))


@cli.command("upcoming")
def predict():
    """get upcoming fixtures from eloratings.net"""
    run_script("upcoming.py")


@cli.command("predict")
def predict_played():
    """predict outcomes for upcoming fixtures"""
    run_script("inference.py")


@cli.command("retro")
def retro_predict():
    """day-by-day retroactive predictions for past WC2026 games (clean model, no leakage)"""
    run_script("inference_retro.py")


@cli.command("dash")
@click.option("--port", default=8050, show_default=True, help="Port to listen on")
@click.option("--debug", is_flag=True, default=False, help="Enable Dash debug mode")
def dash_app(port, debug):
    """launch the Dash prediction UI"""
    import subprocess

    subprocess.run(
        [sys.executable, str(ROOT / "dash_app.py")],
        cwd=ROOT,
        env={
            **__import__("os").environ,
            "DASH_PORT": str(port),
            "DASH_DEBUG": "1" if debug else "0",
        },
    )


@cli.command("pipeline")
def pipeline():
    """run entire pipeline"""
    for s in [
        "data_collection/elo.py",
        "data_collection/stats.py",
        "data_collection/features.py",
        "model.py",
        "upcoming.py",
        "inference.py",
        "inference_retro.py",
    ]:
        run_script(s)


if __name__ == "__main__":
    cli()
