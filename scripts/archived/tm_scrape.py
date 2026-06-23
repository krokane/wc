import argparse
import os
import subprocess
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTFILE = "./data/tm_scraped_data/tm_squads.csv"


def run_R(teams, append: bool):
    r_script = os.path.join(SCRIPT_DIR, "tools/tm.R")
    cmd = ["Rscript", r_script]
    if teams:
        cmd.append(f"--teams={teams}")
    if append:
        cmd.append("--append")

    print(f"\nRunning: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("Rscript not found on PATH. Is R installed?")
    except subprocess.CalledProcessError as e:
        sys.exit(f"R script failed with exit code {e.returncode}")


def summarize():
    if not os.path.exists(OUTFILE):
        print(f"\nOutput file not found: {OUTFILE}")
        return

    df = pd.read_csv(OUTFILE)
    if df.empty:
        print("\nOutput file is empty.")
        return

    print(f"\n{'=' * 50}")
    print(f"  {OUTFILE}")
    print(f"{'=' * 50}")
    print(f"  Total players : {len(df)}")
    print(f"  Teams         : {df['team_name'].nunique()}")
    print(f"  Columns       : {', '.join(df.columns.tolist())}")

    counts = df.groupby("team_name").size().reset_index(name="players")
    print(f"\n  Players per team:")
    for _, row in counts.iterrows():
        print(f"    {row['team_name']:<30} {row['players']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfermarkt WC 2026 squad scraper")
    parser.add_argument(
        "--teams", metavar="CW,CD,CZ", help="Comma-separated team abbreviations"
    )
    parser.add_argument(
        "--append", action="store_true", help="Append to existing output"
    )
    args = parser.parse_args()

    run_R(teams=args.teams, append=args.append)
    summarize()
