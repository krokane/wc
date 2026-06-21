"""
Fetch upcoming WC2026 fixtures from eloratings.net and predict them
with the deploy model.

For each game not yet in features.csv, we build a feature row by:
  1. Taking the team's most recent row from features.csv (their current lag/hist state)
  2. Rolling all lags forward one game (current → lag1, lag1 → lag2, ...)
  3. Overriding Elo values with live values from the fixture feed
"""
import io
import datetime
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from scipy.stats import poisson

from tools.columns import LAG_COLS
from tools.maps import WC26_TEAMS

FIXTURES_URL = "https://www.eloratings.net/2026_World_Cup_fixtures.tsv"
FEATURES_PATH = "./data/features.csv"
MODEL_PATH = "./models/xgb_goals_deploy.txt"
N_LAGS = 5

_FIXTURE_COLS = [
    "year", "month", "day", "t1", "t2", "comp", "host",
    "t1_rank", "t2_rank", "t1_elo", "t2_elo",
] + [f"_x{i}" for i in range(12)]

DROP_COLS = [
    "id", "game_id", "team", "opponent",
    "goals_scored", "goals_conceded", "outcome", "year",
]


def _fetch_raw() -> pd.DataFrame:
    r = requests.get(FIXTURES_URL, timeout=15)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), sep="\t", header=None, names=_FIXTURE_COLS)


def _game_id(row) -> str:
    yy = str(row["year"])[-2:]
    mm = str(int(row["month"])).zfill(2)
    dd = str(int(row["day"])).zfill(2)
    pair = sorted([row["t1"], row["t2"]])
    return f"{yy}_{mm}_{dd}_{pair[0]}_{pair[1]}"


def _home_away(team: str, opponent: str, host: str) -> int:
    if host == team:
        return 1
    if host == opponent:
        return -1
    return 0


def _shift_lags(base: pd.Series) -> pd.Series:
    """Roll all lag columns forward one game.

    The team is about to play their next game. Their most recent game's stats
    (currently in the base columns) should become lag1; existing lag1 → lag2, etc.
    lag5 from the base row falls off — it becomes lag6, which the model doesn't use.
    """
    base = base.copy()
    for col in LAG_COLS:
        for lag in range(N_LAGS, 1, -1):
            base[f"{col}_lag{lag}"] = base[f"{col}_lag{lag - 1}"]
        base[f"{col}_lag1"] = base[col]
    return base


def get_upcoming_fixtures(features_path: str = FEATURES_PATH) -> pd.DataFrame:
    """
    Return a features-compatible DataFrame for upcoming WC2026 games
    not already present in features.csv (i.e., not yet played + scraped).
    """
    features = pd.read_csv(features_path)
    existing_ids = set(features["game_id"])

    latest_map = features.sort_values("game_id").groupby("team").last()

    raw = _fetch_raw()
    raw = raw[raw["comp"] == "WC"]
    raw = raw[raw["day"] != 0]  # drop knockout placeholders with no confirmed date
    today = datetime.date.today()
    raw["_date"] = pd.to_datetime(
        {"year": raw["year"], "month": raw["month"], "day": raw["day"]}
    ).dt.date
    raw = raw[raw["_date"] >= today]

    rows = []
    for _, fix in raw.iterrows():
        gid = _game_id(fix)
        if gid in existing_ids:
            continue

        t1, t2 = fix["t1"], fix["t2"]
        if t1 not in WC26_TEAMS or t2 not in WC26_TEAMS:
            continue

        for team, opp, elo, opp_elo in [
            (t1, t2, fix["t1_elo"], fix["t2_elo"]),
            (t2, t1, fix["t2_elo"], fix["t1_elo"]),
        ]:
            if team not in latest_map.index:
                continue

            base = _shift_lags(latest_map.loc[team])
            base["id"] = f"{gid}_{team}"
            base["game_id"] = gid
            base["team"] = team
            base["opponent"] = opp
            base["home_away_neutral"] = _home_away(team, opp, fix["host"])
            base["pre_match_elo"] = elo
            base["opponent_pm_elo"] = opp_elo
            base["goals_scored"] = np.nan
            base["goals_conceded"] = np.nan
            base["outcome"] = np.nan
            base["world_cup"] = 1
            base["continent"] = 0
            base["wc_quali"] = 0
            base["cont_quali"] = 0
            base["nat_league"] = 0
            base["friendly"] = 0
            base["other_comp"] = 0
            rows.append(base)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).reset_index(drop=True)


def predict_match(team_lambda, opp_lambda, max_goals=8):
    grid = np.outer(
        poisson.pmf(range(max_goals + 1), team_lambda),
        poisson.pmf(range(max_goals + 1), opp_lambda),
    )
    return np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum(), grid


def predict_upcoming(
    features_path: str = FEATURES_PATH,
    model_path: str = MODEL_PATH,
) -> pd.DataFrame:
    feats = get_upcoming_fixtures(features_path)
    if feats.empty:
        return pd.DataFrame()

    feats["year"] = feats["game_id"].str.split("_").str[0].astype(int)

    booster = xgb.Booster()
    booster.load_model(model_path)

    X = feats.drop(columns=DROP_COLS)[booster.feature_names]
    feats["exp_goals"] = booster.predict(xgb.DMatrix(X))

    pmap = feats.set_index(["game_id", "team"])["exp_goals"]
    pmap = pmap[~pmap.index.duplicated(keep="first")]

    out = []
    for _, r in feats.iterrows():
        opp = pmap.get((r["game_id"], r["opponent"]), np.nan)
        if pd.isna(opp):
            continue
        p_win, p_draw, p_loss, grid = predict_match(r["exp_goals"], opp)
        i, j = np.unravel_index(grid.argmax(), grid.shape)
        out.append({
            "game_id": r["game_id"],
            "team": r["team"],
            "opponent": r["opponent"],
            "exp_goals": round(r["exp_goals"], 2),
            "opp_exp_goals": round(opp, 2),
            "p_win": round(p_win, 3),
            "p_draw": round(p_draw, 3),
            "p_loss": round(p_loss, 3),
            "ml_score": f"{i}-{j}",
        })

    return pd.DataFrame(out)


if __name__ == "__main__":
    results = predict_upcoming()
    if results.empty:
        print("No upcoming fixtures found (all games may already be in features.csv).")
    else:
        print(results.sort_values("game_id").to_string(index=False))
