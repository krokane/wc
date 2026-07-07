import datetime
import io

import numpy as np
import pandas as pd
import requests
from tools.columns import LAG_COLS
from tools.maps import WC26_TEAMS
from tools.net import pin_ipv4

pin_ipv4()

FIXTURES_URL = "https://www.eloratings.net/2026_World_Cup_fixtures.tsv"
FEATURES_PATH = "./data/features.csv"
N_LAGS = 5

_FIXTURE_COLS = [
    "year",
    "month",
    "day",
    "t1",
    "t2",
    "comp",
    "host",
    "t1_rank",
    "t2_rank",
    "t1_elo",
    "t2_elo",
] + [f"_x{i}" for i in range(12)]


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
    base = base.copy()
    for col in LAG_COLS:
        for lag in range(N_LAGS, 1, -1):
            base[f"{col}_lag{lag}"] = base[f"{col}_lag{lag - 1}"]
        base[f"{col}_lag1"] = base[col]
    return base


def get_upcoming_fixtures(features_path: str = FEATURES_PATH) -> pd.DataFrame:
    features = pd.read_csv(features_path)
    existing_ids = set(features["game_id"])
    latest_map = features.sort_values("game_id").groupby("team").last()

    raw = _fetch_raw()
    raw = raw[raw["comp"] == "WC"]
    raw = raw[raw["day"] != 0]
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


if __name__ == "__main__":
    feats = get_upcoming_fixtures()
    if feats.empty:
        print("No upcoming fixtures found (all games may already be in features.csv).")
    else:
        cols = ["game_id", "team", "opponent", "pre_match_elo", "opponent_pm_elo"]
        print(feats[cols].sort_values("game_id").to_string(index=False))
