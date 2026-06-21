import os

import pandas as pd
from tools.columns import (
    HIST_COLS,
    LAG_COLS,
    elo_cols,
    elo_ind,
    elo_t1,
    elo_t2,
    fs_cols,
    fs_ind,
    fs_t1,
    fs_t2,
)

elo_raw = pd.read_csv("./data/elo_results.csv")
fs_raw = pd.read_csv("./data/fs_results.csv")
tm_raw = pd.read_csv("./data/tm_results.csv")


def stack_teams(df, ind, t1_cols, t2_cols, out_cols):
    a = df[ind + t1_cols].rename(columns=dict(zip(t1_cols, out_cols)))
    b = df[ind + t2_cols].rename(columns=dict(zip(t2_cols, out_cols)))

    return pd.concat([a, b], ignore_index=True)


def elim_non_wc(row):
    game_id = row["game_id"]

    if len(game_id.split("_")) == 4:
        team = game_id.split("_")[3]
    elif len(game_id.split("_")) == 5:
        return "keep"
    else:
        print(f"HELP -- {game_id}")
        return "keep"

    return "drop" if team != row["team"] else "keep"


# elo adjustments
def elo_restructure():
    df = stack_teams(elo_raw, elo_ind, elo_t1, elo_t2, elo_cols)
    df = df.sort_values("game_id").reset_index(drop=True)
    df = df[df.apply(elim_non_wc, axis=1) != "drop"]
    df["id"] = df["game_id"].astype(str) + "_" + df["team"].astype(str)

    df = df[
        [
            "id",
            "game_id",
            "team",
            "opponent",
            "home_away_neutral",
            "pre_match_elo",
            "opponent_pm_elo",
            "goals_scored",
            "goals_conceded",
            "outcome",
            "world_cup",
            "continent",
            "wc_quali",
            "cont_quali",
            "nat_league",
            "friendly",
            "other_comp",
        ]
    ]

    return df


# footy stats adjustments
def fs_restructure():
    df = stack_teams(fs_raw, fs_ind, fs_t1, fs_t2, fs_cols)
    df = df.sort_values("game_id").reset_index(drop=True)
    df = df[df.apply(elim_non_wc, axis=1) != "drop"]

    df["id"] = df["game_id"].astype(str) + "_" + df["team"].astype(str)

    return df


def main():
    elo = elo_restructure()
    fs = fs_restructure()

    tm = tm_raw
    tm_opp = tm_raw.copy()

    tm_opp.columns = [
        c + "_opponent" if c != "team_abbr" else c for c in tm_opp.columns
    ]

    final = pd.merge(elo, fs, how="left", on=["id", "game_id", "team"])

    final["year"] = final["game_id"].str.split("_").str[0].astype(int)
    final1 = final[final["year"] < 24]
    final2 = final[final["year"] >= 24]

    # final2 = pd.merge(final2, tm, how="left", left_on="team", right_on="team_abbr")
    # final2 = pd.merge(
    #     final2, tm_opp, how="left", left_on="opponent", right_on="team_abbr"
    # )

    final = (
        pd.concat([final1, final2], ignore_index=True)
        .sort_values("id")
        .reset_index(drop=True)
    )

    final = final.drop(
        columns=[
            # "team_abbr_x",
            # "team_abbr_y",
            "year",
        ]
    )

    # apply manual xG overrides before hist computation (fills gaps for WC2026 games)
    manual_path = "./data/manual_xg.csv"
    if os.path.exists(manual_path):
        manual = pd.read_csv(manual_path)
        final = final.merge(
            manual[["game_id", "team", "xg", "xg_conceded"]],
            on=["game_id", "team"],
            how="left",
            suffixes=("", "_m"),
        )
        final["xg"] = final["xg"].fillna(final["xg_m"])
        final["xg_conceded"] = final["xg_conceded"].fillna(final["xg_conceded_m"])
        final = final.drop(columns=["xg_m", "xg_conceded_m"])

    # retreiving lag game results
    N_LAGS = 5

    final = final.sort_values(["team", "game_id"]).reset_index(drop=True)

    lag_frames = []
    for lag in range(1, N_LAGS + 1):
        shifted = final.groupby("team")[LAG_COLS].shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in LAG_COLS]
        lag_frames.append(shifted)

    hist_frames = []
    for col in HIST_COLS:
        s = final.groupby("team")[col].transform(
            lambda x: x.expanding().mean().shift(N_LAGS + 1)
        )
        hist_frames.append(s.rename(f"{col}_hist_avg"))

    final = pd.concat([final] + lag_frames + hist_frames, axis=1)
    final = final.dropna(subset=[f"goals_scored_lag{N_LAGS}"])

    final = final.drop(columns=fs_cols[1:])

    final.to_csv("./data/features.csv", index=False)


if __name__ == "__main__":
    main()
