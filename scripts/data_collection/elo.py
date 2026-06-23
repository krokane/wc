import io

import pandas as pd
import requests

from tools.maps import COMP_MAP, WC26_TEAMS

COLS = [
    "year",
    "month",
    "day",
    "t1",
    "t2",
    "t1_score",
    "t2_score",
    "comp",
    "venue",
    "elo_delt",
    "t1_elo",
    "t2_elo",
    "t1_rank_delt",
    "t2_rank_delt",
    "t1_rank",
    "t2_rank",
    "year_2",
]


def get_elo_yr(yr):
    url = f"https://www.eloratings.net/{yr}_results.tsv"
    r = requests.get(url)

    return pd.read_csv(io.StringIO(r.text), sep="\t", names=COLS)


def sort_teams(t1, t2):
    if t1 in WC26_TEAMS and t2 in WC26_TEAMS:
        pair = sorted([t1, t2])
        return f"{pair[0]}_{pair[1]}"
    elif t1 in WC26_TEAMS:
        return f"{t1}"
    elif t2 in WC26_TEAMS:
        return f"{t2}"
    else:
        return "drop"


def get_venue_status(t1, t2, venue):
    if venue == t1:
        return 1
    elif venue == t2:
        return -1
    else:
        return 0


def find_outcomes(score_1, score_2):
    if score_1 > score_2:
        return pd.Series([1, -1])
    elif score_1 < score_2:
        return pd.Series([-1, 1])
    else:
        return pd.Series([0, 0])


def scrape_clean_elo():
    dfs = [get_elo_yr(y) for y in range(2003, 2027)]
    df = pd.concat(dfs)

    df["game_id"] = (
        df["year"].astype(str).str[-2:]
        + "_"
        + df["month"].astype(str).str.zfill(2)
        + "_"
        + df["day"].astype(str).str.zfill(2)
        + "_"
        + df.apply(lambda x: sort_teams(x["t1"], x["t2"]), axis=1)
    )

    df["t1"] = df["t1"].replace("SC", "SQ")
    df["t2"] = df["t2"].replace("SC", "SQ")

    df["world_cup"] = (df["comp"].map(COMP_MAP) == "wc").astype(int)
    df["continent"] = (df["comp"].map(COMP_MAP) == "cont").astype(int)
    df["wc_quali"] = (df["comp"].map(COMP_MAP) == "wc_qualifier").astype(int)
    df["cont_quali"] = (df["comp"].map(COMP_MAP) == "cont_quali").astype(int)
    df["nat_league"] = (df["comp"].map(COMP_MAP) == "nations_league").astype(int)
    df["friendly"] = (df["comp"].map(COMP_MAP) == "friendly").astype(int)

    any_known = df[
        ["world_cup", "continent", "wc_quali", "cont_quali", "nat_league", "friendly"]
    ].any(axis=1)
    df["other_comp"] = (~any_known).astype(int)

    df["pre_match_elo_t1"] = df["t1_elo"] - df["elo_delt"]
    df["pre_match_elo_t2"] = df["t2_elo"] + df["elo_delt"]

    df = df[df["game_id"].str[-4:] != "drop"]

    df[["t1_outcome", "t2_outcome"]] = df[["t1_score", "t2_score"]].apply(
        lambda x: find_outcomes(x["t1_score"], x["t2_score"]),
        axis=1,
    )

    df["t1_venue_status"] = df.apply(
        lambda x: get_venue_status(x["t1"], x["t2"], x["venue"]), axis=1
    )
    df["t2_venue_status"] = -df["t1_venue_status"]

    df = df.drop(
        columns=[
            "year",
            "month",
            "day",
            "comp",
            "venue",
            "elo_delt",
            "t1_elo",
            "t2_elo",
            "t1_rank_delt",
            "t2_rank_delt",
            "t1_rank",
            "t2_rank",
            "year_2",
        ]
    )

    df = df[
        [
            "game_id",
            "t1",
            "t1_venue_status",
            "pre_match_elo_t1",
            "t1_score",
            "t1_outcome",
            "t2",
            "t2_venue_status",
            "pre_match_elo_t2",
            "t2_score",
            "t2_outcome",
            "world_cup",
            "continent",
            "wc_quali",
            "cont_quali",
            "nat_league",
            "friendly",
            "other_comp",
        ]
    ]

    df.to_csv("./data/elo_results.csv", index=False)


if __name__ == "__main__":
    scrape_clean_elo()
