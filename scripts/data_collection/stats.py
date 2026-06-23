import glob

import pandas as pd

from tools.columns import FS_UNUSED_COLS
from tools.maps import FS_TO_CANONICAL, WC26_ABBR, WC26_TEAMS

file_paths = glob.glob("./data/footy_stats_data/*.csv")


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


FS_ABBR = {
    **WC26_ABBR,
    **{fs: WC26_ABBR[can] for fs, can in FS_TO_CANONICAL.items()},
}


def clean_csvs(file_paths):
    dfs = []
    len1 = 0
    for path in file_paths:
        df = pd.read_csv(path)
        len1 += len(df)

        dt = pd.to_datetime(df["timestamp"], unit="s")
        date_str = dt.dt.strftime("%y_%m_%d")

        df["game_id"] = (
            date_str
            + "_"
            + df.apply(
                lambda x: sort_teams(
                    FS_ABBR.get(x["home_team_name"], x["home_team_name"]),
                    FS_ABBR.get(x["away_team_name"], x["away_team_name"]),
                ),
                axis=1,
            )
        )
        df["home_team"] = df["home_team_name"].map(lambda x: FS_ABBR.get(x, x))
        df["away_team"] = df["away_team_name"].map(lambda x: FS_ABBR.get(x, x))

        df = df[df["game_id"].str[-4:] != "drop"]
        df = df.drop(columns=FS_UNUSED_COLS)
        df = df.rename(
            columns={"team_a_xg": "home_team_xg", "team_b_xg": "away_team_xg"}
        )

        df = df[
            [
                "game_id",
                "home_team",
                "home_team_possession",
                "home_team_shots",
                "home_team_shots_on_target",
                "home_team_xg",
                "home_team_corner_count",
                "home_team_yellow_cards",
                "home_team_red_cards",
                "home_team_fouls",
                "away_team",
                "away_team_possession",
                "away_team_shots",
                "away_team_shots_on_target",
                "away_team_xg",
                "away_team_corner_count",
                "away_team_yellow_cards",
                "away_team_red_cards",
                "away_team_fouls",
            ]
        ]

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    df = clean_csvs(file_paths)
    df.to_csv("./data/fs_results.csv", index=False)
