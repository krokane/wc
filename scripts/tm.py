import pandas as pd
from tools.maps import POS_GROUP_MAP, TM_TO_CANONICAL, WC26_ABBR


def parse_mkt_value(value):
    value = str(value).strip("€")
    if "k" in value:
        return int(float(value.strip("k")) * 1_000)
    return int(float(value.strip("m")) * 1_000_000)


def top_n_value(df, position_group, n):
    top_n = (
        df[df["position_group"] == position_group]
        .groupby("team_abbr")["market_value"]
        .apply(lambda x: x.nlargest(n).sum())
    )
    return df["team_abbr"].map(top_n)


def pos_avg_age(df, position_group):
    avg = df[df["position_group"] == position_group].groupby("team_abbr")["age"].mean()
    return df["team_abbr"].map(avg)


def main():
    df = pd.read_csv("./data/tm_scraped_data/tm_squads.csv")

    TM_ABBR = {
        **WC26_ABBR,
        **{fs: WC26_ABBR[can] for fs, can in TM_TO_CANONICAL.items()},
    }

    df["team_abbr"] = df["team_name"].map(TM_ABBR)

    df = df.drop(
        columns=["team_name", "shirt_number", "player_url", "nationality", "kader_url"]
    )

    # clean up existing columns, add positional group
    df["market_value"] = df["market_value"].map(parse_mkt_value)
    df["age"] = df["age"].astype(int)
    df["position_group"] = df["position"].map(POS_GROUP_MAP)

    # squad value stats
    df["team_value"] = df.groupby("team_abbr")["market_value"].transform("sum")
    df["att_value"] = (
        df["market_value"]
        .where(df["position_group"] == "attack", 0)
        .groupby(df["team_abbr"])
        .transform("sum")
    )
    df["mid_value"] = (
        df["market_value"]
        .where(df["position_group"] == "midfield", 0)
        .groupby(df["team_abbr"])
        .transform("sum")
    )
    df["def_value"] = (
        df["market_value"]
        .where(df["position_group"] == "defense", 0)
        .groupby(df["team_abbr"])
        .transform("sum")
    )
    df["gk_value"] = (
        df["market_value"]
        .where(df["position_group"] == "goal", 0)
        .groupby(df["team_abbr"])
        .transform("sum")
    )

    # approx starting lineup value
    df["start_gk_value"] = top_n_value(df, "goal", 1)
    df["start_def_value"] = top_n_value(df, "defense", 4)
    df["start_mid_value"] = top_n_value(df, "midfield", 3)
    df["start_att_value"] = top_n_value(df, "attack", 3)

    # value dist
    df["best_player"] = df.groupby("team_abbr")["market_value"].transform("max")
    df["worst_player"] = df.groupby("team_abbr")["market_value"].transform("min")
    df["value_std"] = df.groupby("team_abbr")["market_value"].transform("std")

    # squad age
    df["average_age"] = df.groupby("team_abbr")["age"].transform("mean")
    df["att_avg_age"] = pos_avg_age(df, "attack")
    df["mid_avg_age"] = pos_avg_age(df, "midfield")
    df["def_avg_age"] = pos_avg_age(df, "defense")
    df["gk_avg_age"] = pos_avg_age(df, "goal")
    df["u23_count"] = df.groupby("team_abbr")["age"].transform(
        lambda x: (x <= 23).sum()
    )
    df["vet_count"] = df.groupby("team_abbr")["age"].transform(
        lambda x: (x >= 30).sum()
    )

    # row per team
    drop_player_cols = [
        "player_name",
        "position",
        "age",
        "position_group",
        "market_value",
    ]
    df = (
        df.drop(columns=drop_player_cols)
        .drop_duplicates(subset="team_abbr")
        .reset_index(drop=True)
    )

    df.loc[df["team_abbr"] == "SC", "team_abbr"] = "SQ"

    df.to_csv("./data/tm_results.csv", index=False)


if __name__ == "__main__":
    main()
