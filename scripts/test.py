import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Compare the two trained goal models against the 2026 World Cup matches we
# already have results for (year == 26, world_cup == 1), with an Elo baseline.
#
#   eval   -> models/xgb_goals.txt          (trained on year < 22)
#   deploy -> models/xgb_goals_deploy.txt    (trained on ALL non-WC2026 data)
#   elo    -> logistic on elo_diff, fit on all non-WC2026 matches
#
# NEITHER xgb model trained on the WC2026 games, so this is a clean holdout for
# both -- the eval-vs-deploy gap shows whether the extra training data (2022-25
# + 2026 friendlies) that deploy sees actually helps on real WC matches.
#
# For each xgb model we predict each team's expected goals, turn the two
# team/opp predictions into a 1X2 (win/draw/loss) distribution via independent
# Poisson, and score it with RPS against the actual result. Elo predicts the
# 1X2 distribution directly from the pre-match Elo gap.
# ---------------------------------------------------------------------------

FEATURES = "./data/features.csv"

DROP_COLS = [
    "id",
    "game_id",
    "team",
    "opponent",
    "goals_scored",
    "goals_conceded",
    "outcome",
    "year",
]

XGB_MODELS = [
    ("eval", "./models/xgb_goals.txt"),
    ("deploy", "./models/xgb_goals_deploy.txt"),
]

ALL_NAMES = [name for name, _ in XGB_MODELS] + ["elo"]
XGB_NAMES = [name for name, _ in XGB_MODELS]
OUTCOME_LABEL = {1: "W", 0: "D", -1: "L"}

OUT_COLS = [
    "model",
    "game_id",
    "team",
    "opponent",
    "outcome",
    "goals_scored",
    "pred_goals",
    "opp_pred_goals",
    "p_win",
    "p_draw",
    "p_loss",
    "pred_outcome",
    "rps",
]


def predict_match(team, opp, max_goals=8):
    """Independent-Poisson scoreline grid -> (p_win, p_draw, p_loss)."""
    grid = np.outer(
        poisson.pmf(range(max_goals + 1), team),
        poisson.pmf(range(max_goals + 1), opp),
    )
    p_win = np.tril(grid, -1).sum()
    p_draw = np.trace(grid)
    p_loss = np.triu(grid, 1).sum()
    return p_win, p_draw, p_loss


def rps(probs, outcome):
    cumulative = np.cumsum(probs)
    outcome_v = np.cumsum([outcome == 1, outcome == 0, outcome == -1])
    return np.mean((cumulative - outcome_v) ** 2)


def load_features():
    df = pd.read_csv(FEATURES)
    df["year"] = df["game_id"].str.split("_").str[0].astype(int)
    return df


def wc2026_mask(df):
    return (df["year"] == 26) & (df["world_cup"] == 1)


def attach_probs_and_rps(sub):
    """Add pred_outcome and rps from existing p_win/p_draw/p_loss columns."""
    sub["rps"] = sub.apply(
        lambda r: rps([r["p_win"], r["p_draw"], r["p_loss"]], r["outcome"]), axis=1
    )
    pred_idx = sub[["p_win", "p_draw", "p_loss"]].values.argmax(axis=1)
    sub["pred_outcome"] = np.array([1, 0, -1])[pred_idx]
    return sub


def score_xgb(name, model_path):
    df = load_features()
    sub = df[wc2026_mask(df)].copy().reset_index(drop=True)

    booster = xgb.Booster()
    booster.load_model(model_path)

    X = sub.drop(columns=DROP_COLS)[booster.feature_names]  # align to training
    sub["pred_goals"] = booster.predict(xgb.DMatrix(X))

    pmap = sub.set_index(["game_id", "team"])["pred_goals"]
    pmap = pmap[~pmap.index.duplicated(keep="first")]
    sub["opp_pred_goals"] = [
        pmap.get((g, o), np.nan) for g, o in zip(sub["game_id"], sub["opponent"])
    ]

    probs = sub.apply(
        lambda r: predict_match(r["pred_goals"], r["opp_pred_goals"])
        if pd.notna(r["opp_pred_goals"])
        else (np.nan, np.nan, np.nan),
        axis=1,
        result_type="expand",
    )
    sub[["p_win", "p_draw", "p_loss"]] = probs

    sub = attach_probs_and_rps(sub)
    sub["model"] = name
    return sub[OUT_COLS]


def score_elo(name="elo"):
    df = load_features()
    df["elo_diff"] = df["pre_match_elo"] - df["opponent_pm_elo"]

    wc = wc2026_mask(df)
    train = df[~wc]  # everything that isn't a WC2026 game
    sub = df[wc].copy().reset_index(drop=True)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(train[["elo_diff"]], train["outcome"])
    proba = clf.predict_proba(sub[["elo_diff"]])
    classes = list(clf.classes_)
    sub["p_win"] = proba[:, classes.index(1)]
    sub["p_draw"] = proba[:, classes.index(0)]
    sub["p_loss"] = proba[:, classes.index(-1)]

    sub["pred_goals"] = np.nan
    sub["opp_pred_goals"] = np.nan
    sub = attach_probs_and_rps(sub)
    sub["model"] = name
    return sub[OUT_COLS]


def main():
    results = pd.concat(
        [score_xgb(n, p) for n, p in XGB_MODELS] + [score_elo()],
        ignore_index=True,
    )

    n_games = results["game_id"].nunique()
    print(
        f"2026 World Cup matches scored: {n_games} games "
        f"({len(results) // len(ALL_NAMES)} team-rows)\n"
    )

    # ---- summary per model ------------------------------------------------
    summary = (
        results.groupby("model", sort=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "mean_rps": g["rps"].mean(),
                    "accuracy": (g["pred_outcome"] == g["outcome"]).mean(),
                    "mean_pred_goals": g["pred_goals"].mean(),
                }
            ),
            include_groups=False,
        )
        .round(4)
    )
    print("Per-model summary (lower RPS = better):")
    print(summary.to_string())

    # ---- xgb models vs Elo: RPS delta ------------------------------------
    elo_rps = summary.loc["elo", "mean_rps"]
    print("\nRPS improvement over Elo (positive = beats Elo):")
    for name in XGB_NAMES:
        print(f"  {name:7s}: {elo_rps - summary.loc[name, 'mean_rps']:+.4f}")

    # ---- per-match win-probability comparison -----------------------------
    pw = results.pivot_table(
        index=["game_id", "team", "opponent", "outcome"],
        columns="model",
        values="p_win",
    ).reset_index()
    pw = pw.sort_values("game_id").reset_index(drop=True)
    pw["result"] = pw["outcome"].map(OUTCOME_LABEL)
    show = pw[["game_id", "team", "opponent", "result"] + ALL_NAMES].copy()
    show[ALL_NAMES] = show[ALL_NAMES].round(3)
    print("\nP(team wins) by model, per match (result = actual, team perspective):")
    print(show.to_string(index=False))

    # ---- per-match RPS comparison -----------------------------------------
    pr = results.pivot_table(
        index=["game_id", "team", "outcome"], columns="model", values="rps"
    ).reset_index()
    pr = pr.sort_values("game_id").reset_index(drop=True)
    pr["result"] = pr["outcome"].map(OUTCOME_LABEL)
    showr = pr[["game_id", "team", "result"] + ALL_NAMES].copy()
    showr[ALL_NAMES] = showr[ALL_NAMES].round(4)
    print("\nRPS by model, per team-row:")
    print(showr.to_string(index=False))

    results.to_csv("./data/wc2026_model_comparison.csv", index=False)
    print("\nSaved row-level comparison -> ./data/wc2026_model_comparison.csv")


if __name__ == "__main__":
    main()
