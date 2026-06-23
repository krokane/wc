import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()
parser.add_argument(
    "--eval", action="store_true", help="train and report the eval model"
)
args = parser.parse_args()


def attach_match_probs(meta):
    opp_map = meta.set_index(["game_id", "team"])["prediction"]
    opp_map = opp_map[~opp_map.index.duplicated(keep="first")]
    meta["opp_prediction"] = meta.apply(
        lambda r: opp_map.get((r["game_id"], r["opponent"]), np.nan), axis=1
    )
    meta[["p_win", "p_draw", "p_loss"]] = meta.apply(match_probs, axis=1)
    meta["rps_score"] = meta.apply(
        lambda r: rps([r["p_win"], r["p_draw"], r["p_loss"]], r["outcome"]), axis=1
    )
    return meta


def comp_bucket(row):
    if row["world_cup"] == 1 and row["year"] == 18:
        return "WC 2018"
    if row["world_cup"] == 1 and row["year"] == 22:
        return "WC 2022"
    if row["continent"] == 1:
        return f"Continental {int(row['year'])}"
    return "other"


def elo_baseline_probs(train_df, eval_df):
    tr = train_df.copy()
    ev = eval_df.copy()
    tr["elo_diff"] = tr["pre_match_elo"] - tr["opponent_pm_elo"]
    ev["elo_diff"] = ev["pre_match_elo"] - ev["opponent_pm_elo"]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(tr[["elo_diff"]], tr["outcome"])
    proba = clf.predict_proba(ev[["elo_diff"]])
    classes = list(clf.classes_)
    return (
        proba[:, classes.index(1)],
        proba[:, classes.index(0)],
        proba[:, classes.index(-1)],
    )


def match_probs(row):
    if pd.isna(row["opp_prediction"]):
        return pd.Series({"p_win": np.nan, "p_draw": np.nan, "p_loss": np.nan})
    p_win, p_draw, p_loss, _ = predict_match(row["prediction"], row["opp_prediction"])
    return pd.Series({"p_win": p_win, "p_draw": p_draw, "p_loss": p_loss})


def predict_match(team, opp, max_goals=8):
    grid = np.outer(
        poisson.pmf(range(max_goals + 1), team),
        poisson.pmf(range(max_goals + 1), opp),
    )
    p_win = np.tril(grid, -1).sum()
    p_draw = np.trace(grid)
    p_loss = np.triu(grid, 1).sum()
    return p_win, p_draw, p_loss, grid


def rps(probs, outcome):
    cumulative = np.cumsum(probs)
    outcome_v = np.cumsum([outcome == 1, outcome == 0, outcome == -1])
    return np.mean((cumulative - outcome_v) ** 2)


df = pd.read_csv("./data/features.csv")
df["year"] = df["game_id"].str.split("_").str[0].astype(int)

test_mask = df["year"] == 26
wc_proxy = (df["world_cup"] == 1) & (df["year"] == 22)
cont_proxy = (df["continent"] == 1) & (df["year"] == 24)
val_mask = wc_proxy | cont_proxy
train_mask = (~val_mask) & (~test_mask) & (df["year"] < 22)

drop_cols = [
    "id",
    "game_id",
    "team",
    "opponent",
    "goals_scored",
    "goals_conceded",
    "outcome",
    "year",
]

X = df.drop(columns=drop_cols)
Y = df["goals_scored"]

X_train, Y_train = X[train_mask], Y[train_mask]
X_val, Y_val = X[val_mask], Y[val_mask]

train_xgb = xgb.DMatrix(X_train, label=Y_train)
val_xgb = xgb.DMatrix(X_val, label=Y_val)

params = {
    "objective": "count:poisson",
    "eval_metric": "poisson-nloglik",
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 50,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "verbosity": 0,
}

# always run early stopping to determine best round count for deploy model
eval_model = xgb.train(
    params,
    train_xgb,
    num_boost_round=2000,
    evals=[(val_xgb, "val")],
    callbacks=[xgb.callback.EarlyStopping(rounds=50)],
    verbose_eval=False,
)
best_stopping = eval_model.best_iteration + 1

if args.eval:
    eval_model.save_model("./models/xgb_goals.txt")

    val_meta = (
        df[val_mask][
            ["game_id", "team", "opponent", "outcome", "world_cup", "continent", "year"]
        ]
        .reset_index(drop=True)
        .copy()
    )
    val_meta["prediction"] = eval_model.predict(val_xgb)
    val_meta = attach_match_probs(val_meta)
    val_meta["bucket"] = val_meta.apply(comp_bucket, axis=1)

    usable = val_meta["rps_score"].notna()
    print(f"\nusable val rows (opponent present): {usable.sum()} / {len(val_meta)}")

    train_df = df[train_mask][["pre_match_elo", "opponent_pm_elo", "outcome"]]
    p_w, p_d, p_l = elo_baseline_probs(train_df, df[val_mask])
    val_meta["elo_rps"] = [
        rps([p_w[i], p_d[i], p_l[i]], o) for i, o in enumerate(val_meta["outcome"])
    ]

    model_rps = val_meta.loc[usable, "rps_score"].mean()
    elo_rps = val_meta.loc[usable, "elo_rps"].mean()
    print(f"\nModel RPS (WC proxy): {model_rps:.4f}")
    print(f"Elo baseline RPS:     {elo_rps:.4f}")
    print(f"Improvement:          {elo_rps - model_rps:+.4f} RPS")

    print("\nPer-bucket (usable rows only):")
    grp = (
        val_meta[usable]
        .groupby("bucket")
        .agg(
            n=("rps_score", "size"),
            model=("rps_score", "mean"),
            elo=("elo_rps", "mean"),
        )
    )
    grp["delta"] = grp["elo"] - grp["model"]
    print(grp.round(4).to_string())

# deploy model — train on all played games including completed WC2026 matches
wc26_unplayed = (df["world_cup"] == 1) & (df["year"] == 26) & df["goals_scored"].isna()
valid_deploy = ~wc26_unplayed & Y.notna()

deploy_model = xgb.train(
    params,
    xgb.DMatrix(X[valid_deploy], label=Y[valid_deploy]),
    num_boost_round=best_stopping,
)

deploy_model.save_model("./models/xgb_goals_deploy.txt")
print(f"Deploy model saved (rounds={best_stopping}, rows={valid_deploy.sum()})")

# # importance plot
# _, ax = plt.subplots(figsize=(10, 10))
# xgb.plot_importance(model, ax=ax, max_num_features=25, importance_type="gain")
# plt.tight_layout()
# plt.savefig("./data/plots/feature_importance.png", dpi=150)
# plt.show()
