"""
Day-by-day retroactive predictions for past WC2026 games.

For each WC2026 match date D, trains a model on:
  - All pre-2026 historical data (years 03-25)
  - Any WC2026 games already played before date D (their results + xG lags)

Then predicts the games played on date D.

This mirrors what the model would have actually known on each match day,
including xG lag information from prior WC2026 games.

Saves to data/retro_predictions.csv with a pred_date column.
"""
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson

DROP_COLS = [
    "id", "game_id", "team", "opponent",
    "goals_scored", "goals_conceded", "outcome", "year", "game_date",
]


def _date_from_gid(gid):
    p = gid.split("_")
    return datetime.date(2000 + int(p[0]), int(p[1]), int(p[2]))


def predict_match(team, opp, max_goals=8):
    grid = np.outer(
        poisson.pmf(range(max_goals + 1), team),
        poisson.pmf(range(max_goals + 1), opp),
    )
    return np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum(), grid


def main():
    df = pd.read_csv("./data/features.csv")
    df["year"] = df["game_id"].str.split("_").str[0].astype(int)
    df["game_date"] = df["game_id"].apply(_date_from_gid)

    today = datetime.date.today()
    test_mask = df["year"] == 26

    # Only backfill games that have been played (goals_scored present) and are in the past
    played_mask = test_mask & df["goals_scored"].notna() & (df["game_date"] < today)

    if not played_mask.any():
        print("No played WC2026 games found in features.csv.")
        return

    wc26_dates = sorted(df[played_mask]["game_date"].unique())
    n_days = len(wc26_dates)
    print(f"Found {n_days} WC2026 match day(s) to backfill.")

    # Compute best_iteration once via early stopping (same logic as model.py)
    wc_proxy = (df["world_cup"] == 1) & (df["year"] == 22)
    cont_proxy = (df["continent"] == 1) & (df["year"] == 24)
    val_mask = wc_proxy | cont_proxy
    train_mask_eval = (~val_mask) & (~test_mask) & (df["year"] < 22)

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

    X_all = df.drop(columns=DROP_COLS)
    Y_all = df["goals_scored"]

    X_train_eval = X_all[train_mask_eval]
    Y_train_eval = Y_all[train_mask_eval]
    X_val = X_all[val_mask]
    Y_val = Y_all[val_mask]

    print("Computing best iteration via early stopping...")
    eval_model = xgb.train(
        params,
        xgb.DMatrix(X_train_eval, label=Y_train_eval),
        num_boost_round=2000,
        evals=[(xgb.DMatrix(X_val, label=Y_val), "val")],
        callbacks=[xgb.callback.EarlyStopping(rounds=50)],
        verbose_eval=False,
    )
    best_iter = eval_model.best_iteration + 1
    print(f"Best iteration: {best_iter}")

    all_out = []

    for match_date in wc26_dates:
        # Training data: all non-WC2026 games + WC2026 games played strictly before this date
        day_train_mask = (
            (~test_mask)
            | (test_mask & df["goals_scored"].notna() & (df["game_date"] < match_date))
        )

        X_day = X_all[day_train_mask]
        Y_day = Y_all[day_train_mask]
        valid = Y_day.notna()

        retro_model = xgb.train(
            params,
            xgb.DMatrix(X_day[valid], label=Y_day[valid]),
            num_boost_round=best_iter,
        )

        # Predict WC2026 games on this date that have been played
        games_today = df[
            test_mask & (df["game_date"] == match_date) & df["goals_scored"].notna()
        ].copy()

        X_today = games_today.drop(columns=DROP_COLS)[retro_model.feature_names]
        games_today["exp_goals"] = retro_model.predict(xgb.DMatrix(X_today))

        pmap = games_today.set_index(["game_id", "team"])["exp_goals"]
        pmap = pmap[~pmap.index.duplicated(keep="first")]

        n_games = 0
        for _, r in games_today.iterrows():
            opp = pmap.get((r["game_id"], r["opponent"]), np.nan)
            if pd.isna(opp):
                continue
            p_win, p_draw, p_loss, grid = predict_match(r["exp_goals"], opp)
            i, j = np.unravel_index(grid.argmax(), grid.shape)
            all_out.append({
                "game_id": r["game_id"],
                "team": r["team"],
                "opponent": r["opponent"],
                "exp_goals": round(r["exp_goals"], 2),
                "opp_exp_goals": round(opp, 2),
                "p_win": round(p_win, 3),
                "p_draw": round(p_draw, 3),
                "p_loss": round(p_loss, 3),
                "ml_score": f"{i}-{j}",
                "pred_date": match_date.isoformat(),
            })
            n_games += 1

        n_unique = n_games // 2
        print(f"  {match_date}  →  {n_unique} game(s) predicted")

    if not all_out:
        print("No predictions generated.")
        return

    result_df = pd.DataFrame(all_out)
    result_df.to_csv("./data/retro_predictions.csv", index=False)
    print(f"\nSaved {len(result_df)} rows → data/retro_predictions.csv")


if __name__ == "__main__":
    main()
