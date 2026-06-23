import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson
from upcoming import get_upcoming_fixtures

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

MODEL_PATH = "./models/xgb_goals_deploy.txt"


def predict_match(team, opp, max_goals=8):
    grid = np.outer(
        poisson.pmf(range(max_goals + 1), team), poisson.pmf(range(max_goals + 1), opp)
    )
    return np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum(), grid


def predict_fixtures(feats, model_path=MODEL_PATH):
    booster = xgb.Booster()
    booster.load_model(model_path)

    feats = feats.copy()
    feats["year"] = feats["game_id"].str.split("_").str[0].astype(int)
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
        out.append(
            {
                "game_id": r["game_id"],
                "team": r["team"],
                "opponent": r["opponent"],
                "exp_goals": round(r["exp_goals"], 2),
                "opp_exp_goals": round(opp, 2),
                "p_win": round(p_win, 3),
                "p_draw": round(p_draw, 3),
                "p_loss": round(p_loss, 3),
                "ml_score": f"{i}-{j}",
            }
        )
    return pd.DataFrame(out)


def predict_upcoming(features_path="./data/features.csv", model_path=MODEL_PATH):
    feats = get_upcoming_fixtures(features_path)
    if feats.empty:
        return pd.DataFrame()
    return predict_fixtures(feats, model_path)


if __name__ == "__main__":
    results = predict_upcoming()
    if results.empty:
        print("No upcoming fixtures found (all games may already be in features.csv).")
    else:
        print(results.sort_values("game_id").to_string(index=False))
