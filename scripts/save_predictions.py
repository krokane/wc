"""
Snapshot today's upcoming predictions to data/prediction_snapshots/YYYY-MM-DD.csv.

Run this after daily retraining (before new results are scraped) to capture
the model's pre-game predictions. These snapshots are shown on the History
page alongside the retroactive predictions.
"""
import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fixtures import predict_upcoming

SNAPSHOTS_DIR = ROOT / "data" / "prediction_snapshots"


def main():
    SNAPSHOTS_DIR.mkdir(exist_ok=True)

    preds = predict_upcoming()
    if preds.empty:
        print("No upcoming fixtures to snapshot.")
        return

    date_str = datetime.date.today().isoformat()
    out_path = SNAPSHOTS_DIR / f"{date_str}.csv"
    preds.to_csv(out_path, index=False)
    print(f"Saved {len(preds)} predictions → {out_path}")


if __name__ == "__main__":
    main()
