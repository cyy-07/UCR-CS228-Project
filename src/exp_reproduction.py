"""
src/exp_reproduction.py — Workstream 0: Reproduce Candanedo et al. (2017)

The original paper's setup (Energy and Buildings 140, 81-97):
  • single-step prediction (predict next 10-min Appliances value)
  • 75/25 chronological split (we use 70/30 for compatibility)
  • models: Linear Regression, SVR, Random Forest, Gradient Boosting Machine
  • metrics: R², RMSE, MAE on test set

Paper SOTA (test set, all features):
                R²      RMSE    MAE
   Linear     0.16     93      52
   SVM-rad    0.22     70      31
   RF         0.52     68      31
   GBM        0.57     66.65   29.6     ← reference target

Run
---
    python src/exp_reproduction.py
    nohup python -u src/exp_reproduction.py > results/log_reproduction.txt 2>&1 &
"""

import os, sys, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
from training import save_csv, print_table

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

# ──────────────────────────────────────────────────────────────
#  Data prep — single-step setup matching the original paper
# ──────────────────────────────────────────────────────────────

def load_paper_setup(csv_path: str = "data/energydata_complete.csv",
                     train_ratio: float = 0.75,
                     drop_cols=("date", "rv1", "rv2")):
    """
    Build features X and target y for SINGLE-STEP prediction:
      X_t = all sensor readings at time t
      y_t = Appliances at time t   (no lookback, just regression)

    Then add time-of-day features (NSM, weekday) like the paper does.
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Time features (paper used NSM = number of seconds from midnight)
    df["nsm"]      = df["date"].dt.hour * 3600 + df["date"].dt.minute * 60
    df["weekday"]  = df["date"].dt.weekday
    df["is_wknd"]  = (df["weekday"] >= 5).astype(int)

    drop = list(drop_cols)
    y = df["Appliances"].values
    X = df.drop(columns=["Appliances"] + drop).values

    n_tr = int(len(df) * train_ratio)
    return (X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:],
            df.drop(columns=["Appliances"] + drop).columns.tolist())


def fit_eval(model, X_tr, y_tr, X_te, y_te, name: str):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    pred_te = model.predict(X_te)
    pred_tr = model.predict(X_tr)
    rec = {
        "model":     name,
        "train_R2":  round(r2_score(y_tr, pred_tr), 3),
        "test_R2":   round(r2_score(y_te, pred_te), 3),
        "test_MAE":  round(mean_absolute_error(y_te, pred_te), 3),
        "test_RMSE": round(np.sqrt(mean_squared_error(y_te, pred_te)), 3),
        "fit_s":     round(time.time() - t0, 1),
    }
    print(f"  {name:<20} train_R2={rec['train_R2']:.3f}  "
          f"test_R2={rec['test_R2']:.3f}  "
          f"test_RMSE={rec['test_RMSE']:.2f}  "
          f"test_MAE={rec['test_MAE']:.2f}  "
          f"({rec['fit_s']}s)")
    return rec


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("WORKSTREAM 0 — Literature reproduction (Candanedo et al. 2017)")
    print("=" * 60)

    X_tr, y_tr, X_te, y_te, feat_names = load_paper_setup()
    print(f"\n  Train: {X_tr.shape}  Test: {X_te.shape}  "
          f"n_features={X_tr.shape[1]}")

    # standardize for SVM/Linear (tree models are scale-invariant)
    sx = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = sx.transform(X_tr), sx.transform(X_te)

    print(f"\n  Paper SOTA (test set, GBM): R²=0.57  RMSE≈66.65  MAE≈29.6")
    print(f"  ────────────────────────────────────────────────────────\n")

    rows = []
    rows.append(fit_eval(LinearRegression(),
                         X_tr_s, y_tr, X_te_s, y_te, "LinearRegression"))
    rows.append(fit_eval(SVR(C=10.0, gamma="scale"),
                         X_tr_s, y_tr, X_te_s, y_te, "SVR-rbf"))
    rows.append(fit_eval(
        RandomForestRegressor(n_estimators=200, max_depth=None,
                              n_jobs=-1, random_state=42),
        X_tr, y_tr, X_te, y_te, "RandomForest"))
    rows.append(fit_eval(
        GradientBoostingRegressor(n_estimators=300, max_depth=5,
                                  learning_rate=0.05, random_state=42),
        X_tr, y_tr, X_te, y_te, "GBM"))

    save_csv(rows, os.path.join(RESULTS, "reproduction.csv"))
    print_table(rows, ["model","train_R2","test_R2","test_RMSE","test_MAE","fit_s"])

    print("\n  Comparison vs paper:")
    paper = {"GBM": (0.57, 66.65, 29.6), "RandomForest": (0.52, 68.0, 31.0)}
    for r in rows:
        if r["model"] in paper:
            p_r2, p_rmse, p_mae = paper[r["model"]]
            print(f"    {r['model']:<14} "
                  f"R² {r['test_R2']:.2f} (paper {p_r2:.2f})  "
                  f"RMSE {r['test_RMSE']:.1f} (paper {p_rmse:.1f})  "
                  f"MAE {r['test_MAE']:.1f} (paper {p_mae:.1f})")
    print("\nDone.")


if __name__ == "__main__":
    main()
