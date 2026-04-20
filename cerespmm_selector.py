"""
CereSpMM — Automatic Algorithm Selection
=========================================
Decision tree model that selects the optimal SpMM algorithm
(BCSR / BCOO / BDIA) for a given sparse matrix on Cerebras CS-3.

This file accompanies the paper:
    "CereSpMM: Accelerating General Sparse Matrix-Matrix Multiplication
     on Cerebras CS-3"  (SC 2026)

Repository: https://github.com/AnonymousSC2026/CereSpMM

Workflow
--------
OFFLINE (run once, before deployment):
    1. Extract features from your matrices using extract_features.py
       → produces features.csv  (columns: file, row, col, ..., label)
    2. Train the decision tree:
          python cerespMM_selector.py --mode train \\
                 --data features.csv --save model.pkl
    3. Evaluate with Leave-One-Out CV (printed automatically).

ONLINE (at runtime, for each new matrix):
    4. Load the saved model and call predict():
          python cerespMM_selector.py --mode predict \\
                 --load model.pkl --mtx my_matrix.mtx

Requirements
------------
    pip install numpy scipy scikit-learn pandas matplotlib joblib
"""

import argparse
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

# The four features selected by feature-importance analysis.
# Full 12-feature set is available in features.csv; only these
# four are used at runtime to minimise prediction overhead.
SELECTED_FEATURES = ["average", "CV", "dia_ratio", "dia_pad"]

# All 12 features produced by extract_features.py
ALL_FEATURES = [
    "row", "col", "nnz", "nnz_ratio",
    "max", "min", "average", "VAR", "CV",
    "dia_num", "dia_ratio", "dia_pad",
]

ALGORITHMS = ["BCSR", "BCOO", "BDIA"]

# Decision tree hyper-parameters (fixed for reproducibility)
TREE_PARAMS = dict(
    max_depth=3,
    min_samples_leaf=2,
    criterion="gini",
    random_state=42,
)


# ──────────────────────────────────────────────────────────────
# 1. Feature extraction  (used in --mode predict for new .mtx)
# ──────────────────────────────────────────────────────────────

def extract_features_from_mtx(mtx_path: str) -> dict:
    """
    Extract the four runtime features from a single .mtx file.
    Returns a dict with keys matching SELECTED_FEATURES.

    Parameters
    ----------
    mtx_path : str
        Path to a Matrix Market (.mtx) file.

    Returns
    -------
    dict with keys: average, CV, dia_ratio, dia_pad
    """
    try:
        from scipy.io import mmread
        from scipy.sparse import csr_matrix as scipy_csr
    except ImportError:
        sys.exit("[ERROR] scipy is required for .mtx loading: pip install scipy")

    raw = mmread(mtx_path)
    A = scipy_csr(raw)
    n_rows, n_cols = A.shape
    nnz = A.nnz

    if nnz == 0:
        sys.exit(f"[ERROR] Matrix {mtx_path} has no nonzero entries.")

    # Per-row nonzero counts
    row_nnz = np.diff(A.indptr).astype(np.float64)
    avg_nnz = row_nnz.mean()
    cv_nnz  = row_nnz.std() / avg_nnz if avg_nnz > 0 else 0.0

    # Diagonal structure
    ri, ci     = A.nonzero()
    diag_off   = ci.astype(np.int64) - ri.astype(np.int64)
    dia_num    = len(np.unique(diag_off))
    all_diags  = n_rows + n_cols - 1
    dia_ratio  = dia_num / all_diags
    dia_pad    = 1.0 - nnz / (dia_num * max(n_rows, n_cols)) if dia_num > 0 else 1.0

    return {
        "average":   round(avg_nnz, 4),
        "CV":        round(cv_nnz,  4),
        "dia_ratio": round(dia_ratio, 6),
        "dia_pad":   round(max(0.0, dia_pad), 6),
    }


# ──────────────────────────────────────────────────────────────
# 2. Load training data from features.csv
# ──────────────────────────────────────────────────────────────

def load_dataset(csv_path: str, features: list = SELECTED_FEATURES):
    """
    Load the labelled feature CSV produced by extract_features.py.

    The CSV must contain:
      - Columns listed in `features`
      - A column named 'label' with values in {BCSR, BCOO, BDIA}

    Rows with empty or NaN labels are dropped.

    Returns
    -------
    X : np.ndarray  shape (n_samples, n_features)
    y : np.ndarray  shape (n_samples,)  integer-encoded labels
    le : LabelEncoder  (fitted)
    df : pd.DataFrame  (full, for reporting)
    """
    if not os.path.isfile(csv_path):
        sys.exit(f"[ERROR] Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Drop rows with missing labels
    df = df[df["label"].notna() & (df["label"].astype(str).str.strip() != "")]
    df = df.reset_index(drop=True)

    missing = [f for f in features if f not in df.columns]
    if missing:
        sys.exit(f"[ERROR] Missing columns in CSV: {missing}")
    if "label" not in df.columns:
        sys.exit("[ERROR] 'label' column not found in CSV.")

    X  = df[features].values.astype(np.float64)
    le = LabelEncoder()
    y  = le.fit_transform(df["label"].astype(str).str.strip().values)

    return X, y, le, df


# ──────────────────────────────────────────────────────────────
# 3. Train
# ──────────────────────────────────────────────────────────────

def train(X, y, le):
    """
    Train a shallow decision tree and return the fitted model.
    """
    model = DecisionTreeClassifier(**TREE_PARAMS)
    model.fit(X, y)
    return model


# ──────────────────────────────────────────────────────────────
# 4. Evaluate (Leave-One-Out CV)
# ──────────────────────────────────────────────────────────────

def evaluate(X, y, le, features=SELECTED_FEATURES):
    """
    Run Leave-One-Out cross-validation and print a full report.
    LOO-CV is the standard protocol for small datasets (n < 30 per class).
    """
    model_loo = DecisionTreeClassifier(**TREE_PARAMS)
    loo       = LeaveOneOut()

    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in loo.split(X):
        m = DecisionTreeClassifier(**TREE_PARAMS)
        m.fit(X[train_idx], y[train_idx])
        y_true_all.append(y[test_idx[0]])
        y_pred_all.append(m.predict(X[test_idx])[0])

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    overall_acc = np.mean(y_true == y_pred)

    print("\n" + "=" * 56)
    print("  Leave-One-Out Cross-Validation Results")
    print("=" * 56)
    print(f"  n = {len(X)} matrices,  depth = {TREE_PARAMS['max_depth']}")
    print(f"  Overall accuracy: {overall_acc:.1%}  "
          f"({int(overall_acc * len(X))}/{len(X)})\n")

    print(classification_report(
        y_true, y_pred,
        target_names=le.classes_,
        digits=2,
        zero_division=0,
    ))

    # Per-class breakdown
    print("  Per-class detail:")
    for i, cls in enumerate(le.classes_):
        mask    = y_true == i
        correct = int(np.sum(y_pred[mask] == i))
        total   = int(np.sum(mask))
        acc_c   = correct / total if total > 0 else 0.0
        print(f"    {cls:6s}  {correct}/{total}  ({acc_c:.0%})")
    print()

    return overall_acc


# ──────────────────────────────────────────────────────────────
# 5. Predict (online, single matrix)
# ──────────────────────────────────────────────────────────────

def predict(model, le, feature_dict: dict, features=SELECTED_FEATURES) -> str:
    """
    Online algorithm selection for a single matrix.

    Parameters
    ----------
    model       : fitted DecisionTreeClassifier
    le          : fitted LabelEncoder
    feature_dict: dict mapping feature names → float values
                  (output of extract_features_from_mtx or a manual dict)

    Returns
    -------
    str : predicted algorithm label, one of {BCSR, BCOO, BDIA}
    """
    x     = np.array([[feature_dict[f] for f in features]])
    enc   = model.predict(x)[0]
    proba = model.predict_proba(x)[0]
    label = le.inverse_transform([enc])[0]

    confidence = proba.max()
    print(f"  Predicted algorithm : {label}  (confidence {confidence:.0%})")
    print(f"  Class probabilities : "
          + "  ".join(f"{le.classes_[i]}={proba[i]:.2f}"
                      for i in range(len(le.classes_))))
    return label


# ──────────────────────────────────────────────────────────────
# 6. Print human-readable tree rules
# ──────────────────────────────────────────────────────────────

def print_tree_rules(model, le, features=SELECTED_FEATURES):
    print("\n" + "=" * 56)
    print("  Decision Tree Rules")
    print("=" * 56)
    print(export_text(model, feature_names=features))


# ──────────────────────────────────────────────────────────────
# 7. Visualise (saves PNG)
# ──────────────────────────────────────────────────────────────

def visualise(model, le, df, out_dir=".", features=SELECTED_FEATURES):
    """
    Save two figures:
      cerespMM_tree.png         — tree structure
      cerespMM_importance.png   — feature importance bar chart
    """
    # ── Tree structure ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_tree(
        model,
        feature_names=features,
        class_names=le.classes_,
        filled=True, rounded=True, fontsize=10,
        ax=ax,
    )
    ax.set_title("CereSpMM Algorithm Selection — Decision Tree", fontsize=13)
    fig.tight_layout()
    tree_path = os.path.join(out_dir, "cerespMM_tree.png")
    fig.savefig(tree_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Tree diagram saved → {tree_path}")

    # ── Feature importance ────────────────────────────────────
    importances = model.feature_importances_
    imp_df = (pd.DataFrame({"feature": features, "importance": importances})
                .sort_values("importance", ascending=True))

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#3B82F6" if v > 0 else "#CBD5E1" for v in imp_df["importance"]]
    ax.barh(imp_df["feature"], imp_df["importance"], color=colors)
    ax.set_xlabel("Gini Importance")
    ax.set_title("Feature Importance (CereSpMM)")
    ax.set_xlim(0, imp_df["importance"].max() * 1.15)
    for i, (feat, val) in enumerate(zip(imp_df["feature"], imp_df["importance"])):
        if val > 0:
            ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    imp_path = os.path.join(out_dir, "cerespMM_importance.png")
    fig.savefig(imp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Importance chart saved → {imp_path}")


# ──────────────────────────────────────────────────────────────
# 8. Main entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CereSpMM automatic algorithm selection (decision tree)."
    )
    parser.add_argument(
        "--mode", choices=["train", "predict"], default="train",
        help="'train': fit model on CSV and evaluate.  "
             "'predict': load saved model and classify one matrix.",
    )
    parser.add_argument(
        "--data", default="features.csv",
        help="Path to labelled feature CSV (required for --mode train).",
    )
    parser.add_argument(
        "--features", nargs="+", default=SELECTED_FEATURES,
        help="Feature columns to use (default: average CV dia_ratio dia_pad).",
    )
    parser.add_argument(
        "--save", default=None,
        help="Save trained model to this path (e.g. model.pkl).",
    )
    parser.add_argument(
        "--load", default=None,
        help="Load a previously saved model (required for --mode predict).",
    )
    parser.add_argument(
        "--mtx", default=None,
        help="Path to a .mtx file to classify (used with --mode predict).",
    )
    parser.add_argument(
        "--out_dir", default=".",
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--no_plot", action="store_true",
        help="Skip figure generation.",
    )
    args = parser.parse_args()

    features = args.features

    # ── TRAIN mode ────────────────────────────────────────────
    if args.mode == "train":
        print("\nCereSpMM — Algorithm Selection Training")
        print("=" * 56)

        X, y, le, df = load_dataset(args.data, features)
        print(f"  Dataset : {args.data}")
        print(f"  Samples : {len(df)}")
        print(f"  Features: {features}")
        print(f"  Labels  :\n{df['label'].value_counts().to_string()}\n")

        # Evaluate first (LOO-CV)
        evaluate(X, y, le, features)

        # Train on full dataset
        model = train(X, y, le)
        print_tree_rules(model, le, features)

        # Save model
        if args.save:
            payload = {"model": model, "le": le, "features": features}
            joblib.dump(payload, args.save)
            print(f"  Model saved → {args.save}\n")

        # Visualise
        if not args.no_plot:
            visualise(model, le, df, args.out_dir, features)

    # ── PREDICT mode ──────────────────────────────────────────
    elif args.mode == "predict":
        if args.load is None:
            sys.exit("[ERROR] --load <model.pkl> is required for predict mode.")
        if args.mtx is None:
            sys.exit("[ERROR] --mtx <matrix.mtx> is required for predict mode.")

        payload  = joblib.load(args.load)
        model    = payload["model"]
        le       = payload["le"]
        features = payload["features"]

        print(f"\n  Loading matrix : {args.mtx}")
        feat_dict = extract_features_from_mtx(args.mtx)
        print(f"  Extracted features: {feat_dict}")
        print()
        predict(model, le, feat_dict, features)


if __name__ == "__main__":
    main()