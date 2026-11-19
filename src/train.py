"""Fit the baselines and the LightGBM model, tune it, and export artifacts.

Run from the repository root (after the data pipeline):

    python -m src.train

What happens here, in order:

1. Load the training and validation partitions. The test partition is never
   opened in this file.
2. Fit three reference points on the training data: a dummy prior, the
   segment rule, and a class-weighted logistic regression.
3. Tune LightGBM with 5-fold stratified cross-validation on the training
   data only, scored by PR-AUC.
4. Pick each model's decision threshold on the validation set (best F1-macro).
5. Choose the production model by validation PR-AUC and save everything
   the API needs: the fitted pipelines and one metadata JSON file.

Class imbalance is handled with class weights and threshold tuning. I do not
oversample. Copied churners add no new information, and if the copies are
made before cross-validation, the same customer can sit in both the fitting
fold and the scoring fold, which makes the tuning scores look better than
they are. Class weights rebalance the loss without touching the rows. Either
way the output is shifted upwards, so it is a risk score and not a calibrated
probability.
"""
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import lightgbm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.data_pipeline import load_split
from src.evaluate import best_f1_macro_threshold, compute_metrics
from src.fallback import fit_segment_table, segment_scores

MODEL_VERSION = "1.0.0"


def build_preprocessor() -> ColumnTransformer:
    """Imputation, scaling and one-hot encoding.

    This object sits inside the model pipeline, so its statistics (medians,
    most frequent categories, means, category lists) are learned only from
    the rows passed to fit(), which is always training data.
    """
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": a category never seen in training becomes
        # all zeros instead of raising an error at serving time.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_steps, config.MODEL_NUMERIC),
        ("categorical", categorical_steps, config.MODEL_CATEGORICAL),
    ])


def build_logreg() -> Pipeline:
    """Interpretable baseline: class-weighted logistic regression."""
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=config.RANDOM_SEED
    )
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def build_lightgbm() -> Pipeline:
    """Gradient boosted trees with class weights for the imbalance."""
    model = LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
        n_jobs=config.MODEL_THREADS,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def tune_lightgbm(X_train, y_train):
    """Random search over a small grid, 5-fold stratified CV, scored by PR-AUC.

    The data is small (about 4,900 training rows), so I keep the trees small
    and the learning rate low to avoid overfitting.
    """
    search_space = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__learning_rate": [0.01, 0.02, 0.05, 0.1],
        "model__num_leaves": [4, 8, 15, 31],
        "model__max_depth": [3, 4, 6, -1],
        "model__min_child_samples": [20, 40, 80],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__subsample_freq": [1],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__reg_lambda": [0.0, 1.0, 5.0, 10.0],
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    search = RandomizedSearchCV(
        build_lightgbm(),
        param_distributions=search_space,
        n_iter=25,
        scoring="average_precision",
        cv=folds,
        random_state=config.RANDOM_SEED,
        n_jobs=1,
        refit=True,  # refit the best setting on all training rows
    )
    search.fit(X_train, y_train)
    return search


def plot_threshold_curve(y_val, scores, chosen, path) -> None:
    """How precision, recall and F1-macro move as the threshold changes."""
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f1_macro = [f1_score(y_val, scores >= t, average="macro") for t in thresholds]
    precision = [precision_score(y_val, scores >= t, zero_division=0) for t in thresholds]
    recall = [recall_score(y_val, scores >= t, zero_division=0) for t in thresholds]

    # A small figure at a high dpi keeps the text readable when the image
    # is placed in the report at about half the page width.
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.plot(thresholds, f1_macro, label="F1-macro", color="#1d4ed8", linewidth=2)
    ax.plot(thresholds, precision, label="Precision (churn)", color="#059669", linewidth=1.5)
    ax.plot(thresholds, recall, label="Recall (churn)", color="#d97706", linewidth=1.5)
    ax.axvline(chosen, color="#dc2626", linestyle="--", linewidth=1.2,
               label=f"Chosen threshold {chosen:.2f}")
    ax.axvline(0.5, color="#9ca3af", linestyle=":", linewidth=1.2, label="Default 0.50")
    ax.set_xlabel("Decision threshold on the model score", fontsize=9)
    ax.set_ylabel("Metric value", fontsize=9)
    ax.set_title("Threshold tuning on the validation set", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, labelspacing=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    _, X_train, y_train = load_split("train")
    _, X_val, y_val = load_split("val")
    print(f"Train rows: {len(X_train):,}  Validation rows: {len(X_val):,}")

    validation = {}

    # ---- reference point 1: dummy prior -------------------------------
    prior = float(y_train.mean())
    validation["dummy_prior"] = compute_metrics(y_val, np.full(len(y_val), prior), 0.5)

    # ---- reference point 2: segment rule (also the serving fallback) ---
    segment_table = fit_segment_table(X_train, y_train)
    rule_val_scores = segment_scores(X_val, segment_table)
    rule_threshold, _ = best_f1_macro_threshold(y_val, rule_val_scores)
    validation["segment_rule"] = compute_metrics(y_val, rule_val_scores, rule_threshold)

    # ---- reference point 3: logistic regression ------------------------
    logreg = build_logreg().fit(X_train, y_train)
    logreg_val_scores = logreg.predict_proba(X_val)[:, 1]
    logreg_threshold, _ = best_f1_macro_threshold(y_val, logreg_val_scores)
    validation["logreg"] = compute_metrics(y_val, logreg_val_scores, logreg_threshold)

    # ---- main model: tuned LightGBM -----------------------------------
    print("Tuning LightGBM (25 settings x 5 folds) ...")
    search = tune_lightgbm(X_train, y_train)
    lgbm = search.best_estimator_
    lgbm_val_scores = lgbm.predict_proba(X_val)[:, 1]
    lgbm_threshold, _ = best_f1_macro_threshold(y_val, lgbm_val_scores)
    validation["lightgbm"] = compute_metrics(y_val, lgbm_val_scores, lgbm_threshold)
    validation["lightgbm_at_default_0.5"] = compute_metrics(y_val, lgbm_val_scores, 0.5)

    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    cv_mean = float(search.cv_results_["mean_test_score"][search.best_index_])
    cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
    print(f"Best CV PR-AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")
    print(f"Best parameters: {best_params}")

    # ---- choose the production model ----------------------------------
    # Rule: LightGBM is deployed only if it beats the simpler logistic
    # regression on validation PR-AUC. A tie goes to the simpler model.
    if validation["lightgbm"]["pr_auc"] > validation["logreg"]["pr_auc"]:
        production = "lightgbm"
    else:
        production = "logreg"
    print(f"Production model chosen on validation PR-AUC: {production}")

    # ---- save artifacts -----------------------------------------------
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(logreg, config.ARTIFACT_DIR / "model_logreg.joblib")
    joblib.dump(lgbm, config.ARTIFACT_DIR / "model_lightgbm.joblib")

    # Facts about the training data that the serving code needs.
    numeric_ranges = {
        col: {"min": float(X_train[col].min()), "max": float(X_train[col].max())}
        for col in config.RAW_NUMERIC
    }
    known_categories = {
        col: sorted(X_train[col].dropna().unique().tolist())
        for col in config.MODEL_CATEGORICAL
    }

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "production_model": production,
        "candidates": {
            "logreg": {"artifact": "model_logreg.joblib", "threshold": logreg_threshold},
            "lightgbm": {"artifact": "model_lightgbm.joblib", "threshold": lgbm_threshold},
        },
        "threshold_rule": "maximise F1-macro on the validation set",
        "features": {
            "numeric": config.MODEL_NUMERIC,
            "categorical": config.MODEL_CATEGORICAL,
        },
        "numeric_ranges": numeric_ranges,
        "known_categories": known_categories,
        "fallback": {"segment_table": segment_table, "threshold": rule_threshold},
        "training": {
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "train_churn_rate": round(prior, 4),
            "random_seed": config.RANDOM_SEED,
            "lightgbm_best_params": best_params,
            "lightgbm_cv_pr_auc_mean": round(cv_mean, 4),
            "lightgbm_cv_pr_auc_std": round(cv_std, 4),
        },
        "validation_metrics": validation,
        "library_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }
    with open(config.ARTIFACT_DIR / config.METADATA_FILE, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    # ---- reports -------------------------------------------------------
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    production_scores = lgbm_val_scores if production == "lightgbm" else logreg_val_scores
    production_threshold = lgbm_threshold if production == "lightgbm" else logreg_threshold
    plot_threshold_curve(y_val, production_scores, production_threshold,
                         config.FIGURE_DIR / "threshold_tuning_val.png")

    # Keep the five best search settings so the tuning can be inspected.
    order = np.argsort(search.cv_results_["rank_test_score"])[:5]
    top_settings = [
        {
            "rank": int(search.cv_results_["rank_test_score"][i]),
            "cv_pr_auc_mean": round(float(search.cv_results_["mean_test_score"][i]), 4),
            "cv_pr_auc_std": round(float(search.cv_results_["std_test_score"][i]), 4),
            "params": {k.replace("model__", ""): v
                       for k, v in search.cv_results_["params"][i].items()},
        }
        for i in order
    ]
    with open(config.REPORT_DIR / "metrics_validation.json", "w", encoding="utf-8") as handle:
        json.dump({"production_model": production, "validation": validation,
                   "lightgbm_top_settings": top_settings}, handle, indent=2)

    print("\nValidation results")
    print(f"{'model':26s} {'PR-AUC':>7s} {'ROC-AUC':>8s} {'R@P60':>6s} {'thr':>5s} {'F1-mac':>7s} {'prec':>6s} {'recall':>7s}")
    for name, m in validation.items():
        print(f"{name:26s} {m['pr_auc']:7.4f} {m['roc_auc']:8.4f} {m['recall_at_precision']:6.3f} "
              f"{m['threshold']:5.2f} {m['f1_macro']:7.4f} {m['precision_churn']:6.3f} {m['recall_churn']:7.3f}")
    print(f"\nArtifacts saved to artifacts/. Training took {time.perf_counter() - started:.1f} s.")


if __name__ == "__main__":
    main()
