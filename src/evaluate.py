"""Metric functions and the one-time evaluation on the held-out test set.

Run from the repository root (after training):

    python -m src.evaluate

Why these metrics for churn:

- PR-AUC (average precision). Only about a quarter of customers churn. PR-AUC
  looks at the churn class only, so it is not flattered by the easy majority
  class the way accuracy is.
- Recall at a fixed precision. The retention team can only contact so many
  people. This answers: if at least 60% of contacted customers must be real
  churners, what share of all churners do we reach?
- F1-macro. The average of the F1 score of both classes, so the model has to
  do reasonably on churners and on loyal customers at the chosen threshold.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config
from src.data_pipeline import load_split
from src.fallback import segment_scores


def recall_at_precision(y_true, scores, min_precision=config.TARGET_PRECISION) -> float:
    """Highest recall that still keeps precision at or above min_precision."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    allowed = recall[precision >= min_precision]
    # The curve always ends with the point (precision 1, recall 0).
    return float(allowed.max()) if len(allowed) else 0.0


def best_f1_macro_threshold(y_true, scores):
    """Try thresholds from 0.05 to 0.95 and keep the best F1-macro one."""
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        score = f1_score(y_true, scores >= threshold, average="macro")
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def compute_metrics(y_true, scores, threshold) -> dict:
    """All metrics for one model at one decision threshold."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "recall_at_precision": round(recall_at_precision(y_true, scores), 4),
        "threshold": round(float(threshold), 2),
        "f1_macro": round(float(f1_score(y_true, predicted, average="macro")), 4),
        "precision_churn": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        "recall_churn": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        "accuracy": round(float((predicted == y_true).mean()), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def plot_pr_curves(y_true, score_by_model, operating_point, path) -> None:
    """Precision-recall curves on one chart, with the chosen threshold marked.

    Each curve is drawn as steps between the cut-offs that really exist in
    the scores. precision_recall_curve() always adds one extra end point
    (precision 1, recall 0) that no cut-off produces. I leave it out, because
    a straight line to that point would show precision levels a model with
    only a few distinct scores (the segment rule) can never reach.
    """
    fig, ax = plt.subplots(figsize=(4.9, 3.75))
    colors = {"Logistic regression": "#6b7280", "LightGBM": "#1d4ed8", "Segment rule": "#d97706"}
    for name, scores in score_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall[:-1], precision[:-1], drawstyle="steps-post",
                label=f"{name} (PR-AUC {ap:.3f})", color=colors.get(name), linewidth=1.6)
    prevalence = float(np.mean(y_true))
    ax.axhline(prevalence, color="#9ca3af", linestyle="--", linewidth=1,
               label=f"No-skill prior ({prevalence:.3f})")
    ax.scatter([operating_point["recall"]], [operating_point["precision"]],
               color="#dc2626", zorder=5, s=45,
               label=f"Deployed threshold {operating_point['threshold']:.2f}")
    ax.set_xlabel("Recall (share of churners caught)", fontsize=9)
    ax.set_ylabel("Precision (share of flags that are churners)", fontsize=9)
    ax.set_title("Precision-recall curve, held-out test set", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, labelspacing=0.3)
    fig.tight_layout()
    # A small figure at a high dpi keeps the text readable when the image
    # is placed in the report at about half the page width.
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion(confusion: dict, title: str, path) -> None:
    """Confusion matrix with counts."""
    matrix = np.array([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]])
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    ax.imshow(matrix, cmap="Blues")
    labels = ["Stayed", "Churned"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    for row in range(2):
        for col in range(2):
            value = matrix[row, col]
            color = "white" if value > matrix.max() / 2 else "black"
            ax.text(col, row, f"{value}", ha="center", va="center", color=color, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    with open(config.ARTIFACT_DIR / config.METADATA_FILE, encoding="utf-8") as handle:
        metadata = json.load(handle)

    # This is the only place in the project that uses the test labels.
    _, X_test, y_test = load_split("test")
    print(f"Test rows: {len(X_test):,}, churn rate {y_test.mean():.4f}")

    results = {}
    score_by_model = {}

    # Dummy prior: every customer gets the training churn rate.
    prior = metadata["fallback"]["segment_table"]["global_rate"]
    prior_scores = np.full(len(y_test), prior)
    results["dummy_prior"] = compute_metrics(y_test, prior_scores, 0.5)

    # Segment rule baseline (also the serving fallback).
    rule_scores = segment_scores(X_test, metadata["fallback"]["segment_table"])
    results["segment_rule"] = compute_metrics(y_test, rule_scores, metadata["fallback"]["threshold"])
    score_by_model["Segment rule"] = rule_scores

    # Trained pipelines. Thresholds were fixed on the validation set.
    display_names = {"logreg": "Logistic regression", "lightgbm": "LightGBM"}
    for key, info in metadata["candidates"].items():
        pipeline = joblib.load(config.ARTIFACT_DIR / info["artifact"])
        scores = pipeline.predict_proba(X_test)[:, 1]
        results[key] = compute_metrics(y_test, scores, info["threshold"])
        score_by_model[display_names[key]] = scores

    production = metadata["production_model"]
    results["production_model"] = production

    # Also show the production model at the naive 0.5 threshold for contrast.
    production_scores = score_by_model[display_names[production]]
    results["production_at_default_0.5"] = compute_metrics(y_test, production_scores, 0.5)

    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    final = results[production]
    plot_pr_curves(
        y_test,
        score_by_model,
        {"recall": final["recall_churn"], "precision": final["precision_churn"],
         "threshold": final["threshold"]},
        config.FIGURE_DIR / "pr_curve_test.png",
    )
    plot_confusion(
        final["confusion"],
        f"{display_names[production]} on test, threshold {final['threshold']:.2f}",
        config.FIGURE_DIR / "confusion_matrix_test.png",
    )

    with open(config.REPORT_DIR / "metrics_test.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nProduction model: {production}")
    header = f"{'model':28s} {'PR-AUC':>7s} {'ROC-AUC':>8s} {'R@P60':>6s} {'thr':>5s} {'F1-mac':>7s} {'prec':>6s} {'recall':>7s}"
    print(header)
    for name in ["dummy_prior", "segment_rule", "logreg", "lightgbm", "production_at_default_0.5"]:
        m = results[name]
        print(f"{name:28s} {m['pr_auc']:7.4f} {m['roc_auc']:8.4f} {m['recall_at_precision']:6.3f} "
              f"{m['threshold']:5.2f} {m['f1_macro']:7.4f} {m['precision_churn']:6.3f} {m['recall_churn']:7.3f}")
    print("\nSaved reports/metrics_test.json and figures in reports/figures/")


if __name__ == "__main__":
    main()
