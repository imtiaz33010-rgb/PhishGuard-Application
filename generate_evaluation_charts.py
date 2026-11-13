
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
)

# ==========================================================
# 1. PATHS AND SETTINGS
# ==========================================================

PROJECT_DIR = Path(r"C:\phising_detection")
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "evaluation_results"

PREDICTIONS_FILE = MODELS_DIR / "test_predictions.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Legitimate", "Phishing"]

MODELS = {
    "Random Forest": {
        "prediction": "random_forest_prediction",
        "probability": "random_forest_phishing_probability",
        "filename": "random_forest",
    },
    "Logistic Regression": {
        "prediction": "logistic_regression_prediction",
        "probability": "logistic_regression_phishing_probability",
        "filename": "logistic_regression",
    },
    "Multinomial Naive Bayes": {
        "prediction": "multinomial_naive_bayes_prediction",
        "probability": "multinomial_naive_bayes_phishing_probability",
        "filename": "multinomial_naive_bayes",
    },
}

sns.set_theme(style="whitegrid")


def normalize_labels(series):
    """Convert supported class labels to 0=Legitimate, 1=Phishing."""
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="raise").astype(int)
        if not set(values.unique()).issubset({0, 1}):
            raise ValueError("Numeric labels must be 0 or 1.")
        return values

    mapping = {
        "0": 0, "legitimate": 0, "legit": 0,
        "benign": 0, "safe": 0, "normal": 0,
        "1": 1, "phishing": 1, "phish": 1,
        "malicious": 1, "unsafe": 1,
    }

    values = series.astype(str).str.strip().str.lower().map(mapping)

    if values.isna().any():
        unknown = series[values.isna()].unique().tolist()
        raise ValueError(f"Unknown label values: {unknown}")

    return values.astype(int)


# ==========================================================
# 2. LOAD EXISTING TEST PREDICTIONS
# ==========================================================

if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find existing predictions:\n{PREDICTIONS_FILE}\n"
        "This script needs test_predictions.csv and does not train models."
    )

df = pd.read_csv(PREDICTIONS_FILE)

required = ["actual_label"]
for columns in MODELS.values():
    required.extend([columns["prediction"], columns["probability"]])

missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(
        "Missing required CSV columns:\n"
        + "\n".join(missing)
        + "\nAvailable columns:\n"
        + ", ".join(df.columns)
    )

df = df.dropna(subset=required).copy()

if df.empty:
    raise ValueError("No complete prediction records were found.")

y_true = normalize_labels(df["actual_label"])

if y_true.nunique() != 2:
    raise ValueError("The test data must contain both classes.")

print("=" * 65)
print("PHISHGUARD EVALUATION CHARTS")
print("=" * 65)
print(f"Input:  {PREDICTIONS_FILE}")
print(f"Rows:   {len(df):,}")
print(f"Output: {OUTPUT_DIR}")


# ==========================================================
# 3. CONFUSION MATRICES AND INDIVIDUAL ROC CURVES
# ==========================================================

accuracies = {}
roc_results = {}

for model_name, config in MODELS.items():

    # ----- Confusion matrix -----
    y_pred = normalize_labels(df[config["prediction"]])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    accuracy = accuracy_score(y_true, y_pred) * 100
    accuracies[model_name] = accuracy

    plt.figure(figsize=(7, 5.5))

    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 13},
    )

    plt.title(f"{model_name} - Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    cm_path = OUTPUT_DIR / f"{config['filename']}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Also save the numerical confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Legitimate", "Actual_Phishing"],
        columns=["Predicted_Legitimate", "Predicted_Phishing"],
    )
    cm_df.to_csv(
        OUTPUT_DIR / f"{config['filename']}_confusion_matrix.csv"
    )

    # ----- Individual ROC curve -----
    scores = pd.to_numeric(
        df[config["probability"]], errors="coerce"
    )

    valid = scores.notna()
    actual = y_true.loc[valid]
    scores = scores.loc[valid]

    if actual.nunique() == 2:
        fpr, tpr, _ = roc_curve(actual, scores, pos_label=1)
        roc_auc = auc(fpr, tpr)

        roc_results[model_name] = (fpr, tpr, roc_auc)

        plt.figure(figsize=(7, 6))
        plt.plot(
            fpr, tpr, linewidth=2,
            label=f"AUC = {roc_auc:.4f}"
        )
        plt.plot(
            [0, 1], [0, 1],
            linestyle="--", linewidth=1,
            label="Random classifier"
        )

        plt.xlim(0, 1)
        plt.ylim(0, 1.02)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} - ROC Curve", fontsize=14)
        plt.legend(loc="lower right")
        plt.tight_layout()

        roc_path = OUTPUT_DIR / f"{config['filename']}_roc_curve.png"
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(
            f"{model_name}: accuracy={accuracy:.2f}%, "
            f"AUC={roc_auc:.4f}"
        )
    else:
        print(f"Warning: ROC curve skipped for {model_name}.")

    print(f"  Saved: {cm_path.name}")


# ==========================================================
# 4. ACCURACY COMPARISON CHART
# ==========================================================

model_names = list(accuracies.keys())
accuracy_values = list(accuracies.values())

lowest = min(accuracy_values)
highest = max(accuracy_values)
spread = highest - lowest
margin = max(spread * 0.25, 0.2)

y_min = max(0, lowest - margin)
y_max = min(100, highest + margin)

if y_max <= y_min:
    y_min = max(0, lowest - 0.5)
    y_max = min(100, highest + 0.5)

plt.figure(figsize=(10, 6))

bars = plt.bar(
    model_names,
    accuracy_values,
    width=0.55,
    edgecolor="black",
    linewidth=0.8,
)

plt.ylim(y_min, y_max)
plt.title("Accuracy Comparison of PhishGuard Models", fontsize=14)
plt.xlabel("Classification Model")
plt.ylabel("Accuracy (%) - zoomed scale")
plt.xticks(rotation=10, ha="right")

for bar, value in zip(bars, accuracy_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + (y_max - y_min) * 0.025,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "model_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ==========================================================
# 5. COMBINED ROC CURVE
# ==========================================================

if roc_results:
    plt.figure(figsize=(8, 6.5))

    for model_name, (fpr, tpr, roc_auc) in roc_results.items():
        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} (AUC = {roc_auc:.4f})",
        )

    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        linewidth=1,
        label="Random classifier",
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Combined ROC Curves for the Three Classifiers", fontsize=14)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "combined_roc_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ==========================================================
# 6. SAVE SUMMARY AND FINISH
# ==========================================================

summary = pd.DataFrame([
    {
        "Model": model_name,
        "Accuracy (%)": round(accuracies[model_name], 4),
        "ROC-AUC": (
            round(roc_results[model_name][2], 4)
            if model_name in roc_results else np.nan
        ),
    }
    for model_name in MODELS
])

summary.to_csv(OUTPUT_DIR / "evaluation_summary.csv", index=False)

print("\n" + "=" * 65)
print("EVALUATION CHARTS GENERATED")
print("=" * 65)
print(summary.to_string(index=False))
print(f"\nOpen this folder:\n{OUTPUT_DIR}")
print("\nNo model training was performed.")
