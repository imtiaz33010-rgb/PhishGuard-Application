"""
Error analysis of the three PhishGuard models on the final test set.

Reads models/test_predictions.csv (one row per test URL, with the true
label and each model's prediction/probability) and reports:

    1. Majority-vote performance (accuracy, precision, recall, F1).
    2. How often the models disagree, and what the disagreements contain.
    3. Logistic Regression error rate broken down by URL characteristics.
    4. Errors that ALL THREE models make together.
    5. Precision of each model at realistic (low) phishing prevalence.
    6. A CSV with sample false negatives and false positives for manual review.

Labels: 0 = Legitimate, 1 = Phishing.
"""

import re
import numpy as np, pandas as pd
from urllib.parse import urlparse

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

# Test-set predictions saved during evaluation
d = pd.read_csv("models/test_predictions.csv")

# Ground-truth labels (0 = Legitimate, 1 = Phishing)
y = d["actual_label"]

# Short model name -> column prefix used in the CSV
M = {"RF": "random_forest", "LR": "logistic_regression", "NB": "multinomial_naive_bayes"}

# For every model, flag rows where its prediction differs from the truth
# (creates columns RF_err, LR_err, NB_err as True/False)
for k, v in M.items():
    d[k + "_err"] = d[v + "_prediction"] != y

# Majority vote: since predictions are 0/1, the sum is the number of
# models that said "phishing" (0 to 3)
d["votes"] = d[[v + "_prediction" for v in M.values()]].sum(axis=1)

# Final ensemble prediction: phishing if at least 2 of 3 models agree
d["vote_pred"] = (d["votes"] >= 2).astype(int)


# ------------------------------------------------------------
# URL-DERIVED GROUPING FEATURES
# (used later to see which kinds of URLs cause errors)
# ------------------------------------------------------------

def host(u):
    """Return the lowercase host (netloc) of a URL, or "" if unparseable."""
    try:
        # Add a scheme if missing so urlparse puts the host in netloc
        return urlparse(u if "://" in u else "http://" + u).netloc.lower()
    except Exception:
        return ""

# Total URL length in characters
d["len"] = d["url"].str.len()

# Does the URL use HTTPS?
d["https"] = d["url"].str.lower().str.startswith("https")

# Is the host a raw IP address (only digits, dots, colons)?
# split("@")[-1] drops any "user:pass@" prefix first.
d["ip_host"] = d["url"].map(lambda u: bool(re.fullmatch(r"[\d.:]+", host(u).split("@")[-1])))

# Number of dots in the host (a rough measure of subdomain depth)
d["dots"] = d["url"].map(lambda u: host(u).count("."))

# Number of digit characters in the URL
d["digits"] = d["url"].str.count(r"\d")

# Does the URL contain "@"? (a classic obfuscation trick)
d["has_at"] = d["url"].str.contains("@")

# Split URL length into 4 equal-sized groups (quartiles);
# duplicates="drop" avoids errors if several bin edges are identical
d["lenbin"] = pd.qcut(d["len"], 4, duplicates="drop")


# ------------------------------------------------------------
# 1. MAJORITY-VOTE PERFORMANCE
# ------------------------------------------------------------

print("== Majority vote on final test set ==")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
p = d["vote_pred"]
print("acc %.4f prec %.4f rec %.4f f1 %.4f" % (accuracy_score(y, p), precision_score(y, p), recall_score(y, p), f1_score(y, p)))


# ------------------------------------------------------------
# 2. MODEL DISAGREEMENT
# ------------------------------------------------------------

print("\n== Where do the models disagree? ==")

# Split votes = 1 or 2 phishing votes (the models did not agree).
# 0 or 3 votes means all three models gave the same answer.
dis = d[d["votes"].isin([1, 2])]
print("split votes:", len(dis), "of", len(d), "(%.1f%%)" % (100 * len(dis) / len(d)))

# Mean of the 0/1 label = fraction of truly phishing URLs among the disagreements
print("phishing share among split votes: %.1f%%" % (100 * dis["actual_label"].mean()))

# Same breakdown separately for 1-vote and 2-vote cases
print(dis.groupby("votes")["actual_label"].agg(["count", "mean"]).rename(columns={"mean": "phishing_share"}))


# ------------------------------------------------------------
# 3. LOGISTIC REGRESSION ERROR RATE BY GROUP
# ------------------------------------------------------------

# For each grouping column: number of rows and error rate (mean of True/False)
print("\n== LR error rate by group ==")
for col in ["actual_label", "source_dataset", "https", "ip_host", "has_at", "lenbin"]:
    print(d.groupby(col)["LR_err"].agg(["count", "mean"]).round(4), "\n")


# ------------------------------------------------------------
# 4. ERRORS SHARED BY ALL MODELS
# ------------------------------------------------------------

print("== Errors made by ALL THREE models ==")

# Rows where RF, LR and NB were all wrong (the hardest cases)
all_err = d[d["RF_err"] & d["LR_err"] & d["NB_err"]]

# Missed phishing = true label 1; false alarms = true label 0
print("missed phishing:", (all_err["actual_label"] == 1).sum(), " false alarms:", (all_err["actual_label"] == 0).sum())


# ------------------------------------------------------------
# 5. PRECISION AT REALISTIC PHISHING PREVALENCE
# ------------------------------------------------------------

# The test set is usually balanced, but in real traffic phishing is rare.
# Using each model's recall (TPR) and false-positive rate (FPR), Bayes' rule
# gives the precision it would have at a given prevalence:
#     precision = pv*TPR / (pv*TPR + (1-pv)*FPR)
print("\n== Precision at realistic phishing prevalence ==")
for k in M:
    # Confusion-matrix counts for this model
    tp = ((d[M[k] + "_prediction"] == 1) & (y == 1)).sum(); fn = ((d[M[k] + "_prediction"] == 0) & (y == 1)).sum()
    fp = ((d[M[k] + "_prediction"] == 1) & (y == 0)).sum(); tn = ((d[M[k] + "_prediction"] == 0) & (y == 0)).sum()

    # tpr = recall (share of phishing caught); fpr = share of legitimate flagged
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)

    # Expected precision for prevalence values of 1%, 5%, 10% and 20%
    out = {f"{int(pv*100)}%": round(pv * tpr / (pv * tpr + (1 - pv) * fpr), 3) for pv in (0.01, 0.05, 0.10, 0.2)}
    print(k, "recall %.3f FPR %.3f" % (tpr, fpr), out)


# ------------------------------------------------------------
# 6. SAMPLE ERRORS FOR MANUAL INSPECTION
# ------------------------------------------------------------

# Columns to keep in the exported file
cols = ["url", "domain", "source_dataset", "actual_label", "logistic_regression_phishing_probability", "votes"]

# 12 random Logistic Regression false negatives (phishing it missed)
# NOTE: this reuses the name "fn", overwriting the integer from the loop above
fn = d[d["LR_err"] & (y == 1)].sample(12, random_state=42)[cols]

# 12 random Logistic Regression false positives (legitimate flagged as phishing)
fp = d[d["LR_err"] & (y == 0)].sample(12, random_state=42)[cols]

# Stack both samples, label each row with its error type, and save as CSV.
# random_state=42 makes the sample reproducible.
pd.concat([fn.assign(kind="FALSE NEGATIVE (missed phishing)"), fp.assign(kind="FALSE POSITIVE (false alarm)")]
          ).to_csv("error_samples.csv", index=False, encoding="utf-8")
print("\nSaved error_samples.csv (12 FN + 12 FP, seed 42)")