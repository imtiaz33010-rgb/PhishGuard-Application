"""
PhishGuard: URL-Based Phishing Detection
Training with domain-grouped evaluation.

Datasets:
  dataset/balanced_urls.csv
  dataset/Phishing_Legitimate_balanced.csv

Existing feature extractor:
  services/feature_extractor.py

Run:
  python train_models.py
"""

# --- Standard library imports ---
from pathlib import Path                      # Object-oriented file paths
from urllib.parse import urlsplit, urlunsplit  # Split / rebuild URLs
from collections import Counter               # (imported, not used below)
import json                                   # Save reports as JSON
import re                                     # Regular expressions (scheme check)
import warnings                               # Control warning messages

# --- Third-party imports ---
import joblib                                 # Save/load trained models
import numpy as np                            # Numeric arrays
import pandas as pd                           # DataFrames / CSV handling
from scipy.sparse import hstack, csr_matrix   # Combine sparse feature matrices

# --- scikit-learn: splitting, preprocessing, models, metrics ---
from sklearn.model_selection import StratifiedGroupKFold       # Split by group + class
from sklearn.preprocessing import MinMaxScaler                 # Scale features to 0-1
from sklearn.feature_extraction.text import TfidfVectorizer    # Text -> TF-IDF features
from sklearn.ensemble import RandomForestClassifier            # Model 1
from sklearn.linear_model import LogisticRegression            # Model 2
from sklearn.naive_bayes import MultinomialNB                  # Model 3
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --- Plotting (non-interactive backend so it works without a display) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Used to find the registrable domain (e.g. example.co.uk)
import tldextract

# Your own feature extractor: function + ordered list of feature names
from services.feature_extractor import calculate_features, FEATURE_NAMES


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Folder paths, all relative to where this script lives
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
REPORT_DIR = MODEL_DIR / "reports"

# The two input CSV files
DATASET_1 = DATASET_DIR / "Phishing_Legitimate_balanced.csv"
DATASET_2 = DATASET_DIR / "balanced_urls.csv"

# Fixed seed so results are reproducible
RANDOM_STATE = 42
# Target share of data that should end up in the test set (20%)
TEST_SIZE = 0.20

# Preserves the original script's structural sampling limit.
# Compare results with an uncapped experiment if computationally feasible.
MAX_GROUP_SIZE = 15000

# Number of domain-grouped folds. One fold is selected as the test set.
N_SPLITS = 5

# Create output folders if they do not exist yet
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Readable names for the numeric labels (used in reports and plots)
LABEL_NAMES = {
    0: "Legitimate",
    1: "Phishing",
}


# ============================================================
# 2. LABEL NORMALIZATION
# ============================================================

def normalize_label(value):
    """Normalize supported label formats to 0=legitimate, 1=phishing."""

    # Missing label -> keep as NaN so it can be removed later
    if pd.isna(value):
        return np.nan

    # If the label is already a number, accept 0 or 1 directly
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = int(value)

        if number in (0, 1):
            return number

    # Otherwise treat it as text: trim spaces and lowercase it
    value = str(value).strip().lower()

    # All text spellings that mean "phishing"
    phishing_values = {
        "phishing", "phish", "malicious", "malware",
        "bad", "1", "true"
    }

    # All text spellings that mean "legitimate"
    legitimate_values = {
        "legitimate", "benign", "safe", "good",
        "0", "false"
    }

    # Map text to the standard numeric label
    if value in legitimate_values:
        return 0

    if value in phishing_values:
        return 1

    # Unknown label: stop loudly instead of silently training on bad data
    raise ValueError(
        f"Unknown label {value!r}. "
        "Check the label meanings in the source dataset."
    )


# ============================================================
# 3. URL NORMALIZATION AND DOMAIN EXTRACTION
# ============================================================

def canonical_url(url):
    """
    Normalize URL scheme/host and remove fragments.
    Path and query are retained.
    """

    # Missing value -> empty string (marked invalid later)
    if pd.isna(url):
        return ""

    # Convert to text and remove surrounding spaces
    url = str(url).strip()

    # Empty after trimming -> invalid
    if not url:
        return ""

    # If there is no scheme (http://, https://, ftp://...), add http://
    # so the URL parser can read the host correctly
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url

    try:
        # Break the URL into scheme, netloc, path, query, fragment
        parts = urlsplit(url)

        # Scheme and hostname are case-insensitive -> lowercase them
        scheme = parts.scheme.lower()
        # Also remove a trailing dot (example.com. -> example.com)
        hostname = (parts.hostname or "").lower().rstrip(".")

        # No host means the URL is unusable
        if not hostname:
            return ""

        # Preserve an explicit port, if valid.
        try:
            port = parts.port
        except ValueError:
            # Invalid port number -> ignore the port
            port = None

        # Rebuild network location (host, plus port when present)
        netloc = hostname

        if port is not None:
            netloc = f"{hostname}:{port}"

        # Reassemble the URL; empty path becomes "/"
        return urlunsplit((
            scheme,
            netloc,
            parts.path or "/",
            parts.query,
            "",  # Remove fragment.
        ))

    except Exception:
        # Any parsing failure -> treat as invalid URL
        return ""


# Use the bundled Public Suffix List snapshot where available.
# No network request is required by this configuration.
extract_domain = tldextract.TLDExtract(suffix_list_urls=())


def get_domain(url):
    """
    Return registrable domain, e.g. login.example.co.uk -> example.co.uk.
    IP addresses and single-label hosts are retained as host identifiers.
    """

    try:
        # Clean the URL first, then pull out the hostname
        normalized = canonical_url(url)
        hostname = (urlsplit(normalized).hostname or "").lower().rstrip(".")

        if not hostname:
            return ""

        # Split the hostname into subdomain / domain / suffix
        extracted = extract_domain(hostname)

        # Normal case: domain + known suffix -> "domain.suffix"
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}".lower()

        # IPs and single-label hosts have no suffix -> use hostname itself
        return hostname

    except Exception:
        return ""


def get_url_structure(url):
    """Extract the structural attributes used by the original sampling method."""

    normalized = canonical_url(url)

    try:
        parts = urlsplit(normalized)
        hostname = (parts.hostname or "").lower()
        # Split host into labels, e.g. "a.b.example.com" -> 4 labels
        labels = [part for part in hostname.split(".") if part]

        return {
            "https": int(parts.scheme.lower() == "https"),   # 1 if HTTPS
            "www": int(hostname.startswith("www.")),         # 1 if starts with www.
            "subdomain": int(len(labels) > 2),               # 1 if more than 2 labels
        }

    except Exception:
        # If parsing fails, return all zeros
        return {
            "https": 0,
            "www": 0,
            "subdomain": 0,
        }


# ============================================================
# 4. LOAD AND STANDARDIZE BOTH DATASETS
# ============================================================

def load_datasets():
    # Make sure both dataset files exist before doing anything
    for path in (DATASET_1, DATASET_2):
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                "Check the project folder and filenames."
            )

    # Read both CSV files
    df1 = pd.read_csv(DATASET_1)
    df2 = pd.read_csv(DATASET_2)

    # Columns each dataset must have
    required_1 = {"url", "type"}
    required_2 = {"url", "label"}

    # Validate dataset 1 columns
    if not required_1.issubset(df1.columns):
        raise ValueError(
            f"{DATASET_1.name} must contain columns: {required_1}. "
            f"Found: {list(df1.columns)}"
        )

    # Validate dataset 2 columns
    if not required_2.issubset(df2.columns):
        raise ValueError(
            f"{DATASET_2.name} must contain columns: {required_2}. "
            f"Found: {list(df2.columns)}"
        )

    # Keep only needed columns and give the label column the same name
    df1 = df1[["url", "type"]].rename(columns={"type": "raw_label"})
    df2 = df2[["url", "label"]].rename(columns={"label": "raw_label"})

    # Remember which file each row came from
    df1["source_dataset"] = DATASET_1.name
    df2["source_dataset"] = DATASET_2.name

    # Stack both datasets into one table
    df = pd.concat([df1, df2], ignore_index=True)

    # Record the original row counts.
    initial_count = len(df)

    # Clean URL text, standardize labels, build canonical URL
    df["url"] = df["url"].astype("string").str.strip()
    df["label"] = df["raw_label"].apply(normalize_label)
    df["canonical_url"] = df["url"].apply(canonical_url)

    # Remove missing URLs, invalid URLs and missing labels.
    invalid_mask = (
        df["url"].isna()
        | df["url"].eq("")
        | df["canonical_url"].eq("")
        | df["label"].isna()
    )

    # Count then drop the invalid rows
    invalid_count = int(invalid_mask.sum())
    df = df.loc[~invalid_mask].copy()
    df["label"] = df["label"].astype(int)

    # Identify URLs assigned conflicting labels across the sources.
    # (same canonical URL appearing as both phishing and legitimate)
    labels_per_url = df.groupby("canonical_url")["label"].nunique()
    conflicting_urls = set(labels_per_url[labels_per_url > 1].index)

    conflict_rows = int(df["canonical_url"].isin(conflicting_urls).sum())

    # Drop all rows of conflicting URLs (their true label is unknown)
    df = df.loc[
        ~df["canonical_url"].isin(conflicting_urls)
    ].copy()

    # Remove duplicate canonical URLs.
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["canonical_url"], keep="first").copy()
    duplicate_rows = before_dedup - len(df)

    # Add the registrable domain (used later for grouped splitting)
    df["domain"] = df["canonical_url"].apply(get_domain)

    # Drop rows where no domain could be found
    df = df.loc[df["domain"].ne("")].copy()

    # Summary of everything removed, for auditing / the thesis
    cleaning_report = {
        "combined_input_rows": int(initial_count),
        "invalid_or_missing_rows_removed": invalid_count,
        "rows_removed_for_conflicting_url_labels": conflict_rows,
        "duplicate_rows_removed": int(duplicate_rows),
        "cleaned_rows_before_structural_sampling": int(len(df)),
        "class_distribution_before_sampling": {
            LABEL_NAMES[int(k)]: int(v)
            for k, v in df["label"].value_counts().sort_index().items()
        },
        "unique_domains_before_sampling": int(df["domain"].nunique()),
    }

    return df.reset_index(drop=True), cleaning_report


# ============================================================
# 5. STRUCTURAL SAMPLING WITH REPORTS
# ============================================================

def apply_structural_sampling(df):
    """
    Cap each label + HTTPS + WWW + subdomain structural group.

    IMPORTANT: this is structural sampling, NOT domain grouping.
    Domain grouping is applied separately during train/test splitting.
    """

    df = df.copy()

    # Compute the https / www / subdomain flags for every URL
    structure = df["canonical_url"].apply(get_url_structure)
    structure_df = pd.DataFrame(structure.tolist(), index=df.index)

    # Attach the flags as columns
    for column in ("https", "www", "subdomain"):
        df[column] = structure_df[column].astype(int)

    # A "group" = same label + same structural flags
    group_columns = ["label", "https", "www", "subdomain"]

    # Count how many rows each group has BEFORE sampling
    before = (
        df.groupby(group_columns, dropna=False)
        .size()
        .rename("records_before")
        .reset_index()
    )

    sampled_parts = []

    # Go through each group; randomly shrink it if it is too large
    for _, group in df.groupby(group_columns, sort=True, dropna=False):
        if len(group) > MAX_GROUP_SIZE:
            group = group.sample(
                n=MAX_GROUP_SIZE,
                random_state=RANDOM_STATE,
            )

        sampled_parts.append(group)

    # Join all groups back together, then shuffle the rows
    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    # Count how many rows each group has AFTER sampling
    after = (
        sampled.groupby(group_columns, dropna=False)
        .size()
        .rename("records_after")
        .reset_index()
    )

    # Build a per-group before/after/removed report and save it
    report = before.merge(after, on=group_columns, how="left")
    report["records_after"] = report["records_after"].fillna(0).astype(int)
    report["records_removed"] = (
        report["records_before"] - report["records_after"]
    )

    report.to_csv(
        REPORT_DIR / "structural_sampling_report.csv",
        index=False,
    )

    # Overall summary of the sampling step
    summary = {
        "sampling_method": "Random sampling within label/HTTPS/WWW/subdomain groups",
        "max_group_size": MAX_GROUP_SIZE,
        "rows_before_sampling": int(len(df)),
        "rows_after_sampling": int(len(sampled)),
        "rows_removed_by_sampling": int(len(df) - len(sampled)),
        # max(..., 1) avoids division by zero on an empty DataFrame
        "reduction_percentage": round(
            100 * (len(df) - len(sampled)) / max(len(df), 1), 2
        ),
        "class_distribution_after_sampling": {
            LABEL_NAMES[int(k)]: int(v)
            for k, v in sampled["label"].value_counts().sort_index().items()
        },
    }

    return sampled, summary


# ============================================================
# 6. DOMAIN-GROUPED TRAIN/TEST SPLIT
# ============================================================

def make_domain_grouped_split(df):
    """
    Assign each registrable domain entirely to train OR test.

    Try stratified group folds and select a test fold closest to the
    requested size and overall class proportions.
    """

    # Need at least as many domains as folds, otherwise splitting fails
    if df["domain"].nunique() < N_SPLITS:
        raise ValueError(
            f"Only {df['domain'].nunique()} unique domains remain. "
            f"At least {N_SPLITS} are required for grouped cross-validation."
        )

    # y = labels, groups = domain of each row
    y = df["label"].to_numpy()
    groups = df["domain"].to_numpy()

    # Splitter keeps each domain in one fold and balances the classes
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Share of phishing rows in the whole dataset (target for the test set)
    overall_phishing_rate = float(np.mean(y))
    best = None
    best_score = float("inf")

    # Check every fold; the fold used as "test" in each iteration is a candidate
    for train_idx, test_idx in splitter.split(
        np.zeros(len(df)), y, groups
    ):
        test_fraction = len(test_idx) / len(df)
        test_phishing_rate = float(np.mean(y[test_idx]))

        # Prefer folds close to 20% and the overall phishing proportion.
        score = (
            abs(test_fraction - TEST_SIZE)
            + abs(test_phishing_rate - overall_phishing_rate)
        )

        # Keep the candidate with the lowest (best) score
        if score < best_score:
            best_score = score
            best = (train_idx, test_idx)

    train_idx, test_idx = best

    # Build the actual train and test tables
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # Safety check: no domain may appear in both sets (data leakage)
    train_domains = set(train_df["domain"])
    test_domains = set(test_df["domain"])
    overlap = train_domains.intersection(test_domains)

    if overlap:
        raise RuntimeError(
            f"Domain leakage detected: {len(overlap)} domains occur "
            "in both training and test sets."
        )

    # Summary of the split, saved for auditing
    split_report = {
        "split_method": "StratifiedGroupKFold; one fold selected for test",
        "requested_test_fraction": TEST_SIZE,
        "actual_test_fraction": round(len(test_df) / len(df), 4),
        "training_records": int(len(train_df)),
        "test_records": int(len(test_df)),
        "unique_training_domains": int(train_df["domain"].nunique()),
        "unique_test_domains": int(test_df["domain"].nunique()),
        "overlapping_domains": int(len(overlap)),
        "training_class_distribution": {
            LABEL_NAMES[int(k)]: int(v)
            for k, v in train_df["label"].value_counts().sort_index().items()
        },
        "test_class_distribution": {
            LABEL_NAMES[int(k)]: int(v)
            for k, v in test_df["label"].value_counts().sort_index().items()
        },
    }

    # Write the split report to disk as JSON
    with open(
        REPORT_DIR / "domain_split_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(split_report, file, indent=4)

    return train_df, test_df, split_report


# ============================================================
# 7. FEATURE EXTRACTION
# ============================================================

def get_numeric_features(urls):
    """
    Uses your existing feature extractor.
    Checks the returned feature vector length for consistency.
    """

    rows = []

    # Loop over URLs; start=1 so the counter matches human counting
    for i, url in enumerate(urls, start=1):
        try:
            # Compute features for this URL using your extractor
            features = calculate_features(url)

            # If a dict is returned, order values by FEATURE_NAMES;
            # otherwise assume it is already an ordered list
            if isinstance(features, dict):
                row = [features[name] for name in FEATURE_NAMES]
            else:
                row = list(features)

            # Every row must have exactly the expected number of features
            if len(row) != len(FEATURE_NAMES):
                raise ValueError(
                    f"Expected {len(FEATURE_NAMES)} features, "
                    f"received {len(row)}."
                )

            rows.append(row)

        except Exception as exc:
            # Re-raise with the failing URL so it is easy to debug
            raise RuntimeError(
                f"Feature extraction failed at URL {i}: {url!r}. "
                "Check services/feature_extractor.py. "
                f"Original error: {exc}"
            ) from exc

        # Progress message every 5,000 URLs
        if i % 5000 == 0:
            print(f"Extracted features for {i:,} URLs...")

    # Build a table: one row per URL, one column per feature
    result = pd.DataFrame(rows, columns=FEATURE_NAMES)
    # Replace infinity / missing values with 0 so models can use the data
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)

    return result.astype(float)


# ============================================================
# 8. METRICS AND CONFUSION MATRIX
# ============================================================

def evaluate_model(name, model, X_test, y_test):
    # Predict labels for the test set
    predictions = model.predict(X_test)

    # Standard metrics, with phishing (1) as the positive class
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
    }

    # Print a per-class report (precision / recall / F1 / support)
    print(f"\n{name} classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Phishing"],
            zero_division=0,
        )
    )

    # Confusion matrix: rows = actual, columns = predicted
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    # Draw the confusion matrix as a heat-map
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ax.imshow(cm)
    ax.set_title(f"{name} - Domain-Grouped Test Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], ["Legitimate", "Phishing"])
    ax.set_yticks([0, 1], ["Legitimate", "Phishing"])

    # Write the count inside each cell
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center")

    # Add colour bar, save the image, and free the memory
    fig.colorbar(display, ax=ax)
    fig.tight_layout()
    fig.savefig(
        REPORT_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png",
        dpi=200,
    )
    plt.close(fig)

    return metrics, predictions


# ============================================================
# 9. MAIN TRAINING PIPELINE
# ============================================================

def main():
    # Hide noisy FutureWarning messages from libraries
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("=" * 70)
    print("PHISHGUARD DOMAIN-GROUPED TRAINING")
    print("=" * 70)

    # Load and clean.
    df, cleaning_report = load_datasets()

    print("\nCleaning summary:")
    print(json.dumps(cleaning_report, indent=4))

    # Save the cleaning report
    with open(
        REPORT_DIR / "cleaning_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(cleaning_report, file, indent=4)

    # Save cleaned data before sampling for auditability.
    df.to_csv(
        REPORT_DIR / "cleaned_data_before_sampling.csv",
        index=False,
    )

    # Structural sampling.
    df, sampling_summary = apply_structural_sampling(df)

    print("\nSampling summary:")
    print(json.dumps(sampling_summary, indent=4))

    # Save the sampling summary
    with open(
        REPORT_DIR / "sampling_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(sampling_summary, file, indent=4)

    # Domain-grouped split.
    train_df, test_df, split_report = make_domain_grouped_split(df)

    print("\nDomain split summary:")
    print(json.dumps(split_report, indent=4))

    # Save the exact split assignments for later auditing.
    train_df[["url", "label", "domain"]].to_csv(
        REPORT_DIR / "training_split_urls.csv",
        index=False,
    )
    test_df[["url", "label", "domain"]].to_csv(
        REPORT_DIR / "test_split_urls.csv",
        index=False,
    )

    # Fit feature transformations using training data only.
    print("\nExtracting numeric URL features...")

    # Handcrafted numeric features for train and test
    X_train_num_raw = get_numeric_features(train_df["url"].tolist())
    X_test_num_raw = get_numeric_features(test_df["url"].tolist())

    # Scale to 0-1: fit on TRAIN only, then apply the same scaling to TEST
    scaler = MinMaxScaler()
    X_train_num = scaler.fit_transform(X_train_num_raw)
    X_test_num = scaler.transform(X_test_num_raw)

    # Character n-gram TF-IDF: captures patterns inside the URL text
    print("Fitting character TF-IDF...")
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),       # sequences of 3 to 5 characters
        max_features=50000,       # keep the 50,000 most useful n-grams
        min_df=2,                 # ignore n-grams seen in fewer than 2 URLs
        sublinear_tf=True,        # dampen very frequent terms (log scaling)
        dtype=np.float32,         # smaller memory use
    )

    # Learn vocabulary on training URLs only; transform both sets
    X_train_char = char_vectorizer.fit_transform(train_df["url"].astype(str))
    X_test_char = char_vectorizer.transform(test_df["url"].astype(str))

    # Word TF-IDF: captures meaningful words/tokens in the URL
    print("Fitting word TF-IDF...")
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",   # tokens = runs of word characters
        ngram_range=(1, 2),             # single words and word pairs
        max_features=30000,
        min_df=2,
        sublinear_tf=True,
        dtype=np.float32,
    )

    X_train_word = word_vectorizer.fit_transform(train_df["url"].astype(str))
    X_test_word = word_vectorizer.transform(test_df["url"].astype(str))

    # Numeric features are nonnegative after MinMax scaling.
    # (needed because MultinomialNB cannot accept negative values)
    # Combine numeric + char + word features side by side into one matrix
    X_train = hstack(
        [
            csr_matrix(X_train_num, dtype=np.float32),
            X_train_char,
            X_train_word,
        ],
        format="csr",
    )

    X_test = hstack(
        [
            csr_matrix(X_test_num, dtype=np.float32),
            X_test_char,
            X_test_word,
        ],
        format="csr",
    )

    # Target labels for training and testing
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    print(f"\nTraining matrix: {X_train.shape}")
    print(f"Testing matrix:  {X_test.shape}")

    # Train models on the same training and test split.
    models = {
        # Ensemble of 400 decision trees; balanced weights handle class imbalance
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,                  # use all CPU cores
            class_weight="balanced",
        ),
        # Linear model; liblinear works well on large sparse data
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        # Simple probabilistic baseline, suited to TF-IDF-style features
        "Multinomial Naive Bayes": MultinomialNB(),
    }

    # Containers for metrics and predictions of every model
    results = []
    predictions_by_model = {}

    for name, model in models.items():
        print(f"\n{'=' * 70}\nTraining {name}...\n")

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate it on the domain-grouped test set
        metrics, predictions = evaluate_model(
            name, model, X_test, y_test
        )

        results.append(metrics)
        predictions_by_model[name] = predictions

        # File name used to save each trained model
        model_filename = {
            "Random Forest": "random_forest_model.joblib",
            "Logistic Regression": "logistic_regression_model.joblib",
            "Multinomial Naive Bayes": "multinomial_nb_model.joblib",
        }[name]

        joblib.dump(model, MODEL_DIR / model_filename)

    # Save transformations and feature names.
    # (needed later to prepare new URLs the same way before predicting)
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(char_vectorizer, MODEL_DIR / "char_vectorizer.joblib")
    joblib.dump(word_vectorizer, MODEL_DIR / "word_vectorizer.joblib")
    joblib.dump(list(FEATURE_NAMES), MODEL_DIR / "feature_names.joblib")

    # Results table.
    results_df = pd.DataFrame(results)
    results_df.to_csv(MODEL_DIR / "model_results.csv", index=False)

    print("\nFinal domain-grouped results:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Save prediction details for further analysis.
    # Start with the URL info and the true labels
    prediction_df = pd.DataFrame({
        "url": test_df["url"].to_numpy(),
        "domain": test_df["domain"].to_numpy(),
        "source_dataset": test_df["source_dataset"].to_numpy(),
        "actual_label": y_test,
    })

    # Add one prediction column (and probability column) per model
    for name, model in models.items():
        prefix = {
            "Random Forest": "random_forest",
            "Logistic Regression": "logistic_regression",
            "Multinomial Naive Bayes": "multinomial_naive_bayes",
        }[name]

        prediction_df[f"{prefix}_prediction"] = predictions_by_model[name]

        # Probability of "phishing" for each URL, if the model supports it
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)
            # Find which column corresponds to class 1 (phishing)
            phishing_column = list(model.classes_).index(1)

            prediction_df[
                f"{prefix}_phishing_probability"
            ] = probabilities[:, phishing_column]

    prediction_df.to_csv(
        MODEL_DIR / "test_predictions.csv",
        index=False,
    )

    # Save metadata needed to understand this experiment.
    training_info = {
        "random_state": RANDOM_STATE,
        "requested_test_size": TEST_SIZE,
        "split_method": "StratifiedGroupKFold grouped by registrable domain",
        "label_mapping": LABEL_NAMES,
        "max_group_size": MAX_GROUP_SIZE,
        "n_splits": N_SPLITS,
        "cleaning_report": cleaning_report,
        "sampling_summary": sampling_summary,
        "split_report": split_report,
        "feature_names": list(FEATURE_NAMES),
        "numeric_feature_count": len(FEATURE_NAMES),
        "character_tfidf_feature_count": int(X_train_char.shape[1]),
        "word_tfidf_feature_count": int(X_train_word.shape[1]),
        "training_matrix_shape": list(X_train.shape),
        "test_matrix_shape": list(X_test.shape),
        "results": results,
    }

    # Save as .joblib (for code) and .json (human-readable)
    joblib.dump(training_info, MODEL_DIR / "training_info.joblib")

    with open(
        REPORT_DIR / "training_info.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(training_info, file, indent=4)

    print("\nTraining completed.")
    print(f"Models and predictions saved to: {MODEL_DIR}")
    print(f"Reports saved to: {REPORT_DIR}")
    print("\nReview the domain split report before using results in your thesis.")


# Run main() only when this file is executed directly (not when imported)
if __name__ == "__main__":
    main()