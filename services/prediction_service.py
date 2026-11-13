"""
PhishGuard URL prediction service.

This module loads three pre-trained machine-learning models (Random Forest,
Logistic Regression, Multinomial Naive Bayes) together with the text
vectorizers and numeric scaler that were fitted during training. For any
given URL it:

    1. Extracts hand-crafted numeric features from the URL.
    2. Converts the URL into character-level and word-level TF-IDF vectors.
    3. Combines everything into a single sparse feature matrix.
    4. Runs all three models and reports each one's verdict.
    5. Returns the Random Forest result as the DEFAULT top-level verdict.
       NOTE: in the full application, app.py (apply_ensemble_decision)
       overrides this default with a majority vote of all three models.
       The individual model results are always kept under "models".
    6. Optionally performs a live website check as secondary evidence.
"""

import os
import re
from urllib.parse import urlsplit  # Splits a URL into scheme/host/path/query
import joblib                      # Loads the saved (pickled) sklearn objects
import numpy as np
from scipy.sparse import hstack, csr_matrix  # Used to join dense + sparse features

# Custom feature extraction: computes numeric features from a URL
# and exposes the ordered list of feature names used in training.
from services.feature_extractor import (
    calculate_features,
    FEATURE_NAMES,
)

# Helper that visits the URL in real time and returns security observations.
from services.live_url_checker import (
    check_live_url,
)


# ============================================================
# PATHS
# ============================================================

# BASE_DIR = the project root, i.e. two directory levels above this file
# (this file -> its folder -> project root).
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Folder that holds all the saved model artifacts.
MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# MODEL FILES
# ============================================================

# Trained classifiers
RF_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest_tfidf_model.joblib"
)

LR_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "logistic_regression_tfidf_model.joblib"
)

NB_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "naive_bayes_tfidf_model.joblib"
)

# Fitted TF-IDF vectorizers (character n-grams and word tokens)
CHAR_VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "char_tfidf_vectorizer.joblib"
)

WORD_VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "word_tfidf_vectorizer.joblib"
)

# Scaler fitted on the numeric features during training
SCALER_PATH = os.path.join(
    MODEL_DIR,
    "feature_scaler.joblib"
)

# Ordered list of numeric feature names used at training time
FEATURE_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "feature_names.joblib"
)


# ============================================================
# LOAD MODELS
# ============================================================

# Models are loaded once, when this module is first imported,
# so that each prediction request does not pay the loading cost.
# NOTE: the .joblib files must be loaded with the same scikit-learn
# version that created them (scikit-learn==1.6.1 in requirements.txt).
print("Loading PhishGuard models...")

random_forest_model = joblib.load(
    RF_MODEL_PATH
)

logistic_model = joblib.load(
    LR_MODEL_PATH
)

naive_bayes_model = joblib.load(
    NB_MODEL_PATH
)

char_vectorizer = joblib.load(
    CHAR_VECTORIZER_PATH
)

word_vectorizer = joblib.load(
    WORD_VECTORIZER_PATH
)

numeric_scaler = joblib.load(
    SCALER_PATH
)

saved_feature_names = joblib.load(
    FEATURE_NAMES_PATH
)


# ============================================================
# FEATURE SCHEMA CHECK
# ============================================================

# Safety check: the feature names (and their order) used during training
# must be identical to what feature_extractor.py produces now. If not, the
# numeric columns would be misaligned and predictions would be meaningless,
# so we stop immediately instead of silently giving wrong results.
if list(saved_feature_names) != list(FEATURE_NAMES):

    raise ValueError(
        "Feature schema mismatch between "
        "feature_extractor.py and trained model."
    )


# Number of hand-crafted numeric features
EXPECTED_NUMERIC_FEATURES = len(
    FEATURE_NAMES
)


# ============================================================
# MODEL DIMENSION CHECK
# ============================================================

# Number of text features = size of the character vocabulary
# plus size of the word vocabulary.
EXPECTED_TEXT_FEATURES = (
    len(char_vectorizer.vocabulary_)
    +
    len(word_vectorizer.vocabulary_)
)

# Total input width every model must expect:
# numeric features + character TF-IDF + word TF-IDF.
EXPECTED_TOTAL_FEATURES = (
    EXPECTED_NUMERIC_FEATURES
    +
    EXPECTED_TEXT_FEATURES
)


# Verify each model was trained on exactly this many features.
# A mismatch usually means the models and vectorizers come from
# different training runs.
if random_forest_model.n_features_in_ != EXPECTED_TOTAL_FEATURES:

    raise ValueError(
        "Random Forest feature count mismatch. "
        f"Model expects "
        f"{random_forest_model.n_features_in_}, "
        f"but pipeline produces "
        f"{EXPECTED_TOTAL_FEATURES}."
    )


if logistic_model.n_features_in_ != EXPECTED_TOTAL_FEATURES:

    raise ValueError(
        "Logistic Regression feature count mismatch. "
        f"Model expects "
        f"{logistic_model.n_features_in_}, "
        f"but pipeline produces "
        f"{EXPECTED_TOTAL_FEATURES}."
    )


if naive_bayes_model.n_features_in_ != EXPECTED_TOTAL_FEATURES:

    raise ValueError(
        "Naive Bayes feature count mismatch. "
        f"Model expects "
        f"{naive_bayes_model.n_features_in_}, "
        f"but pipeline produces "
        f"{EXPECTED_TOTAL_FEATURES}."
    )


# Startup confirmation / diagnostics
print("Models loaded successfully.")
print("Numeric features:", EXPECTED_NUMERIC_FEATURES)
print("Total features:", EXPECTED_TOTAL_FEATURES)



# ============================================================
# DOMAIN-AWARE TF-IDF TEXT PREPROCESSING
# ============================================================

def prepare_tfidf_text(url):
    """
    Convert a raw URL into a structured text string for TF-IDF.

    The output looks like:
        "DOMAIN example.com PATH login verify QUERY id 123"

    The marker words (DOMAIN / PATH / QUERY) let the vectorizers
    distinguish where a token came from.

    Must match the preprocessing used during training.
    Handles malformed URLs safely.
    """

    # Reject anything that is not a string (None, numbers, etc.)
    if not isinstance(url, str):
        return ""

    # Normalise: remove surrounding whitespace and lowercase everything
    value = url.strip().lower()

    # Empty URL -> empty text
    if not value:
        return ""

    parse_value = value

    # urlsplit only identifies the host correctly when a scheme is present,
    # so add "http://" if the URL doesn't already start with one
    # (e.g. "example.com/login" -> "http://example.com/login").
    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        parse_value
    ):
        parse_value = "http://" + parse_value

    try:
        # Split the URL into components
        parsed = urlsplit(parse_value)

        domain = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""

    except ValueError:
        # urlsplit can raise ValueError on badly formed URLs
        # (e.g. invalid IPv6 brackets). In that case, fall back to
        # manually splitting the string.

        # Remove the scheme (http://, https://, ftp://, ...)
        fallback = re.sub(
            r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
            "",
            value
        )

        # Discard the fragment (everything after '#')
        fallback = fallback.split("#", 1)[0]

        # Separate the query string from the rest
        if "?" in fallback:
            main_part, query = fallback.split("?", 1)
        else:
            main_part = fallback
            query = ""

        # Separate domain from path at the first '/'
        if "/" in main_part:
            domain, path = main_part.split("/", 1)
            path = "/" + path
        else:
            domain = main_part
            path = ""

    # Strip a leading "www." so "www.site.com" and "site.com" look the same
    domain = re.sub(
        r"^www\.",
        "",
        domain
    )

    # Turn path separators into spaces so each path segment becomes a word
    path = path.replace("/", " ")

    # Turn query-string delimiters into spaces so keys and values become words
    query = query.replace("&", " ")
    query = query.replace("=", " ")

    # Collapse repeated whitespace into a single space and trim each part
    domain = re.sub(
        r"\s+",
        " ",
        domain
    ).strip()

    path = re.sub(
        r"\s+",
        " ",
        path
    ).strip()

    query = re.sub(
        r"\s+",
        " ",
        query
    ).strip()

    # Assemble the final text with section markers
    parts = [
        "DOMAIN",
        domain,
        "PATH",
        path,
        "QUERY",
        query,
    ]

    # Join non-empty parts only (empty strings are skipped)
    return " ".join(
        part for part in parts if part
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(url):
    """
    Build the complete feature matrix for one URL.

    Returns a tuple:
        (combined_features, raw_numeric_features_dict)
    where combined_features is a 1-row sparse matrix ready for the models.
    """

    # ========================================================
    # NUMERICAL FEATURES
    # ========================================================

    # Compute hand-crafted features (URL length, digit count, etc.)
    # as a dictionary {feature_name: value}.
    features = calculate_features(
        url
    )

    # Put the values in the exact order used during training.
    # Missing features default to 0 so the vector length never changes.
    numeric_values = [
        float(
            features.get(
                name,
                0
            )
        )
        for name in FEATURE_NAMES
    ]

    # Convert to a 2D numpy array of shape (1, n_features),
    # which is the format sklearn expects for a single sample.
    numeric_array = np.asarray(
        numeric_values,
        dtype=np.float32
    ).reshape(
        1,
        -1
    )

    # Apply the same scaling that was learned from the training data.
    # (Values outside the training range can scale outside the usual
    # range; see the review notes if the scaler is a MinMaxScaler.)
    numeric_scaled = numeric_scaler.transform(
        numeric_array
    )


    # ========================================================
    # TF-IDF TEXT
    #
    # IMPORTANT:
    # This MUST match train_balanced.py exactly.
    # ========================================================

    # Convert the URL into the DOMAIN/PATH/QUERY text format
    tfidf_text = prepare_tfidf_text(url)


    # ========================================================
    # CHARACTER TF-IDF
    # ========================================================

    # Character n-gram vectors capture sub-word patterns such as
    # suspicious fragments, typosquatting, random-looking strings.
    char_tfidf = char_vectorizer.transform(
        [tfidf_text]
    )


    # ========================================================
    # WORD TF-IDF
    # ========================================================

    # Word-level vectors capture meaningful tokens such as
    # "login", "secure", "verify", "account".
    word_tfidf = word_vectorizer.transform(
        [tfidf_text]
    )


    # ========================================================
    # COMBINE ALL FEATURES
    #
    # Training order:
    #
    # numeric
    # +
    # character TF-IDF
    # +
    # word TF-IDF
    # ========================================================

    # The TF-IDF outputs are sparse, so convert the dense numeric array
    # to sparse too, allowing everything to be stacked horizontally.
    sparse_numeric = csr_matrix(
        numeric_scaled
    )

    # Column order MUST be: numeric | char TF-IDF | word TF-IDF
    combined_features = hstack(
        [
            sparse_numeric,
            char_tfidf,
            word_tfidf,
        ],
        format="csr"
    )


    # ========================================================
    # FINAL DIMENSION CHECK
    # ========================================================

    # Last safeguard: confirm the column count matches what the
    # models were trained on before attempting a prediction.
    expected_features = (
        random_forest_model.n_features_in_
    )

    if combined_features.shape[1] != expected_features:

        raise ValueError(
            "Prediction feature dimension mismatch. "
            f"Model expects "
            f"{expected_features}, "
            f"but prediction pipeline produced "
            f"{combined_features.shape[1]}."
        )


    # Return the model-ready matrix and the readable feature dictionary
    return (
        combined_features,
        features,
    )


# ============================================================
# MODEL RESULT
# ============================================================

def get_model_result(
    model,
    features
):
    """
    Run a single model on the prepared features and return a
    standardised result dictionary (label, confidence, probabilities).

    All probability values in the returned dictionary are PERCENTAGES
    on a 0-100 scale, not fractions between 0 and 1.
    """

    # Probability of each class for the first (only) sample
    probabilities = model.predict_proba(
        features
    )[0]

    # Class labels in the same order as the probabilities
    classes = list(
        model.classes_
    )

    # Map class label -> probability, e.g. {0: 0.12, 1: 0.88}
    probability_map = {
        int(cls): float(prob)
        for cls, prob in zip(
            classes,
            probabilities
        )
    }


    # ========================================================
    # LABEL MAPPING
    #
    # 0 = Legitimate
    # 1 = Phishing
    # ========================================================

    # Convert probabilities to percentages (0-100).
    # .get(..., 0.0) guards against a missing class.
    legitimate_probability = (
        probability_map.get(
            0,
            0.0
        ) * 100
    )

    phishing_probability = (
        probability_map.get(
            1,
            0.0
        ) * 100
    )


    # The model's own predicted class (0 or 1)
    prediction = int(
        model.predict(
            features
        )[0]
    )


    # Pick the human-readable label, and use the probability of the
    # predicted class as the confidence value.
    if prediction == 1:

        label = "Phishing"

        confidence = phishing_probability

    else:

        label = "Legitimate"

        confidence = legitimate_probability


    # Return everything rounded to 2 decimal places for clean output
    return {

        "prediction":
            prediction,

        "label":
            label,

        "confidence":
            round(
                confidence,
                2
            ),

        "phishing_probability":
            round(
                phishing_probability,
                2
            ),

        "legitimate_probability":
            round(
                legitimate_probability,
                2
            ),
    }


# ============================================================
# FINAL URL PREDICTION
# ============================================================

def predict_url(
    url,
    perform_live_check=True
):
    """
    Main entry point: analyse a URL with all three models.

    Args:
        url: the URL string to analyse.
        perform_live_check: if True, also visit the site and attach the
            result as extra evidence (does not change the ML verdict).

    Returns:
        A dictionary with a default verdict (from the Random Forest),
        the per-model results, the optional live check, and the
        extracted features.

        IMPORTANT: app.py later replaces the top-level "prediction" and
        "label" with the majority vote of the three models. The top-level
        confidence and probability fields still come from the Random
        Forest unless app.py also updates them.
    """

    # Build the feature matrix once and reuse it for all three models
    (
        combined_features,
        features,
    ) = prepare_features(
        url
    )


    # ========================================================
    # MACHINE LEARNING
    # ========================================================

    # Run each model on the same features
    rf_result = get_model_result(
        random_forest_model,
        combined_features
    )

    lr_result = get_model_result(
        logistic_model,
        combined_features
    )

    nb_result = get_model_result(
        naive_bayes_model,
        combined_features
    )


    # ========================================================
    # MODEL RESULTS
    # ========================================================

    # Collect every model's output. app.py reads these three entries
    # to compute the majority-vote decision.
    models = {

        "random_forest":
            rf_result,

        "logistic_regression":
            lr_result,

        "naive_bayes":
            nb_result,
    }


    # ========================================================
    # DEFAULT VERDICT: RANDOM FOREST
    # ========================================================

    # Within this module, the Random Forest supplies the default
    # top-level verdict. This is NOT the final decision of the
    # application: app.py combines all three models by majority vote
    # and overwrites "prediction" and "label".
    final_prediction = (
        rf_result["prediction"]
    )

    final_label = (
        rf_result["label"]
    )

    final_confidence = (
        rf_result["confidence"]
    )


    # ========================================================
    # LIVE WEBSITE CHECK
    #
    # Secondary evidence only.
    # It does NOT override ML.
    # ========================================================

    live_result = None

    if perform_live_check:

        try:

            # Visit the URL and gather real-time security information
            live_result = check_live_url(
                url
            )

        except Exception as live_error:

            # A failing live check (timeout, DNS error, etc.) must never
            # break the prediction, so record the error and continue.
            live_result = {
                "error": str(
                    live_error
                )
            }


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        # Default verdict (from Random Forest; may be overridden
        # by the majority vote in app.py)
        "prediction":
            final_prediction,

        "label":
            final_label,

        "confidence":
            round(
                final_confidence,
                2
            ),

        # Class probabilities from the Random Forest (0-100 scale)
        "phishing_probability":
            round(
                rf_result[
                    "phishing_probability"
                ],
                2
            ),

        "legitimate_probability":
            round(
                rf_result[
                    "legitimate_probability"
                ],
                2
            ),

        # Individual results of all three models
        "models":
            models,

        # Live website analysis (or None / error dict)
        "live_check":
            live_result,

        # Number of numeric features and their raw values
        "feature_count":
            len(features),

        "features":
            features,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info():
    """
    Return static metadata about the loaded models and pipeline,
    useful for an "about" / health-check API endpoint.

    NOTE: this function is not currently called by app.py.
    """

    return {

        # Model whose result is used as the default top-level verdict
        # (the final application decision is a three-model majority vote)
        "primary_model":
            "Random Forest",

        # Other models reported in the "models" dictionary
        "additional_models": [
            "Logistic Regression",
            "Multinomial Naive Bayes",
        ],

        # Count of hand-crafted numeric features
        "numeric_feature_count":
            len(FEATURE_NAMES),

        # Count of all features (numeric + char TF-IDF + word TF-IDF)
        "total_feature_count":
            random_forest_model.n_features_in_,

        # Whether the live website check is supported
        "live_detection":
            True,

        "live_detection_role":
            "Secondary live webpage security check",

        # Meaning of the numeric class labels
        "label_mapping": {
            "0": "Legitimate",
            "1": "Phishing",
        },
    }