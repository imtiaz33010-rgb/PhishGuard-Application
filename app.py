"""
PhishGuard Flask web application.

Responsibilities of this module:
    1. Configure Flask (random per-start secret key, secure session cookies).
    2. Configure optional Google OAuth sign-in.
    3. Combine the three ML models' outputs with majority voting.
    4. Expose HTTP endpoints:
         /                    home page
         /inspect             inspection page
         /login, /logout      authentication pages
         /auth/google[...]    Google OAuth flow
         /predict             ML prediction + live website check
         /predict-ml          ML-only prediction
         /website-preview     safe screenshot of a website
         /health              service status
"""

import os
import base64
import ipaddress
import socket
from datetime import datetime, timezone
from urllib.parse import urlsplit

from dotenv import load_dotenv

# Read variables from the .env file into the environment.
# This must run before any code that reads os.environ.
load_dotenv()

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session,
)

# Authlib handles the OAuth 2.0 / OpenID Connect protocol with Google
from authlib.integrations.flask_client import OAuth
from authlib.integrations.base_client.errors import OAuthError

# Playwright drives a headless Chromium browser to take screenshots
from playwright.sync_api import sync_playwright

# Main prediction function (feature extraction + the three ML models)
from services.prediction_service import predict_url


# =========================================================
# FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)

# The secret key signs the session cookie so it cannot be forged.
# A random key is generated every time the app starts, so no key is
# stored in the project or in the .env file. Consequence: all users are
# signed out whenever the server restarts, and the app must run as a
# single process (multiple workers would each get a different key).
app.secret_key = os.urandom(32)

# Session cookie hardening:
# HttpOnly -> JavaScript cannot read the cookie (limits XSS damage)
app.config["SESSION_COOKIE_HTTPONLY"] = True
# SameSite=Lax -> cookie is not sent on most cross-site requests (CSRF defence)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# Secure -> cookie only sent over HTTPS; enable via COOKIE_SECURE=true
app.config["SESSION_COOKIE_SECURE"] = (
    os.environ.get("COOKIE_SECURE", "false").lower() == "true"
)
# Permanent sessions expire after 3600 seconds (1 hour)
app.config["PERMANENT_SESSION_LIFETIME"] = 3600


# =========================================================
# GOOGLE OAUTH CONFIGURATION
# =========================================================

oauth = OAuth(app)

# Credentials come from environment variables, never hard-coded
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")

# Stays None when credentials are missing, so the rest of the app
# can check "google is None" to know whether login is available.
google = None

if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    google = oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        # Google's discovery document: Authlib reads endpoints and keys from it
        server_metadata_url=(
            "https://accounts.google.com/.well-known/"
            "openid-configuration"
        ),
        client_kwargs={
            # Request the user's identity, email and basic profile
            "scope": "openid email profile",
        },
    )

    app.logger.info("Google OAuth credentials are configured.")

else:
    app.logger.warning(
        "Google OAuth is not configured. Check GOOGLE_CLIENT_ID "
        "and GOOGLE_CLIENT_SECRET in your .env file."
    )


# =========================================================
# MODEL DECISION POLICY
# 0 = Legitimate
# 1 = Phishing
# =========================================================

# The three models that must be present in the prediction result.
MODEL_KEYS = (
    "random_forest",
    "logistic_regression",
    "naive_bayes",
)


def get_model_prediction(model_result):
    """Return 0 for Legitimate, 1 for Phishing, or None."""

    # The model result must be a dictionary
    if not isinstance(model_result, dict):
        return None

    # 1st choice: interpret the text label (accepts a few synonyms)
    label = model_result.get("label")

    if isinstance(label, str):
        normalized = label.strip().lower()

        if normalized in ("legitimate", "legit", "safe"):
            return 0

        if normalized in ("phishing", "phish", "malicious"):
            return 1

    # 2nd choice: use the numeric prediction (0 or 1).
    # bool is excluded because True/False are technically ints in Python.
    raw_prediction = model_result.get("prediction")

    if (
        isinstance(raw_prediction, int)
        and not isinstance(raw_prediction, bool)
        and raw_prediction in (0, 1)
    ):
        return raw_prediction

    # 3rd choice: compare the two class probabilities.
    # NOTE: this fallback expects values on a 0-1 scale.
    try:
        legitimate_probability = float(
            model_result["legitimate_probability"]
        )
        phishing_probability = float(
            model_result["phishing_probability"]
        )

        # Values outside 0-1 are considered invalid
        if not (
            0 <= legitimate_probability <= 1
            and 0 <= phishing_probability <= 1
        ):
            return None

        # The higher probability wins
        return (
            1
            if phishing_probability > legitimate_probability
            else 0
        )

    except (KeyError, TypeError, ValueError):
        # Missing or non-numeric fields -> cannot decide
        return None


def apply_ensemble_decision(result):
    """
    Combine the three model outputs using majority voting.

    0 = Legitimate
    1 = Phishing
    None = Inconclusive
    """

    # Validate the structure returned by the prediction service
    if not isinstance(result, dict):
        raise ValueError(
            "The prediction service did not return a valid object."
        )

    models = result.get("models")

    if not isinstance(models, dict):
        raise ValueError(
            "The prediction service must return a 'models' dictionary."
        )

    # Collect each model's vote: {"random_forest": 1, ...}
    individual_predictions = {}

    for model_key in MODEL_KEYS:
        if model_key not in models:
            raise ValueError(
                f"Missing model result: {model_key}"
            )

        prediction = get_model_prediction(models[model_key])

        if prediction is None:
            raise ValueError(
                f"Could not determine the prediction for {model_key}. "
                "Check its label mapping and probability fields."
            )

        individual_predictions[model_key] = prediction

    # Count the votes for each class
    phishing_votes = sum(
        value == 1
        for value in individual_predictions.values()
    )

    legitimate_votes = sum(
        value == 0
        for value in individual_predictions.values()
    )

    total_votes = len(individual_predictions)

    # -----------------------------------------------------
    # MAJORITY-VOTING DECISION
    # -----------------------------------------------------

    if phishing_votes > legitimate_votes:
        final_prediction = 1
        final_label = "Phishing"

        message = (
            f"Classified as phishing by majority vote "
            f"({phishing_votes} of {total_votes} models)."
        )

    elif legitimate_votes > phishing_votes:
        final_prediction = 0
        final_label = "Legitimate"

        message = (
            f"Classified as legitimate by majority vote "
            f"({legitimate_votes} of {total_votes} models)."
        )

    else:
        # Equal votes (only possible if the number of models is even,
        # or if the voting rule is changed later)
        final_prediction = None
        final_label = "Suspicious / Inconclusive"

        message = (
            "The models are tied. Verify the URL before proceeding."
        )

    # -----------------------------------------------------
    # PRESERVE INDIVIDUAL MODEL RESULTS
    # -----------------------------------------------------

    # Overwrite the top-level verdict with the ensemble result.
    # The per-model details remain untouched inside result["models"].
    result["prediction"] = final_prediction
    result["label"] = final_label
    result["final_prediction"] = final_prediction
    result["final_label"] = final_label
    result["message"] = message

    # Human-readable explanation of the rule, shown in the UI
    result["decision_rule"] = (
        "Majority voting: the classification supported by at least "
        "two of the three models is selected."
    )

    # Each model's vote as a readable label
    result["individual_predictions"] = {
        key: "Phishing" if value == 1 else "Legitimate"
        for key, value in individual_predictions.items()
    }

    # Vote tally
    result["vote_counts"] = {
        "phishing": phishing_votes,
        "legitimate": legitimate_votes,
        "total": total_votes,
    }

    return result


# =========================================================
# LOGIN PAGE
# =========================================================

@app.route("/login")
def login():

    # Already signed in -> no need to show the login page
    if session.get("user"):
        return redirect(url_for("home"))

    return render_template("login.html")


# =========================================================
# GOOGLE LOGIN
# =========================================================

@app.route("/auth/google")
def auth_google():
    """Redirect the user to Google's authentication page."""

    # Already signed in -> go home
    if session.get("user"):
        return redirect(url_for("home"))

    # OAuth credentials missing -> 503 Service Unavailable
    if google is None:
        app.logger.error(
            "Google login requested without OAuth credentials."
        )

        return (
            "Google login is not configured. Check your .env file.",
            503,
        )

    try:
        # Absolute URL that Google will send the user back to.
        # It must match the redirect URI registered in Google Cloud Console.
        redirect_uri = url_for(
            "auth_google_callback",
            _external=True,
        )

        app.logger.info(
            "Starting Google OAuth. Redirect URI: %s",
            redirect_uri,
        )

        # Sends the browser to Google (also stores a "state" value in the
        # session to protect against CSRF attacks)
        return google.authorize_redirect(redirect_uri)

    except Exception:
        # Full details go to the server log; the user sees a generic message
        app.logger.exception(
            "Unable to start Google OAuth authorization."
        )

        return (
            "Unable to start Google login. "
            "Check the Flask terminal for details.",
            500,
        )


# =========================================================
# GOOGLE OAUTH CALLBACK
# =========================================================

@app.route("/auth/google/callback")
def auth_google_callback():
    """Validate Google's response and establish the user session."""

    if google is None:
        app.logger.error(
            "Google OAuth callback received but OAuth is not configured."
        )

        return "Google login is not configured.", 503

    # The user pressed Deny, or Google returned an error.
    # This is an expected case, not a server fault.
    if request.args.get("error"):
        app.logger.info(
            "Google OAuth callback: user denied access or Google "
            "returned an error."
        )

        return redirect(url_for("login"))

    try:
        # Logging server UTC time helps diagnose token timing errors
        # (clock skew causes "token used too early/late" failures)
        app.logger.info(
            "Google OAuth callback started. Server UTC time: %s",
            datetime.now(timezone.utc).isoformat(),
        )

        # Exchange the authorization code for tokens and verify the
        # state and ID token signature.
        token = google.authorize_access_token()

        # Preferred: user info is already inside the ID token
        user_info = token.get("userinfo")

        # Otherwise request it from Google's userinfo endpoint
        if not user_info:
            user_info = google.userinfo()

        if not user_info:
            app.logger.error(
                "Google did not return user information."
            )

            return (
                "Unable to retrieve your Google account information.",
                400,
            )

        # Only accept accounts whose email Google has verified
        email_verified = user_info.get("email_verified", False)

        if email_verified not in (True, "true", 1):
            app.logger.warning(
                "Google login rejected because email verification "
                "was not confirmed."
            )

            return (
                "Please sign in with a verified Google email address.",
                403,
            )

        email = user_info.get("email")

        if not email:
            app.logger.error(
                "Google user information did not include an email."
            )

            return (
                "Google did not provide an email address.",
                400,
            )

        # Start from a clean session (prevents session fixation),
        # then mark it as permanent so the 1-hour lifetime applies.
        session.clear()
        session.permanent = True

        # Store only the minimum data needed by the interface
        session["user"] = {
            "email": email,
            # Fall back to the part of the email before "@" if no name
            "name": user_info.get("name") or email.split("@")[0],
            "picture": user_info.get("picture", ""),
        }

        app.logger.info("Google OAuth login succeeded.")

        return redirect(url_for("home"))

    except OAuthError as exc:
        # Forged or missing state, bad or missing code, and similar
        # problems with the request itself. These are client errors.
        app.logger.warning(
            "Google OAuth callback rejected. Exception type: %s",
            type(exc).__name__,
        )

        return "Sign-in could not be verified. Please try again.", 400

    except Exception as exc:
        # A real unexpected fault. Details go to the log only.
        app.logger.exception(
            "Google OAuth login failed. Exception type: %s",
            type(exc).__name__,
        )

        return "Google login failed. Please try again later.", 500


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():
    """Clear the user's session and return to the homepage."""

    session.clear()

    return redirect(url_for("home"))


# =========================================================
# URL VALIDATION FOR WEBSITE PREVIEW
# =========================================================

def is_public_http_url(url):
    """Allow HTTP(S) URLs resolving exclusively to public IP addresses."""

    # This is an SSRF (Server-Side Request Forgery) defence: it stops
    # users from making the server open internal addresses such as
    # localhost, 192.168.x.x or cloud metadata endpoints.

    try:
        parsed = urlsplit(url)

        # Only web schemes are allowed (blocks file://, ftp://, etc.)
        if parsed.scheme.lower() not in ("http", "https"):
            return False

        # A hostname is required
        if not parsed.hostname:
            return False

        # Reject credentials embedded in the URL (user:pass@host)
        if parsed.username or parsed.password:
            return False

        host = parsed.hostname.lower()

        # Block localhost and any *.localhost name
        if host == "localhost" or host.endswith(".localhost"):
            return False

        # If the host is a literal IP address, check it directly.
        # is_global is False for private, loopback, link-local, etc.
        try:
            literal_ip = ipaddress.ip_address(host)
            return literal_ip.is_global

        except ValueError:
            # Not an IP literal -> it is a domain name; resolve it below
            pass

        # Use the explicit port, or the default for the scheme
        port = parsed.port or (
            443 if parsed.scheme.lower() == "https" else 80
        )

        # Resolve the domain to all its IP addresses (IPv4 and IPv6)
        addresses = socket.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
        )

        if not addresses:
            return False

        # EVERY resolved address must be public; one private address
        # is enough to reject the URL (blocks DNS tricks).
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])

            if not ip.is_global:
                return False

        return True

    except (ValueError, OSError):
        # Malformed URL/port or DNS failure -> treat as not allowed
        return False


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# INSPECTION PAGE
# =========================================================

@app.route("/inspect")
def inspect():
    return render_template("inspect.html")


# =========================================================
# SHARED URL INPUT VALIDATION
# =========================================================

# Upper bound on URL length to prevent abuse and heavy processing
MAX_PREDICTION_URL_LENGTH = 2048


def get_validated_prediction_url():
    """Extract, length-check, and validate a URL from a JSON request."""

    # Returns a tuple: (clean_url, None) on success,
    # or (None, (json_response, status_code)) on failure.

    # silent=True returns None instead of raising on invalid JSON
    data = request.get_json(silent=True) or {}
    url = data.get("url", "")

    # Must be a non-empty string
    if not isinstance(url, str) or not url.strip():
        return None, (
            jsonify({"error": "Please enter a website URL."}),
            400,
        )

    url = url.strip()

    # Reject oversized input BEFORE parsing, feature extraction, or prediction.
    if len(url) > MAX_PREDICTION_URL_LENGTH:
        return None, (
            jsonify({
                "error": (
                    f"URL exceeds the maximum allowed length of "
                    f"{MAX_PREDICTION_URL_LENGTH} characters."
                )
            }),
            400,
        )

    try:
        parsed = urlsplit(url)

        # If the user typed "example.com", assume HTTPS
        if not parsed.scheme:
            url = "https://" + url
            parsed = urlsplit(url)

        # Check again after normalization because adding a scheme changes length.
        if len(url) > MAX_PREDICTION_URL_LENGTH:
            return None, (
                jsonify({
                    "error": (
                        f"URL exceeds the maximum allowed length of "
                        f"{MAX_PREDICTION_URL_LENGTH} characters."
                    )
                }),
                400,
            )

        # Only web URLs are supported
        if parsed.scheme.lower() not in ("http", "https"):
            return None, (
                jsonify({
                    "error": (
                        "Only HTTP and HTTPS website URLs are supported."
                    )
                }),
                400,
            )

        # Reject malformed host/port combinations early.
        if not parsed.hostname:
            return None, (
                jsonify({"error": "Please enter a valid website URL."}),
                400,
            )

        # Accessing .port raises ValueError if the port is invalid
        # (e.g. non-numeric or out of range); the value itself is unused.
        _ = parsed.port

    except ValueError:
        return None, (
            jsonify({"error": "Please enter a valid website URL."}),
            400,
        )

    return url, None


# =========================================================
# MAIN PREDICTION ENDPOINT
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():
    """Full analysis: ML models plus the live website check."""

    try:
        # Validate and normalise the submitted URL
        url, error_response = get_validated_prediction_url()

        if error_response:
            return error_response

        # Run the ML models and the live check
        result = predict_url(
            url,
            perform_live_check=True,
        )

        # Replace the single-model verdict with the majority vote
        result = apply_ensemble_decision(result)

        return jsonify(result)

    except ValueError as exc:
        # Raised when the model output is malformed (server-side problem)
        app.logger.exception(
            "Prediction response or model result invalid."
        )

        return jsonify({
            "error": str(exc)
        }), 500

    except Exception:
        # Any other unexpected failure; hide details from the client
        app.logger.exception("Prediction failed.")

        return jsonify({
            "error": (
                "Prediction failed. Check the Flask console for details."
            )
        }), 500


# =========================================================
# ML-ONLY PREDICTION ENDPOINT
# =========================================================

@app.route("/predict-ml", methods=["POST"])
def predict_ml():
    """Same as /predict but skips the (slower) live website check."""

    try:
        url, error_response = get_validated_prediction_url()

        if error_response:
            return error_response

        # perform_live_check=False -> no network request to the target site
        result = predict_url(
            url,
            perform_live_check=False,
        )

        result = apply_ensemble_decision(result)

        return jsonify(result)

    except ValueError as exc:
        app.logger.exception(
            "ML prediction response or model result invalid."
        )

        return jsonify({
            "error": str(exc)
        }), 500

    except Exception:
        app.logger.exception("ML prediction failed.")

        return jsonify({
            "error": (
                "ML prediction failed. Check the Flask console for details."
            )
        }), 500


# =========================================================
# WEBSITE SCREENSHOT ENDPOINT
# =========================================================

@app.route("/website-preview", methods=["POST"])
def website_preview():
    """Capture a screenshot separately from the ML prediction."""

    try:
        url, error_response = get_validated_prediction_url()

        if error_response:
            return error_response

        # SSRF check before the server visits the URL
        if not is_public_http_url(url):
            return jsonify({
                "error": (
                    "This URL is invalid or its destination "
                    "is not publicly accessible."
                )
            }), 400

        # Start Playwright and a headless (invisible) Chromium browser
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)

            try:
                # Isolated browser context with a desktop-size viewport
                context = browser.new_context(
                    viewport={
                        "width": 1365,
                        "height": 900,
                    },
                    device_scale_factor=1,
                    # Refuse sites with invalid SSL certificates
                    ignore_https_errors=False,
                )

                page = context.new_page()

                def guard_request(route):
                    """Check every request the page makes, not just the first."""
                    target = route.request.url

                    # Allow only public destinations; this also covers
                    # redirects and sub-resources (images, scripts, iframes)
                    if is_public_http_url(target):
                        route.continue_()
                    else:
                        route.abort()

                # Intercept ALL requests made by the page
                page.route("**/*", guard_request)

                # Load the page (20 s timeout), stopping at DOM ready
                response = page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=20000,
                )

                # Short pause so images and scripts can render
                page.wait_for_timeout(1200)

                # URL after any redirects
                final_url = page.url

                # Re-validate the final destination as an extra safeguard
                if not is_public_http_url(final_url):
                    return jsonify({
                        "error": (
                            "The website redirected to a blocked "
                            "destination."
                        )
                    }), 400

                # Capture only the visible area as a compressed JPEG
                screenshot_bytes = page.screenshot(
                    type="jpeg",
                    quality=75,
                    full_page=False,
                    timeout=10000,
                )

                return jsonify({
                    "success": True,
                    "requested_url": url,
                    "final_url": final_url,
                    "domain": urlsplit(final_url).hostname or "",
                    "page_title": page.title() or "Title unavailable",
                    "http_status": (
                        response.status if response else None
                    ),
                    # Treat anything below 500 as "reachable"
                    # (even 404 means a server answered)
                    "reachable": (
                        response is not None and response.status < 500
                    ),
                    # Embed the image as a data URI so the frontend can
                    # place it directly in an <img> tag
                    "screenshot": (
                        "data:image/jpeg;base64,"
                        + base64.b64encode(
                            screenshot_bytes
                        ).decode("ascii")
                    ),
                })

            finally:
                # Always close the browser, even after errors,
                # to avoid leaking processes
                browser.close()

    except Exception:
        # Timeouts, blocked automation, crashes, etc.
        app.logger.exception(
            "Website screenshot capture failed."
        )

        # 502 Bad Gateway: the upstream website could not be captured
        return jsonify({
            "error": (
                "A screenshot could not be captured. The website may "
                "block automated browsers or take too long to load."
            )
        }), 502


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():
    """Simple status endpoint for monitoring and quick checks."""

    return jsonify({
        "status": "running",
        "models": [
            "Random Forest",
            "Logistic Regression",
            "Multinomial Naive Bayes",
        ],
        "decision_policy": (
            "Majority voting: at least two of three models determine "
            "the final classification."
        ),
        "live_detection": True,
        "website_preview": True,
        # True only if Google credentials were found at startup
        "google_login_configured": google is not None,
    })


# =========================================================
# RUN APPLICATION
# =========================================================

# Runs only when the file is executed directly (python app.py),
# not when it is imported by another module or a WSGI server.
if __name__ == "__main__":

    print("=" * 60)
    print("PHISHGUARD")
    print("Phishing Website Detection System")
    print("Three-model majority-voting decision policy enabled")
    print("At least two matching predictions determine the result")
    print(f"Google login configured: {google is not None}")
    print("=" * 60)

    app.run(
        host="127.0.0.1",   # Listen on localhost only (not exposed to the network)
        port=5000,
        debug=False,        # Debug mode off: safer, no interactive debugger
    )