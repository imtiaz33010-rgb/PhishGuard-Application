# PhishGuard — Phishing Website Detection System

PhishGuard is a Flask web application that checks whether a URL is phishing or legitimate.

- **Three machine-learning models** analyse the URL: Random Forest, Logistic Regression and Multinomial Naive Bayes, using hand-crafted features plus character-level and word-level TF-IDF.
- **Majority voting** decides the final result (at least 2 of 3 models must agree).
- **Live website check** adds secondary evidence by inspecting the real site.
- **Safe website preview** takes a screenshot with a headless browser, protected against SSRF.
- **Optional Google sign-in** (OAuth 2.0).

Labels used everywhere: `0 = Legitimate`, `1 = Phishing`.

---

## 1. Project structure

```
phishguard/
├── app.py                          # Flask application (run this)
├── train_models.py                 # Trains the three models
├── generate_evaluation_charts.py   # Confusion matrices, ROC curves
├── ssrf_check.py                   # Error analysis of test predictions
├── security_tests.py               # 49-case security test suite
├── requirements.txt
├── .env                            # Your secrets (you create this)
├── services/
│   ├── feature_extractor.py
│   ├── prediction_service.py
│   └── live_url_checker.py
├── templates/
│   ├── index.html
│   ├── inspect.html
│   └── login.html
├── dataset/
│   ├── Phishing_Legitimate_balanced.csv   # columns: url, type
│   └── balanced_urls.csv                  # columns: url, label
└── models/                         # trained models and vectorizers
```

The code imports `services.prediction_service` and `services.feature_extractor`, so the three service files must sit inside the `services/` folder. Flask also requires the `templates/` folder.

---

## 2. Requirements

- Python 3.10 or newer (developed with Python 3.13)
- About 1 GB of free disk space (Chromium for screenshots, plus models)
- Internet access (for installing packages, the live check, screenshots and Google sign-in)

---

## 3. Installation

### Step 1 — Open the project folder

```bash
cd phishguard
```

### Step 2 — Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
cmd
python -m venv venv
venv\Scripts\activate.bat


**macOS / Linux:**
bash
python3 -m venv venv
source venv/bin/activate


### Step 3 — Install dependencies

bash
pip install -r requirements.txt


The main packages are Flask, python-dotenv, Authlib, requests, Playwright, NumPy, SciPy, pandas, scikit-learn 1.6.1, joblib, beautifulsoup4, tldextract and dnspython.

The saved models must be loaded with the same scikit-learn version that created them, which is why `requirements.txt` pins `scikit-learn==1.6.1`.

If you also want to retrain the models or generate the evaluation charts (Sections 5 and 6), install the plotting libraries:

bash
pip install matplotlib seaborn


### Step 4 — Install the headless browser (for website screenshots)

bash
playwright install chromium


On Linux you may also need system libraries: `playwright install-deps chromium`.

### Step 5 — Create the `.env` file (required)

The application reads its Google sign-in settings from a `.env` file. Create a file named `.env` in the project root (the same folder as `app.py`):

env
# Required for "Sign in with Google"
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# Optional: set to true only when serving over HTTPS
COOKIE_SECURE=false


No session secret key is needed. The app generates a random one each time it starts, so users are signed out whenever the server restarts. Google credentials come from your own Google Cloud project (see "Setting up Google sign-in" in Section 4).

### Step 6 — Check the trained models

The `models/` folder must be in the project root and contain the trained classifiers, the two TF-IDF vectorizers, the feature scaler and the feature-name list:


models/
├── random_forest_tfidf_model.joblib
├── logistic_regression_tfidf_model.joblib
├── naive_bayes_tfidf_model.joblib
├── char_tfidf_vectorizer.joblib
├── word_tfidf_vectorizer.joblib
├── feature_scaler.joblib
└── feature_names.joblib


---

## 4. Running the application

bash
python app.py

You should see the PhishGuard banner in the terminal. Then open:

**http://127.0.0.1:5000**

| Page / Endpoint         | Method | Purpose                                |
|-------------------------|--------|----------------------------------------|
| `/`                     | GET    | Home page                              |
| `/inspect`              | GET    | URL inspection page                    |
| `/login`, `/logout`     | GET    | Sign-in page / sign out                |
| `/auth/google`          | GET    | Start Google sign-in                   |
| `/auth/google/callback` | GET    | Google OAuth callback                  |
| `/predict`              | POST   | ML prediction + live website check     |
| `/predict-ml`           | POST   | ML prediction only (faster)            |
| `/website-preview`      | POST   | Screenshot of a public website         |
| `/health`               | GET    | Status check                           |

The server listens on `127.0.0.1:5000` only (not exposed to your network) and runs with debug mode off.

### Quick test from the command line

**Check the server is running:**
bash
curl http://127.0.0.1:5000/health


**Analyse a URL:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d "{\"url\": \"https://example.com\"}"
```

(On Windows PowerShell, use `curl.exe` instead of `curl`.)

### Setting up Google sign-in

To get the Google credentials for `.env`:

1. Open the [Google Cloud Console](https://console.cloud.google.com/) and go to **APIs & Services → Credentials**.
2. Create an **OAuth client ID** (type: *Web application*).
3. Add this **Authorized redirect URI** exactly:
   ```
   http://127.0.0.1:5000/auth/google/callback
   ```
4. Copy the client ID and secret into `.env`, then restart the app.

Open the app using `127.0.0.1`, not `localhost`, so the redirect URI matches. When sign-in is configured, `/health` shows `"google_login_configured": true`.

---

## 5. Training the models (optional)

Only needed if you want to reproduce the training process.

1. Place both CSV files in `dataset/`:
   - `Phishing_Legitimate_balanced.csv` with columns `url` and `type`
   - `balanced_urls.csv` with columns `url` and `label`
2. Run:
   ```bash
   python train_models.py
   ```

The script:

1. Loads and cleans both datasets and normalises the labels.
2. Splits the data by registrable domain, so the same domain never appears in both the training and test sets.
3. Extracts numeric URL features and fits character-level and word-level TF-IDF vectorizers.
4. Trains Random Forest, Logistic Regression and Multinomial Naive Bayes.
5. Saves the models, vectorizers, scaler, results and test predictions to `models/`, and detailed reports to `models/reports/`.

---

## 6. Evaluation and analysis scripts (optional)

These read `models/test_predictions.csv`, which is produced during training. Run them from the project root.

**Evaluation charts** (confusion matrices, ROC curves, accuracy comparison):

```bash
python generate_evaluation_charts.py
```

Set `PROJECT_DIR` near the top of the script to the location of your project folder. Output is saved to `evaluation_results/`.

**Error analysis** (majority-vote metrics, model disagreement, error rates by URL characteristics, precision at realistic phishing rates):

```bash
python ssrf_check.py
```

This also writes `error_samples.csv` with sample false positives and false negatives for manual review.

---

## 7. Security test suite (optional)

A 49-case suite that checks input validation, SSRF protection, the OAuth callback and error handling.

1. Start the app in one terminal:
   ```bash
   python app.py
   ```
2. In a second terminal (same virtual environment), run:
   ```bash
   python security_tests.py
   ```

Notes:

- It temporarily opens a listener on **port 8000**, so make sure that port is free.
- Cases S15 and S16 need internet access (they use httpbin.org and httpbingo.org).
- Results are saved to `security_results.csv`, and any failed cases are printed at the end.

---

## 8. How it works

1. The URL is validated (HTTP/HTTPS only, maximum 2048 characters).
2. `feature_extractor.py` computes hand-crafted numeric features.
3. Character-level and word-level TF-IDF vectors are built from the URL.
4. All features are combined and passed to the three models.
5. `app.py` applies **majority voting** to produce the final verdict: *Phishing*, *Legitimate* or *Suspicious / Inconclusive*.
6. For `/predict`, `live_url_checker.py` adds a secondary live check of the website.

Security measures included:

- SSRF protection on `/website-preview` (blocks private, loopback and link-local addresses, credentials in URLs and unsafe redirects; every browser request is re-checked)
- HttpOnly, SameSite=Lax session cookies (optionally Secure), signed with a random key generated at startup
- Sessions expire after 1 hour
- Debug mode disabled

---

## 9. Production notes

`python app.py` uses Flask's development server and is meant for local use. For deployment:

- Set `COOKIE_SECURE=true` (serve over HTTPS).
- Run behind a production WSGI server and reverse proxy, using a single worker, for example `gunicorn -w 1 app:app`. The session key is generated per process, so multiple workers would break Google sign-in.
- Update the Google OAuth redirect URI to your real HTTPS domain.
- Keep `.env` out of version control (add it to `.gitignore`).
