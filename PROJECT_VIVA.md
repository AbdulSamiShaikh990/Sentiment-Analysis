# Employee Review Rating Predictor — Viva Guide

## Project Summary
- **Purpose**: A lightweight ML system that predicts an employee review’s overall rating (1–5) from free‑text fields (`summary`, `pros`, `cons`).
- **End users**: HR analysts, product teams, or students who want quick sentiment/quality signals from textual reviews.
- **Tech stack**: Streamlit UI, scikit‑learn (Logistic Regression), TF‑IDF features, Python (pandas/numpy).
- **Data**: Trains locally from `employee-review-train.csv` (root) or `data/employee_reviews.csv`. Missing text is handled safely.
- **Pipeline**: Clean text → TF‑IDF vectorization (max 5000 features, English stop‑words) → Multiclass Logistic Regression → Cached for fast prediction.
- **Usage modes**: Single prediction via form inputs; Batch prediction by uploading a CSV with `summary, pros, cons` (and optional `id`).
- **Outputs**: Predicted class 1–5; single mode can show class probabilities.
- **Why this approach**: Fast, interpretable baseline that works well on short reviews and small/medium datasets; simple to demo and discuss in a viva.
- **What it is not**: It’s not a deep language model; it prioritizes clarity, speed, and reproducibility over state‑of‑the‑art semantics.

## 1) What is this project?
- **Goal**: Predict the overall employee review rating (1–5) from free‑text review fields: `summary`, `pros`, and `cons`.
- **Type**: End‑to‑end ML + UI app built with **Streamlit** and **scikit‑learn**.
- **Inputs**:
  - Single prediction: text fields from the UI.
  - Batch prediction: CSV with columns `id (optional)`, `summary`, `pros`, `cons`.
- **Output**: A predicted `overall-ratings` class (1–5). In single mode, class probabilities are also shown (if available from the model).

## 2) Does it use a pre‑trained model or train now?
- **It trains on startup (first run)** using local data, then caches the trained artifacts in memory for the session using `@st.cache_resource`.
- Training data is loaded from the first available source:
  1. `employee-review-train.csv` in project root, or
  2. `data/employee_reviews.csv` (then the app selects and sanitizes required columns).
- No model file is stored on disk by default; training happens in memory and is fast for this dataset size.

## 3) How does it work (pipeline)?
1. **Load Data**: Reads CSV and ensures the columns `summary`, `pros`, `cons` exist (fills missing with empty strings). Uses `overall-ratings` as the target.
2. **Text Cleaning**: `clean_text()` performs:
   - Remove user mentions: `@username` → space
   - Keep letters only: non‑alphabetic replaced with spaces
   - Collapse extra whitespace and lowercase
3. **Feature Engineering (TF‑IDF)**:
   - Concatenate `summary + pros + cons` into one text string per row.
   - `TfidfVectorizer(max_features=5000, stop_words='english')` transforms text into numeric vectors.
4. **Model**: `LogisticRegression(max_iter=2000)` is trained on the TF‑IDF features to predict the `overall-ratings` class (1–5).
5. **Caching**: The vectorizer and model are cached via `@st.cache_resource` so subsequent predictions don’t retrain.
6. **Prediction**:
   - Single: Clean and vectorize input text, then `model.predict()`; optionally show `predict_proba`.
   - Batch: Read uploaded CSV, clean and vectorize each row, return predicted ratings, and allow a download of results.

## 4) Why these choices?
- **TF‑IDF**: Simple and strong baseline for text classification; fast, interpretable, and works well with limited data.
- **Logistic Regression**: Efficient for multi‑class classification, robust to high‑dimensional sparse features, quick to train.
- **Max features = 5000**: Controls dimensionality to balance accuracy and speed.
- **English stop‑words**: Removes common high‑frequency words that add little signal.
- **Streamlit UI**: Rapid prototyping, easy to demo single and batch predictions.

## 5) Project files overview
- `app.py`: Streamlit app. Contains data loading, cleaning, training, and prediction logic.
- `employee-review-train.csv`: Default training dataset (root).
- `data/employee_reviews.csv`: Alternate dataset location and schema (app maps to needed columns).
- `requirements-ui.txt`: Minimal dependencies (streamlit, scikit‑learn, pandas, numpy).
- `sample_reviews.csv`: Example batch file with 5 rows.

## 6) How to run
- Create/activate venv (optional but recommended) and install from `requirements-ui.txt`.
- Run: `python -m streamlit run app.py`
- Browser opens to the app. First run trains and caches the model.

## 7) Inputs/Outputs details
- **Training target**: `overall-ratings` must be integers 1–5.
- **Single Prediction**: Any of `summary`, `pros`, `cons` can be blank; text is concatenated and cleaned.
- **Batch Prediction**: CSV columns required: `summary`, `pros`, `cons`. `id` is optional and if present, it is passed through to outputs.
- **Outputs**: Predicted integer in [1..5]. In single mode, a probability distribution over classes may be shown if available.

## 8) Key code snippets (conceptual)
- Vectorization:
  - `vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')`
- Model:
  - `model = LogisticRegression(max_iter=2000)`
- Caching:
  - `@st.cache_resource(show_spinner=False)` on `train_model()`

## 9) Assumptions & constraints
- Text language is primarily English (due to English stop‑word list and cleaning).
- Ratings are discrete classes 1–5 (classification, not regression).
- Data quality: Missing text is handled; non‑textual tokens are stripped.
- This is a lightweight demo; no persistence of trained model between app restarts.

## 10) Evaluation (high‑level)
- The app currently focuses on deployability and interactivity rather than offline evaluation.
- You can add a quick split and report metrics via `train_test_split` and `classification_report` if needed for your viva.

## 11) Limitations
- TF‑IDF + Logistic Regression doesn’t capture deep semantics, sarcasm, or long‑range context.
- No hyperparameter search; defaults chosen for speed and simplicity.
- No class imbalance handling if the dataset is skewed.
- Model isn’t persisted to disk; retrains on cold start.

## 12) Possible improvements
- Persist model/vectorizer with `joblib` and load instead of retraining each run.
- Add proper train/validation split with metrics display.
- Try n‑grams, class weights, or calibration for better probabilities.
- Experiment with modern NLP (e.g., sentence transformers, small LMs) if compute permits.
- Add simple explainability (top tokens by class weights).

## 13) Data privacy & ethics
- Keep datasets local; do not upload sensitive employee data.
- Remove personally identifiable information (PII) if present.
- Be transparent: predictions are probabilistic and may be biased by training data.

## 14) Troubleshooting
- "No module named streamlit": install deps or ensure venv is active.
- App cannot find training data: place `employee-review-train.csv` in root or `data/employee_reviews.csv` with the required columns (`overall-ratings`, `summary`, `pros`, `cons`).
- Long first run: that’s training; subsequent runs are cached.

## 15) Viva Q&A (quick bullets)
- **Q: Pre‑trained or trained now?**
  - A: Trained at runtime from local CSV, cached via Streamlit. No pre‑saved model file.
- **Q: Why TF‑IDF + Logistic Regression?**
  - A: Fast, interpretable baseline that performs well on short texts; minimal compute.
- **Q: How do you handle missing fields?**
  - A: Fill with empty strings; concatenate and clean uniformly.
- **Q: What preprocessing?**
  - A: Mention/emoji removal (non‑letters), whitespace normalization, lowercase.
- **Q: What features?**
  - A: Unigram TF‑IDF up to 5000 features with English stop‑words.
- **Q: Multi‑class handling?**
  - A: scikit‑learn’s multinomial logistic regression handles 1–5 classes natively.
- **Q: Limitations?**
  - A: Shallow model, no persistence, no imbalance handling; scope is a functional demo.
- **Q: How to extend?**
  - A: Persist models with joblib, add metrics, tune hyperparameters, try better embeddings.

---
This document summarizes what the app is, how it works, and why design choices were made, aimed at viva preparation. Use `sample_reviews.csv` for quick demos of the batch predictor.
