import os
import re
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def load_train_data():
    if os.path.exists('employee-review-train.csv'):
        df = pd.read_csv('employee-review-train.csv')
    elif os.path.exists(os.path.join('data', 'employee_reviews.csv')):
        df = pd.read_csv(os.path.join('data', 'employee_reviews.csv'))
        if 'overall-ratings' not in df.columns:
            st.error('overall-ratings column not found in data/employee_reviews.csv')
            st.stop()
        for col in ['summary', 'pros', 'cons']:
            if col not in df.columns:
                df[col] = ''
        df = df[['overall-ratings', 'summary', 'pros', 'cons']]
    else:
        st.error('Training data not found. Place employee-review-train.csv in the project root or data/employee_reviews.csv.')
        st.stop()
    for col in ['summary', 'pros', 'cons']:
        if col not in df.columns:
            df[col] = ''
    df[['summary', 'pros', 'cons']] = df[['summary', 'pros', 'cons']].fillna('')
    return df


def clean_text(s: str) -> str:
    s = re.sub(r"@[\w]*", " ", s)
    s = re.sub(r"[^a-zA-Z#]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


@st.cache_resource(show_spinner=False)
def train_model():
    df = load_train_data()
    df['combined'] = (df['summary'].astype(str) + ' ' + df['pros'].astype(str) + ' ' + df['cons'].astype(str)).str.strip()
    df['combined'] = df['combined'].apply(clean_text)
    y = df['overall-ratings'].astype(int)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['combined'])

    model = LogisticRegression(max_iter=2000, n_jobs=None)
    model.fit(X, y)
    return vectorizer, model


def predict_single(vectorizer, model, summary, pros, cons):
    txt = f"{summary or ''} {pros or ''} {cons or ''}"
    txt = clean_text(txt)
    X = vectorizer.transform([txt])
    y_pred = model.predict(X)[0]
    probs = None
    try:
        probs = model.predict_proba(X)[0]
    except Exception:
        pass
    return int(y_pred), probs


def predict_batch(vectorizer, model, df):
    for col in ['summary', 'pros', 'cons']:
        if col not in df.columns:
            df[col] = ''
    df[['summary', 'pros', 'cons']] = df[['summary', 'pros', 'cons']].fillna('')
    combined = (df['summary'].astype(str) + ' ' + df['pros'].astype(str) + ' ' + df['cons'].astype(str)).str.strip()
    combined = combined.apply(clean_text)
    X = vectorizer.transform(combined)
    preds = model.predict(X)
    return preds.astype(int)


st.set_page_config(page_title='Employee Review Rating Predictor', page_icon='📊', layout='centered')

# Global stylistic enhancements: gradient background, glass cards, modern controls
st.markdown(
        """
        <style>
            :root {
                --primary: #7C3AED; /* violet-600 */
                --accent:  #22D3EE; /* cyan-400 */
                --accent2: #F472B6; /* pink-400 */
                --bg1:     #0f172a; /* slate-900 */
                --bg2:     #1e293b; /* slate-800 */
                --fg:      #e5e7eb; /* gray-200 */
            }

            /* Ambient gradient background */
            .stApp {
                background: radial-gradient(1200px 800px at 15% 10%, var(--bg2) 0%, var(--bg1) 55%) no-repeat fixed;
                color: var(--fg);
            }
            .stApp:before,
            .stApp:after {
                content: "";
                position: fixed;
                width: 420px;
                height: 420px;
                border-radius: 50%;
                filter: blur(60px);
                opacity: 0.25;
                z-index: 0;
            }
            .stApp:before { top: -80px; right: -60px; background: radial-gradient(circle, var(--accent) 0%, transparent 60%); }
            .stApp:after  { bottom: -80px; left: -60px; background: radial-gradient(circle, var(--accent2) 0%, transparent 60%); }

            /* Gradient headings */
            h1, h2 { 
                background: linear-gradient(90deg, var(--accent), var(--primary), var(--accent2));
                -webkit-background-clip: text; background-clip: text; color: transparent; 
            }

            /* Glass card container */
            .glass-card {
                backdrop-filter: saturate(120%) blur(14px);
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.12);
                box-shadow: 0 18px 60px rgba(0, 0, 0, 0.45);
                border-radius: 22px;
                padding: 22px 22px;
                margin: 12px auto;
                position: relative;
                z-index: 1;
            }

            /* Inputs and text areas */
            input[type="text"], textarea {
                background: rgba(255,255,255,0.06) !important;
                color: var(--fg) !important;
                border: 1px solid rgba(255,255,255,0.20) !important;
                border-radius: 14px !important;
            }
            label { color: #cbd5e1 !important; }

            /* Buttons */
            .stButton > button, .stDownloadButton > button {
                background: linear-gradient(135deg, var(--accent), var(--primary)) !important;
                color: #0b1020 !important;
                border: none !important;
                border-radius: 14px !important;
                padding: 12px 16px !important;
                font-weight: 600 !important;
                box-shadow: 0 10px 28px rgba(124, 58, 237, 0.35) !important;
                transition: transform .08s ease, box-shadow .2s ease !important;
            }
            .stButton > button:hover, .stDownloadButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 16px 36px rgba(34, 211, 238, 0.32) !important;
            }

            /* Dataframe styling */
            .stDataFrame { border-radius: 16px; overflow: hidden; box-shadow: 0 12px 32px rgba(0,0,0,0.35); }
            .stDataFrame [class*="blank"] { background: rgba(255,255,255,0.03); }

            /* Dividers and accents */
            hr { border-color: rgba(255,255,255,0.12); }
            .gradient-separator { height: 2px; background: linear-gradient(90deg, var(--accent), var(--primary), var(--accent2)); border: 0; margin: 18px 0; }
            .pill { display:inline-block; padding: 6px 12px; border-radius: 999px; background: rgba(34,211,238,0.18); border: 1px solid rgba(34,211,238,0.40); color:#cffafe; font-weight:600; }
        </style>
        """,
        unsafe_allow_html=True,
)

# Hero accent and subtle separator
st.markdown('<span class="pill">AI‑Powered Sentiment Insights</span>', unsafe_allow_html=True)
st.title('Employee Review Rating Predictor')
st.markdown('<div class="gradient-separator"></div>', unsafe_allow_html=True)

with st.spinner('Training model (first run only)...'):
    vec, mdl = train_model()

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader('Single Prediction')
col1, col2 = st.columns(2)
with col1:
    summary = st.text_input('Summary', '')
    pros = st.text_area('Pros', '')
with col2:
    cons = st.text_area('Cons', '')

if st.button('Predict Rating', type='primary'):
    pred, probs = predict_single(vec, mdl, summary, pros, cons)
    st.success(f'Predicted overall rating: {pred}')
    # Visual: quick metric and star display
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(label='Predicted Rating', value=int(pred))
        # Star visualization (filled stars = predicted rating)
        filled = int(pred)
        empty = max(0, 5 - filled)
        stars_html = '<span style="font-size:22px; letter-spacing:2px;">' + '★' * filled + '<span style="opacity:.35;">' + '★' * empty + '</span></span>'
        st.markdown(stars_html, unsafe_allow_html=True)
    with c2:
        if probs is not None:
            labels = [1, 2, 3, 4, 5]
            prob_map = {str(lbl): float(probs[i]) if i < len(probs) else None for i, lbl in enumerate(labels)}
            # Probability bar chart
            prob_df = pd.DataFrame({'Class': labels, 'Probability': [prob_map[str(lbl)] for lbl in labels]})
            prob_df = prob_df.set_index('Class')
            st.bar_chart(prob_df, height=180)
            st.caption('Class probability distribution')
        else:
            st.caption('Model does not expose probabilities for this solver.')

st.markdown('</div>', unsafe_allow_html=True)
st.divider()
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader('Batch Prediction (CSV Upload)')
st.caption('Template columns: id (optional), summary, pros, cons')
upload = st.file_uploader('Upload CSV', type=['csv'])
if upload is not None:
    try:
        udf = pd.read_csv(upload)
    except Exception as e:
        st.error(f'Failed to read CSV: {e}')
        st.stop()
    preds = predict_batch(vec, mdl, udf)
    out = udf.copy()
    out['predicted_overall_ratings'] = preds
    # Charts: distribution of predicted ratings
    dist = out['predicted_overall_ratings'].value_counts().sort_index()
    st.bar_chart(dist.rename('Count'))
    # Summary stats
    st.caption(f"Mean predicted rating: {float(out['predicted_overall_ratings'].mean()):.2f} | Total rows: {len(out)}")
    st.dataframe(out.head(20))
    st.download_button('Download predictions CSV', data=out.to_csv(index=False).encode('utf-8'), file_name='predictions.csv', mime='text/csv')

st.markdown('</div>', unsafe_allow_html=True)
st.divider()
st.caption('Tip: This UI trains from employee-review-train.csv or data/employee_reviews.csv if available.')
