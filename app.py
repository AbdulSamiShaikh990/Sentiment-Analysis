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
st.title('Employee Review Rating Predictor')

with st.spinner('Training model (first run only)...'):
    vec, mdl = train_model()

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
    if probs is not None:
        labels = [1, 2, 3, 4, 5]
        prob_map = {str(lbl): float(probs[i]) if i < len(probs) else None for i, lbl in enumerate(labels)}
        st.write('Class probabilities:', prob_map)

st.divider()
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
    st.dataframe(out.head(20))
    st.download_button('Download predictions CSV', data=out.to_csv(index=False).encode('utf-8'), file_name='predictions.csv', mime='text/csv')

st.divider()
st.caption('Tip: This UI trains from employee-review-train.csv or data/employee_reviews.csv if available.')
