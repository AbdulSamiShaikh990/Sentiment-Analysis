import os
import re
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Global variables to store trained model and vectorizer
vectorizer = None
model = None

def load_train_data():
    """Load training data from available CSV files"""
    if os.path.exists('employee-review-train.csv'):
        df = pd.read_csv('employee-review-train.csv')
    elif os.path.exists(os.path.join('data', 'employee_reviews.csv')):
        df = pd.read_csv(os.path.join('data', 'employee_reviews.csv'))
        if 'overall-ratings' not in df.columns:
            raise Exception('overall-ratings column not found in data/employee_reviews.csv')
        for col in ['summary', 'pros', 'cons']:
            if col not in df.columns:
                df[col] = ''
        df = df[['overall-ratings', 'summary', 'pros', 'cons']]
    else:
        raise Exception('Training data not found. Place employee-review-train.csv in the project root or data/employee_reviews.csv.')
    
    for col in ['summary', 'pros', 'cons']:
        if col not in df.columns:
            df[col] = ''
    df[['summary', 'pros', 'cons']] = df[['summary', 'pros', 'cons']].fillna('')
    return df

def clean_text(s: str) -> str:
    """Clean and preprocess text for sentiment analysis"""
    s = re.sub(r"@[\w]*", " ", s)
    s = re.sub(r"[^a-zA-Z#]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def train_sentiment_model():
    """Train the sentiment analysis model"""
    global vectorizer, model
    
    print("Training sentiment model...")
    df = load_train_data()
    df['combined'] = (df['summary'].astype(str) + ' ' + df['pros'].astype(str) + ' ' + df['cons'].astype(str)).str.strip()
    df['combined'] = df['combined'].apply(clean_text)
    y = df['overall-ratings'].astype(int)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['combined'])

    model = LogisticRegression(max_iter=2000, n_jobs=None)
    model.fit(X, y)
    print("Model trained successfully!")

def analyze_sentiment(text):
    """
    Main sentiment analysis function
    Args:
        text (str): Input text to analyze
    Returns:
        dict: sentiment result with rating and confidence
    """
    global vectorizer, model
    
    if vectorizer is None or model is None:
        raise Exception("Model not trained. Please train the model first.")
    
    # Clean the input text
    cleaned_text = clean_text(text)
    
    # Transform text to features
    X = vectorizer.transform([cleaned_text])
    
    # Predict sentiment rating
    predicted_rating = model.predict(X)[0]
    
    # Get prediction probabilities
    try:
        probabilities = model.predict_proba(X)[0]
        confidence = float(max(probabilities))
        
        # Create probability distribution
        class_probabilities = {}
        for i, prob in enumerate(probabilities):
            class_probabilities[f"rating_{i+1}"] = float(prob)
            
    except Exception:
        confidence = None
        class_probabilities = None
    
    # Map rating to sentiment label
    sentiment_mapping = {
        1: "Very Negative",
        2: "Negative", 
        3: "Neutral",
        4: "Positive",
        5: "Very Positive"
    }
    
    result = {
        "predicted_rating": int(predicted_rating),
        "sentiment_label": sentiment_mapping.get(int(predicted_rating), "Unknown"),
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "input_text": text[:100] + "..." if len(text) > 100 else text  # Truncate for response
    }
    
    return result

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": (vectorizer is not None and model is not None)
    })

@app.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment_endpoint():
    """
    Main sentiment analysis endpoint
    Expected JSON payload: {"text": "your text here"}
    Returns: JSON with sentiment analysis results
    """
    try:
        # Check if model is loaded
        if vectorizer is None or model is None:
            return jsonify({
                "error": "Model not loaded. Please check server logs."
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing 'text' field in request body"
            }), 400
        
        input_text = data['text']
        
        if not input_text or not input_text.strip():
            return jsonify({
                "error": "Empty text provided"
            }), 400
        
        # Analyze sentiment
        result = analyze_sentiment(input_text)
        
        return jsonify({
            "success": True,
            "result": result
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint with sample text"""
    try:
        sample_text = "This company has great work culture and amazing team members. Really enjoyed working here."
        result = analyze_sentiment(sample_text)
        
        return jsonify({
            "success": True,
            "test_input": sample_text,
            "result": result
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    try:
        # Train model on startup
        train_sentiment_model()
        print("✅ Sentiment API Server Ready!")
        print("📊 Endpoints available:")
        print("   GET  /health - Health check")
        print("   POST /analyze-sentiment - Main sentiment analysis")
        print("   GET  /test - Test with sample data")
        print("🚀 Starting server on http://localhost:5000")
        
        # Start Flask server
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")