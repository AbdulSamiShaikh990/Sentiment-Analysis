import os
import re
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

app = Flask(__name__)

# Global variables to store trained model and vectorizer
vectorizer = None
model = None
vader_analyzer = SentimentIntensityAnalyzer()

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
    # Remove mentions and URLs
    s = re.sub(r"@[\w]*", " ", s)
    s = re.sub(r"http\S+", " ", s)
    
    # Preserve important punctuation for sentiment
    s = re.sub(r"[^a-zA-Z\s!'.]", " ", s)
    
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)
    
    return s.strip().lower()

def preprocess_for_sentiment(text: str) -> str:
    """Advanced preprocessing that preserves sentiment-important patterns"""
    # Handle negations - add prefix to following words
    negation_pattern = r'\b(not|no|never|neither|none|nobody|nothing|nowhere|hardly|scarcely|barely)\s+'
    text = re.sub(negation_pattern, lambda m: m.group(0) + 'NOT_', text, flags=re.IGNORECASE)
    
    # Handle intensifiers
    intensifiers = {
        'very': 'VERY_',
        'really': 'REALLY_',
        'extremely': 'EXTREMELY_',
        'highly': 'HIGHLY_',
        'quite': 'QUITE_',
        'totally': 'TOTALLY_',
        'completely': 'COMPLETELY_'
    }
    
    for intensifier, prefix in intensifiers.items():
        pattern = r'\b' + intensifier + r'\s+'
        text = re.sub(pattern, prefix, text, flags=re.IGNORECASE)
    
    return text

def get_stop_words():
    """Get stop words but keep negation words for better sentiment analysis"""
    stop_words = set(ENGLISH_STOP_WORDS)
    # Keep negation words as they're important for sentiment
    for word in ("no", "not", "nor", "never", "neither", "none"):
        stop_words.discard(word)
    return stop_words

def map_rating_to_sentiment(rating: int) -> str:
    """Map 1-5 rating to 3-class sentiment"""
    if rating <= 2:
        return "Negative"
    if rating == 3:
        return "Neutral"
    return "Positive"

def apply_neutral_overrides(sentiment_label: str, class_probabilities_3, confidence_3):
    """Override to Neutral only for very uncertain cases"""
    if class_probabilities_3 is None or confidence_3 is None:
        return sentiment_label

    # Only neutral for extremely uncertain cases
    if confidence_3 < 0.30:
        return "Neutral"

    # Only neutral if all three probabilities are very close
    probs = sorted(class_probabilities_3.values(), reverse=True)
    if len(probs) >= 3:
        max_prob = probs[0]
        second_prob = probs[1] 
        third_prob = probs[2]
        
        # If top 3 are all very close (within 5%), then it's truly ambiguous
        if (max_prob - third_prob) < 0.05:
            return "Neutral"

    return sentiment_label

def train_sentiment_model():
    """Train the sentiment analysis model"""
    global vectorizer, model
    
    print("Training improved sentiment model...")
    df = load_train_data()
    
    # Combine text fields
    df['combined'] = (df['summary'].astype(str) + ' ' + df['pros'].astype(str) + ' ' + df['cons'].astype(str)).str.strip()
    
    # Apply advanced preprocessing
    df['processed'] = df['combined'].apply(lambda x: preprocess_for_sentiment(clean_text(x)))
    
    y = df['overall-ratings'].astype(int)

    # Use improved TF-IDF with phrase awareness
    vectorizer = TfidfVectorizer(
        max_features=8000,
        stop_words=get_stop_words(),
        ngram_range=(1, 3),  # Include trigrams for better phrase capture
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95,  # Ignore terms that appear in more than 95% of documents
        sublinear_tf=True  # Apply sublinear TF scaling
    )
    
    X = vectorizer.fit_transform(df['processed'])

    # Use LogisticRegression with L2 regularization
    model = LogisticRegression(
        max_iter=3000, 
        C=1.0,  # Regularization strength
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    )
    model.fit(X, y)
    print("Enhanced sentiment model trained successfully!")
    print(f"Training completed with {len(df)} samples and {X.shape[1]} features")

def analyze_sentiment(text):
    """
    Hybrid sentiment analysis using VADER + ML model for better phrase understanding
    Args:
        text (str): Input text to analyze
    Returns:
        dict: comprehensive sentiment result with multiple analysis methods
    """
    global vectorizer, model, vader_analyzer
    
    if vectorizer is None or model is None:
        raise Exception("Model not trained. Please train the model first.")
    
    # VADER analysis for rule-based sentiment understanding
    vader_scores = vader_analyzer.polarity_scores(text)
    vader_compound = vader_scores['compound']
    
    # Map VADER compound score to sentiment
    def vader_to_sentiment(compound):
        if compound >= 0.05:
            return "Positive"
        elif compound <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    # Map VADER compound to 1-5 rating scale
    def vader_to_rating(compound):
        if compound >= 0.6:
            return 5
        elif compound >= 0.2:
            return 4
        elif compound >= -0.2:
            return 3
        elif compound >= -0.6:
            return 2
        else:
            return 1
    
    # ML Model Analysis
    processed_text = preprocess_for_sentiment(clean_text(text))
    X = vectorizer.transform([processed_text])
    ml_predicted_rating = model.predict(X)[0]
    
    # Get ML model probabilities
    try:
        probabilities = model.predict_proba(X)[0]
        ml_confidence_5 = float(max(probabilities))
        
        class_probabilities_5 = {
            f"rating_{i+1}": float(prob)
            for i, prob in enumerate(probabilities)
        }
        
        # Calculate 3-class probabilities
        prob_negative = float(sum(probabilities[0:2]))
        prob_neutral = float(probabilities[2])
        prob_positive = float(sum(probabilities[3:5]))
        
        class_probabilities_3 = {
            "negative": prob_negative,
            "neutral": prob_neutral,
            "positive": prob_positive,
        }
        ml_confidence_3 = float(max(class_probabilities_3.values()))
        
    except Exception:
        ml_confidence_5 = None
        ml_confidence_3 = None
        class_probabilities_5 = None
        class_probabilities_3 = None
    
    # Hybrid Decision Logic
    vader_rating = vader_to_rating(vader_compound)
    vader_sentiment = vader_to_sentiment(vader_compound)
    ml_sentiment = map_rating_to_sentiment(int(ml_predicted_rating))
    
    # Determine final sentiment using hybrid approach
    final_sentiment = ml_sentiment
    final_rating = int(ml_predicted_rating)
    confidence = ml_confidence_3 if ml_confidence_3 else 0.5
    
    # Apply VADER override for clear cases or when ML model is uncertain
    if ml_confidence_3 and ml_confidence_3 < 0.4:
        # ML model is uncertain, trust VADER more
        final_sentiment = vader_sentiment
        final_rating = vader_rating
        confidence = abs(vader_compound)
    elif abs(vader_compound) > 0.7 and vader_sentiment != ml_sentiment:
        # VADER is very confident and disagrees with ML, give VADER more weight
        if abs(vader_compound) > 0.8:
            final_sentiment = vader_sentiment
            final_rating = vader_rating
        else:
            # Blend the scores
            blended_rating = int((vader_rating + ml_predicted_rating) / 2)
            final_rating = blended_rating
            final_sentiment = map_rating_to_sentiment(blended_rating)
    
    # Apply neutral overrides for mixed signals
    final_sentiment = apply_neutral_overrides(
        final_sentiment,
        class_probabilities_3,
        confidence
    )
    
    # Sentiment mappings
    sentiment_mapping_5 = {
        1: "Very Negative",
        2: "Negative",
        3: "Neutral", 
        4: "Positive",
        5: "Very Positive"
    }
    
    result = {
        "predicted_rating": final_rating,
        "sentiment_label": final_sentiment,
        "sentiment_label_5": sentiment_mapping_5.get(final_rating, "Unknown"),
        "confidence": confidence,
        "confidence_5": ml_confidence_5,
        "class_probabilities": class_probabilities_3,
        "class_probabilities_5": class_probabilities_5,
        "vader_analysis": {
            "compound": vader_compound,
            "positive": vader_scores['pos'],
            "neutral": vader_scores['neu'], 
            "negative": vader_scores['neg'],
            "vader_sentiment": vader_sentiment,
            "vader_rating": vader_rating
        },
        "ml_analysis": {
            "ml_rating": int(ml_predicted_rating),
            "ml_sentiment": ml_sentiment,
            "ml_confidence": ml_confidence_3
        },
        "input_text": text[:100] + "..." if len(text) > 100 else text
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