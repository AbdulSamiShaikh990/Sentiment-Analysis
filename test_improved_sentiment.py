#!/usr/bin/env python3
"""Test script for improved sentiment analysis"""

import requests
import json

def test_sentiment_analysis(text, description=""):
    """Test sentiment analysis with given text"""
    print(f"\n=== {description} ===")
    print(f"Input: {text}")
    print("-" * 50)
    
    try:
        response = requests.post(
            "http://localhost:5000/analyze-sentiment",
            json={"text": text},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()['result']
            
            print(f"🎯 Final Result: {result['sentiment_label']} (Rating: {result['predicted_rating']}/5)")
            print(f"📊 Confidence: {result['confidence']:.1%}")
            
            print(f"\n🤖 VADER Analysis:")
            vader = result['vader_analysis']
            print(f"   Compound: {vader['compound']:.3f} → {vader['vader_sentiment']} (Rating: {vader['vader_rating']})")
            print(f"   Details: {vader['positive']:.1%} pos, {vader['neutral']:.1%} neu, {vader['negative']:.1%} neg")
            
            print(f"\n🧠 ML Model Analysis:")
            ml = result['ml_analysis'] 
            print(f"   Rating: {ml['ml_rating']}/5 → {ml['ml_sentiment']}")
            print(f"   Confidence: {ml['ml_confidence']:.1%}")
            
            print(f"\n📈 Class Probabilities:")
            probs = result['class_probabilities']
            print(f"   Negative: {probs['negative']:.1%}")
            print(f"   Neutral:  {probs['neutral']:.1%}")
            print(f"   Positive: {probs['positive']:.1%}")
            
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    # Test cases from Nexora feedback
    test_cases = [
        {
            "text": "Team collaboration is very good as our team members are very cooperative with one another. Managers are also very supportive, as they support their team mates in every situation. Work environment is good, but it needs to be more fair. No macro management by CEO.",
            "description": "Samad's Mixed Feedback (Previously: Negative)"
        },
        {
            "text": "Office atmosphere is good but team work is not good and also time restriction is very hard means there is no relaxation",
            "description": "Moosa's Mixed Feedback (Previously: Positive)"
        },
        {
            "text": "Time restriction is very hard, and also attendance marking is also very strict but overall office is good.",
            "description": "Tayyab's Mixed Feedback (Previously: Negative)"
        },
        {
            "text": "This company is absolutely terrible. Worst management ever. Hate working here.",
            "description": "Clearly Negative Example"
        },
        {
            "text": "Amazing company! Love the culture, great benefits, fantastic team!",
            "description": "Clearly Positive Example"
        },
        {
            "text": "The company is okay. Nothing special but decent place to work.",
            "description": "Clearly Neutral Example"
        }
    ]
    
    print("🚀 Testing Improved Sentiment Analysis")
    print("=" * 80)
    
    for case in test_cases:
        test_sentiment_analysis(case["text"], case["description"])