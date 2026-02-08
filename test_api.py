# Test Flask Sentiment API

import requests
import json

# Test endpoints
BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("🔍 Health Check:")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_sample():
    """Test with sample endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/test")
        print("📝 Sample Test:")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print()
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Sample test failed: {e}")
        return False

def test_sentiment_analysis():
    """Test main sentiment analysis endpoint"""
    test_texts = [
        "This company is absolutely amazing! Great work culture and excellent benefits.",
        "Terrible management and poor work life balance. Would not recommend.",
        "The company is okay, nothing special but decent place to work."
    ]
    
    for i, text in enumerate(test_texts, 1):
        try:
            payload = {"text": text}
            response = requests.post(
                f"{BASE_URL}/analyze-sentiment", 
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"🧠 Sentiment Test {i}:")
            print(f"Input: {text[:50]}...")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    analysis = result['result']
                    print(f"Predicted Rating: {analysis['predicted_rating']}/5")
                    print(f"Sentiment: {analysis['sentiment_label']}")
                    print(f"Confidence: {analysis['confidence']:.2f}" if analysis['confidence'] else "Confidence: N/A")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
            else:
                print(f"Failed with status {response.status_code}")
                print(f"Response: {response.text}")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
            print("-" * 50)

if __name__ == "__main__":
    print("🚀 Testing Flask Sentiment Analysis API")
    print("=" * 50)
    
    # Test health
    if test_health():
        print("✅ Health check passed")
    else:
        print("❌ Health check failed - API might not be running")
        exit(1)
    
    # Test sample endpoint
    if test_sample():
        print("✅ Sample test passed")
    else:
        print("❌ Sample test failed")
    
    # Test main sentiment analysis  
    print("Testing main sentiment analysis endpoint...")
    test_sentiment_analysis()
    
    print("🎉 Testing completed!")