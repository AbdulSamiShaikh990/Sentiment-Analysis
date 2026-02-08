# 🚀 Sentiment Analysis Quick Guide

## 📋 Commands to Run

### Start Streamlit UI:
```bash
streamlit run app.py
```
- Opens: http://localhost:8501

### Start Backend API:
```bash
C:/Users/hp/AppData/Local/Programs/Python/Python37/python.exe sentiment_api.py
```
- API: http://localhost:5000

### Test API:
```bash
C:/Users/hp/AppData/Local/Programs/Python/Python37/python.exe test_improved_sentiment.py
```

## 🎯 Expected Results

### ✅ **POSITIVE** - Kab aayega:
- "Great company, amazing culture, love working here"
- "Excellent benefits, fantastic team, really enjoyed"
- Clear positive words without major complaints

### ❌ **NEGATIVE** - Kab aayega:
- "Terrible management, hate working here, worst company"
- "Very hard restrictions, not good work culture, no relaxation"
- Clear negative words dominating the text

### 😐 **NEUTRAL** - Kab aayega:
- "Company is okay, nothing special but decent"
- Very mixed feedback with **equal** positive/negative
- Model confidence < 30% (very uncertain)

## 📊 API Response Structure

```json
{
  "sentiment_label": "Positive/Negative/Neutral",
  "predicted_rating": "1-5",
  "confidence": "0-100%",
  "vader_analysis": {...},
  "ml_analysis": {...}
}
```

## 🔧 Key Features

- **Hybrid Model**: VADER + ML combination
- **Phrase Understanding**: "not good", "very hard" properly handled
- **Negation Aware**: Understands context, not just word count
- **3-Class Output**: Simple Positive/Negative/Neutral for Nexora

## ⚡ Quick Test Examples

| Text | Expected Result |
|------|----------------|
| "Office good but team work not good, very hard restrictions" | **Negative** |
| "Amazing company! Great benefits!" | **Positive** |
| "Company is okay, decent place" | **Neutral** |

---
**For Nexora Integration**: Use POST to `http://localhost:5000/analyze-sentiment` with `{"text": "..."}`