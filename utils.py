import joblib

# Load trained model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def detect_contradiction(s1, s2):
    try:
        text = s1 + " " + s2
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]

        if pred == 0:
            return "❌ Contradiction", 0.85
        elif pred == 1:
            return "✅ Entailment", 0.85
        else:
            return "⚪ Neutral", 0.85

    except Exception as e:
        return str(e), 0.0


def highlight_words(s1, s2):
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())
    return words1 - words2, words2 - words1