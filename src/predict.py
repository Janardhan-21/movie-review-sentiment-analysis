import joblib

def predict_sentiment(text):
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]
