from src.load_data import load_movie_review_data
from src.train_model import train_and_evaluate_model
from src.predict import predict_sentiment

# Step 1: Load and Train
df = load_movie_review_data()
train_and_evaluate_model(df)

# Step 2: Make Predictions
print(predict_sentiment("I absolutely loved this movie! It was fantastic."))
print(predict_sentiment("It was a terrible film. I hated it."))
print(predict_sentiment("The movie was okay, nothing special."))
