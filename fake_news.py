import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import numpy as np

# Load LIAR dataset
liar_dataset_url = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv"
liar_data = pd.read_csv(liar_dataset_url, sep='\t', header=None)

# Extract necessary information
X = liar_data[2]  # Text
y = liar_data[1]  # Labels

# Preprocessing
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression())
])

# Train model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model 
joblib.dump(pipeline, 'fake_news_detection_model.pkl')
