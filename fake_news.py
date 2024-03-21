import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Loading the dataset
def read_dataframe(tsv_file: str) -> pd.DataFrame:
    data = pd.read_csv(tsv_file, delimiter='\t', dtype=object)
    data.fillna("", inplace=True)
    data.columns = [
        'id',                # Column 1: the ID of the statement ([ID].json).
        'label',            
        'text',         
        'subjects',          
        'speaker',           
        'speaker_job_title', 
        'state_info',        
        'party_affiliation', 
        'barelyTrueCount', 
        'falseCount', 
        'halfTruecCount', 
        'mostlyTrueCount',
        'pantsOnFireCunt', 
        'context' # the context (venue / location of the speech or statement).
    ]
    return data

data= read_dataframe(train.tsv)
data.info()

# Preprocessing the data
data['text'] = data['text'].str.lower()
data['text'] = data['text'].str.replace(r'[^\w\s]', '')
data['text'] = data['text'].str.replace(r'\s+', ' ')

data['label'] = data['label'].str.lower()
data['label'] = data['label'].str.replace(r'[^\w\s]', '')
data['label'] = data['label'].str.replace(r'\s+', ' ')


# Split the data into training and testing sets- is already split into files: train, test, valid

# Define the model
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.3f}')

# Use PassiveAgressive Classifier
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', PassiveAgressiveClassifier())
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.3f}')
