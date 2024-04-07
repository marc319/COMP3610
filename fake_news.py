import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Loading the dataset
def read_dataframe() -> pd.DataFrame:
    liar_dataset_url = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv"
    data = pd.read_csv(liar_dataset_url, sep='\t', header=None)
    #data = pd.read_csv(tsv_file, delimiter='\t', dtype=object)
    data.fillna("", inplace=True)
    data.columns = [
        'id',                # Column 1: the ID of the statement ([ID].json).
        'label',            
        'statement',         
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

data= read_dataframe()
data.info()

def TF(value):
    if value == 'pants-fire':
        return 0
    elif value == 'false':
        return 1
    elif value == 'barely-true':
        return 2
    elif value == 'half-true':
        return 3
    elif value == 'mostly-true':
        return 4
    else:
        return 5

data['T/F Rating'] = data['label'].apply(TF)

def sentiment(value):
    if value >= 3:
        return 1
    else:
        return 0

sentiment1 = data['T/F Rating'].apply(sentiment)
data['T/F'] = sentiment1

# Preprocessing the data
#data['statement'] = data['statement'].str.lower()
#data['statement'] = data['statement'].str.replace(r'[^\w\s]', '')
#data['statement'] = data['statement'].str.replace(r'\s+', ' ')

#data['label'] = data['label'].str.lower()
#data['label'] = data['label'].str.replace(r'[^\w\s]', '')
#data['label'] = data['label'].str.replace(r'\s+', ' ')

#label_onehot = pd.get_dummies(data['label'].explode()).groupby(level=0).sum()
#data = pd.concat([data, label_onehot], axis=1)
#data.drop('label', axis=1, inplace=True)
print(data)

vectorizer = TfidfVectorizer()
#X = vectorizer.fit_transform(data['statement'])
#y = data['label']

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Train the model
#model.fit(X_train, y_train)

# Evaluate the model
#y_pred = model.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#print(f'Accuracy: {accuracy:.3f}')

# Use PassiveAgressive Classifier
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', PassiveAggressiveClassifier())
])

# Train the model
#model.fit(X_train, y_train)

# Evaluate the model
#y_pred = model.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#print(f'Accuracy: {accuracy:.3f}')

tfidf_vec = TfidfVectorizer()
tfidf = tfidf_vec.fit_transform(data)
print(pd.DataFrame(tfidf.A, columns=tfidf_vec.get_feature_names_out()).to_string())


