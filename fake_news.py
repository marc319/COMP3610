import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Loading the dataset
def read_dataframe() -> pd.DataFrame:
    liar_dataset_url = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv"
    data = pd.read_csv(liar_dataset_url, sep='\t', header=None)
    #data = pd.read_csv(tsv_file, delimiter='\t', dtype=object)
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
        'pantsOnFireCount', 
        'context' # the context (venue / location of the speech or statement).
    ]
    return data

data= read_dataframe()
data.info()

sns.set()

def getPercent(x):
        return x * 100

def chartForLabel(input_data: pd.DataFrame, title: str = "LIAR Dataset") -> None:
    
    label_freqs = input_data['label'].value_counts(normalize=True) #gets count for each label
    
    label_freqs = label_freqs.apply(getPercent) #finds the percentage of the label from the dataset
    
    labels = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
    colors = ['#1F77B4', '#2CA02C', '#D62728', '#9467BD', '#E377C2', '#F7B6D2']
    
    label_freqs = label_freqs.reindex(index = labels)
    axis = label_freqs.plot(kind='bar', figsize=(12, 8), color=colors);
    axis.set_title(f"Distribution of label values ({title}, sample_size={len(input_data)})", size=20);
    
chartForLabel(data) # graph to show the frequency in % of each label in the dataset

# Tokenization, stemming, removing stopwords, lowercasing, and removing punctuations
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word.isalnum()]
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['processed_text'] = data['text'].apply(preprocess_text)

print(data['processed_text']);

from sklearn.feature_extraction.text import CountVectorizer

# Creating a Bag-Of-Words model using CountVectorizer
count_vectorizer = CountVectorizer()
train_countVec = count_vectorizer.fit_transform(data['processed_text'])

print(count_vectorizer.vocabulary_)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()

# tokenize, build vocab and encode training data
train_tfidTransf = tfidf_transformer.fit_transform(train_countVec)
tfidf_transformer.transform(train_countVec) #tfid_score







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
