import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

#Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

#Load the LIAR dataset
def read_dataframe() -> pd.DataFrame:
    liar_dataset_url = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv"
    data = pd.read_csv(liar_dataset_url, sep='\t', header=None)
    data.fillna("", inplace=True)
    data.columns = [
        'id',
        'label',
        'statement',
        'subjects',
        'speaker',
        'speaker_job_title',
        'state_info',
        'party_affiliation',
        'barelyTrueCount',
        'falseCount',
        'halfTrueCount',
        'mostlyTrueCount',
        'pantsOnFireCount',
        'context'
    ]
    return data

data = read_dataframe()

#Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stopwords
    return text

#Preprocess the statement column
data['statement'] = data['statement'].apply(preprocess_text)

#TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['statement'])
y = data['label']

#Train SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X, y)

#Save trained SVM model
joblib.dump(svm_classifier, 'svm_fake_news_detection_model.pkl')

#Function to extract text from a website link
def extract_text_from_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

#Function to predict truth of news
def predict_truthfulness(input_data):
    if input_data.startswith('http'):
        # Extract text from website link
        text = extract_text_from_website(input_data)
    else:
        text = input_data
    
    #Preprocess text
    preprocessed_text = preprocess_text(text)
    
    #Vectorize the text
    text_vectorized = vectorizer.transform([preprocessed_text])
    
    #Predict using the trained SVM model
    prediction = svm_classifier.predict(text_vectorized)[0]
    
    return prediction

input_data = input("Enter text or website link separated by comma: ")
inputs = [x.strip() for x in input_data.split(',')]

for i, input_item in enumerate(inputs):
    prediction = predict_truthfulness(input_item)
    print(f"Prediction {i+1}: {prediction}")
