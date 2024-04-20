import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Loading the dataset
def read_dataframe(url) -> pd.DataFrame:
    liar_dataset_url = f"https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/{url}"
    data = pd.read_csv(liar_dataset_url, sep='\t', header=None)
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
        'halfTrueCount', 
        'mostlyTrueCount',
        'pantsOnFireCount', 
        'context' # the context (venue / location of the speech or statement).
    ]
    
    data['label'] = data['label'].apply(lambda x: 1 if x == 'true' else 0)
    #data['statement'] = data['statement'].apply(clean_text)
    return data

train_data= read_dataframe("train.tsv")
test_data = read_dataframe("test.tsv")
valid_data = read_dataframe("valid.tsv")

data = pd.concat([train_data, test_data, valid_data], axis=0)

# Tokenization, stemming, removing stopwords, lowercasing, and removing punctuations
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word.isalnum()]
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['statement'] = data['statement'].apply(preprocess_text)

print(data['statement']);

# encode the data 

dataFrame = data.drop(columns=['id'])
dataFrame= dataFrame.dropna()
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

labelEnc = LabelEncoder()
dataFrame["label"] = labelEnc.fit_transform(dataFrame["label"])
dataFrame.head()

dataFrame['label'] = dataFrame['label'].astype('int')
sns.countplot(x='label', data=dataFrame, palette='hls')

df= dataFrame.drop(columns=['subjects', 'speaker_job_title', 'state_info', 'party_affiliation', 'barelyTrueCount', 'falseCount', 'halfTrueCount', 'mostlyTrueCount','pantsOnFireCount'])
df.head()


dataFrame= dataFrame.drop(columns=['subjects', 'speaker_job_title', 'state_info', 'party_affiliation', 'barelyTrueCount', 'falseCount', 'halfTrueCount', 'mostlyTrueCount','pantsOnFireCount'])
dataFrame.info()

X = dataFrame.iloc[:,1].values #stores text
Y= dataFrame.iloc[:,0].values #stores label
print(X)
print(Y)

print(type(X))
print(type(Y))
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.25, random_state = 0)

# use TFIDVectorizer to make the text into numerical values
vectorizer= TfidfVectorizer()
tfid_fit = vectorizer.fit(X_train)
X_train_tf = tfid_fit.transform(X_train)
X_test_tf = tfid_fit.transform(X_test)

X_train_tf.toarray().shape

def logRegModel(X_train, y_train):
    log = LogisticRegression(random_state=0)
    X_train_tf_array = X_train_tf.toarray()
    log.fit(X_train_tf_array, y_train)
    print('Logistic Regression Accuracy of train data: ', log.score(X_train_tf_array, y_train))
    return log

from sklearn.tree import DecisionTreeClassifier

def decistreeModel(X_train_tf, y_train):
    dtree = DecisionTreeClassifier(random_state=0)
    dtree.fit(X_train_tf, y_train)
    print('Decision Tree Accuracy on train data: ', dtree.score(X_train_tf, y_train))
    return dtree

lrModel = logRegModel(X_train, y_train)

print(lrModel)


dtree = decistreeModel(X_train_tf, y_train)

print(dtree)
