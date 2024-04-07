import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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
        return 0
    elif value == 'barely-true':
        return 0
    elif value == 'half-true':
        return 1
    elif value == 'mostly-true':
        return 1
    else:
        return 1

data['T/F'] = data['label'].apply(TF)

x = data['statement']
y = data['T/F']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
LR.score(xv_test, y_test)
print(classification_report(y_test, pred_lr))


SVM = SVC()
SVM.fit(xv_train, y_train)
pred_svm = SVM.predict(xv_test)
SVM.score(xv_test, y_test)
print(classification_report(y_test, pred_svm))

def clean(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '',text)
    return text

def labels(outcome):
    if outcome == 0:
        return "Fake News"
    elif outcome == 1:
        return "Real News"

def testing(news):
    expriment_news = {"text":[news]}
    new_def_test = pd.DataFrame(expriment_news)
    new_def_test["text"] = new_def_test["text"].apply(clean)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_SVM = SVM.predict(new_xv_test)
    return print ("\n\nLR Prediction: {} \nSVM Prediction: {}".format(labels(pred_LR[0]), labels(pred_SVM[0])))


for x in data['statement']:
    testing(x)
