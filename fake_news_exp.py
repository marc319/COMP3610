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






############################################




import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

# Load the LIAR dataset
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
        'halfTrueCount', 
        'mostlyTrueCount',
        'pantsOnFireCount', 
        'context' # the context (venue / location of the speech or statement).
    ]
    return data

liar_data= read_dataframe()

# Step by step multilabel preprocessing
mlb = MultiLabelBinarizer()
liar_data['labels'] = liar_data['label'].apply(lambda x: set(x.split(',')))
labels_encoded = mlb.fit_transform(liar_data['labels'])



from sklearn.multiclass import OneVsRestClassifier

# Feature Extraction
features = ['subjects', 'speaker', 'speaker_job_title', 'state_info', 'party_affiliation',
            'barelyTrueCount', 'falseCount', 'halfTrueCount', 'mostlyTrueCount', 'pantsOnFireCount', 'context']

X = liar_data[features].apply(lambda x: ' '.join(x.astype(str)), axis=1)
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)




# Data Modeling - Train Logistic Regression and Support Vector Machines
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, labels_encoded, test_size=0.2, random_state=42)



lr_model = LogisticRegression()
lr_model = OneVsRestClassifier(lr_model)
lr_model.fit(X_train, y_train)
svm_model = SVC()
svm_model = OneVsRestClassifier(svm_model)
svm_model.fit(X_train, y_train)

# Model Evaluation
lr_pred = lr_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_pred))
print("\nSVM Classification Report:")
print(classification_report(y_test, svm_pred))
print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test.argmax(axis=1), lr_pred.argmax(axis=1)))
print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test.argmax(axis=1), svm_pred.argmax(axis=1)))

# Performance Testing
test_url = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv"
test_data = pd.read_csv(test_url, sep='\t', header=None)
X_test_data = test_data[features].apply(lambda x: ' '.join(x.astype(str)), axis=1)
X_test_tfidf = tfidf.transform(X_test_data)
test_labels_encoded = mlb.transform(test_data['labels'])
test_pred_lr = lr_model.predict(X_test_tfidf)
test_pred_svm = svm_model.predict(X_test_tfidf)

# Validation
valid_url = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/valid.tsv"
valid_data = pd.read_csv(valid_url, sep='\t', header=None)
X_valid_data = valid_data[features].apply(lambda x: ' '.join(x.astype(str)), axis=1)
X_valid_tfidf = tfidf.transform(X_valid_data)
valid_labels_encoded = mlb.transform(valid_data['labels'])
valid_pred_lr = lr_model.predict(X_valid_tfidf)
valid_pred_svm = svm_model.predict(X_valid_tfidf)

# Data Visualization
# Bar Charts
liar_data['party_affiliation'].value_counts().plot(kind='bar')
plt.title('Party Affiliation Distribution')
plt.xlabel('Party Affiliation')
plt.ylabel('Count')
plt.show()

# Heatmap
sns.heatmap(confusion_matrix(y_test.argmax(axis=1), lr_pred.argmax(axis=1)), annot=True, fmt='d')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# ROC Curves
fpr, tpr, _ = roc_curve(y_test.ravel(), lr_pred.ravel())
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
