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
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# Load the LIAR dataset
def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\<.*?\>', '', text)
    text = re.sub(r'&.*?;', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = text.lower()
    text = text.strip()
    return text

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
    data['statement'] = data['statement'].apply(clean_text)
    return data

train_data= read_dataframe("train.tsv")
test_data = read_dataframe("test.tsv")
valid_data = read_dataframe("valid.tsv")

X_train, X_test, y_train, y_test = train_test_split(train_data['statement'], train_data['label'], test_size=0.2, random_state=42)

def build_model():
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    lr_model = LogisticRegression()
    svm_model = SVC()
    lr_model.fit(X_train_tfidf, y_train)
    svm_model.fit(X_train_tfidf, y_train)
    return lr_model, svm_model, vectorizer

lr_model, svm_model, vectorizer = build_model()

X_test_tfidf = vectorizer.transform(X_test)
lr_pred = lr_model.predict(X_test_tfidf)

accuracy = lr_model.score(X_test_tfidf, y_test)
print(f'Model accuracy: {accuracy}')

svm_pred = svm_model.predict(X_test_tfidf)

accuracy = svm_model.score(X_test_tfidf, y_test)
print(f'Model accuracy: {accuracy}')

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_pred))
print("\nSVM Classification Report:")
print(classification_report(y_test, svm_pred))
print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))
print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test, svm_pred))


# Performance Testing
X_test_data = test_data['statement']
y_test_1 = test_data['label']
X_test_tfidf = vectorizer.transform(X_test_data)
test_pred_lr = lr_model.predict(X_test_tfidf)
test_pred_svm = svm_model.predict(X_test_tfidf)

print("Logistic Regression Accuracy:", accuracy_score(y_test_1, test_pred_lr))
print("SVM Accuracy:", accuracy_score(y_test_1, test_pred_svm))
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test_1, test_pred_lr))
print("\nSVM Classification Report:")
print(classification_report(y_test_1, test_pred_svm))
print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test_1, test_pred_lr))
print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test_1, test_pred_svm))


X_valid_data = valid_data['statement']
y_test_2 = valid_data['label']
X_valid_tfidf = vectorizer.transform(X_valid_data)
valid_pred_lr = lr_model.predict(X_valid_tfidf)
valid_pred_svm = svm_model.predict(X_valid_tfidf)

print("Logistic Regression Accuracy:", accuracy_score(y_test_2, valid_pred_lr))
print("SVM Accuracy:", accuracy_score(y_test_2, valid_pred_svm))
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test_2, valid_pred_lr))
print("\nSVM Classification Report:")
print(classification_report(y_test_2, valid_pred_svm))
print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test_2, valid_pred_lr))
print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test_2, valid_pred_svm))

# Data Visualization
# Bar Graph - Model Accuracy
models = ['Logistic Regression', 'SVM']
accuracies = [accuracy_score(y_test, lr_pred), accuracy_score(y_test, svm_pred)]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()

# Heat Map - Confusion Matrix (Logistic Regression)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, svm_pred), annot=True, cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - SVM')
plt.show()

fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, lr_pred)
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, svm_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
plt.plot(fpr_svm, tpr_svm, label='SVM')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

auc_lr = roc_auc_score(y_test, lr_pred)
auc_svm = roc_auc_score(y_test, svm_pred)

print(f'Logistic Regression AUC Score: {auc_lr}')
print(f'SVM AUC Score: {auc_svm}')
