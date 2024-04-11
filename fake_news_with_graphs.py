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

def read_dataframe(url) -> pd.DataFrame:
    liar_dataset_url = f"https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/{url}"
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
        'pantsOnFireCount', 
        'context' # the context (venue / location of the speech or statement).
    ]
    return data

data= read_dataframe("train.tsv")
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
    return pred_LR, pred_SVM
    

test_data= read_dataframe("test.tsv")
LR_array = []
SVM_array =[]

for x in test_data['statement']:
    pred_LR, pred_SVM = testing(x)
    LR_array.append(pred_LR)
    SVM_array.append(pred_SVM)
    print ("\n\nLR Prediction: {} \nSVM Prediction: {}".format(labels(pred_LR[0]), labels(pred_SVM[0])))

test_data['LR_Prediction'] = LR_array
test_data['SVM_Prediction'] = SVM_array


LR_real_pred=0
real =0
LR_fake_pred =0
fake = 0

for data in test_data['LR_Prediction']:
    if data == 1:
        LR_real_pred += 1
    if data == 0:
        LR_fake_pred += 1
                    
val_data= read_dataframe("valid.tsv")

val_data['T/F'] = val_data['label'].apply(TF)

for data in val_data['T/F']:
    if data == 1:
        real += 1
    if data == 0:
        fake += 1

categories = ['Real News', 'Fake News']
actual_values = [real, fake]
predicted_values = [LR_real_pred, LR_fake_pred]

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, actual_values, width, label='Actual')
rects2 = ax.bar(x + width/2, predicted_values, width, label='Predicted')

ax.set_ylabel('Count')
ax.set_title('Real News vs Predicted Real News and Fake News vs Predicted Fake News using Logistic Regression Model')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

fig.tight_layout()

plt.show()


SVM_real_pred=0
SVM_fake_pred =0

for data in test_data['SVM_Prediction']:
    if data == 1:
        SVM_real_pred += 1
    if data == 0:
        SVM_fake_pred += 1
                    

categories = ['Real News', 'Fake News']
actual_values = [real, fake]
predicted_values = [SVM_real_pred, SVM_fake_pred]

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, actual_values, width, label='Actual')
rects2 = ax.bar(x + width/2, predicted_values, width, label='Predicted')

ax.set_ylabel('Count')
ax.set_title('Real News vs Predicted Real News and Fake News vs Predicted Fake News using SVM Model')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

fig.tight_layout()

plt.show()


from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(y_test, pred_lr)

cm_svm = confusion_matrix(y_test, pred_svm)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(cm_lr, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1, 2, 2)
sns.heatmap(cm_svm, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.title("Confusion Matrix - Support Vector Machines")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


conf_matrix = pd.crosstab(y_test, pred_lr, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(conf_matrix, annot=True, cmap='coolwarm')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()



from sklearn.metrics import roc_curve, auc

fpr_lr, tpr_lr, _ = roc_curve(y_test, pred_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

fpr_svm, tpr_svm, _ = roc_curve(y_test, pred_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression (AUC = %0.2f)" % roc_auc_lr)
plt.plot(fpr_svm, tpr_svm, label="Support Vector Machines (AUC = %0.2f)" % roc_auc_svm)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()



from sklearn.metrics import precision_recall_curve

precision_lr, recall_lr, _ = precision_recall_curve(y_test, pred_lr)

precision_svm, recall_svm, _ = precision_recall_curve(y_test, pred_svm)

plt.figure(figsize=(8, 6))
plt.plot(recall_lr, precision_lr, label="Logistic Regression")
plt.plot(recall_svm, precision_svm, label="Support Vector Machines")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.show()
