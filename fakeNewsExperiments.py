import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#preprocess

datatrain = pd.read_csv('https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv', sep='\t')
datatest = pd.read_csv('https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv', sep='\t')
dataValid = pd.read_csv('https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/valid.tsv', sep='\t')
print(datatrain.shape)
print(datatest.shape)
print(dataValid.shape)
datatrain.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'JobTitle', 'State', 'Party', 'barelyTrueCount', 'falseCount', 'halfTruecCount', 'mostlyTrueCount', 'pantsOnFireCount', 'context']
datatest.columns =  ['id', 'label', 'statement', 'subject', 'speaker', 'JobTitle', 'State', 'Party', 'barelyTrueCount', 'falseCount', 'halfTruecCount', 'mostlyTrueCount', 'pantsOnFireCount', 'context']
dataValid.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'JobTitle', 'State', 'Party', 'barelyTrueCount', 'falseCount', 'halfTruecCount', 'mostlyTrueCount', 'pantsOnFireCount', 'context']
data = pd.concat([datatrain, datatest, dataValid], axis=0)
print(data.shape)
data.head()

# encode the data 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
dataCopy=data.copy()
dataCopy.drop(['statement','subject'],axis=1,inplace=True)
dataCopy['id'] = dataCopy['id'].apply(lambda x: x[:-5])
dataCopy['id'] = dataCopy['id'].astype('int64')
le = LabelEncoder()
dataCopy['label'] = le.fit_transform(dataCopy['label'])
dataCopy['speaker'] = le.fit_transform(dataCopy['speaker'])
dataCopy['JobTitle'] = le.fit_transform(dataCopy['JobTitle'])
dataCopy['State'] = le.fit_transform(dataCopy['State'])
dataCopy['Party'] = le.fit_transform(dataCopy['Party'])
print(dataCopy.head())

del dataCopy['context']
#correlation analysis 
correlate_matrix = dataCopy.corr()
plt.figure(figsize=(11,11))
sns.heatmap(correlate_matrix, annot=True, cmap='coolwarm')
plt.show()


data = data.drop(['id', 'JobTitle', 'State', 'barelyTrueCount', 'falseCount', 'halfTruecCount', 'mostlyTrueCount', 'pantsOnFireCount', 'context'], axis=1)
data.head() 


data['label'] = data['label'].map({'true': 1, 'half-true': 1, 'mostly-true': 1, 'false': 0, 'pants-fire': 0, 'barely-true': 0})
data.head()

data['text'] = data['subject'] + ' ' + data['statement']
data = data.drop(['subject', 'statement'], axis=1)
data.head()

print("Number of missing values in each column:")
print(data.isnull().sum())
print("We drop the missing values")
data = data.dropna()
print("The shape of the dataset is now: ", data.shape)


print("Number of missing values in each column:")
print(data.isnull().sum())
print("We drop the missing values")
data = data.dropna()
print("The shape of the dataset is now: ", data.shape)
