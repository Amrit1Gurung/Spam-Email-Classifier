import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# load data
dataset = pd.read_csv('spam_ham_dataset.csv')

def preprocessing(dataset):
  data = dataset.copy()

  #drop unlwnated columns
  data = data.drop('label',axis=1)
  data = data.drop('Unnamed: 0',axis=1)
   
  # reomve special characters
  data['cleaned_text'] = data['text'].str.replace("[^a-zA-Z#]", " ")

  # remove short words 
  data['cleaned_text'] = data['cleaned_text'].apply(lambda x : ' '.join([w for w in x.split() if len(w)>3]))
  
  # tokenize words
  data['cleaned_text'] = data['cleaned_text'].apply(lambda x: word_tokenize(x))
  
  # reomove stipwords
  data['cleaned_text'] = data['cleaned_text'].apply(lambda x: [w for w in x if w not in stopwords.words('english')])

  # stemming
  stemmer = PorterStemmer()
  data['cleaned_text'] = data['cleaned_text'].apply(lambda x: [stemmer.stem(i) for i in x])

  # join data...token is removed and words joined with one space between each words
  data['cleaned_text'] = data['cleaned_text'].apply(lambda x: ' '.join(w for w in x))

  return data


data = preprocessing(dataset)

# applying countVectorizer / creating bag-of-word model
cv = CountVectorizer(max_features=4000,ngram_range=(1,3))
X = cv.fit_transform(data['cleaned_text']).toarray()

y = data['label_num']  # target feature

# train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size=0.3, random_state=0)

# model building
classifier = MultinomialNB()

# train model
classifier.fit(xtrain,ytrain)  

# predict classes
pred = classifier.predict(xtest)

# Evaluating model
# train accuracy
train_pred = classifier.predict(xtrain)
print('train score: ', metrics.accuracy_score(ytrain,train_pred))

# test score
print('accuracy score: ', metrics.accuracy_score(ytest,pred))
# classifcation matrix and classification reports
cm = metrics.confusion_matrix(ytest,pred)
print('Classification matrix:\n', cm)
clf_report = metrics.classification_report(ytest,pred)
print('\nClassification report:\n', clf_report)