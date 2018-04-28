from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn import preprocessing
from sklearn import svm
import numpy as np
import pandas as pd
import csv

data = {"Naive Bayes":[0,0,0,0],"Random Forest":[0,0,0,0],"SVM":[0,0,0,0],"KNN":[0,0,0,0],"My Method":[0,0,0,0]}
df = pd.DataFrame(data,columns = ["Naive Bayes","Random Forest","SVM","KNN","My Method"], index=["Accuracy","Precision","Recall","F1"])
print df
input_file = open("datasets/train_set.csv")
train_data = pd.read_csv(input_file, header = 0, delimiter = "\t")
train_data = train_data[0:500]
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])
count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
x = count_vectorizer.fit_transform(train_data['Content'])
svd = TruncatedSVD(n_components=100)
svdmatrix = svd.fit_transform(x)

clf = MultinomialNB()
clf.fit(x,y)
scores1 = cross_val_score(clf, x,y,cv=10,scoring = 'accuracy')
scores1 = scores1.mean()
scores2 = cross_val_score(clf, x,y,cv=10,scoring = 'precision_macro')
scores2 = scores2.mean()
scores3 = cross_val_score(clf, x,y,cv=10,scoring = 'recall_macro')
scores3 = scores3.mean()
scores4 = cross_val_score(clf, x,y,cv=10,scoring = 'f1_macro')
scores4 = scores4.mean()
df.ix['Accuracy','Naive Bayes'] = scores1
df.ix['Precision','Naive Bayes'] = scores2
df.ix['Recall','Naive Bayes'] = scores3
df.ix['F1','Naive Bayes'] = scores4
print df

clf = RandomForestClassifier()
clf.fit(svdmatrix,y)
scores1 = cross_val_score(clf, svdmatrix,y,cv=10,scoring = 'accuracy')
scores1 = scores1.mean()
scores2 = cross_val_score(clf, svdmatrix,y,cv=10,scoring = 'precision_macro')
scores2 = scores2.mean()
scores3 = cross_val_score(clf, svdmatrix,y,cv=10,scoring = 'recall_macro')
scores3 = scores3.mean()
scores4 = cross_val_score(clf, svdmatrix,y,cv=10,scoring = 'f1_macro')
scores4 = scores4.mean()
df.ix['Accuracy','Random Forest'] = scores1
df.ix['Precision','Random Forest'] = scores2
df.ix['Recall','Random Forest'] = scores3
df.ix['F1','Random Forest'] = scores4
print df

clf = svm.SVC()
clf.fit(svdmatrix,y)
scores1 = cross_val_score(clf, svdmatrix,y,cv=10,scoring = 'accuracy')
scores1 = scores1.mean()
scores2 = cross_val_score(clf, svdmatrix,y,cv=10,scoring = 'precision_macro')
scores2 = scores2.mean()
scores3 = cross_val_score(clf, svdmatrix,y,cv=10,scoring = 'recall_macro')
scores3 = scores3.mean()
scores4 = cross_val_score(clf, svdmatrix,y,cv=10,scoring = 'f1_macro')
scores4 = scores4.mean()
df.ix['Accuracy','SVM'] = scores1
df.ix['Precision','SVM'] = scores2
df.ix['Recall','SVM'] = scores3
df.ix['F1','SVM'] = scores4
print df

df.to_csv('EvaluationMetric_10fold.csv')