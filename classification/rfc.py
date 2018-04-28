from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.scorer import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn import preprocessing
from sklearn import svm
import numpy as np
import pandas as pd

input_file = open("datasets/train_set.csv")
train_data = pd.read_csv(input_file, header = 0, delimiter = "\t")
#train_data = train_data[0:500]
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])
count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
x = count_vectorizer.fit_transform(train_data['Content'])
svd = TruncatedSVD(n_components=100)
svdmatrix = svd.fit_transform(x)
clf = RandomForestClassifier()
clf.fit(svdmatrix,y)
scores = cross_val_score(clf, svdmatrix,y,cv=10,scoring = 'accuracy')
print "Accuracy ", scores.mean()
scores = cross_val_score(clf, svdmatrix,y,cv=10,scoring = 'precision_macro')
print "Precision ", scores.mean()
scores = cross_val_score(clf, svdmatrix,y,cv=10,scoring = 'recall_macro')
print "Recall ",scores.mean()
scores = cross_val_score(clf, svdmatrix,y,cv=10,scoring = 'f1_macro')
print "F1 ",scores.mean()
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#print scores.mean()