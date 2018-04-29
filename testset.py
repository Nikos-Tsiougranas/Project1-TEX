from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn import svm
import numpy as np
import pandas as pd
import csv

train_set = open("datasets/train_set.csv")
test_set = open("datasets/test_set.csv")
train_data = pd.read_csv(train_set, header = 0, delimiter = "\t")
test_data = pd.read_csv(test_set, header = 0, delimiter = "\t")
train_data = train_data[0:500]
test_data = test_data[0:10]
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])
count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
x = count_vectorizer.fit_transform(train_data['Content'])
testx = count_vectorizer.fit_transform(test_data['Content'])
svd = TruncatedSVD(n_components=10)
svdmatrix = svd.fit_transform(x)
testsvdx = svd.fit_transform(testx)

clf = RandomForestClassifier()
clf.fit(svdmatrix,y)
ypred = clf.predict(testsvdx)
ypred = le.inverse_transform(ypred)
results = np.vstack((test_data['Id'], ypred)).T

print results

first = 0
with open("testSet_categories.csv",'wb') as resultFile:
    wr = csv.writer(resultFile, delimiter='\t')
    header = np.vstack(("Id","Category")).T
    wr.writerows(header)
    wr.writerows(results)