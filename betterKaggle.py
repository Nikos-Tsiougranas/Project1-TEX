from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn import svm
import numpy as np
import pandas as pd
import csv
import time
import text_processing as tp
from sklearn.feature_extraction.text import TfidfVectorizer as tfv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

t=time.time()
input_file = open("../datasets/train_set.csv")
train_data = pd.read_csv(input_file, header = 0, delimiter = "\t")
#train = train_data[0:4000]

test_set = open("../datasets/test_set.csv")
test_data = pd.read_csv(test_set, header = 0, delimiter = "\t")

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])


count_vectorizer = CountVectorizer(ENGLISH_STOP_WORDS)
x = count_vectorizer.fit_transform(train_data['Content'])
testx = count_vectorizer.transform(test_data['Content'])

svd = TruncatedSVD(n_components=200)
#x = svd.fit_transform(x)
#testsvdx = svd.fit_transform(testx)
clf = svm.SVC(kernel="linear")
print "SVM"

# skf = StratifiedKFold(n_splits=10)
# skf.get_n_splits(x, y)

# maxscore = -1
# for train_index, test_index in skf.split(x, y):
#     x_train, x_test = x[train_index], x[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     clf.fit(x_train,y_train)
#     ypred = clf.predict(x_test)
#     ypred = le.inverse_transform(ypred)
#     score = accuracy_score(y_test,ypred)
#     if(score > maxscore):
#         best_x_train = x_train
#         best_y_train = y_train

# clf.fit(best_x_train,best_y_train)
clf.fit(x,y)
ypred = clf.predict(testx)
ypred = le.inverse_transform(ypred)
results = np.vstack((test_data['Id'], ypred)).T

print results

with open("testSet_categories.csv",'wb') as resultFile:
    wr = csv.writer(resultFile, delimiter=',')
    header = np.vstack(("Id","Category")).T
    wr.writerows(header)
    wr.writerows(results)

print time.time()-t