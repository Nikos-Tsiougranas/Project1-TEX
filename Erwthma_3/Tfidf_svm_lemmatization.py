from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn import svm
import KNN
import time
from sklearn.neighbors import KNeighborsClassifier
import text_processing as tp
from sklearn.feature_extraction.text import TfidfVectorizer as tfv
def scoring(clf,x,y):
    scores = cross_val_score(clf, x,y,cv=10,scoring = 'accuracy')
    print "Accuracy ", scores.mean()
    scores = cross_val_score(clf, x,y,cv=10,scoring = 'precision_macro')
    print "Precision ", scores.mean()
    scores = cross_val_score(clf, x,y,cv=10,scoring = 'recall_macro')
    print "Recall ",scores.mean()
    scores = cross_val_score(clf, x,y,cv=10,scoring = 'f1_macro')
    print "F1 ",scores.mean()

t=time.time()
input_file = open("datasets/train_set.csv")
train_data = pd.read_csv(input_file, header = 0, delimiter = "\t")
train_data=tp.lemmatization(train_data)
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])
tf_vectoriser = tfv(stop_words=lemstopwords())
x = tf_vectoriser.fit_transform(train_data['Content']+3*train_data['Title'])
svd = TruncatedSVD(n_components=200)
x = svd.fit_transform(x)
clf = svm.SVC(kernel='linear')
print "SVM"
scoring(clf,x,y)
print time.time()-t
