from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import svm
import numpy as np
import pandas as pd
import time
from collections import Counter
from sklearn.neighbors.base import \
    _check_weights, _get_weights, \
    NeighborsBase, KNeighborsMixin,\
    RadiusNeighborsMixin, SupervisedIntegerMixin
from sklearn.base import ClassifierMixin

class KNN(NeighborsBase, KNeighborsMixin,SupervisedIntegerMixin, ClassifierMixin):
    def __init__(self,n_neighbors=5, **kwargs):
        self.n_neighbors=n_neighbors

    def fit(self,train_set,categories):
        self.train_set=train_set
        self.categories=categories
        
    def predict(self,test_set):
        t=time.clock()
        test_categories=list()
        for doc in test_set:
            y=time.clock()
            x=list()
            for traindoc in self.train_set:
                x.append(np.linalg.norm(doc-traindoc))
            smallest=np.argpartition(x, self.n_neighbors)[:self.n_neighbors]
            categories=list()
            for small in smallest:
                categories.append(self.categories[small])
            counter=Counter(categories)
            mostcommon=counter.most_common(1)
            test_categories.append(mostcommon[0][0])
        return test_categories



input_file = open("datasets/train_set.csv")
train_data = pd.read_csv(input_file, header = 0, delimiter = "\t")
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])
count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
x = count_vectorizer.fit_transform(train_data['Content'])
svd = TruncatedSVD(n_components=10)
svdmatrix = svd.fit_transform(x)
clf = KNN(20)
t=time.time()
scores = cross_val_score(clf, svdmatrix,y,cv=10)
print time.time()-t
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))