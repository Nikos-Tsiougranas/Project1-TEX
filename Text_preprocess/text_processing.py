import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
ps = PorterStemmer()

def stemming(train_data):
    for i in range(0,len(train_data)):
        train_data[i][3]= re.sub(r'[^\x00-\x7F]',' ', train_data[i][3])
        train_data[i][3]= [ps.stem(word) for word in train_data[i][3].split( )]
        train_data[i][3]= ' '.join(train_data[i][3])
        train_data[i][3]=train_data[i][3].encode('ascii')
    return train_data

def lemmatization(train_data):
    for i in range(0,len(train_data)):
        train_data[i][3]= re.sub(r'[^\x00-\x7F]',' ', train_data[i][3])
        train_data[i][3]= [lmtzr.lemmatize(word) for word in train_data[i][3].split( )]
        train_data[i][3]= ' '.join(train_data[i][3])
        train_data[i][3]=train_data[i][3].encode('ascii')
    return train_data
    
def lemstopwords():
    stopwords=list(ENGLISH_STOP_WORDS)
    stopwords.extend(['will','never','make','one','say','says','many','much','said','enough','although','among','see','still','come','set','good','may'])
    stopwords= [lmtzr.lemmatize(word) for word in stopwords]
    return stopwords

def stemstopwords():
    stopwords=list(ENGLISH_STOP_WORDS)
    stopwords.extend(['will','never','make','one','say','says','many','much','said','enough','although','among','see','still','come','set','good','may'])
    stopwords= [ps.stem(word) for word in stopwords]
    return stopwords
