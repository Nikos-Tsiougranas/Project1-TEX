from wordcloud import WordCloud, STOPWORDS
import os
from PIL import Image
import numpy as np
import pandas as pd
import nltk
currdir=os.path.dirname(__file__)

def create_wordcloud(text,image):
    mask=np.array(Image.open(os.path.join(currdir,"mask.png")))
    stopwords = list(STOPWORDS)
    stopwords.extend(['will','never','make','one','say','says','many','much','said','enough','although','among','see','still','come','set','good','may'])
    wc=WordCloud(background_color="white",mask=mask,stopwords=stopwords)
    wc.generate(text)
    wc.to_file(os.path.join(currdir,image))

topiclist=[]
textlist=[]
for x in range(5):
    textlist.append("")
df = pd.read_csv("train_set.csv", header = 0, delimiter = "\t")
df=df.values.tolist()
for d in df:
    if d[4] not in topiclist:
        topiclist.append(d[4])
    for x in range(0,len(topiclist)):
        if d[4] == topiclist[x]:
            textlist[x]+=" "+d[3]
            break
create_wordcloud(textlist[0],"wordcloud1.png")
create_wordcloud(textlist[1],"wordcloud2.png")
create_wordcloud(textlist[2],"wordcloud3.png")
create_wordcloud(textlist[3],"wordcloud4.png")
create_wordcloud(textlist[4],"wordcloud5.png")