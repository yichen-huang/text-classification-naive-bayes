import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()

# that what you have inside data
categories = ['alt.atheism', 'comp.graphics'
                , 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'
                , 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale'
                , 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball'
                , 'rec.sport.hockey', 'sci.crypt', 'sci.electronics'
                , 'sci.med', 'sci.space', 'soc.religion.christian'
                , 'talk.politics.guns', 'talk.politics.mideast'
                , 'talk.politics.misc', 'talk.religion.misc']

# train.data and test.data have all articles
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
#print(test.data[5])

# from tokenizing to training
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)

labels = model.predict(test.data)

# creating the confusion matrix and heat map
# for vizualisation
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False
                , xticklabels=train.target_names
                , yticklabels=train.target_names)

plt.xlabel('true labels')
plt.ylabel('predicted label')

# predicting category on new data based on trained model
def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

(predict_category('Jesus Christ'))