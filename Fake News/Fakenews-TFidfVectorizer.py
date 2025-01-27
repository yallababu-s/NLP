import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt


df=pd.read_csv('news.csv', index_col=None)
df


dataset=df.drop("Unnamed: 0",axis=1)
dataset

y=dataset["label"]

X_train, X_test, y_train, y_test = train_test_split(dataset['text'], y, test_size=0.33, random_state=53)

Tfidf_Vectorizer =TfidfVectorizer(stop_words='english')
Tfidf_train = Tfidf_Vectorizer.fit_transform(X_train)
print(Tfidf_train)
Tfidf_test =Tfidf_Vectorizer.transform(X_test)


len(Tfidf_Vectorizer.get_feature_names_out())

print(Tfidf_train.toarray())

clf = MultinomialNB()

clf.fit(Tfidf_train, y_train)
pred = clf.predict(Tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)

cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])


from sklearn.metrics import classification_report

report=classification_report(y_test, pred)


print(report)

dataset["text"][0]

