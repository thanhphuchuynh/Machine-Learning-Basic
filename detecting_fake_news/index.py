import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("news.csv")
print(df.shape)
print(df.head(5))

labels = df.label

print(labels.head(5))

X_train, X_test, y_train, y_test  = train_test_split(
        df['text'],labels,test_size=0.2, random_state=2)
print(X_train, X_test, y_train, y_test)

vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)

tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

print(tfidf_train, tfidf_test)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train,y_train)
y_pred = model.predict(tfidf_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: ",accuracy)
print("Confusion_matrix: ",confusion_matrix(y_test, y_pred,labels=['FAKE','REAL']))

