
# https://www.lucypark.kr/courses/2015-dm/text-mining.html

import numpy
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
import os
import csv


data_path = '/Users/baeseongho/webdriver/review_list_1.csv'
df = pd.read_csv(data_path)
df.head()

dataset = pd.DataFrame()
dataset["text"]=df["comment"]
dataset["text"].head()

dataset["label_text"]=df["rating"]
dataset["label_text"].head()
dataset["label_text"].unique()

dataset["label"]=df["rating"]
dataset["label"].head()
dataset["label"].unique()

dataset.groupby(["label_text"]).count()

print(dataset.head())
print(dataset.groupby(["label_text"]).count())

dataset["label"] = np.where((dataset.label=="별표 5개 만점에 1개를 받았습니다."),0,dataset.label)
dataset["label"] = np.where((dataset.label=="별표 5개 만점에 2개를 받았습니다."),0,dataset.label)
dataset["label"] = np.where((dataset.label=="별표 5개 만점에 4개를 받았습니다."),1,dataset.label)
dataset["label"] = np.where((dataset.label=="별표 5개 만점에 5개를 받았습니다."),1,dataset.label)

dataset = dataset[dataset['label'] != "별표 5개 만점에 3개를 받았습니다."]
dataset.groupby(['label']).count()

# 라벨 인코딩 ,,
# creating instance of labelencoder
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
dataset["label"] = labelencoder.fit_transform(dataset["label"])
dataset.head(-5)

# Map original label to number
integer_mapping = {l: i for i, l in enumerate(labelencoder.classes_)}
print(integer_mapping)

# Split the dataset into training and validation datasets
train_x, test_x, y_train, y_test = model_selection.train_test_split(dataset['text'], dataset['label'])

count_vect = CountVectorizer()
count_vect.fit(dataset['text'])
X_train = count_vect.transform(train_x)
X_test = count_vect.transform(test_x)

#modeling
names = ["Logistic Regression", "Decision Tree", "KNeighborsClassifier",
         "RandomForestClassifier", "BernoulliNB", "SVM"]

classifiers = [LogisticRegression(random_state=42, max_iter=1000),
               DecisionTreeClassifier(random_state=42),
               KNeighborsClassifier(),
               RandomForestClassifier(random_state=42),
               BernoulliNB(),
               SVC(random_state=42),
               MLPClassifier(random_state=42)]

performances = []

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_predict)
    print("Accuracy:", acc)
    precision = metrics.precision_score(y_test, y_predict)
    print("Precision:", precision)
    recall = metrics.recall_score(y_test, y_predict)
    print("Recall:", precision)
    f1 = metrics.f1_score(y_test, y_predict)
    print("F1-score:", precision)
    performances.append([name,acc, precision, recall,f1])

with open(os.getcwd()+'/count_vec_result.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['algorithm','accuracy', 'precision', 'recall', 'f1-score', ])
    for per in performances:
        csv_out.writerow(per)








