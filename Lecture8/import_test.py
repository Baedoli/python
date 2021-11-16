
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import metrics
import csv

data_path ="/Users/baeseongho/webdriver/corpus.txt"
data = open(data_path, encoding='utf8').read()

labels, texts = [], []

labels = []
texts = []

rs = data.split("\n")
print(rs[0])

for i, line in enumerate(data.split("\n")):
    content = line.split()
    labels.append(content[0])
    texts.append(" ".join(content[1:]))

print(labels[0])
print(texts[0])

trainDF = pd.DataFrame()
trainDF["texts"] = texts
trainDF["labels"] = labels

trainDF.head(10)
trainDF.shape

X_train, X_test, y_train, y_test = model_selection.train_test_split(trainDF["texts"],
                                                                    trainDF["labels"],
                                                                    test_size=0.25,
                                                                    random_state=42)

print(X_train.shape)
print(X_test.shape)

'''
Feature 생성
'''
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(X_train)
xtrain_count =  count_vect.transform(X_train)
xvalid_count =  count_vect.transform(X_test)


'''
Modeling
'''
from sklearn.tree import DecisionTreeClassifier

names = ["Logistic Regression", "Decision Tree"]
classifiers = [LogisticRegression(random_state=42, max_iter=500),
               DecisionTreeClassifier(random_state=42)]

performances = []

for name, clf in zip(names, classifiers):
    clf.fit(xtrain_count, y_train)
    y_predict = clf.predict(xvalid_count)
    acc = metrics.accuracy_score(y_test, y_predict)
    print("Accuracy:", acc)
    precision = metrics.precision_score(y_test, y_predict, pos_label="__label__1")
    print("Precision:", precision)
    recall = metrics.recall_score(y_test, y_predict, pos_label="__label__1")
    print("Recall:", precision)
    f1 = metrics.f1_score(y_test, y_predict, pos_label="__label__1")
    print("F1-score:", precision)
    performances.append([name,acc, precision, recall,f1])


 print(performances)


 import os
 print(os.getcwd())

with open(os.getcwd()+'/count_vec_result.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['algorithm','accuracy', 'precision', 'recall', 'f1-score', ])
    for per in performances:
        csv_out.writerow(per)