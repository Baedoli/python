
# import packages for analysis
import pandas as pd
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier

from nltk.corpus import stopwords

# Text vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# 성과지표
from sklearn import metrics
import csv

def build_classifier_1(model, x_train, x_test, y_train, y_test, name, method):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Algorithm Name:", name, "Method:", method)
    # print("Accuracy:", clf.best_score_)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, pos_label="__label__1")
    recall = metrics.recall_score(y_test, y_pred, pos_label="__label__1")
    f1 = metrics.f1_score(y_test, y_pred, pos_label="__label__1")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:",recall)
    print("f1-score:", f1)
    return [name, accuracy, precision, recall, f1, method]

#Load Data
data_path ="/Users/baeseongho/webdriver/corpus.txt"
data = open(data_path, encoding='utf8').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    labels.append(content[0])
    texts.append(" ".join(content[1:]))

#print(labels[11])
#print(texts[10])

# 판다스
trainDF = pd.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

# Split the dataset into training and validation datasets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF["text"])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)

# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)

#print(tfidf_vect_ngram.vocabulary_)

import nltk
#nltk.download()
stopwords.words('english')[:10]

#modeling
names = ["Logistic Regression", "KNeighborsClassifier",
         "RandomForestClassifier", "BernoulliNB", "SVM"]

classifiers = [LogisticRegression(random_state=42, max_iter=1000),
               KNeighborsClassifier(),
               RandomForestClassifier(random_state=42),
               BernoulliNB(),
               SVC(random_state=42),
               MLPClassifier(random_state=42)]


def build_classifier(classifiers,train_x, test_x, train_y, test_y, names):
    for name, clf in zip(names, classifiers):
        clf.fit(xtrain_tfidf, train_y)
        y_pred = clf.predict(xvalid_tfidf)
        accuracy = metrics.accuracy_score(valid_y,y_pred)
        print(name, " accuracy :",accuracy)

build_classifier(classifiers,xtrain_tfidf,xvalid_tfidf,train_y,valid_y,names)
build_classifier(classifiers,xtrain_tfidf_ngram,xvalid_tfidf_ngram,train_y,valid_y,names)


#performances = []
#for name, model in zip(names, classifiers):
#    rs1 = build_classifier(model, xtrain_tfidf, xvalid_tfidf, train_y, valid_y, name, "tf-idf")
#    performances.append(rs1)
#    rs2 = build_classifier(model, xtrain_tfidf_ngram, xvalid_tfidf_ngram, train_y, valid_y, name, 'tf-idf-ngram')
#   performances.append(rs2)

names = ["SVC"]

classifiers = [SVC(random_state=42)]

# Set the parameters for parameter optimization
tuned_parameters = [{"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},{"kernel": ["linear"], "C": [1, 10, 100, 1000]}]

# Set score option
scores = ["f1"]
#scores = {"accuracy","f1"}

for score in scores:
    #clf = model_selection.GridSearchCV(model, tuned_parameters, scoring="%s_macro" % score)
    # clf = GridSearchCV(SVC(random_state=42), cv=5, tuned_parameters, scoring="%s_macro" % score)
    clf = model_selection.GridSearchCV(SVC(random_state=42), tuned_parameters, scoring="%s_macro" % score)
    clf.fit(xtrain_tfidf,train_y)
    print("Best Parameter :",clf.best_params_)
    y_pred = clf.predict(xvalid_tfidf)
    accuarcy = metrics.accuracy_score(valid_y,y_pred)
    print("accuarcy :",accuarcy)






