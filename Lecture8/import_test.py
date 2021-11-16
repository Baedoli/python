
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
