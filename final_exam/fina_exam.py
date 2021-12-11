
# 기말고사 과제 ...
# Import packages for analysis
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics


# Load Data
data_path = '/Users/baeseongho/webdriver/review_list_1.csv'
data = open(data_path, encoding='utf8').read()
labels, texts = [], []

for i, line in enumerate(data.split("\n")):
    content = line.split()
    labels.append(content[0])
    texts.append(" ".join(content[1:]))

# Create data frame for text and label
# text label 을 프레임으로 만든다 ..
dataset_df = pd.DataFrame()
dataset_df['text'] = texts
dataset_df['label'] = labels

dataset_df.head()

print(dataset_df.head())