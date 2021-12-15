import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

data_path = '/Users/baeseongho/webdriver/corpus.txt'
data = open(data_path, encoding='utf8').read()
labels, texts = [], []

for i, line in enumerate(data.split("\n")):
    content = line.split()
    labels.append(content[0])
    texts.append(" ".join(content[1:]))

# Create data frame for text and labe

dataset_df = pd.DataFrame()
dataset_df['text'] = texts
dataset_df['label_new'] = labels
dataset_df['label_old'] = labels

dataset_df.head()

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

dataset_df['label_new'] = labelencoder.fit_transform(dataset_df['label_new'])
dataset_df.head(-5)

integer_mapping = {l: i for i, l in enumerate(labelencoder.classes_)}
print(integer_mapping)
