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
dataset_df['label'] = labels

dataset_df.head()

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column

dataset_df['label'] = labelencoder.fit_transform(dataset_df['label'])
dataset_df.head(-5)
print(dataset_df['label'])
# Map original label to number

integer_mapping = {l: i for i, l in enumerate(labelencoder.classes_)}
print(integer_mapping)

# Split the dataset into training and validation datasets
train_x, test_x, y_train, y_test = model_selection.train_test_split(dataset_df['text'], dataset_df['label'])

from nltk.corpus import stopwords
print(stopwords.words('english')[:10])
count_vect = CountVectorizer(stop_words=stopwords.words('english'))
count_vect.fit(dataset_df['text'])
X_train = count_vect.transform(train_x)
X_test = count_vect.transform(test_x)


names = ["Logistic Regression"]
classifiers = [LogisticRegression(random_state=42, max_iter=5000)]
clf = LogisticRegression()
clf.fit(X_train, y_train)


print("Detailed classification report:")
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)

print(accuracy, precision, recall,f1 )


