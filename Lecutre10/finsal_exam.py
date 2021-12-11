
###
# Using multiple classifiers for classification
###

# Import packages for analysis
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

#Load Data
# 여기서 과제 파일 로딩 ..
data_path = '/Users/baeseongho/webdriver/corpus.txt'
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

# 라벨 인코딩 ,,
# creating instance of labelencoder
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
dataset_df['label'] = labelencoder.fit_transform(dataset_df['label'])
print(dataset_df['label'])
# Map original label to number
# 0 과 1로 매핑 .. ( 평점이 3점이상 1 평점이 2점 이하 0 으로 매핑 ... )
integer_mapping = {l: i for i, l in enumerate(labelencoder.classes_)}
print(integer_mapping)

# Split the dataset into training and validation datasets
train_x, test_x, y_train, y_test = model_selection.train_test_split(dataset_df['text'], dataset_df['label'])

# Add stopwords
# stopword 처리 ..
from nltk.corpus import stopwords
print(stopwords.words('english')[:10])
count_vect = CountVectorizer(stop_words=stopwords.words('english'))
count_vect.fit(dataset_df['text'])
X_train = count_vect.transform(train_x)
X_test = count_vect.transform(test_x)

#Modeling ...
names = ["Logistic Regression"]
# 예제 copy 후 max_iter 값을 지정 하지 않으면 오류가 난다...
classifiers = [LogisticRegression(random_state=42, max_iter=200)]
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Permormance ..
print("Detailed classification report:")
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)

print(accuracy)





