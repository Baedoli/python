import numpy
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

data_path = '/Users/baeseongho/webdriver/review_list_1.csv'
df = pd.read_csv(data_path)
df.shape
df.columns
df.head()

dataset = pd.DataFrame()

dataset["text"]=df["comment"]
dataset["text"].head()

dataset["y"]=df["rating"]
dataset["y"].head()
dataset["y"].unique()

dataset.groupby(["y"]).count()

dataset[y] = np.where((dataset.y=="별표 5개 만점에 1개를 받았습니다."),0,dataset.y)
dataset[y] = np.where((dataset.y=="별표 5개 만점에 2개를 받았습니다."),0,dataset.y)
dataset[y] = np.where((dataset.y=="별표 5개 만점에 4개를 받았습니다."),1,dataset.y)
dataset[y] = np.where((dataset.y=="별표 5개 만점에 5개를 받았습니다."),1,dataset.y)

dataset_filterd = dataset[dataset['y'] != "별표 5개 만점에 3개를 받았습니다."]

dataset_filterd.groupby(['y']).count()

print(dataset_filterd.groupby(['y']).count())

# 라벨 인코딩 ,,
# creating instance of labelencoder
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
dataset_df['y'] = labelencoder.fit_transform(dataset_df['y'])
dataset_filterd.head(-5)
# Map original label to number
integer_mapping = {l: i for i, l in enumerate(labelencoder.classes_)}
print(integer_mapping)

# Split the dataset into training and validation datasets
train_x, test_x, y_train, y_test = model_selection.train_test_split(dataset_df['text'], dataset_df['y'])

# Add stopwords
# stopword 처리 ..
# 여기서 한글 처리 해결 해야 함. vect 처리 말고 다른 방향으로 생각 해야 함.
count_vect = CountVectorizer()
count_vect.fit(dataset_df['text'])
X_train = count_vect.transform(train_x)
X_test = count_vect.transform(test_x)

#Modeling ...
#names = ["Logistic Regression"]
# 예제 copy 후 max_iter 값을 지정 하지 않으면 오류가 난다...
#classifiers = [LogisticRegression(max_iter=200)]
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