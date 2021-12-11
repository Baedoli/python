
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
data_path = '/Users/baeseongho/webdriver/corpus.txt'
data = open(data_path, encoding='utf8').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    labels.append(content[0])
    texts.append(" ".join(content[1:]))

# Create data frame for text and label
dataset_df = pd.DataFrame()
dataset_df['text'] = texts
dataset_df['label'] = labels

dataset_df.head()

# creating instance of labelencoder
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

# Assigning numerical values and storing in another column
dataset_df['label'] = labelencoder.fit_transform(dataset_df['label'])
print(dataset_df['label'])

# Map original label to number
integer_mapping = {l: i for i, l in enumerate(labelencoder.classes_)}
print(integer_mapping)

# Split the dataset into training and validation datasets
train_x, test_x, y_train, y_test = model_selection.train_test_split(dataset_df['text'], dataset_df['label'])

# Add stopwords
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

# 신경망 학습  ...

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
tf.keras.backend.clear_session()

# Keras can't work with csr_matrix.
# Convert to a numpy array.
X_train = X_train.toarray()
X_test = X_test.toarray()

input_dim = X_train.shape[1]
# Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=10,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=10)

# Performance
tf.keras.backend.clear_session()

print("Detailed classification report:")
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# Visualize Results
import matplotlib.pyplot as plt

plt.style.use('ggplot')
def plot_history(trian_value, test_vaule, measure):
    x = range(1, len(trian_value) + 1)
    plt.figure(figsize=(5, 5))
    plt.plot(x, trian_value, 'b', label="Train " + measure)
    plt.plot(x, test_vaule, 'r', label="Test "+ measure)
    plt.title(measure)
    plt.legend()

plot_history(history.history['accuracy'], history.history['val_accuracy'], "accuracy")
plot_history(history.history['loss'], history.history['val_loss'], "loss")



